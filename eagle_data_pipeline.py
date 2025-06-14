#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eagle_data_pipeline.py - Modified to support staged execution and data splitting
"""

import argparse
import asyncio
import json
import logging
import os
import pathlib
import sys
import subprocess
import shutil
from typing import Dict, List, Optional

import torch
import httpx
from tqdm import tqdm

# We'll import the actual functions rather than copying them
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from build_dataset import build_dataset
from preprocess_data import preprocess_dataset

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


class TeacherFeatureExtractor:
    """Extract logprobs from teacher model via vLLM API."""
    
    def __init__(self, url: str, model_path: str, k: int = 20,
                 concurrency: int = 32, max_tokens: int = 1):
        self.url = url
        self.model_path = model_path
        self.k = k
        self.max_tokens = max_tokens
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self.sem = asyncio.Semaphore(concurrency)
    
    async def extract_features(self, prompts_dir: str, output_dir: str, resume: bool = True):
        """Extract teacher features for all prompts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect prompts
        prompts = []
        for jsonl_file in pathlib.Path(prompts_dir).rglob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if 'turns' in record and record['turns']:
                            prompts.append({
                                'id': record['id'],
                                'text': record['turns'][0]
                            })
                    except json.JSONDecodeError:
                        continue
        
        log.info(f"Extracting teacher features for {len(prompts)} prompts")
        
        # Process prompts
        tasks = [self._fetch_one(p['id'], p['text'], output_dir, resume) for p in prompts]
        
        # Process with progress bar
        completed = 0
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Teacher extraction"):
            try:
                await fut
                completed += 1
            except Exception as e:
                log.error(f"Task failed: {e}")
        
        log.info(f"Successfully processed {completed}/{len(prompts)} prompts")
    
    async def _fetch_one(self, prompt_id: str, text: str, output_dir: str, resume: bool):
        """Fetch features for a single prompt."""
        save_path = pathlib.Path(output_dir) / f"{prompt_id}_teacher.pt"
        
        if resume and save_path.exists():
            return
        
        payload = {
            "model": self.model_path,
            "prompt": text,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }
        
        if self.k > 0:
            payload["logprobs"] = self.k
            payload["echo"] = True
        
        try:
            async with self.sem:
                response = await self.client.post(self.url, json=payload)
                response.raise_for_status()
                data = response.json()
            
            choice = data["choices"][0]
            record = {
                "prompt": text,
                "prompt_id": prompt_id
            }
            
            if "logprobs" in choice:
                lp = choice["logprobs"]
                record.update({
                    "tokens": lp["tokens"],
                    "token_logprobs": lp["token_logprobs"],
                    "top_logprobs": lp.get("top_logprobs", [])
                })
            
            torch.save(record, save_path)
            
        except Exception as e:
            log.error(f"Error fetching features for {prompt_id}: {e}")
            raise
    
    async def close(self):
        await self.client.aclose()


def merge_with_teacher_features(base_dir: str, teacher_dir: str, output_dir: str):
    """
    Merge base model hidden states with teacher logprobs.
    This creates an enhanced dataset with teacher guidance.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_files = list(pathlib.Path(base_dir).glob("*.pt"))
    log.info(f"Found {len(base_files)} base feature files")
    
    merged_count = 0
    missing_teacher = 0
    
    for base_file in tqdm(base_files, desc="Merging with teacher features"):
        prompt_id = base_file.stem
        teacher_file = pathlib.Path(teacher_dir) / f"{prompt_id}_teacher.pt"
        
        try:
            # Load base features
            base_data = torch.load(base_file, map_location='cpu')
            
            # Check if teacher features exist
            if teacher_file.exists():
                teacher_data = torch.load(teacher_file, map_location='cpu')
                
                # Add teacher features to the base data
                base_data['teacher_logprobs'] = teacher_data.get('top_logprobs', [])
                base_data['teacher_tokens'] = teacher_data.get('tokens', [])
                base_data['teacher_token_logprobs'] = teacher_data.get('token_logprobs', [])
            else:
                missing_teacher += 1
            
            # Save merged file
            output_file = output_path / f"{prompt_id}.pt"
            torch.save(base_data, output_file)
            merged_count += 1
            
        except Exception as e:
            log.error(f"Error processing {prompt_id}: {e}")
    
    log.info(f"Merged {merged_count} files. Missing teacher features: {missing_teacher}")


async def run_pipeline(args):
    """Run the complete EAGLE data preparation pipeline - now supports staged execution."""
    
    # Create directory structure
    base_dir = pathlib.Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute based on stage
    if args.stage == "build":
        # Step 1: Build dataset from sources with train/eval split
        log.info("=" * 70)
        log.info("Step 1: Building dataset from sources")
        log.info("=" * 70)
        
        build_dataset(
            spec=args.spec,
            output_root=base_dir / "prompts",
            seed=args.seed,
            cn_weight=args.cn_weight,
            stream=args.stream,
            eval_split_ratio=args.eval_split_ratio
        )
        
    elif args.stage == "extract_base":
        # Step 2: Extract base model hidden states (always required)
        log.info("=" * 70)
        log.info("Step 2: Extracting base model hidden states and logits")
        log.info("=" * 70)
        
        preprocess_dataset(
            model_name=args.base_model,
            prompts_dir=args.prompts_dir,
            output_dir=args.output_dir,
            max_length=args.max_length,
            device=args.device,
            resume=args.resume,
            save_logits=True  # Always save logits for offline KL
        )
        
    elif args.stage == "extract_teacher":
        # Step 3: Extract teacher features (optional)
        if not args.teacher_model or not args.teacher_url:
            log.error("Teacher model and URL required for extract_teacher stage")
            return
            
        log.info("=" * 70)
        log.info("Step 3: Extracting teacher model features")
        log.info("=" * 70)
        
        extractor = TeacherFeatureExtractor(
            url=args.teacher_url,
            model_path=args.teacher_model,
            k=args.teacher_k,
            concurrency=args.concurrency,
            max_tokens=args.teacher_max_tokens
        )
        
        await extractor.extract_features(
            prompts_dir=args.prompts_dir,
            output_dir=args.output_dir,
            resume=args.resume
        )
        
        await extractor.close()
        
    elif args.stage == "merge":
        # Step 4: Merge features
        log.info("=" * 70)
        log.info("Step 4: Merging base and teacher features")
        log.info("=" * 70)
        
        merge_with_teacher_features(
            base_dir=args.base_dir,
            teacher_dir=args.teacher_dir,
            output_dir=args.output_dir
        )
        
    else:
        log.error(f"Unknown stage: {args.stage}")
        sys.exit(1)
    
    log.info("Stage completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Complete EAGLE training data preparation pipeline - staged execution",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Stage control
    parser.add_argument("--stage", required=True,
                       choices=["build", "extract_base", "extract_teacher", "merge"],
                       help="Pipeline stage to execute")
    
    # Dataset building arguments
    parser.add_argument("--spec",
                       help="Dataset specification, e.g., 'dataset1:1000,dataset2:2000'")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for current stage")
    parser.add_argument("--seed", type=int, default=2025,
                       help="Random seed for sampling")
    parser.add_argument("--cn-weight", type=float, default=1.0,
                       help="Up-weight Chinese samples (>1.0)")
    parser.add_argument("--stream", action="store_true",
                       help="Use HuggingFace streaming mode")
    parser.add_argument("--eval-split-ratio", type=float, default=0.0,
                       help="Ratio of data to use for evaluation")
    
    # Base model arguments (for extract_base stage)
    parser.add_argument("--base-model",
                       help="Base model name or path (e.g., Qwen/Qwen2-7B-Instruct)")
    parser.add_argument("--prompts-dir",
                       help="Directory containing .jsonl files")
    parser.add_argument("--max-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--device", default="auto",
                       help="Device placement for base model")
    
    # Teacher model arguments (for extract_teacher stage)
    parser.add_argument("--teacher-model",
                       help="Teacher model path for vLLM server")
    parser.add_argument("--teacher-url",
                       default="http://localhost:8500/v1/completions",
                       help="vLLM server URL for teacher model")
    parser.add_argument("--teacher-k", type=int, default=20,
                       help="Top-k logprobs to extract from teacher")
    parser.add_argument("--teacher-max-tokens", type=int, default=1,
                       help="Max tokens to generate from teacher")
    parser.add_argument("--concurrency", type=int, default=32,
                       help="Concurrent requests to vLLM server")
    
    # Merge arguments (for merge stage)
    parser.add_argument("--base-dir",
                       help="Directory with base features")
    parser.add_argument("--teacher-dir",
                       help="Directory with teacher features")
    
    # Other arguments
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing files")
    
    args = parser.parse_args()
    
    # Run the pipeline stage
    try:
        if args.stage in ["extract_teacher"]:
            # Need async for teacher extraction
            asyncio.run(run_pipeline(args))
        else:
            # Can run synchronously for other stages
            asyncio.run(run_pipeline(args))
    except KeyboardInterrupt:
        log.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()