#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_dataset.py - Modified to support data splitting and return data instead of writing directly
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split

import datasets

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def sample_and_format_dataset(
    dataset_name: str,
    num_samples: int,
    config_name: Optional[str] = None,
    seed: int = 2025,
    cn_weight: float = 1.0,
    stream: bool = False
) -> List[Dict[str, Any]]:
    """Sample and format a single dataset, returning data instead of writing."""
    
    # 修改日志，如果存在config，就打印出来
    if config_name:
        log.info(f"Processing {dataset_name} (config: {config_name}) with {num_samples} samples")
    else:
        log.info(f"Processing {dataset_name} with {num_samples} samples")

    # 修改 load_dataset 调用，以支持 config_name
    try:
        ds = datasets.load_dataset(
            dataset_name,
            name=config_name,
            split="train",
            streaming=stream
        )
    except Exception as e:
        log.error(f"Failed to load dataset {dataset_name} with config '{config_name}': {e}")
        return []

    # Sample data
    sampled_data = []
    rng = random.Random(seed)

    if stream:
        # For streaming, we need to iterate
        # Use a simple counter for streaming sampling
        taken_samples = list(ds.take(num_samples))
        for sample in taken_samples:
            formatted = format_sample(sample, dataset_name)
            if formatted:
                sampled_data.append(formatted)
        # Ensure we don't log a warning if the stream has fewer samples than requested
        if len(taken_samples) < num_samples:
             log.warning(f"Stream for {dataset_name} had only {len(taken_samples)} samples, less than the {num_samples} requested.")

    else:
        # For non-streaming, we can sample randomly
        num_available = len(ds)
        if num_available == 0:
            log.warning(f"Dataset {dataset_name} is empty.")
            return []
        
        indices = list(range(num_available))
        rng.shuffle(indices)
        
        for i in range(min(num_samples, num_available)):
            sample = ds[indices[i]]
            formatted = format_sample(sample, dataset_name)
            if formatted:
                sampled_data.append(formatted)
        
        if num_available < num_samples:
            log.warning(f"Dataset {dataset_name} has only {num_available} samples, less than the {num_samples} requested.")


    # Apply Chinese weighting if needed
    if cn_weight > 1.0:
        chinese_samples = [s for s in sampled_data if contains_chinese(s.get("turns", [""])[0])]
        if chinese_samples: # Avoid error if no Chinese samples are found
            extra_cn = int(len(chinese_samples) * (cn_weight - 1))
            if extra_cn > 0:
                extra_samples = rng.choices(chinese_samples, k=extra_cn)
                sampled_data.extend(extra_samples)
                log.info(f"Added {extra_cn} extra Chinese samples due to cn_weight={cn_weight}")

    return sampled_data



def format_sample(sample: Dict, dataset_name: str) -> Optional[Dict[str, Any]]:
    """Format a single sample into EAGLE format."""
    dataset_lower = dataset_name.lower()
    
    try:
        # Create unique ID
        sample_id = f"{dataset_name.replace('/', '_')}_{hash(str(sample)) % 1000000}"
        
        # Format based on dataset type
        if "alpaca" in dataset_lower:
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")
            
            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}"
            else:
                prompt = instruction
                
            return {
                "id": sample_id,
                "turns": [prompt],
                "reference": [output]
            }
            
        elif "gsm8k" in dataset_lower:
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            return {
                "id": sample_id,
                "turns": [question],
                "reference": [answer]
            }
            
        elif "hh-rlhf" in dataset_lower or "anthropic" in dataset_lower:
            chosen = sample.get("chosen", "")
            if "\n\nHuman:" in chosen and "\n\nAssistant:" in chosen:
                parts = chosen.split("\n\nHuman:")[1].split("\n\nAssistant:")
                if len(parts) >= 2:
                    human_msg = parts[0].strip()
                    assistant_msg = parts[1].strip()
                    return {
                        "id": sample_id,
                        "turns": [human_msg],
                        "reference": [assistant_msg]
                    }
                    
        elif "webgpt" in dataset_lower:
            question = sample.get("question", {}).get("full_text", "")
            answer = sample.get("answer_0", "") or sample.get("answer_1", "")
            
            if question and answer:
                return {
                    "id": sample_id,
                    "turns": [question],
                    "reference": [answer]
                }
                
        elif "train_1M_CN" in dataset_name or "belle" in dataset_lower:
            conversations = sample.get("conversations", [])
            if len(conversations) >= 2:
                human_msg = conversations[0].get("value", "")
                assistant_msg = conversations[1].get("value", "")
                return {
                    "id": sample_id,
                    "turns": [human_msg],
                    "reference": [assistant_msg]
                }
                
        else:
            # Generic format
            if "question" in sample and "answer" in sample:
                return {
                    "id": sample_id,
                    "turns": [sample["question"]],
                    "reference": [sample["answer"]]
                }
            elif "prompt" in sample and "response" in sample:
                return {
                    "id": sample_id,
                    "turns": [sample["prompt"]],
                    "reference": [sample["response"]]
                }
                
    except Exception as e:
        log.warning(f"Failed to format sample from {dataset_name}: {e}")
        
    return None


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def build_dataset(
    spec: str,
    output_root: str,
    seed: int = 2025,
    cn_weight: float = 1.0,
    stream: bool = False,
    eval_split_ratio: float = 0.0,
    return_data: bool = False
) -> Optional[Tuple[List[Dict], List[Dict]]]:
    """
    Build dataset from specification.

    Args:
        spec: Dataset specification string
        output_root: Output directory
        seed: Random seed
        cn_weight: Weight for Chinese samples
        stream: Use streaming mode
        eval_split_ratio: Ratio of data to use for evaluation (0.0 means no split)
        return_data: If True, return data instead of writing to disk

    Returns:
        If return_data is True, returns (train_data, eval_data) tuple
    """
    # Parse specification with support for 'repo:config:samples' and 'repo:samples'
    dataset_specs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        
        parts = part.split(':')
        
        name, config, count_str = "", None, ""
        
        if len(parts) == 3:
            # Format: repo_id:config_name:num_samples
            name, config, count_str = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            # Format: repo_id:num_samples
            name, count_str = parts[0], parts[1]
        else:
            log.error(f"Invalid specification part: '{part}'. Expected 'repo:samples' or 'repo:config:samples'.")
            continue
            
        try:
            count = int(count_str)
            dataset_specs.append({'name': name, 'config': config, 'count': count})
        except ValueError:
            log.error(f"Invalid sample count in spec part: '{part}'")
            continue

    log.info(f"Building dataset from {len(dataset_specs)} sources")

    # Collect all data
    all_data = []
    for spec_item in dataset_specs:
        samples = sample_and_format_dataset(
            dataset_name=spec_item['name'],
            num_samples=spec_item['count'],
            config_name=spec_item['config'],
            seed=seed,
            cn_weight=cn_weight,
            stream=stream
        )
        all_data.extend(samples)

    # Shuffle all data
    rng = random.Random(seed)
    rng.shuffle(all_data)

    # Split data if requested
    if eval_split_ratio > 0 and eval_split_ratio < 1.0:
        if len(all_data) > 1:
            train_data, eval_data = train_test_split(
                all_data,
                test_size=eval_split_ratio,
                random_state=seed
            )
            log.info(f"Split data: {len(train_data)} train, {len(eval_data)} eval")
        else:
            log.warning("Not enough data to create a train/eval split. All data will be used for training.")
            train_data = all_data
            eval_data = []
    else:
        train_data = all_data
        eval_data = []

    if return_data:
        return train_data, eval_data

    # Write to disk
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write train data
    if train_data:
        train_dir = output_path / "train"
        train_dir.mkdir(exist_ok=True)
        train_file = train_dir / "data.jsonl"

        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        log.info(f"Wrote {len(train_data)} training samples to {train_file}")

    # Write eval data
    if eval_data:
        eval_dir = output_path / "eval"
        eval_dir.mkdir(exist_ok=True)
        eval_file = eval_dir / "data.jsonl"

        with open(eval_file, 'w', encoding='utf-8') as f:
            for sample in eval_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        log.info(f"Wrote {len(eval_data)} evaluation samples to {eval_file}")

    log.info(f"Dataset building complete. Total samples: {len(all_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, help="Dataset specification")
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--cn-weight", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--eval-split-ratio", type=float, default=0.0)
    
    args = parser.parse_args()
    
    build_dataset(
        spec=args.spec,
        output_root=args.output_root,
        seed=args.seed,
        cn_weight=args.cn_weight,
        stream=args.stream,
        eval_split_ratio=args.eval_split_ratio
    )