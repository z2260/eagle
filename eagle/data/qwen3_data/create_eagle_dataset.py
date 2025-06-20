#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_eagle_dataset.py

A comprehensive script for creating training datasets for EAGLE models.
This script integrates the complete workflow of pulling data from Hugging Face Hub,
formatting data, and generating hidden_states using base large language models.

Features:
- Multiple dataset format support
- Batch processing for memory efficiency
- Checkpointing and resume capability
- Comprehensive error handling
- Progress tracking and saving
"""

import argparse
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
from datetime import datetime

import torch
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

# --- Logging Configuration ---
def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration with optional file output."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
    )

log = logging.getLogger(__name__)


# --- Constants and Configuration ---
SUPPORTED_DATASET_FORMATS = {
    "alpaca": ["instruction", "input", "output"],
    "dolly": ["instruction", "context", "response"],
    "hh-rlhf": ["chosen", "rejected"],
    "gsm8k": ["question", "answer"],
    "sharegpt": ["conversations"],
    "openassistant": ["messages"],
    "wizardlm": ["conversations"],
    "ultrachat": ["messages"],
}

DEFAULT_BATCH_SIZE = 8
MAX_RETRIES = 3
CHECKPOINT_FREQUENCY = 100


# --- Data Formatting Functions ---

class DatasetFormatter:
    """Handles formatting of various dataset types to unified conversation format."""
    
    @staticmethod
    def detect_format(sample: Dict, dataset_name: str) -> Optional[str]:
        """Detect the format of a dataset sample."""
        dataset_lower = dataset_name.lower()
        
        # Check for specific dataset names
        for format_name in ["alpaca", "dolly", "hh-rlhf", "gsm8k", "sharegpt", "openassistant"]:
            if format_name in dataset_lower:
                return format_name
        
        # Check for format by keys
        sample_keys = set(sample.keys())
        for format_name, required_keys in SUPPORTED_DATASET_FORMATS.items():
            if all(key in sample_keys for key in required_keys[:1]):  # Check at least first key
                return format_name
        
        return None
    
    @staticmethod
    def format_sample(sample: Dict, dataset_name: str, sample_idx: int = 0) -> Optional[Dict[str, Any]]:
        """Format a sample to unified conversation format."""
        formatter = DatasetFormatter()
        format_type = formatter.detect_format(sample, dataset_name)
        
        if not format_type:
            log.warning(f"Unknown format for dataset '{dataset_name}', attempting generic formatting")
            return formatter._format_generic(sample, dataset_name, sample_idx)
        
        format_method = getattr(formatter, f"_format_{format_type}", None)
        if format_method:
            return format_method(sample, dataset_name, sample_idx)
        else:
            return formatter._format_generic(sample, dataset_name, sample_idx)
    
    def _generate_id(self, dataset_name: str, sample_idx: int) -> str:
        """Generate a unique ID for a sample."""
        clean_name = dataset_name.replace('/', '_').replace('-', '_')
        return f"{clean_name}_{sample_idx:06d}"
    
    def _format_alpaca(self, sample: Dict, dataset_name: str, sample_idx: int) -> Optional[Dict]:
        """Format Alpaca-style datasets."""
        try:
            instruction = sample.get("instruction", "").strip()
            input_text = sample.get("input", "").strip()
            output = sample.get("output", "").strip()
            
            if not instruction or not output:
                return None
            
            if input_text:
                human_msg = f"{instruction}\n\nInput: {input_text}"
            else:
                human_msg = instruction
            
            return {
                "id": self._generate_id(dataset_name, sample_idx),
                "conversations": [
                    {"from": "human", "value": human_msg},
                    {"from": "gpt", "value": output}
                ]
            }
        except Exception as e:
            log.debug(f"Error formatting Alpaca sample: {e}")
            return None
    
    def _format_dolly(self, sample: Dict, dataset_name: str, sample_idx: int) -> Optional[Dict]:
        """Format Dolly-style datasets."""
        try:
            instruction = sample.get("instruction", "").strip()
            context = sample.get("context", "").strip()
            response = sample.get("response", "").strip()
            
            if not instruction or not response:
                return None
            
            if context:
                human_msg = f"{instruction}\n\nContext: {context}"
            else:
                human_msg = instruction
            
            return {
                "id": self._generate_id(dataset_name, sample_idx),
                "conversations": [
                    {"from": "human", "value": human_msg},
                    {"from": "gpt", "value": response}
                ]
            }
        except Exception as e:
            log.debug(f"Error formatting Dolly sample: {e}")
            return None
    
    def _format_hh_rlhf(self, sample: Dict, dataset_name: str, sample_idx: int) -> Optional[Dict]:
        """Format HH-RLHF (Anthropic) style datasets."""
        try:
            chosen = sample.get("chosen", "")
            
            # Parse multi-turn conversation
            conversations = []
            turns = chosen.split("\n\nHuman:")
            
            for i, turn in enumerate(turns[1:]):  # Skip first empty split
                if "\n\nAssistant:" in turn:
                    human_part, assistant_part = turn.split("\n\nAssistant:", 1)
                    conversations.append({"from": "human", "value": human_part.strip()})
                    conversations.append({"from": "gpt", "value": assistant_part.strip()})
            
            if not conversations:
                return None
            
            return {
                "id": self._generate_id(dataset_name, sample_idx),
                "conversations": conversations
            }
        except Exception as e:
            log.debug(f"Error formatting HH-RLHF sample: {e}")
            return None
    
    def _format_gsm8k(self, sample: Dict, dataset_name: str, sample_idx: int) -> Optional[Dict]:
        """Format GSM8K math datasets."""
        try:
            question = sample.get("question", "").strip()
            answer = sample.get("answer", "").strip()
            
            if not question or not answer:
                return None
            
            return {
                "id": self._generate_id(dataset_name, sample_idx),
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": answer}
                ]
            }
        except Exception as e:
            log.debug(f"Error formatting GSM8K sample: {e}")
            return None
    
    def _format_sharegpt(self, sample: Dict, dataset_name: str, sample_idx: int) -> Optional[Dict]:
        """Format ShareGPT style datasets."""
        try:
            conversations = sample.get("conversations", [])
            
            if not conversations or len(conversations) < 2:
                return None
            
            # Normalize conversation format
            normalized_convs = []
            for conv in conversations:
                from_role = conv.get("from", "")
                if from_role in ["human", "user"]:
                    from_role = "human"
                elif from_role in ["gpt", "assistant", "bot"]:
                    from_role = "gpt"
                else:
                    continue
                
                value = conv.get("value", "").strip()
                if value:
                    normalized_convs.append({"from": from_role, "value": value})
            
            if len(normalized_convs) < 2:
                return None
            
            return {
                "id": self._generate_id(dataset_name, sample_idx),
                "conversations": normalized_convs
            }
        except Exception as e:
            log.debug(f"Error formatting ShareGPT sample: {e}")
            return None
    
    def _format_generic(self, sample: Dict, dataset_name: str, sample_idx: int) -> Optional[Dict]:
        """Generic formatter for unknown dataset formats."""
        try:
            # Try to find text-like fields
            text_fields = []
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 10:
                    text_fields.append((key, value))
            
            if len(text_fields) >= 2:
                return {
                    "id": self._generate_id(dataset_name, sample_idx),
                    "conversations": [
                        {"from": "human", "value": text_fields[0][1]},
                        {"from": "gpt", "value": text_fields[1][1]}
                    ]
                }
            
            return None
        except Exception as e:
            log.debug(f"Error in generic formatting: {e}")
            return None


# --- Data Processing Functions ---

class DatasetProcessor:
    """Handles the processing of datasets with checkpointing support."""
    
    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(self, data: List[Dict], dataset_name: str, stage: str):
        """Save a checkpoint of processed data."""
        checkpoint_file = self.checkpoint_dir / f"{dataset_name}_{stage}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        log.info(f"Saved checkpoint: {checkpoint_file}")
    
    def load_checkpoint(self, dataset_name: str, stage: str) -> Optional[List[Dict]]:
        """Load a checkpoint if it exists."""
        checkpoint_file = self.checkpoint_dir / f"{dataset_name}_{stage}.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            log.info(f"Loaded checkpoint: {checkpoint_file}")
            return data
        return None
    
    def fetch_and_format_datasets(self, spec: str, seed: int = 42) -> List[Dict]:
        """Fetch and format datasets based on specification."""
        dataset_specs = self._parse_spec(spec)
        
        all_data = []
        formatter = DatasetFormatter()
        
        for spec_item in dataset_specs:
            dataset_id = f"{spec_item['name']}_{spec_item.get('config', 'default')}"
            
            # Check for checkpoint
            checkpoint_data = self.load_checkpoint(dataset_id, "formatted")
            if checkpoint_data:
                all_data.extend(checkpoint_data)
                continue
            
            # Process dataset
            formatted_data = self._process_single_dataset(spec_item, formatter, seed)
            
            # Save checkpoint
            if formatted_data:
                self.save_checkpoint(formatted_data, dataset_id, "formatted")
                all_data.extend(formatted_data)
        
        # Shuffle all data
        random.Random(seed).shuffle(all_data)
        log.info(f"Successfully formatted {len(all_data)} samples in total")
        
        return all_data
    
    def _parse_spec(self, spec: str) -> List[Dict]:
        """Parse dataset specification string."""
        dataset_specs = []
        
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            
            components = part.split(':')
            
            if len(components) == 2:
                name, count = components
                config = None
            elif len(components) == 3:
                name, config, count = components
            else:
                log.error(f"Invalid spec format: '{part}'. Expected 'name:count' or 'name:config:count'")
                continue
            
            try:
                dataset_specs.append({
                    'name': name,
                    'config': config,
                    'count': int(count)
                })
            except ValueError:
                log.error(f"Invalid sample count in spec: '{part}'")
        
        return dataset_specs
    
    def _process_single_dataset(self, spec_item: Dict, formatter: DatasetFormatter, seed: int) -> List[Dict]:
        """Process a single dataset."""
        name = spec_item['name']
        config = spec_item.get('config')
        count = spec_item['count']
        
        log_msg = f"Processing {name}"
        if config:
            log_msg += f" (config: {config})"
        log_msg += f", sampling {count} examples"
        log.info(log_msg)
        
        try:
            # Load dataset
            ds = datasets.load_dataset(name, name=config, split="train")
            num_available = len(ds)
            
            if num_available == 0:
                log.warning(f"Dataset {name} is empty")
                return []
            
            # Sample indices
            indices = list(range(num_available))
            random.Random(seed).shuffle(indices)
            sample_count = min(count, num_available)
            
            # Format samples
            formatted_data = []
            for i in tqdm(range(sample_count), desc=f"Formatting {name}"):
                sample = ds[indices[i]]
                formatted = formatter.format_sample(sample, name, i)
                if formatted:
                    formatted_data.append(formatted)
            
            success_rate = len(formatted_data) / sample_count * 100
            log.info(f"Successfully formatted {len(formatted_data)}/{sample_count} samples ({success_rate:.1f}%)")
            
            if num_available < count:
                log.warning(f"Dataset {name} has only {num_available} samples, less than requested {count}")
            
            return formatted_data
            
        except Exception as e:
            log.error(f"Failed to load dataset {name}: {e}")
            traceback.print_exc()
            return []


# --- Model Processing ---

class ModelProcessor:
    """Handles model loading and hidden state generation with batch processing."""
    
    def __init__(self, model_path: str, batch_size: int = DEFAULT_BATCH_SIZE, 
                 device: Optional[str] = None, dtype: torch.dtype = torch.bfloat16):
        self.model_path = model_path
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the model and tokenizer."""
        log.info(f"Loading model from {self.model_path}")
        log.info(f"Using device: {self.device}")
        
        try:
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            log.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise
    
    def process_dataset(self, raw_data: List[Dict], output_file: str, 
                       resume_from: int = 0, save_frequency: int = CHECKPOINT_FREQUENCY):
        """Process dataset and generate hidden states."""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare for resume
        processed_ids = set()
        if resume_from > 0 and output_path.exists():
            log.info(f"Resuming from index {resume_from}")
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    processed_ids.add(record['id'])
        
        # Process in batches
        total_processed = 0
        total_errors = 0
        
        with open(output_path, 'a' if resume_from > 0 else 'w', encoding='utf-8') as f_out:
            for batch_start in tqdm(range(resume_from, len(raw_data), self.batch_size), 
                                   desc="Processing batches"):
                batch_end = min(batch_start + self.batch_size, len(raw_data))
                batch_data = raw_data[batch_start:batch_end]
                
                # Process batch
                batch_results = self._process_batch(batch_data, processed_ids)
                
                # Write results
                for result in batch_results:
                    if result is not None:
                        f_out.write(json.dumps(result) + "\n")
                        total_processed += 1
                    else:
                        total_errors += 1
                
                # Save checkpoint
                if (batch_end % save_frequency) == 0:
                    f_out.flush()
                    log.info(f"Checkpoint saved at index {batch_end}")
                
                # Clear GPU cache periodically
                if self.device == "cuda" and (batch_end % (save_frequency * 10)) == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        
        log.info(f"Processing complete! Successfully processed {total_processed}/{len(raw_data)} records")
        if total_errors > 0:
            log.warning(f"Failed to process {total_errors} records")
    
    def _process_batch(self, batch_data: List[Dict], processed_ids: set) -> List[Optional[Dict]]:
        """Process a batch of data."""
        results = []
        
        for record in batch_data:
            if record['id'] in processed_ids:
                results.append(None)
                continue
            
            try:
                result = self._process_single_record(record)
                results.append(result)
            except Exception as e:
                log.error(f"Error processing record '{record['id']}': {e}")
                results.append(None)
        
        return results
    
    def _process_single_record(self, record: Dict) -> Dict:
        """Process a single record to generate hidden states."""
        conversations = record["conversations"]
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )
        
        input_ids = inputs.input_ids.to(self.device)
        
        # Generate hidden states
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1].cpu().to(torch.float32)
        
        # Prepare result
        result = {
            "id": record["id"],
            "input_ids": input_ids.cpu().tolist()[0],
            "hidden_states": hidden_states.squeeze(0).tolist(),
            "metadata": {
                "length": len(input_ids[0]),
                "conversations_count": len(conversations)
            }
        }
        
        return result


# --- Main Script ---

def main():
    parser = argparse.ArgumentParser(
        description="Build and process training datasets for EAGLE models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python create_eagle_dataset.py --spec "HuggingFaceH4/databricks_dolly_15k:5000" --model-path ./models/qwen3-32b --output-file ./data/eagle_train.jsonl
  
  # Multiple datasets with configs
  python create_eagle_dataset.py --spec "tatsu-lab/alpaca:10000,gsm8k:main:5000,Anthropic/hh-rlhf:helpful-base:3000" --model-path ./models/qwen3-32b --output-file ./data/eagle_train.jsonl
  
  # Resume from checkpoint
  python create_eagle_dataset.py --spec "large_dataset:100000" --model-path ./models/qwen3-32b --output-file ./data/eagle_train.jsonl --resume-from 50000
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--spec",
        required=True,
        help="Dataset specification. Format: 'repo1:samples1,repo2:config:samples2,...'"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the base language model (e.g., Qwen3-32B)"
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to the output JSONL file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for model processing (default: 8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling (default: 42)"
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume processing from a specific index (default: 0)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving checkpoints (default: ./checkpoints)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    # Log configuration
    log.info("="*50)
    log.info("EAGLE Dataset Creation Script")
    log.info("="*50)
    log.info(f"Configuration:")
    for key, value in vars(args).items():
        log.info(f"  {key}: {value}")
    log.info("="*50)
    
    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map[args.dtype]
    
    # Step 1: Fetch and format raw data
    processor = DatasetProcessor(checkpoint_dir=args.checkpoint_dir)
    raw_data = processor.fetch_and_format_datasets(spec=args.spec, seed=args.seed)
    
    if not raw_data:
        log.error("No data was successfully formatted. Exiting.")
        return
    
    # Step 2: Process with model
    model_processor = ModelProcessor(
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype
    )
    
    model_processor.process_dataset(
        raw_data=raw_data,
        output_file=args.output_file,
        resume_from=args.resume_from
    )
    
    log.info("="*50)
    log.info("Dataset creation completed successfully!")
    log.info("="*50)


if __name__ == "__main__":
    main()