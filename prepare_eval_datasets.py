# prepare_eval_datasets.py - Download and prepare evaluation datasets

import os
import json
import requests
from pathlib import Path
from datasets import load_dataset

def prepare_mt_bench(output_path: str):
    """Prepare MT-Bench dataset."""
    # This is a simplified version - you'll need the actual MT-Bench data
    sample_data = [
        {
            "turns": [
                "Explain the concept of quantum entanglement to a 10-year-old.",
                "Now explain it to a physics undergraduate student."
            ]
        },
        {
            "turns": [
                "Write a Python function to find the longest palindrome in a string.",
                "Now optimize it for better time complexity."
            ]
        }
    ]
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

def prepare_gsm8k(output_path: str):
    """Prepare GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main", split="test")
    
    with open(output_path, 'w') as f:
        for item in dataset:
            json_line = json.dumps({
                "question": item["question"],
                "answer": item["answer"]
            })
            f.write(json_line + '\n')

def prepare_humaneval(output_path: str):
    """Prepare HumanEval dataset."""
    dataset = load_dataset("openai_humaneval", split="test")
    
    data = []
    for item in dataset:
        data.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"]
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    output_dir = Path("./eval_datasets")
    output_dir.mkdir(exist_ok=True)
    
    print("Preparing evaluation datasets...")
    
    print("Preparing MT-Bench...")
    prepare_mt_bench(output_dir / "mt_bench.json")
    
    print("Preparing GSM8K...")
    prepare_gsm8k(output_dir / "gsm8k_test.jsonl")
    
    print("Preparing HumanEval...")
    prepare_humaneval(output_dir / "humaneval.json")
    
    print(f"All datasets prepared in {output_dir}")

if __name__ == "__main__":
    main()