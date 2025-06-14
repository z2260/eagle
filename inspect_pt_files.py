#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inspect_pt_files.py
-------------------
A debugging tool to inspect the preprocessed .pt files for EAGLE training.
It decodes input_ids, visualizes the loss mask, and checks tensor stats.
"""

import argparse
import random
from pathlib import Path
import torch
from transformers import AutoTokenizer

def inspect_sample(file_path: Path, tokenizer: AutoTokenizer):
    """Loads a single .pt file and prints its contents in a readable format."""
    
    print("=" * 80)
    print(f"🔎 Inspecting Sample: {file_path.name}")
    print("-" * 80)

    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("=" * 80)
        return

    # --- 1. Basic Info & Sanity Checks ---
    print("📋 METADATA:")
    sample_id = data.get('id', 'N/A')
    num_fused = data.get('num_fused_layers', 'N/A')
    print(f"  - ID: {sample_id}")
    print(f"  - Num Fused Layers: {num_fused}")

    # --- 2. Tensor Shapes ---
    print("\n📏 TENSOR SHAPES:")
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key:<15}: {list(value.shape)}")

    # --- 3. Decoded Text and Loss Mask Visualization ---
    input_ids = data.get('input_ids')
    loss_mask = data.get('loss_mask')
    
    if input_ids is not None:
        print("\n📝 DECODED TEXT & LOSS MASK VISUALIZATION:")
        print("   (Green text with [L] indicates where loss is calculated)\n")
        
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"  Full Decoded Text:\n  ---\n  {full_text}\n  ---\n")

        if loss_mask is not None and loss_mask.shape == input_ids.shape:
            # Token-level visualization
            decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            visualized_text = []
            loss_token_count = 0
            for token, mask_val in zip(decoded_tokens, loss_mask):
                if mask_val == 1:
                    # ANSI escape code for green color
                    visualized_text.append(f"\033[92m{token}[L]\033[0m")
                    loss_token_count += 1
                else:
                    visualized_text.append(token)
            
            # Using .replace(' ', ' ') to visualize spaces
            print("  Token-level Loss Mask:")
            print("  " + " ".join(visualized_text).replace(' ', ' '))
            
            total_tokens = len(input_ids)
            print(f"\n  Loss calculated on {loss_token_count} out of {total_tokens} tokens ({loss_token_count/total_tokens:.2%})")
        else:
            print("  ⚠️ Loss mask not found or shape mismatch.")
    
    # --- 4. Hidden States Statistics ---
    hidden_states = data.get('hidden_states')
    if hidden_states is not None:
        print("\n🔬 HIDDEN STATES STATISTICS:")
        has_nans = torch.isnan(hidden_states).any()
        has_infs = torch.isinf(hidden_states).any()
        
        print(f"  - Dtype: {hidden_states.dtype}")
        print(f"  - Min value: {hidden_states.min():.4f}")
        print(f"  - Max value: {hidden_states.max():.4f}")
        print(f"  - Mean value: {hidden_states.mean():.4f}")
        print(f"  - Std dev: {hidden_states.std():.4f}")
        print(f"  - Contains NaNs: \033[91m{has_nans}\033[0m" if has_nans else f"  - Contains NaNs: {has_nans}")
        print(f"  - Contains Infs: \033[91m{has_infs}\033[0m" if has_infs else f"  - Contains Infs: {has_infs}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect preprocessed .pt files for EAGLE training.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--data_dir", 
        required=True, 
        help="Directory containing the .pt files (e.g., ./eagle_qwen_data/final_features)"
    )
    parser.add_argument(
        "--model_name", 
        required=True, 
        help="Base model name or path used for tokenization (e.g., Qwen/Qwen2-7B-Instruct)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=5, 
        help="Number of random samples to inspect."
    )
    parser.add_argument(
        "--sample_id", 
        type=str, 
        default=None, 
        help="Inspect a specific sample by its ID (e.g., 'openai_gsm8k_0').\nIf provided, --num_samples is ignored."
    )
    
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Error: Data directory not found at {data_path}")
        return

    # Find files to inspect
    if args.sample_id:
        target_file = data_path / f"{args.sample_id}.pt"
        if target_file.exists():
            files_to_inspect = [target_file]
        else:
            print(f"❌ Error: Specific sample file not found: {target_file}")
            return
    else:
        all_files = list(data_path.glob("*.pt"))
        if not all_files:
            print(f"❌ Error: No .pt files found in {data_path}")
            return
        
        num_to_sample = min(args.num_samples, len(all_files))
        files_to_inspect = random.sample(all_files, num_to_sample)

    print(f"\nFound {len(files_to_inspect)} samples to inspect.")
    
    # Inspect each selected file
    for file_path in files_to_inspect:
        inspect_sample(file_path, tokenizer)

if __name__ == "__main__":
    main()