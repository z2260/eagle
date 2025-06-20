# diagnose_final.py
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import collections

def diagnose_data_files(data_dir: str, primary_key: str = 'hidden_states'):
    data_path = Path(data_dir)
    if not data_path.is_dir():
        print(f"Error: Directory not found at '{data_dir}'")
        return

    files = list(data_path.rglob("*.pt"))
    if not files:
        print(f"No .pt files found in '{data_dir}'.")
        return

    print(f"Found {len(files)} files. Starting rigorous diagnosis...")

    baseline_dims = 2  # 我们期望 hidden_states 是 2D 张量 [seq_len, hidden_dim]
    baseline_hidden_dim = -1

    corrupted_files = []
    error_reasons = collections.defaultdict(list)

    # 第一次遍历，建立基准hidden_dim
    for f in tqdm(files, desc="Establishing baseline"):
        try:
            data = torch.load(f, map_location='cpu')
            if primary_key in data:
                baseline_hidden_dim = data[primary_key].shape[-1]
                print(f"\nEstablished baseline hidden_dim = {baseline_hidden_dim} from '{f}'")
                break
        except:
            continue
    
    if baseline_hidden_dim == -1:
        print("Could not find a single valid file to establish hidden_dim. Aborting.")
        return

    # 第二次遍历，进行严格检查
    for f in tqdm(files, desc="Rigorously checking files"):
        try:
            data = torch.load(f, map_location='cpu')
            
            if primary_key not in data:
                reason = f"Missing key '{primary_key}'"
                corrupted_files.append(f); error_reasons[f].append(reason); continue
            
            tensor = data[primary_key]

            if tensor.dim() != baseline_dims:
                reason = f"Incorrect number of dimensions. Expected {baseline_dims}, but got {tensor.dim()} with shape {tensor.shape}"
                corrupted_files.append(f); error_reasons[f].append(reason); continue

            if tensor.shape[-1] != baseline_hidden_dim:
                reason = f"Incorrect hidden dimension. Expected {baseline_hidden_dim}, but got {tensor.shape[-1]} with shape {tensor.shape}"
                corrupted_files.append(f); error_reasons[f].append(reason); continue

        except Exception as e:
            reason = f"Failed to load or process file. Error: {e}"
            corrupted_files.append(f); error_reasons[f].append(reason)

    corrupted_files = sorted(list(set(corrupted_files)))

    if corrupted_files:
        print(f"\n--- Diagnosis Complete: Found {len(corrupted_files)} corrupted files! ---")
        for f_path in corrupted_files:
            print(f"\n[File]: {f_path}")
            for reason in error_reasons.get(f_path, []):
                print(f"  - [Reason]: {reason}")
        
        print("\nTo fix this, you MUST delete these corrupted files.")
        delete_command = "rm " + " ".join([f'"{str(f)}"' for f in corrupted_files])
        print("Run this command:\n" + delete_command)
    else:
        print("\n--- Diagnosis Complete. All files passed the rigorous check! ---")