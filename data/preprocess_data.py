#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_data.py
"""

import sys
from pathlib import Path
# 添加eagle_v3_refactored到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Union, Tuple
import json
import logging
from tqdm import tqdm

# 从统一的模块导入
from eagle_core import FUSE_LAYERS, TOPK_PROBS, get_fusion_indices, fuse_hidden_states

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def iter_jsonl_prompts(prompts_dir: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Iterate through all .jsonl files in the directory and extract prompts.
    """
    prompts_dir = Path(prompts_dir)
    all_prompts = []
    
    for jsonl_file in sorted(prompts_dir.rglob("*.jsonl")):
        log.info(f"Reading prompts from {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    if 'turns' in record and record['turns']:
                        prompt_text = record['turns'][0]
                    else:
                        log.warning(f"No 'turns' field in record: {record.get('id', f'line_{line_num}')}")
                        continue
                    
                    reference = record.get('reference', [])
                    
                    all_prompts.append({
                        'id': record.get('id', f"{jsonl_file.stem}_{line_num}"),
                        'prompt': prompt_text,
                        'reference': reference[0] if reference else ""
                    })
                except json.JSONDecodeError as e:
                    log.error(f"Error parsing JSON in {jsonl_file} line {line_num}: {e}")
                    continue
    
    return all_prompts


def extract_topk_probs(logits: torch.Tensor, k: int = 20, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract top-k probabilities from logits.
    
    Args:
        logits: Raw logits tensor [seq_len, vocab_size]
        k: Number of top probabilities to keep
        temperature: Temperature for softmax
        
    Returns:
        topk_indices: [seq_len, k] tensor of token indices
        topk_probs: [seq_len, k] tensor of probabilities
    """
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Convert to probabilities
    probs = F.softmax(scaled_logits, dim=-1)
    
    # Get top-k
    topk_probs, topk_indices = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)
    
    return topk_indices, topk_probs


def preprocess_dataset(
    model_name: str,
    prompts_dir: str,
    output_dir: str,
    max_length: int = 2048,
    device: str = "auto",
    resume: bool = True,
    save_topk: int = 20,
    temperature: float = 1.0,
):
    """Extract and fuse hidden states from base model, save top-k probabilities."""
    
    log.info(f"Loading base model: {model_name}")
    log.info(f"Will save top-{save_topk} probabilities (temperature={temperature})")
    
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration
    num_hidden_layers = model.config.num_hidden_layers
    fusion_indices = get_fusion_indices(model_name, num_hidden_layers)
    
    log.info(f"Model has {num_hidden_layers} layers. Using layers {fusion_indices} for fusion.")
    
    # Load all prompts
    prompts = iter_jsonl_prompts(prompts_dir)
    log.info(f"Found {len(prompts)} prompts to process")
    
    # Process prompts
    processed_count = 0
    skipped_count = 0
    
    for prompt_data in tqdm(prompts, desc="Processing prompts"):
        prompt_id = prompt_data['id']
        prompt_text = prompt_data['prompt']
        reference_text = prompt_data.get('reference', '')
        
        # Determine save path
        save_path = output_dir / f"{prompt_id}.pt"
        
        # Skip if already processed and resume is True
        if resume and save_path.exists():
            skipped_count += 1
            continue
        
        try:
            # Combine prompt and reference for full context
            if reference_text:
                full_text = f"{prompt_text}\n\nAssistant: {reference_text}"
            else:
                full_text = prompt_text
            
            # Tokenize
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            # Move to device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Extract features with gradient disabled
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False
                )
                
                # Fuse hidden states from selected layers
                fused_features = fuse_hidden_states(
                    outputs.hidden_states,
                    fusion_indices
                )
                
                # Extract top-k probabilities
                topk_indices, topk_probs = extract_topk_probs(
                    outputs.logits[0], 
                    k=save_topk,
                    temperature=temperature
                )
            
                loss_mask = torch.ones_like(inputs['input_ids'][0])

                # If we have a reference, mask out the prompt part
                if reference_text and prompt_text:
                    # Tokenize prompt WITHOUT special tokens to get correct length
                    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
                    prompt_len = len(prompt_tokens)

                    # Check if BOS token was added
                    full_ids = inputs['input_ids'][0]
                    bos_offset = 0

                    # Detect BOS token
                    if tokenizer.bos_token_id is not None and len(full_ids) > 0:
                        if full_ids[0].item() == tokenizer.bos_token_id:
                            bos_offset = 1
                            log.debug(f"BOS token detected, offset={bos_offset}")

                    # For Qwen models, also check for role tokens
                    if "qwen" in model_name.lower():
                        # Qwen可能添加额外的role标记
                        # 检查实际的prompt在full_ids中的起始位置
                        prompt_start_idx = None
                        for i in range(min(10, len(full_ids) - len(prompt_tokens))):  # 搜索前10个位置
                            if full_ids[i:i+len(prompt_tokens)].tolist() == prompt_tokens:
                                prompt_start_idx = i
                                break
                            
                        if prompt_start_idx is not None:
                            mask_end_idx = prompt_start_idx + len(prompt_tokens)
                            log.debug(f"Found prompt at position {prompt_start_idx}, masking until {mask_end_idx}")
                        else:
                            # 回退到基于BOS的计算
                            mask_end_idx = bos_offset + prompt_len
                            log.warning(f"Could not find exact prompt position, using BOS offset: {mask_end_idx}")
                    else:
                        # 其他模型使用简单的BOS偏移
                        mask_end_idx = bos_offset + prompt_len

                    # Apply mask
                    loss_mask[:mask_end_idx] = 0

                    # Validation
                    valid_tokens = loss_mask.sum().item()
                    total_tokens = len(loss_mask)
                    log.debug(f"Loss mask: {valid_tokens}/{total_tokens} valid tokens")

                    # Assert we have at least some valid tokens
                    if valid_tokens == 0:
                        log.error(f"No valid tokens in loss mask for prompt {prompt_id}")
                        log.error(f"Prompt length: {prompt_len}, Total length: {total_tokens}, Mask end: {mask_end_idx}")
                        # Set at least the last 10% tokens as valid
                        fallback_start = int(total_tokens * 0.9)
                        loss_mask[fallback_start:] = 1
                        log.warning(f"Applied fallback mask from position {fallback_start}")

                assert loss_mask.sum() > 0, f"Loss mask is all zeros for {prompt_id}"
                
                
                # Also don't save teacher probs for masked positions to save space
                # Set top-k probs to zero for masked positions
                topk_probs = topk_probs * loss_mask.unsqueeze(-1)
            
            # Prepare data to save
            save_data = {
                'id': prompt_id,
                'input_ids': inputs['input_ids'][0].cpu(),
                'hidden_states': fused_features.cpu(),
                'attention_mask': inputs['attention_mask'][0].cpu(),
                'loss_mask': loss_mask.cpu(),
                'num_fused_layers': FUSE_LAYERS,
                'teacher_topk_indices': topk_indices.to(torch.int32).cpu(),  # int32 for token ids
                'teacher_topk_probs': topk_probs.to(torch.float16).cpu(),    # float16 for probs
                'teacher_temperature': temperature,
                'teacher_topk': save_topk,
            }
            
            # Save with compression
            torch.save(save_data, save_path, _use_new_zipfile_serialization=True)
            
            processed_count += 1
            
        except Exception as e:
            log.error(f"Error processing prompt {prompt_id}: {e}")
            continue
    
    log.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    log.info(f"Output saved to {output_dir}")
    
    # Estimate disk usage
    if processed_count > 0:
        sample_file = list(output_dir.glob("*.pt"))[0]
        sample_size = sample_file.stat().st_size / (1024 * 1024)  # MB
        total_estimated = sample_size * len(prompts) / 1024  # GB
        log.info(f"Estimated total disk usage: {total_estimated:.2f} GB")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess dataset for EAGLE training")
    parser.add_argument("--model-name", required=True, help="Model name or path")
    parser.add_argument("--prompts-dir", required=True, help="Directory containing .jsonl files")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--device", default="auto", help="Device placement")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing files")
    parser.add_argument("--save-topk", type=int, default=20, help="Number of top probabilities to save")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for teacher distribution")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        model_name=args.model_name,
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        device=args.device,
        resume=not args.no_resume,
        save_topk=args.save_topk,
        temperature=args.temperature
    )