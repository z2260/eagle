#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess_data.py - Modified to save target logits for offline KL distillation
"""

# =========================================
# 预处理阶段要拼接的隐藏层个数（训练侧是 3）
FUSE_LAYERS = 3
# =========================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from typing import List, Dict, Union, Tuple
import json
import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def get_fusion_indices(model_name: str, num_hidden_layers: int) -> List[int]:
    """
    Determines the correct layer indices for feature fusion based on the model type.
    This mirrors the adapter logic in train_eagle_v3.py.
    """
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        # Qwen3-specific layer selection
        log.info(f"Using Qwen-specific fusion strategy for {model_name}")
        raw = [2, num_hidden_layers // 3, 2 * num_hidden_layers // 3]
    else:
        # Default strategy for Llama-like models
        log.info(f"Using default Llama-like fusion strategy for {model_name}")
        raw = [num_hidden_layers // 4,
               num_hidden_layers // 2,
               num_hidden_layers - 2]
    
    indices = list(dict.fromkeys(raw))  # 去重并保序
    if max(indices) >= num_hidden_layers:
        raise ValueError("fusion index overflow")
    
    if len(indices) != FUSE_LAYERS:
        raise ValueError(
            f"Expected {FUSE_LAYERS} layer, but obtained {indices}; "
            "Please check if num_hidden_layers is too small or if the index is out of bounds"
        )
    return indices


def fuse_hidden_states(hidden_states_tuple: Tuple[torch.Tensor], indices: List[int]) -> torch.Tensor:
    """
    Fuse hidden states from multiple layers.
    
    Args:
        hidden_states_tuple: Tuple of hidden states from all layers (including embedding)
        indices: List of layer indices to fuse
        
    Returns:
        Fused features of shape [seq_len, 3 * hidden_size]
    """
    # hidden_states_tuple[0] is the embedding layer output
    # hidden_states_tuple[1] to hidden_states_tuple[n] are the transformer layer outputs
    
    selected_states = []
    for idx in indices:
        # Add 1 because hidden_states[0] is embedding layer
        layer_hidden = hidden_states_tuple[idx + 1]
        # Remove batch dimension if present
        if layer_hidden.dim() == 3:
            layer_hidden = layer_hidden[0]  # [seq_len, hidden_size]
        selected_states.append(layer_hidden)
    
    # Concatenate along feature dimension
    fused = torch.cat(selected_states, dim=-1)  # [seq_len, 3 * hidden_size]
    exp_dim = selected_states[0].size(-1) * FUSE_LAYERS
    assert fused.size(-1) == exp_dim, \
        f"fused dim {fused.size(-1)} ≠ {exp_dim}"
    
    return fused


def iter_jsonl_prompts(prompts_dir: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Iterate through all .jsonl files in the directory and extract prompts.
    Returns list of dicts with 'id' and 'prompt' keys.
    """
    prompts_dir = Path(prompts_dir)
    all_prompts = []
    
    for jsonl_file in sorted(prompts_dir.rglob("*.jsonl")):
        log.info(f"Reading prompts from {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    # Extract prompt from 'turns' field (EAGLE format)
                    if 'turns' in record and record['turns']:
                        prompt_text = record['turns'][0]
                    else:
                        log.warning(f"No 'turns' field in record: {record.get('id', f'line_{line_num}')}")
                        continue
                    
                    # Also get the reference answer if available
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


def preprocess_dataset(
    model_name: str,
    prompts_dir: str,
    output_dir: str,
    max_length: int = 2048,
    device: str = "auto",
    resume: bool = True,
    save_logits: bool = True,  # New parameter
):
    """Extract and fuse hidden states from base model, optionally save logits."""
    
    log.info(f"Loading base model: {model_name}")
    
    # Load model with appropriate dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True  # For models like Qwen
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()  # Set model to evaluation mode
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration
    num_hidden_layers = model.config.num_hidden_layers
    fusion_indices = get_fusion_indices(model_name, num_hidden_layers)
    
    log.info(f"Model has {num_hidden_layers} layers. Using layers {fusion_indices} for fusion.")
    log.info(f"Save logits: {save_logits}")
    
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
                # Format as a conversation turn
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
            
            # Create loss mask (1 for all positions by default)
            loss_mask = torch.ones_like(inputs['input_ids'][0])
            
            # If we have a reference, we might want to only compute loss on the response part
            if reference_text and prompt_text:
                # Find where the response starts (approximate)
                prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
                prompt_len = len(prompt_tokens)
                # Mask out the prompt part
                loss_mask[:prompt_len] = 0
            
            # Prepare data to save
            save_data = {
                'id': prompt_id,
                'input_ids': inputs['input_ids'][0].cpu(),
                'hidden_states': fused_features.cpu(),  # [seq_len, 3 * hidden_size]
                'attention_mask': inputs['attention_mask'][0].cpu(),
                'loss_mask': loss_mask.cpu(),
                'num_fused_layers': FUSE_LAYERS,  # Number of layers fused
            }
            
            # Save logits if requested (for offline KL distillation)
            if save_logits:
                # Convert logits to bfloat16 to save space
                save_data['target_logits'] = outputs.logits[0].to(dtype=torch.bfloat16).cpu()
            
            # Save the processed sample
            torch.save(save_data, save_path)
            
            processed_count += 1
            
        except Exception as e:
            log.error(f"Error processing prompt {prompt_id}: {e}")
            continue
    
    log.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    log.info(f"Output saved to {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess dataset for EAGLE training")
    parser.add_argument("--model-name", required=True, help="Model name or path")
    parser.add_argument("--prompts-dir", required=True, help="Directory containing .jsonl files")
    parser.add_argument("--output-dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--device", default="auto", help="Device placement")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing files")
    parser.add_argument("--no-save-logits", action="store_true", help="Don't save logits")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        model_name=args.model_name,
        prompts_dir=args.prompts_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        device=args.device,
        resume=not args.no_resume,
        save_logits=not args.no_save_logits
    )