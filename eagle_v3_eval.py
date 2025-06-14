# =============================================================
#  eagle_v3_eval_corrected_debug.py  --  Debug Enhanced EAGLE-3 Evaluation
# =============================================================

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm

from transformers.cache_utils import DynamicCache, Cache

# Import from training script
from train_eagle_v3 import (
    DraftModelV3,
    build_tree_buffers,
    _make_causal_mask,
    FUSE_LAYERS,
    Qwen3ModelAdapter,
    BaseModelAdapter,
    DEFAULT_TREE,
    TOPK,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# For precise GPU timing
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ----------------------------
#    Data Classes
# ----------------------------

@dataclass
class AcceptanceRecord:
    """Record for a single draft-verify cycle."""
    draft_length: int
    accept_length: int
    draft_tokens: List[int]
    accepted_tokens: List[int]
    continuous_accept: bool  # True if all draft tokens were accepted

@dataclass
class EvaluationMetrics:
    """Corrected metrics for EAGLE-3 evaluation."""
    speedup_ratio: float
    avg_accept_len_tau: float  # τ in the paper
    n_alpha_rates: Dict[int, float]  # n-α rates
    tokens_per_second_baseline: float
    tokens_per_second_eagle: float
    latency_ms_baseline: float
    latency_ms_eagle: float
    total_tokens_generated: int
    total_draft_tokens: int
    total_accepted_tokens: int
    total_cycles: int  # Number of draft-verify cycles

@dataclass
class GenerationResult:
    """Result of a single generation."""
    prompt: str
    baseline_output: str
    eagle_output: str
    baseline_time: float
    eagle_time: float
    baseline_tokens: int
    eagle_tokens: int
    acceptance_records: List[AcceptanceRecord] = field(default_factory=list)

# ----------------------------
#    Dataset Loaders (same as before)
# ----------------------------

class EvalDatasetLoader:
    """Load various evaluation datasets."""
    
    @staticmethod
    def load_dataset(dataset_name: str, data_path: str, num_samples: int = 100) -> List[str]:
        """Load evaluation prompts from various datasets."""
        
        if dataset_name == "mt-bench":
            return EvalDatasetLoader._load_mt_bench(data_path, num_samples)
        elif dataset_name == "gsm8k":
            return EvalDatasetLoader._load_gsm8k(data_path, num_samples)
        elif dataset_name == "humaneval":
            return EvalDatasetLoader._load_humaneval(data_path, num_samples)
        elif dataset_name == "alpaca":
            return EvalDatasetLoader._load_alpaca(data_path, num_samples)
        elif dataset_name == "custom":
            return EvalDatasetLoader._load_custom(data_path, num_samples)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def _load_mt_bench(data_path: str, num_samples: int) -> List[str]:
        """Load MT-Bench dataset."""
        prompts = []
        with open(data_path, 'r') as f:
            data = json.load(f)
            for item in data[:num_samples]:
                if isinstance(item, dict) and 'turns' in item:
                    prompts.append(item['turns'][0])
                elif isinstance(item, dict) and 'conversations' in item:
                    prompts.append(item['conversations'][0]['value'])
        return prompts
    
    @staticmethod
    def _load_gsm8k(data_path: str, num_samples: int) -> List[str]:
        """Load GSM8K dataset."""
        prompts = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data = json.loads(line)
                question = data['question']
                prompts.append(f"Question: {question}\nLet's think step by step.")
        return prompts
    
    @staticmethod
    def _load_humaneval(data_path: str, num_samples: int) -> List[str]:
        """Load HumanEval dataset."""
        prompts = []
        with open(data_path, 'r') as f:
            data = json.load(f)
            for item in data[:num_samples]:
                prompt = item['prompt']
                prompts.append(prompt)
        return prompts
    
    @staticmethod
    def _load_alpaca(data_path: str, num_samples: int) -> List[str]:
        """Load Alpaca dataset."""
        prompts = []
        with open(data_path, 'r') as f:
            data = json.load(f)
            for item in data[:num_samples]:
                instruction = item['instruction']
                input_text = item.get('input', '')
                if input_text:
                    prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"
                else:
                    prompt = f"Instruction: {instruction}\nResponse:"
                prompts.append(prompt)
        return prompts
    
    @staticmethod
    def _load_custom(data_path: str, num_samples: int) -> List[str]:
        """Load custom dataset (one prompt per line)."""
        prompts = []
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                prompts.append(line.strip())
        return prompts

# ----------------------------
#    Debug Enhanced EAGLE-3 Inference
# ----------------------------

class EAGLE3InferenceCorrected:
    """Debug enhanced EAGLE-3 speculative decoding implementation."""
    
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        draft_model: DraftModelV3,
        tokenizer: AutoTokenizer,
        adapter: Any,
        device: torch.device,
        tree_choices: List[List[int]] = None,
        max_draft_len: int = 8,
        debug_mode: bool = True,
    ):
        self.base_model = base_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.adapter = adapter
        self.device = device
        self.tree_choices = tree_choices or DEFAULT_TREE
        self.max_draft_len = max_draft_len
        self.debug_mode = debug_mode
        
        # Build tree buffers
        self.tree_buffers = build_tree_buffers(self.tree_choices, device)
        
        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Debug model architecture
        if self.debug_mode:
            self._debug_model_info()
    
    def _debug_model_info(self):
        """Debug model information."""
        logger.info("="*80)
        logger.info("MODEL DEBUG INFORMATION")
        logger.info("="*80)
        
        # Base model info
        logger.info(f"Base model type: {type(self.base_model).__name__}")
        logger.info(f"Base model device: {next(self.base_model.parameters()).device}")
        logger.info(f"Base model dtype: {next(self.base_model.parameters()).dtype}")
        logger.info(f"Base model config: {self.base_model.config}")
        
        # Draft model info
        logger.info(f"Draft model type: {type(self.draft_model).__name__}")
        logger.info(f"Draft model device: {next(self.draft_model.parameters()).device}")
        logger.info(f"Draft model dtype: {next(self.draft_model.parameters()).dtype}")
        
        # Check draft model components
        logger.info("Draft model components:")
        for name, module in self.draft_model.named_children():
            logger.info(f"  - {name}: {type(module).__name__}")
            if hasattr(module, 'config'):
                logger.info(f"    Config: {module.config}")
        
        # Check draft model parameters
        total_params = sum(p.numel() for p in self.draft_model.parameters())
        trainable_params = sum(p.numel() for p in self.draft_model.parameters() if p.requires_grad)
        logger.info(f"Draft model total parameters: {total_params:,}")
        logger.info(f"Draft model trainable parameters: {trainable_params:,}")
        
        # Tokenizer info
        logger.info(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Tokenizer pad token: {self.tokenizer.pad_token_id}")
        logger.info(f"Tokenizer eos token: {self.tokenizer.eos_token_id}")
        
        logger.info("="*80)
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> Tuple[str, List[AcceptanceRecord]]:
        """Generate text using EAGLE-3 speculative decoding with comprehensive debugging."""
        
        if self.debug_mode:
            logger.info(f"\n{'='*60}")
            logger.info(f"STARTING GENERATION")
            logger.info(f"{'='*60}")
            logger.info(f"Prompt: {prompt[:100]}...")
            logger.info(f"Max new tokens: {max_new_tokens}")
            logger.info(f"Temperature: {temperature}")
            logger.info(f"Top-p: {top_p}")
            logger.info(f"Seed: {seed}")
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        if self.debug_mode:
            logger.info(f"Input IDs shape: {input_ids.shape}")
            logger.info(f"Input tokens: {input_ids[0].tolist()}")
            logger.info(f"Decoded input: {self.tokenizer.decode(input_ids[0])}")
        
        # Initialize
        generated_ids = input_ids.clone()
        acceptance_records = []
        
        # Prefill: get initial features and KV cache from base model
        if self.debug_mode:
            logger.info(f"\n--- PREFILL PHASE ---")
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        
        if self.debug_mode:
            logger.info(f"Base model output logits shape: {outputs.logits.shape}")
            logger.info(f"Hidden states layers: {len(outputs.hidden_states)}")
            for i, hs in enumerate(outputs.hidden_states):
                logger.info(f"  Layer {i} hidden state shape: {hs.shape}")
            logger.info(f"Past key values type: {type(past_key_values)}")
            if hasattr(past_key_values, '__len__'):
                logger.info(f"Past key values layers: {len(past_key_values)}")
                if len(past_key_values) > 0:
                    k, v = past_key_values[0]
                    logger.info(f"  First layer KV cache shape: K={k.shape}, V={v.shape}")
        
        # Initialize accumulated hidden states buffer
        all_hidden_states = []
        for layer_hs in outputs.hidden_states:
            # layer_hs.shape: [batch_size, seq_len, hidden_dim]
            all_hidden_states.append(layer_hs.clone())
        
        if self.debug_mode:
            logger.info(f"Initialized hidden states buffer with {len(all_hidden_states)} layers")
        
        # Main generation loop
        position = input_ids.shape[1]
        generation_step = 0
        
        while position < input_ids.shape[1] + max_new_tokens:
            generation_step += 1
            
            if self.debug_mode:
                logger.info(f"\n--- GENERATION STEP {generation_step} (Position: {position}) ---")
            
            # Extract and fuse features from the last generated position
            fused_features = self._fuse_features_from_buffer(all_hidden_states, position - 1)
            
            if self.debug_mode:
                logger.info(f"Fused features shape: {fused_features.shape}")
                logger.info(f"Fused features norm: {fused_features.norm().item():.4f}")
            
            # Draft phase (simplified chain drafting for now)
            draft_tokens, draft_logits = self._draft_tokens_chain(
                fused_features,
                position,
                min(self.max_draft_len, input_ids.shape[1] + max_new_tokens - position)
            )
            
            if self.debug_mode:
                logger.info(f"Draft phase completed:")
                logger.info(f"  Draft tokens: {draft_tokens.tolist() if len(draft_tokens) > 0 else []}")
                logger.info(f"  Draft tokens text: {self.tokenizer.decode(draft_tokens) if len(draft_tokens) > 0 else 'None'}")
                logger.info(f"  Draft length: {len(draft_tokens)}")
                if len(draft_logits) > 0:
                    logger.info(f"  Draft logits shape: {draft_logits.shape}")
                    logger.info(f"  Draft logits norm: {draft_logits.norm(dim=-1).tolist()}")
            
            if len(draft_tokens) == 0:
                if self.debug_mode:
                    logger.info("No draft tokens generated, stopping generation")
                break
            
            # Verification phase with KV cache reuse
            accepted_tokens, new_past_key_values, continuous_accept, new_hidden_states = self._verify_tokens_with_cache(
                generated_ids,
                draft_tokens,
                draft_logits,
                past_key_values,
                position,
                temperature,
                top_p,
            )
            
            if self.debug_mode:
                logger.info(f"Verification phase completed:")
                logger.info(f"  Accepted tokens: {accepted_tokens.tolist()}")
                logger.info(f"  Accepted tokens text: {self.tokenizer.decode(accepted_tokens) if len(accepted_tokens) > 0 else 'None'}")
                logger.info(f"  Accepted length: {len(accepted_tokens)}")
                logger.info(f"  Continuous accept: {continuous_accept}")
                logger.info(f"  Accept rate: {len(accepted_tokens)/len(draft_tokens):.2%}")
            
            # Record acceptance
            record = AcceptanceRecord(
                draft_length=len(draft_tokens),
                accept_length=len(accepted_tokens),
                draft_tokens=draft_tokens.tolist(),
                accepted_tokens=accepted_tokens.tolist(),
                continuous_accept=continuous_accept,
            )
            acceptance_records.append(record)
            
            # Update state
            if len(accepted_tokens) > 0:
                generated_ids = torch.cat([
                    generated_ids, 
                    accepted_tokens.unsqueeze(0)
                ], dim=1)
                position += len(accepted_tokens)
                past_key_values = new_past_key_values
                
                if self.debug_mode:
                    logger.info(f"Updated generated_ids shape: {generated_ids.shape}")
                    logger.info(f"New position: {position}")
                
                # Update accumulated hidden states
                if new_hidden_states is not None:
                    for layer_i, layer_hs in enumerate(new_hidden_states):
                        # Concatenate new hidden states to buffer
                        all_hidden_states[layer_i] = torch.cat(
                            [all_hidden_states[layer_i], layer_hs[:, :len(accepted_tokens), :]],
                            dim=1
                        )
                    
                    if self.debug_mode:
                        logger.info(f"Updated hidden states buffer, new shapes:")
                        for i, hs in enumerate(all_hidden_states[:3]):  # Show first 3 layers
                            logger.info(f"  Layer {i}: {hs.shape}")
                
                # If not all tokens were accepted, need to get new hidden states for the last accepted token
                if len(accepted_tokens) < len(draft_tokens) and new_hidden_states is None:
                    if self.debug_mode:
                        logger.info("Getting hidden states for last accepted token...")
                    
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.base_model(
                            input_ids=generated_ids[:, -1:],
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            use_cache=True,
                            return_dict=True,
                        )
                    # Append the single token hidden state
                    for layer_i, layer_hs in enumerate(outputs.hidden_states):
                        all_hidden_states[layer_i] = torch.cat(
                            [all_hidden_states[layer_i], layer_hs],
                            dim=1
                        )
            else:
                # No tokens accepted - sample one token normally
                if self.debug_mode:
                    logger.info("No tokens accepted, sampling from base model...")
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.base_model(
                        input_ids=generated_ids[:, -1:],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                    )
                
                logits = outputs.logits[:, -1, :]
                next_token = self._sample(logits, temperature, top_p)
                
                if self.debug_mode:
                    logger.info(f"Sampled token: {next_token.item()}")
                    logger.info(f"Sampled token text: {self.tokenizer.decode([next_token.item()])}")
                
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                position += 1
                past_key_values = outputs.past_key_values
                
                # Update accumulated hidden states
                for layer_i, layer_hs in enumerate(outputs.hidden_states):
                    all_hidden_states[layer_i] = torch.cat(
                        [all_hidden_states[layer_i], layer_hs],
                        dim=1
                    )
                
                # Record single token generation
                record = AcceptanceRecord(
                    draft_length=0,
                    accept_length=1,
                    draft_tokens=[],
                    accepted_tokens=[next_token.item()],
                    continuous_accept=True,
                )
                acceptance_records.append(record)
            
            # Check for EOS
            if self.tokenizer.eos_token_id in generated_ids[0, input_ids.shape[1]:]:
                if self.debug_mode:
                    logger.info("EOS token detected, stopping generation")
                break
        
        # Decode output
        output_ids = generated_ids[0, input_ids.shape[1]:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if self.debug_mode:
            logger.info(f"\n--- GENERATION COMPLETED ---")
            logger.info(f"Total generation steps: {generation_step}")
            logger.info(f"Total acceptance records: {len(acceptance_records)}")
            logger.info(f"Output tokens: {len(output_ids)}")
            logger.info(f"Output text: {output_text}")
            
            # Summary statistics
            total_draft = sum(r.draft_length for r in acceptance_records)
            total_accepted = sum(r.accept_length for r in acceptance_records)
            if total_draft > 0:
                overall_accept_rate = total_accepted / total_draft
                logger.info(f"Overall accept rate: {overall_accept_rate:.2%}")
        
        return output_text, acceptance_records
    
    def _fuse_features_from_buffer(self, all_hidden_states: List[torch.Tensor], position: int) -> torch.Tensor:
        """Fuse features from multiple layers at a specific position using the accumulated buffer."""
        n_layers = len(all_hidden_states) - 1  # Exclude embedding layer
        
        # Select layers (low, mid, high)
        low_idx = n_layers // 4
        mid_idx = n_layers // 2
        high_idx = -2  # Second to last
        
        if self.debug_mode:
            logger.info(f"Feature fusion - using layers: {low_idx}, {mid_idx}, {high_idx}")
            logger.info(f"Position: {position}")
            logger.info(f"Buffer sizes: {[hs.shape[1] for hs in all_hidden_states[:3]]}")
        
        # Check bounds
        max_pos = min(hs.shape[1] for hs in all_hidden_states)
        if position >= max_pos:
            if self.debug_mode:
                logger.warning(f"Position {position} >= max buffer size {max_pos}, using last position")
            position = max_pos - 1
        
        # Extract features at the specific position from buffer
        low = all_hidden_states[low_idx][:, position:position+1, :]
        mid = all_hidden_states[mid_idx][:, position:position+1, :]
        high = all_hidden_states[high_idx][:, position:position+1, :]
        
        if self.debug_mode:
            logger.info(f"Extracted features shapes - low: {low.shape}, mid: {mid.shape}, high: {high.shape}")
            logger.info(f"Feature norms - low: {low.norm():.4f}, mid: {mid.norm():.4f}, high: {high.norm():.4f}")
        
        fused = torch.cat([low, mid, high], dim=-1)
        
        if self.debug_mode:
            logger.info(f"Fused features shape: {fused.shape}")
            logger.info(f"Fused features norm: {fused.norm():.4f}")
        
        return fused
    
    def _draft_tokens_chain(
        self,
        fused_features: torch.Tensor,
        start_position: int,
        max_len: int,
        temperature: float = 0.0,  # 添加temperature参数
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate draft tokens using chain drafting WITH proper temperature handling.
        Returns tokens, logits, and trace information.
        """
        
        if self.debug_mode:
            logger.info(f"\n  --- DRAFT PHASE DEBUG (Temperature: {temperature}) ---")
            logger.info(f"  Initial fused features shape: {fused_features.shape}")
            logger.info(f"  Start position: {start_position}, Max draft length: {max_len}")
        
        if max_len <= 0:
            if self.debug_mode:
                logger.warning("  Draft phase skipped: max_len is zero or negative.")
            return torch.tensor([], device=self.device, dtype=torch.long), torch.tensor([], device=self.device), []
    
        draft_tokens = []
        draft_logits = []
        trace_info = []  # 追踪信息
        
        current_features = fused_features
        
        for i in range(max_len):
            if self.debug_mode:
                logger.info(f"    Draft step {i+1}/{max_len}")
    
            position_ids = torch.tensor([[start_position + i]], device=self.device)
            
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.draft_model(
                        current_features,
                        input_ids=None,
                        position_ids=position_ids,
                        is_fused_features=(i == 0)
                    )
                    
                    logits, hidden = outputs
                
                if self.debug_mode:
                    logger.info(f"    Draft model output:")
                    logger.info(f"      Logits shape: {logits.shape}")
                    logger.info(f"      Hidden state output shape: {hidden.shape}")
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.error(f"    NaN or Inf detected in logits! Stopping draft.")
                        break
                    
            except Exception as e:
                logger.error(f"    Error during draft model forward pass: {e}", exc_info=True)
                break
            
            # 修复的采样逻辑 - 明确区分贪婪和采样
            if temperature < 1e-4:  # 贪婪模式
                next_token = logits[0, -1].argmax()
                sampling_method = "greedy"
            else:  # 采样模式
                probs = torch.softmax(logits[0, -1] / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze()
                sampling_method = "sampling"
            
            # 记录trace信息
            trace_step = {
                "draft_step": i,
                "position": start_position + i,
                "draft_token_id": int(next_token),
                "draft_token": self.tokenizer.decode([next_token.item()]),
                "sampling_method": sampling_method,
                "temperature": temperature,
                "logit_max": float(logits[0, -1].max().item()),
                "logit_mean": float(logits[0, -1].mean().item()),
                "logit_std": float(logits[0, -1].std().item()),
            }
            
            # 添加top-5预测信息
            if self.debug_mode:
                probs = F.softmax(logits[0, -1], dim=-1)
                top_k_probs, top_k_indices = torch.topk(probs, 5)
                trace_step["top_5_predictions"] = [
                    {
                        "token_id": int(idx),
                        "token": self.tokenizer.decode([idx.item()]),
                        "prob": float(prob),
                    }
                    for idx, prob in zip(top_k_indices, top_k_probs)
                ]
            
            trace_info.append(trace_step)
            
            if self.debug_mode:
                logger.info(f"    Sampled next token: {next_token.item()} ('{self.tokenizer.decode([next_token.item()])}')")
                logger.info(f"    Sampling method: {sampling_method}")
            
            draft_tokens.append(next_token)
            draft_logits.append(logits[0, -1])
            
            if next_token == self.tokenizer.eos_token_id:
                if self.debug_mode:
                    logger.info(f"    EOS token generated, stopping draft.")
                break
            
            current_features = hidden.detach()
        
        if draft_tokens:
            result_tokens = torch.stack(draft_tokens)
            result_logits = torch.stack(draft_logits)
            
            if self.debug_mode:
                logger.info(f"  Draft phase completed successfully:")
                logger.info(f"    Generated {len(draft_tokens)} tokens.")
                logger.info(f"    Tokens: {result_tokens.tolist()}")
                logger.info(f"    Text: '{self.tokenizer.decode(result_tokens)}'")
            
            return result_tokens, result_logits, trace_info
        else:
            if self.debug_mode:
                logger.warning(f"  Draft phase produced no tokens.")
            return torch.tensor([], device=self.device, dtype=torch.long), torch.tensor([], device=self.device), []
        
    def _verify_tokens_with_cache(
            self,
            prefix_ids: torch.Tensor,
            draft_tokens: torch.Tensor,
            draft_logits: torch.Tensor,
            past_key_values: Any,
            position: int,
            temperature: float,
            top_p: float,
        ) -> Tuple[torch.Tensor, Any, bool, Optional[Tuple[torch.Tensor]]]:
            """Verify draft tokens using base model with KV cache reuse and debugging."""
            
            if self.debug_mode:
                logger.info(f"\n  --- VERIFICATION PHASE DEBUG ---")
                logger.info(f"  Draft tokens to verify: {draft_tokens.tolist()}")
                logger.info(f"  Temperature: {temperature}")
                logger.info(f"  Position: {position}")
            
            if len(draft_tokens) == 0:
                return torch.tensor([], device=self.device, dtype=torch.long), past_key_values, True, None
            
            # Prepare draft sequence for verification
            draft_ids = draft_tokens.unsqueeze(0)
            
            # 生成position_ids - 重要！
            seq_len = draft_ids.shape[1]
            position_ids = torch.arange(
                position, position + seq_len, 
                dtype=torch.long, 
                device=self.device
            ).unsqueeze(0)
            
            if self.debug_mode:
                logger.info(f"  Draft IDs shape: {draft_ids.shape}")
                logger.info(f"  Position IDs: {position_ids}")
            
            # Forward through base model with cached KV
            try:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.base_model(
                        input_ids=draft_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                    )
            except Exception as e:
                logger.error(f"  Error in base model verification: {e}")
                return torch.tensor([], device=self.device, dtype=torch.long), past_key_values, False, None
            
            verify_logits = outputs.logits[0]
            new_past_key_values = outputs.past_key_values
            new_hidden_states = outputs.hidden_states
            
            if self.debug_mode:
                logger.info(f"  Base model verification output:")
                logger.info(f"    Logits shape: {verify_logits.shape}")
                logger.info(f"    Hidden states layers: {len(new_hidden_states)}")
            
            # =========================================================================
            #  关键修复 #1：将新版 Cache 对象转换为旧版元组，以便后续处理
            # =========================================================================
            if isinstance(new_past_key_values, Cache):
                # 将 Cache 对象转换为 (key, value) 元组的列表，以便我们进行迭代和裁剪
                legacy_past_key_values = new_past_key_values.to_legacy_cache()
            else:
                # 如果已经是旧格式，直接使用
                legacy_past_key_values = new_past_key_values
            
            # Verify each token according to speculative sampling algorithm
            accepted_tokens = []
            continuous_accept = True
            
            for i, (draft_token, draft_logit) in enumerate(zip(draft_tokens, draft_logits)):
                if self.debug_mode:
                    logger.info(f"    Verifying token {i+1}: {draft_token.item()} ('{self.tokenizer.decode([draft_token.item()])}')")
                
                # Get probabilities
                if temperature == 0:
                    # Greedy: check if draft token is argmax
                    verify_token = verify_logits[i].argmax()
                    
                    if self.debug_mode:
                        logger.info(f"      Base model predicts: {verify_token.item()} ('{self.tokenizer.decode([verify_token.item()])}')")
                        logger.info(f"      Match: {draft_token == verify_token}")
                    
                    if draft_token == verify_token:
                        accepted_tokens.append(draft_token)
                        if self.debug_mode:
                            logger.info(f"      ✓ ACCEPTED")
                    else:
                        # Reject and resample
                        accepted_tokens.append(verify_token)
                        continuous_accept = False
                        if self.debug_mode:
                            logger.info(f"      ✗ REJECTED, resampled to: {verify_token.item()}")
                        break
                else:
                    # ... (sampling logic remains the same) ...
                    draft_prob = F.softmax(draft_logit / temperature, dim=-1)
                    verify_prob = F.softmax(verify_logits[i] / temperature, dim=-1)
                    
                    if self.debug_mode:
                        logger.info(f"      Draft prob for token: {draft_prob[draft_token]:.6f}")
                        logger.info(f"      Verify prob for token: {verify_prob[draft_token]:.6f}")
                    
                    verify_argmax = verify_prob.argmax()
                    if draft_token == verify_argmax:
                        accepted_tokens.append(draft_token)
                        if self.debug_mode:
                            logger.info(f"      ✓ FAST ACCEPT (argmax match)")
                    else:
                        r = torch.rand(1, device=self.device).item()
                        accept_prob = min(1.0, verify_prob[draft_token] / (draft_prob[draft_token] + 1e-10))
                        
                        if self.debug_mode:
                            logger.info(f"      Random number: {r:.6f}")
                            logger.info(f"      Accept probability: {accept_prob:.6f}")
                        
                        if r < accept_prob:
                            accepted_tokens.append(draft_token)
                            if self.debug_mode:
                                logger.info(f"      ✓ PROBABILISTIC ACCEPT")
                        else:
                            residual = torch.clamp(verify_prob - draft_prob, min=0)
                            residual = residual / residual.sum()
                            
                            new_token = torch.multinomial(residual, 1)
                            accepted_tokens.append(new_token.item())
                            continuous_accept = False
                            
                            if self.debug_mode:
                                logger.info(f"      ✗ REJECTED, rejection sampled to: {new_token.item()}")
                            break
                        
            accepted_tokens = torch.tensor(accepted_tokens, device=self.device, dtype=torch.long)
            
            if self.debug_mode:
                logger.info(f"  Verification completed:")
                logger.info(f"    Accepted tokens: {accepted_tokens.tolist()}")
                logger.info(f"    Accepted text: '{self.tokenizer.decode(accepted_tokens) if len(accepted_tokens) > 0 else ''}'")
                logger.info(f"    Accept rate: {len(accepted_tokens)}/{len(draft_tokens)} = {len(accepted_tokens)/len(draft_tokens):.2%}" if len(draft_tokens)>0 else "N/A")
                logger.info(f"    Continuous accept: {continuous_accept}")
            
            # Trim KV cache to only include accepted tokens
            if len(accepted_tokens) < len(draft_tokens):
                if self.debug_mode:
                    logger.info(f"    Trimming KV cache from {len(draft_tokens)} to {len(accepted_tokens)} tokens")
                
                # 使用我们之前转换的 legacy_past_key_values 进行裁剪
                trimmed_kv = []
                # 注意：这里的 past_key_values 是指 *验证之前* 的缓存，我们需要裁剪的是 *验证之后* 的缓存 `legacy_past_key_values`
                # Qwen3的KV缓存 shape: [batch, num_heads, seq_len, head_dim]
                # 我们只需要裁剪 seq_len 维度
                for layer_kv in legacy_past_key_values:
                    k, v = layer_kv
                    # 裁剪 key 和 value 张量，只保留到被接受的词元的位置
                    k_trimmed = k[:, :, :len(accepted_tokens), :]
                    v_trimmed = v[:, :, :len(accepted_tokens), :]
                    trimmed_kv.append((k_trimmed, v_trimmed))
                
                # =========================================================================
                #  关键修复 #2：将裁剪后的旧版元组转换回新版 DynamicCache 对象
                # =========================================================================
                final_past_key_values = DynamicCache.from_legacy_cache(past_key_values=tuple(trimmed_kv))
                
                return accepted_tokens, final_past_key_values, continuous_accept, None
            else:
                # 所有词元都被接受，直接返回模型输出的 new_past_key_values (它已经是正确的 Cache 对象了)
                if self.debug_mode:
                    logger.info(f"    All tokens accepted, returning hidden states and original new cache")
                return accepted_tokens, new_past_key_values, continuous_accept, new_hidden_states
    
    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample from logits with temperature and top-p."""
        if temperature == 0:
            return logits.argmax(dim=-1)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-p
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find cutoff
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)

# ----------------------------
#    Enhanced Evaluation Functions
# ----------------------------

def precise_cuda_timing(func):
    """Decorator for precise CUDA timing."""
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            
            result = func(*args, **kwargs)
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
        
        return result, elapsed_time
    return wrapper

@torch.no_grad()
def evaluate_baseline(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    batch_size: int = 1,
    seed: int = 42,
    debug_mode: bool = False,
) -> List[GenerationResult]:
    """Evaluate baseline model with precise timing and debugging."""
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Baseline evaluation"):
        batch_prompts = prompts[i:i+batch_size]
        
        if debug_mode:
            logger.info(f"\nBaseline evaluation batch {i//batch_size + 1}")
            logger.info(f"Prompts: {[p[:50] + '...' for p in batch_prompts]}")
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        
        # Generate with timing
        @precise_cuda_timing
        def generate_batch():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            return outputs
        
        outputs, elapsed_time = generate_batch()
        
        if debug_mode:
            logger.info(f"Baseline generation time: {elapsed_time:.3f}s")
        
        # Process results
        for j, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            input_length = inputs.input_ids[j].shape[0]
            output_ids = output[input_length:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            if debug_mode:
                logger.info(f"Sample {j}: {len(output_ids)} tokens generated")
                logger.info(f"Output: {output_text[:100]}...")
            
            result = GenerationResult(
                prompt=prompt,
                baseline_output=output_text,
                eagle_output="",
                baseline_time=elapsed_time / len(batch_prompts),  # Average time per sample
                eagle_time=0,
                baseline_tokens=len(output_ids),
                eagle_tokens=0,
            )
            results.append(result)
    
    return results

@torch.no_grad()
def evaluate_eagle3(
    eagle_inference: EAGLE3InferenceCorrected,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    baseline_results: List[GenerationResult],
    seed: int = 42,
    debug_mode: bool = False,
) -> List[GenerationResult]:
    """Evaluate EAGLE-3 model with corrected metrics and debugging."""
    
    for i, (prompt, baseline_result) in enumerate(tqdm(
        zip(prompts, baseline_results), 
        total=len(prompts),
        desc="EAGLE-3 evaluation"
    )):
        if debug_mode:
            logger.info(f"\n{'='*80}")
            logger.info(f"EAGLE-3 Sample {i+1}/{len(prompts)}")
            logger.info(f"{'='*80}")
        
        # Generate with timing
        @precise_cuda_timing
        def generate_eagle():
            return eagle_inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            )
        
        (output_text, acceptance_records), elapsed_time = generate_eagle()
        
        if debug_mode:
            logger.info(f"EAGLE-3 generation completed in {elapsed_time:.3f}s")
            logger.info(f"Baseline time: {baseline_result.baseline_time:.3f}s")
            logger.info(f"Speedup: {baseline_result.baseline_time/elapsed_time:.2f}x")
        
        # Update result
        baseline_result.eagle_output = output_text
        baseline_result.eagle_time = elapsed_time
        baseline_result.eagle_tokens = len(eagle_inference.tokenizer.encode(output_text))
        baseline_result.acceptance_records = acceptance_records
    
    return baseline_results

def calculate_corrected_metrics(results: List[GenerationResult]) -> EvaluationMetrics:
    """Calculate corrected evaluation metrics matching paper definitions."""
    
    # Timing statistics
    total_baseline_time = sum(r.baseline_time for r in results)
    total_eagle_time = sum(r.eagle_time for r in results)
    
    # Token statistics (using tokenizer, not word count)
    total_baseline_tokens = sum(r.baseline_tokens for r in results)
    total_eagle_tokens = sum(r.eagle_tokens for r in results)
    
    # Speedup ratio
    speedup_ratio = total_baseline_time / total_eagle_time if total_eagle_time > 0 else 0
    
    # Collect all acceptance records
    all_records = []
    for r in results:
        all_records.extend(r.acceptance_records)
    
    # Calculate τ (average accept length per cycle)
    accept_lengths = [rec.accept_length for rec in all_records if rec.draft_length > 0]
    avg_accept_len_tau = np.mean(accept_lengths) if accept_lengths else 0
    
    # Calculate n-α rates (probability of accepting all n+1 tokens when n are from draft)
    n_alpha_rates = {}
    for n in range(8):  # Calculate for n=0 to 7
        # Find all cycles where at least n+1 tokens were drafted
        eligible_records = [rec for rec in all_records if rec.draft_length >= n + 1]
        
        if eligible_records:
            # Count how many accepted at least n+1 tokens continuously
            continuous_accepts = [
                rec for rec in eligible_records 
                if rec.continuous_accept and rec.accept_length >= n + 1
            ]
            n_alpha_rates[n] = len(continuous_accepts) / len(eligible_records)
        else:
            n_alpha_rates[n] = 0.0
    
    # Draft token statistics
    total_draft_tokens = sum(rec.draft_length for rec in all_records)
    total_accepted_tokens = sum(rec.accept_length for rec in all_records)
    total_cycles = len(all_records)
    
    # Performance metrics
    tokens_per_second_baseline = total_baseline_tokens / total_baseline_time if total_baseline_time > 0 else 0
    tokens_per_second_eagle = total_eagle_tokens / total_eagle_time if total_eagle_time > 0 else 0
    
    avg_latency_baseline = (total_baseline_time / len(results)) * 1000 if results else 0
    avg_latency_eagle = (total_eagle_time / len(results)) * 1000 if results else 0
    
    return EvaluationMetrics(
        speedup_ratio=speedup_ratio,
        avg_accept_len_tau=avg_accept_len_tau,
        n_alpha_rates=n_alpha_rates,
        tokens_per_second_baseline=tokens_per_second_baseline,
        tokens_per_second_eagle=tokens_per_second_eagle,
        latency_ms_baseline=avg_latency_baseline,
        latency_ms_eagle=avg_latency_eagle,
        total_tokens_generated=total_eagle_tokens,
        total_draft_tokens=total_draft_tokens,
        total_accepted_tokens=total_accepted_tokens,
        total_cycles=total_cycles,
    )

# ----------------------------
#    Main Function with Debug Options
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Debug Enhanced EAGLE-3 Evaluation')
    
    # Model arguments
    parser.add_argument('--base_model', required=True, help='Base model path')
    parser.add_argument('--draft_ckpt', required=True, help='Draft model checkpoint')
    
    # Dataset arguments
    parser.add_argument('--dataset', choices=['mt-bench', 'gsm8k', 'humaneval', 'alpaca', 'custom'],
                       default='mt-bench', help='Dataset to evaluate on')
    parser.add_argument('--data_path', required=True, help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    
    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_draft_len', type=int, default=8, help='Maximum draft length')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for baseline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Debug arguments
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug_samples', type=int, default=1, help='Number of samples to debug in detail')
    
    # Other arguments
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_file', default='evaluation_results_debug.json')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load models
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model.eval()
    
    # Load draft model
    print(f"Loading draft model from {args.draft_ckpt}")

    # Load draft model config from the checkpoint directory. This is crucial for consistency.
    try:
        draft_config = AutoConfig.from_pretrained(args.draft_ckpt)
        logger.info(f"Successfully loaded draft model config from {args.draft_ckpt}")
    except Exception as e:
        logger.error(f"Failed to load config from {args.draft_ckpt}: {e}")
        logger.error("Please ensure that 'config.json' from the training output is present in the checkpoint directory.")
        return

    # Create the draft model with the loaded config
    draft_model = DraftModelV3(draft_config)

    # Load draft model state
    try:
        draft_state_path = Path(args.draft_ckpt) / "pytorch_model.bin"
        draft_state = torch.load(draft_state_path, map_location="cpu")
        draft_model.load_state_dict(draft_state)
        logger.info(f"Successfully loaded draft model weights from {draft_state_path}")
    except Exception as e:
        logger.error(f"Failed to load model state from {draft_state_path}: {e}")
        logger.error("This is often caused by a mismatch between the model architecture saved during training and the one used here for evaluation.")
        return

    draft_model.to(device=device, dtype=torch.bfloat16)
    draft_model.eval()

    # Create adapter
    adapter_cls = Qwen3ModelAdapter if "qwen" in args.base_model.lower() else BaseModelAdapter
    adapter = adapter_cls(base_model)
    
    # Create EAGLE-3 inference engine
    eagle_inference = EAGLE3InferenceCorrected(
        base_model=base_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        adapter=adapter,
        device=device,
        max_draft_len=args.max_draft_len,
        debug_mode=args.debug,
    )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    prompts = EvalDatasetLoader.load_dataset(
        args.dataset,
        args.data_path,
        args.num_samples
    )
    
    # For debugging, limit to a few samples
    if args.debug and args.debug_samples > 0:
        prompts = prompts[:args.debug_samples]
        print(f"Debug mode: limiting to {len(prompts)} samples")
    
    # Baseline evaluation
    print("Running baseline evaluation...")
    results = evaluate_baseline(
        base_model,
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.temperature,
        device,
        args.batch_size,
        args.seed,
        args.debug,
    )
    
    # EAGLE-3 evaluation
    print("Running EAGLE-3 evaluation...")
    results = evaluate_eagle3(
        eagle_inference,
        prompts,
        args.max_new_tokens,
        args.temperature,
        results,
        args.seed,
        args.debug,
    )
    
    # Calculate corrected metrics
    metrics = calculate_corrected_metrics(results)
    
    # Print results in paper format
    print("\n" + "="*60)
    print("EAGLE-3 Evaluation Results (Debug Enhanced)")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.base_model}")
    print(f"Number of samples: {len(results)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Debug mode: {args.debug}")
    print("-"*60)
    print(f"Speedup ratio: {metrics.speedup_ratio:.2f}x")
    print(f"Average accept length τ: {metrics.avg_accept_len_tau:.2f}")
    print(f"Baseline throughput: {metrics.tokens_per_second_baseline:.1f} tokens/s")
    print(f"EAGLE-3 throughput: {metrics.tokens_per_second_eagle:.1f} tokens/s")
    print(f"Baseline latency: {metrics.latency_ms_baseline:.1f} ms")
    print(f"EAGLE-3 latency: {metrics.latency_ms_eagle:.1f} ms")
    print(f"Total draft tokens: {metrics.total_draft_tokens}")
    print(f"Total accepted tokens: {metrics.total_accepted_tokens}")

    if metrics.total_draft_tokens == 0 :
        print(f"Overall accept rate: NaN")
    else :
        print(f"Overall accept rate: {metrics.total_accepted_tokens/metrics.total_draft_tokens:.2%}")

    print(f"Total draft-verify cycles: {metrics.total_cycles}")
    print("-"*60)
    print("n-α Accept Rates (matching paper definition):")
    for n, rate in sorted(metrics.n_alpha_rates.items()):
        print(f"  {n}-α: {rate:.3f}")
    print("="*60)
    
    # Save detailed results
    output_data = {
        'args': vars(args),
        'metrics': {
            'speedup_ratio': metrics.speedup_ratio,
            'avg_accept_len_tau': metrics.avg_accept_len_tau,
            'n_alpha_rates': metrics.n_alpha_rates,
            'tokens_per_second_baseline': metrics.tokens_per_second_baseline,
            'tokens_per_second_eagle': metrics.tokens_per_second_eagle,
            'latency_ms_baseline': metrics.latency_ms_baseline,
            'latency_ms_eagle': metrics.latency_ms_eagle,
            'total_tokens_generated': metrics.total_tokens_generated,
            'total_draft_tokens': metrics.total_draft_tokens,
            'total_accepted_tokens': metrics.total_accepted_tokens,
            'total_cycles': metrics.total_cycles,
        },
        'sample_results': [
            {
                'prompt': r.prompt[:100] + '...' if len(r.prompt) > 100 else r.prompt,
                'baseline_output': r.baseline_output[:200] + '...' if len(r.baseline_output) > 200 else r.baseline_output,
                'eagle_output': r.eagle_output[:200] + '...' if len(r.eagle_output) > 200 else r.eagle_output,
                'baseline_tokens': r.baseline_tokens,
                'eagle_tokens': r.eagle_tokens,
                'baseline_time': r.baseline_time,
                'eagle_time': r.eagle_time,
                'speedup': r.baseline_time / r.eagle_time if r.eagle_time > 0 else 0,
                'num_cycles': len(r.acceptance_records),
                'avg_accept_len': np.mean([rec.accept_length for rec in r.acceptance_records]) if r.acceptance_records else 0,
                'acceptance_records': [
                    {
                        'draft_length': rec.draft_length,
                        'accept_length': rec.accept_length,
                        'draft_tokens': rec.draft_tokens,
                        'accepted_tokens': rec.accepted_tokens,
                        'continuous_accept': rec.continuous_accept,
                    }
                    for rec in r.acceptance_records
                ] if args.debug else [],
            }
            for r in results
        ]
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to {args.output_file}")

if __name__ == "__main__":
    main() 