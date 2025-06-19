# =============================================================
#  eagle_v3_eval_updated.py  --  Updated EAGLE-3 Evaluation with Enhanced Readability
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
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, GenerationConfig
from tqdm import tqdm
from termcolor import colored  # For colored output
from tabulate import tabulate  # For nice tables

from transformers.cache_utils import DynamicCache, Cache

# Import from training script
from train_eagle_v3 import (
    DraftModelV3,
    build_tree_buffers,
    make_causal_mask,
    DEFAULT_TREE,
    FUSE_LAYERS
)

# Import from eagle_core
from eagle_core import get_fusion_indices

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# For precise GPU timing
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ----------------------------
#    Data Classes
# ----------------------------

@dataclass
class TokenPrediction:
    """Information about a single token prediction."""
    position: int
    predicted_token_id: int
    predicted_token: str
    probability: float
    top_5_predictions: List[Tuple[int, str, float]]  # (token_id, token_text, probability)
    accepted: bool = False
    rejection_reason: Optional[str] = None

@dataclass
class DraftCycle:
    """Information about a single draft-verify cycle."""
    cycle_number: int
    start_position: int
    draft_predictions: List[TokenPrediction]
    accepted_predictions: List[TokenPrediction]
    verification_time: float
    draft_time: float

@dataclass
class AcceptanceRecord:
    """Record for a single draft-verify cycle."""
    draft_length: int
    accept_length: int
    draft_tokens: List[int]
    accepted_tokens: List[int]
    continuous_accept: bool
    draft_cycle: Optional[DraftCycle] = None  # Detailed information

@dataclass
class EvaluationMetrics:
    """Corrected metrics for EAGLE-3 evaluation."""
    speedup_ratio: float
    avg_accept_len_tau: float
    n_alpha_rates: Dict[int, float]
    tokens_per_second_baseline: float
    tokens_per_second_eagle: float
    latency_ms_baseline: float
    latency_ms_eagle: float
    total_tokens_generated: int
    total_draft_tokens: int
    total_accepted_tokens: int
    total_cycles: int

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
    draft_cycles: List[DraftCycle] = field(default_factory=list)  # Detailed cycle information

# ----------------------------
#    Visualization Utilities
# ----------------------------

class VisualizationHelper:
    """Helper class for visualizing the draft-verify process."""
    
    @staticmethod
    def print_draft_cycle(cycle: DraftCycle, tokenizer: AutoTokenizer):
        """Print a single draft-verify cycle in a readable format."""
        print(f"\n{colored('='*80, 'blue')}")
        print(colored(f"DRAFT-VERIFY CYCLE #{cycle.cycle_number}", 'blue', attrs=['bold']))
        print(colored(f"Starting at position: {cycle.start_position}", 'blue'))
        print(colored(f"Draft time: {cycle.draft_time*1000:.2f}ms | Verify time: {cycle.verification_time*1000:.2f}ms", 'blue'))
        print(colored('='*80, 'blue'))
        
        # Create table data
        table_data = []
        for i, pred in enumerate(cycle.draft_predictions):
            is_accepted = pred in cycle.accepted_predictions
            
            # Format top-5 predictions
            top_5_str = " | ".join([
                f"{t[1]} ({t[2]:.1%})" for t in pred.top_5_predictions[:3]
            ])
            
            row = [
                i + 1,
                pred.predicted_token,
                f"{pred.probability:.1%}",
                top_5_str,
                colored("✓", "green") if is_accepted else colored("✗", "red"),
                pred.rejection_reason or "-"
            ]
            table_data.append(row)
        
        headers = ["Step", "Predicted", "Prob", "Top-3 Alternatives", "Accept", "Reason"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Print summary
        accept_rate = len(cycle.accepted_predictions) / len(cycle.draft_predictions) if cycle.draft_predictions else 0
        print(f"\n{colored('Summary:', 'yellow')} Accepted {len(cycle.accepted_predictions)}/{len(cycle.draft_predictions)} tokens ({accept_rate:.1%})")
        
        # Print the generated text
        draft_text = " ".join([p.predicted_token for p in cycle.draft_predictions])
        accepted_text = " ".join([p.predicted_token for p in cycle.accepted_predictions])
        
        print(f"{colored('Draft:', 'cyan')} {draft_text}")
        print(f"{colored('Accepted:', 'green')} {accepted_text}")

    @staticmethod
    def print_generation_summary(result: GenerationResult, tokenizer: AutoTokenizer):
        """Print a summary of the entire generation."""
        print(f"\n{colored('='*100, 'magenta')}")
        print(colored("GENERATION SUMMARY", 'magenta', attrs=['bold']))
        print(colored('='*100, 'magenta'))
        
        print(f"{colored('Prompt:', 'yellow')} {result.prompt[:100]}...")
        print(f"\n{colored('Baseline Output:', 'blue')} {result.baseline_output[:200]}...")
        print(f"{colored('EAGLE Output:', 'green')} {result.eagle_output[:200]}...")
        
        print(f"\n{colored('Performance Metrics:', 'yellow')}")
        speedup = result.baseline_time / result.eagle_time if result.eagle_time > 0 else 0
        print(f"  • Baseline time: {result.baseline_time*1000:.1f}ms ({result.baseline_tokens} tokens)")
        print(f"  • EAGLE time: {result.eagle_time*1000:.1f}ms ({result.eagle_tokens} tokens)")
        print(f"  • Speedup: {colored(f'{speedup:.2f}x', 'green' if speedup > 1 else 'red')}")
        
        # Draft statistics
        total_draft = sum(r.draft_length for r in result.acceptance_records)
        total_accepted = sum(r.accept_length for r in result.acceptance_records)
        overall_accept_rate = total_accepted / total_draft if total_draft > 0 else 0
        
        print(f"\n{colored('Draft Statistics:', 'yellow')}")
        print(f"  • Total cycles: {len(result.acceptance_records)}")
        print(f"  • Total draft tokens: {total_draft}")
        print(f"  • Total accepted tokens: {total_accepted}")
        print(f"  • Overall accept rate: {colored(f'{overall_accept_rate:.1%}', 'green' if overall_accept_rate > 0.5 else 'red')}")

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
#    Updated EAGLE-3 Inference
# ----------------------------

class EAGLE3InferenceUpdated:
    """Updated EAGLE-3 speculative decoding implementation without adapter classes."""
    
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        draft_model: DraftModelV3,
        tokenizer: AutoTokenizer,
        device: torch.device,
        tree_choices: List[List[int]] = None,
        max_draft_len: int = 8,
        debug_mode: bool = True,
        visual_mode: bool = True,  # New: enable visual output
    ):
        self.base_model = base_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.device = device
        self.tree_choices = tree_choices or DEFAULT_TREE
        self.max_draft_len = max_draft_len
        self.debug_mode = debug_mode
        self.visual_mode = visual_mode
        
        # Build tree buffers
        self.tree_buffers = build_tree_buffers(self.tree_choices, device)
        
        # Set pad token if not set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Visualization helper
        self.viz = VisualizationHelper()
            
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
        
        # Draft model info
        logger.info(f"Draft model type: {type(self.draft_model).__name__}")
        logger.info(f"Draft model device: {next(self.draft_model.parameters()).device}")
        logger.info(f"Draft model dtype: {next(self.draft_model.parameters()).dtype}")
        logger.info(f"Draft model base_hidden_size: {self.draft_model.base_hidden_size}")
        logger.info(f"Draft model student_hidden_size: {self.draft_model.student_hidden_size}")
        
        # Check fusion layers
        logger.info(f"Number of fusion layers: {FUSE_LAYERS}")
        
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
    ) -> Tuple[str, List[AcceptanceRecord], List[DraftCycle]]:
        """Generate text using EAGLE-3 speculative decoding with enhanced visualization."""
        
        if self.visual_mode:
            print(f"\n{colored('='*100, 'cyan')}")
            print(colored(f"STARTING EAGLE-3 GENERATION", 'cyan', attrs=['bold']))
            print(colored('='*100, 'cyan'))
            print(f"{colored('Prompt:', 'yellow')} {prompt[:100]}...")
            print(f"{colored('Settings:', 'yellow')} max_tokens={max_new_tokens}, temp={temperature}, top_p={top_p}")
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        
        # Initialize
        generated_ids = input_ids.clone()
        acceptance_records = []
        draft_cycles = []
        
        # Prefill: get initial features and KV cache from base model
        if self.visual_mode:
            print(f"\n{colored('PREFILL PHASE', 'green', attrs=['bold'])}")
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        
        # Initialize accumulated hidden states buffer
        all_hidden_states = []
        for layer_hs in outputs.hidden_states:
            all_hidden_states.append(layer_hs.clone())
        
        # Main generation loop
        position = input_ids.shape[1]
        generation_step = 0
        
        while position < input_ids.shape[1] + max_new_tokens:
            generation_step += 1
            
            if self.visual_mode:
                print(f"\n{colored('-'*80, 'gray')}")
                print(colored(f"Generation Step {generation_step} (Position: {position})", 'yellow', attrs=['bold']))
            
            # Extract and fuse features from the last generated position
            fused_features = self._fuse_features_from_buffer(all_hidden_states, position - 1)
            
            # Measure draft time
            draft_start = time.perf_counter()
            
            # Draft phase
            draft_tokens, draft_logits, draft_predictions = self._draft_tokens_chain_visual(
                fused_features,
                position,
                min(self.max_draft_len, input_ids.shape[1] + max_new_tokens - position),
                temperature
            )
            
            draft_time = time.perf_counter() - draft_start
            
            if len(draft_tokens) == 0:
                if self.visual_mode:
                    print(colored("No draft tokens generated, stopping generation", 'red'))
                break
            
            # Measure verification time
            verify_start = time.perf_counter()
            
            # Verification phase
            accepted_tokens, new_past_key_values, continuous_accept, new_hidden_states, accepted_predictions = self._verify_tokens_with_cache_visual(
                generated_ids,
                draft_tokens,
                draft_logits,
                draft_predictions,
                past_key_values,
                position,
                temperature,
                top_p,
            )
            
            verification_time = time.perf_counter() - verify_start
            
            # Create draft cycle record
            cycle = DraftCycle(
                cycle_number=generation_step,
                start_position=position,
                draft_predictions=draft_predictions,
                accepted_predictions=accepted_predictions,
                draft_time=draft_time,
                verification_time=verification_time
            )
            draft_cycles.append(cycle)
            
            # Visualize the cycle
            if self.visual_mode:
                self.viz.print_draft_cycle(cycle, self.tokenizer)
            
            # Record acceptance
            record = AcceptanceRecord(
                draft_length=len(draft_tokens),
                accept_length=len(accepted_tokens),
                draft_tokens=draft_tokens.tolist(),
                accepted_tokens=accepted_tokens.tolist(),
                continuous_accept=continuous_accept,
                draft_cycle=cycle,
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
                
                # Update accumulated hidden states
                if new_hidden_states is not None:
                    for layer_i, layer_hs in enumerate(new_hidden_states):
                        all_hidden_states[layer_i] = torch.cat(
                            [all_hidden_states[layer_i], layer_hs[:, :len(accepted_tokens), :]],
                            dim=1
                        )
                
                # If not all tokens were accepted, need to get new hidden states
                if len(accepted_tokens) < len(draft_tokens) and new_hidden_states is None:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.base_model(
                            input_ids=generated_ids[:, -1:],
                            past_key_values=past_key_values,
                            output_hidden_states=True,
                            use_cache=True,
                            return_dict=True,
                        )
                    for layer_i, layer_hs in enumerate(outputs.hidden_states):
                        all_hidden_states[layer_i] = torch.cat(
                            [all_hidden_states[layer_i], layer_hs],
                            dim=1
                        )
            else:
                # No tokens accepted - sample one token normally
                if self.visual_mode:
                    print(colored("No tokens accepted, sampling from base model...", 'orange'))
                
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
                if self.visual_mode:
                    print(colored("EOS token detected, stopping generation", 'yellow'))
                break
        
        # Decode output
        output_ids = generated_ids[0, input_ids.shape[1]:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        if self.visual_mode:
            print(f"\n{colored('='*100, 'green')}")
            print(colored("GENERATION COMPLETED", 'green', attrs=['bold']))
            print(colored('='*100, 'green'))
            print(f"{colored('Output:', 'cyan')} {output_text}")
        
        return output_text, acceptance_records, draft_cycles
    
    def _fuse_features_from_buffer(self, all_hidden_states: List[torch.Tensor], position: int) -> torch.Tensor:
        """Fuse features from multiple layers at a specific position."""
        n_layers = len(all_hidden_states) - 1  # Exclude embedding layer
        
        # --- Start of Fix ---
        # 使用统一的融合索引函数，确保与训练时一致
        try:
            # 尝试从模型配置中获取模型名称
            model_name_for_fusion = self.base_model.config._name_or_path
        except AttributeError:
            # 如果配置中没有，可以提供一个默认值或从参数传入
            model_name_for_fusion = "qwen"  # 确保这里的关键字能触发正确的逻辑
            
        indices = get_fusion_indices(model_name_for_fusion, n_layers)
        # --- End of Fix ---
        
        # Check bounds
        max_pos = min(hs.shape[1] for hs in all_hidden_states)
        if position >= max_pos:
            position = max_pos - 1
        
        # Extract and concatenate features
        features = []
        for idx in indices:
            # +1 是因为 all_hidden_states[0] 是 embedding 层
            features.append(all_hidden_states[idx + 1][:, position:position+1, :])
        
        fused = torch.cat(features, dim=-1)
        
        return fused
    
    def _draft_tokens_chain_visual(
        self,
        fused_features: torch.Tensor,
        start_position: int,
        max_len: int,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[TokenPrediction]]:
        """Generate draft tokens with detailed prediction information."""
        
        if max_len <= 0:
            return torch.tensor([], device=self.device, dtype=torch.long), torch.tensor([], device=self.device), []
    
        draft_tokens = []
        draft_logits = []
        predictions = []
        
        current_features = fused_features
        
        for i in range(max_len):
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
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    break
                    
            except Exception as e:
                logger.error(f"Error during draft model forward pass: {e}")
                break
            
            # Get probabilities and top-k predictions
            probs = F.softmax(logits[0, -1], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, min(5, probs.size(-1)))
            
            # Sample next token
            if temperature < 1e-4:  # Greedy
                next_token = top_k_indices[0]
                sampling_prob = top_k_probs[0].item()
            else:  # Sampling
                temp_probs = F.softmax(logits[0, -1] / temperature, dim=-1)
                next_token = torch.multinomial(temp_probs, num_samples=1).squeeze()
                sampling_prob = probs[next_token].item()
            
            # Create prediction record
            pred = TokenPrediction(
                position=start_position + i,
                predicted_token_id=next_token.item(),
                predicted_token=self.tokenizer.decode([next_token.item()]),
                probability=sampling_prob,
                top_5_predictions=[
                    (idx.item(), self.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(top_k_indices, top_k_probs)
                ]
            )
            predictions.append(pred)
            
            draft_tokens.append(next_token)
            draft_logits.append(logits[0, -1])
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            current_features = hidden.detach()
        
        if draft_tokens:
            result_tokens = torch.stack(draft_tokens)
            result_logits = torch.stack(draft_logits)
            return result_tokens, result_logits, predictions
        else:
            return torch.tensor([], device=self.device, dtype=torch.long), torch.tensor([], device=self.device), []
        
    def _verify_tokens_with_cache_visual(
            self,
            prefix_ids: torch.Tensor,
            draft_tokens: torch.Tensor,
            draft_logits: torch.Tensor,
            draft_predictions: List[TokenPrediction],
            past_key_values: Any,
            position: int,
            temperature: float,
            top_p: float,
        ) -> Tuple[torch.Tensor, Any, bool, Optional[Tuple[torch.Tensor]], List[TokenPrediction]]:
            """Verify draft tokens with detailed acceptance information."""
            
            if len(draft_tokens) == 0:
                return torch.tensor([], device=self.device, dtype=torch.long), past_key_values, True, None, []
            
            # Prepare draft sequence for verification
            draft_ids = draft_tokens.unsqueeze(0)
            
            # Generate position_ids
            seq_len = draft_ids.shape[1]
            position_ids = torch.arange(
                position, position + seq_len, 
                dtype=torch.long, 
                device=self.device
            ).unsqueeze(0)
            
            # Forward through base model
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
                logger.error(f"Error in base model verification: {e}")
                return torch.tensor([], device=self.device, dtype=torch.long), past_key_values, False, None, []
            
            verify_logits = outputs.logits[0]
            new_past_key_values = outputs.past_key_values
            new_hidden_states = outputs.hidden_states
            
            # Convert cache if needed
            if isinstance(new_past_key_values, Cache):
                legacy_past_key_values = new_past_key_values.to_legacy_cache()
            else:
                legacy_past_key_values = new_past_key_values
            
            # Verify each token
            accepted_tokens = []
            accepted_predictions = []
            continuous_accept = True
            
            for i, (draft_token, draft_logit, draft_pred) in enumerate(zip(draft_tokens, draft_logits, draft_predictions)):
                if temperature == 0:
                    # Greedy verification
                    verify_token = verify_logits[i].argmax()
                    
                    if draft_token == verify_token:
                        accepted_tokens.append(draft_token)
                        draft_pred.accepted = True
                        accepted_predictions.append(draft_pred)
                    else:
                        # Reject and resample
                        accepted_tokens.append(verify_token)
                        continuous_accept = False
                        draft_pred.accepted = False
                        draft_pred.rejection_reason = f"Expected {self.tokenizer.decode([verify_token.item()])}"
                        
                        # Create new prediction for the resampled token
                        resampled_pred = TokenPrediction(
                            position=draft_pred.position,
                            predicted_token_id=verify_token.item(),
                            predicted_token=self.tokenizer.decode([verify_token.item()]),
                            probability=1.0,  # Greedy
                            top_5_predictions=[],
                            accepted=True
                        )
                        accepted_predictions.append(resampled_pred)
                        break
                else:
                    # Probabilistic verification
                    draft_prob = F.softmax(draft_logit / temperature, dim=-1)
                    verify_prob = F.softmax(verify_logits[i] / temperature, dim=-1)
                    
                    verify_argmax = verify_prob.argmax()
                    if draft_token == verify_argmax:
                        accepted_tokens.append(draft_token)
                        draft_pred.accepted = True
                        accepted_predictions.append(draft_pred)
                    else:
                        r = torch.rand(1, device=self.device).item()
                        accept_prob = min(1.0, verify_prob[draft_token] / (draft_prob[draft_token] + 1e-10))
                        
                        if r < accept_prob:
                            accepted_tokens.append(draft_token)
                            draft_pred.accepted = True
                            accepted_predictions.append(draft_pred)
                        else:
                            # Rejection sampling
                            residual = torch.clamp(verify_prob - draft_prob, min=0)
                            residual = residual / residual.sum()
                            
                            new_token = torch.multinomial(residual, 1)
                            accepted_tokens.append(new_token.item())
                            continuous_accept = False
                            
                            draft_pred.accepted = False
                            draft_pred.rejection_reason = f"Rejection sampled to {self.tokenizer.decode([new_token.item()])}"
                            
                            # Create new prediction for rejection sampled token
                            resampled_pred = TokenPrediction(
                                position=draft_pred.position,
                                predicted_token_id=new_token.item(),
                                predicted_token=self.tokenizer.decode([new_token.item()]),
                                probability=residual[new_token].item(),
                                top_5_predictions=[],
                                accepted=True
                            )
                            accepted_predictions.append(resampled_pred)
                            break
                        
            accepted_tokens = torch.tensor(accepted_tokens, device=self.device, dtype=torch.long)
            
            # Trim KV cache if needed
            if len(accepted_tokens) < len(draft_tokens):
                trimmed_kv = []
                for layer_kv in legacy_past_key_values:
                    k, v = layer_kv
                    k_trimmed = k[:, :, :position - past_key_values[0][0].shape[2] + len(accepted_tokens), :]
                    v_trimmed = v[:, :, :position - past_key_values[0][0].shape[2] + len(accepted_tokens), :]
                    trimmed_kv.append((k_trimmed, v_trimmed))
                
                final_past_key_values = DynamicCache.from_legacy_cache(past_key_values=tuple(trimmed_kv))
                
                return accepted_tokens, final_past_key_values, continuous_accept, None, accepted_predictions
            else:
                return accepted_tokens, new_past_key_values, continuous_accept, new_hidden_states, accepted_predictions
    
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
#    Evaluation Functions (updated)
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
    """Evaluate baseline model with precise timing."""
    
    # Set random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    results = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Baseline evaluation"):
        batch_prompts = prompts[i:i+batch_size]
        
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
        
        # Process results
        for j, (prompt, output) in enumerate(zip(batch_prompts, outputs)):
            input_length = inputs.input_ids[j].shape[0]
            output_ids = output[input_length:]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            
            result = GenerationResult(
                prompt=prompt,
                baseline_output=output_text,
                eagle_output="",
                baseline_time=elapsed_time / len(batch_prompts),
                eagle_time=0,
                baseline_tokens=len(output_ids),
                eagle_tokens=0,
            )
            results.append(result)
    
    return results

@torch.no_grad()
def evaluate_eagle3(
    eagle_inference: EAGLE3InferenceUpdated,
    prompts: List[str],
    max_new_tokens: int,
    temperature: float,
    baseline_results: List[GenerationResult],
    seed: int = 42,
    visual_mode: bool = True,
) -> List[GenerationResult]:
    """Evaluate EAGLE-3 model with visual output."""
    
    for i, (prompt, baseline_result) in enumerate(tqdm(
        zip(prompts, baseline_results), 
        total=len(prompts),
        desc="EAGLE-3 evaluation"
    )):
        if visual_mode:
            print(f"\n{colored('='*120, 'magenta')}")
            print(colored(f"EVALUATING SAMPLE {i+1}/{len(prompts)}", 'magenta', attrs=['bold']))
            print(colored('='*120, 'magenta'))
        
        # Generate with timing
        @precise_cuda_timing
        def generate_eagle():
            return eagle_inference.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                seed=seed,
            )
        
        (output_text, acceptance_records, draft_cycles), elapsed_time = generate_eagle()
        
        # Update result
        baseline_result.eagle_output = output_text
        baseline_result.eagle_time = elapsed_time
        baseline_result.eagle_tokens = len(eagle_inference.tokenizer.encode(output_text))
        baseline_result.acceptance_records = acceptance_records
        baseline_result.draft_cycles = draft_cycles
        
        # Print summary for this sample
        if visual_mode:
            eagle_inference.viz.print_generation_summary(baseline_result, eagle_inference.tokenizer)
    
    return baseline_results

def calculate_corrected_metrics(results: List[GenerationResult]) -> EvaluationMetrics:
    """Calculate corrected evaluation metrics matching paper definitions."""
    
    # Timing statistics
    total_baseline_time = sum(r.baseline_time for r in results)
    total_eagle_time = sum(r.eagle_time for r in results)
    
    # Token statistics
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
    
    # Calculate n-α rates
    n_alpha_rates = {}
    for n in range(8):
        eligible_records = [rec for rec in all_records if rec.draft_length >= n + 1]
        
        if eligible_records:
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
#    Main Function
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='Updated EAGLE-3 Evaluation with Visualization')
    
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
    
    # Visualization arguments
    parser.add_argument('--visual', action='store_true', help='Enable visual output')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Other arguments
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output_file', default='evaluation_results_visual.json')
    
    args = parser.parse_args()
    
    # Check for required packages for visualization
    if args.visual:
        try:
            import termcolor
            import tabulate
        except ImportError:
            print("Please install required packages: pip install termcolor tabulate")
            return
    
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
    
    # Load draft model configuration
    print(f"Loading draft model from {args.draft_ckpt}")
    
    try:
        config_path = Path(args.draft_ckpt) / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"config.json not found in {args.draft_ckpt}")

        with open(config_path, 'r') as f:
            draft_config_data = json.load(f)
        logger.info(f"Successfully loaded draft model config")

        # Extract parameters
        vocab_size = draft_config_data["vocab_size"]
        eagle_config = draft_config_data["_eagle_config"]
        base_hidden_size = eagle_config["base_hidden_size"]
        student_hidden_size = eagle_config["student_hidden_size"]
        num_decoder_layers = eagle_config["num_decoder_layers"]
        use_simple_fusion = eagle_config.get("use_simple_fusion", True)

        # Create the draft model
        draft_model = DraftModelV3(
            vocab_size=vocab_size,
            base_hidden_size=base_hidden_size,
            student_hidden_size=student_hidden_size,
            num_decoder_layers=num_decoder_layers,
            use_simple_fusion=use_simple_fusion,
        )
        logger.info("DraftModelV3 instantiated successfully")

    except Exception as e:
        logger.error(f"Failed to load or instantiate draft model: {e}", exc_info=True)
        return

    # Load draft model weights
    try:
        draft_state_path = Path(args.draft_ckpt) / "pytorch_model.bin"
        draft_state = torch.load(draft_state_path, map_location="cpu")
        draft_model.load_state_dict(draft_state)
        logger.info(f"Successfully loaded draft model weights")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        return

    draft_model.to(device=device, dtype=torch.bfloat16)
    draft_model.eval()

    # Create EAGLE-3 inference engine
    eagle_inference = EAGLE3InferenceUpdated(
        base_model=base_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        device=device,
        max_draft_len=args.max_draft_len,
        debug_mode=args.debug,
        visual_mode=args.visual,
    )
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    prompts = EvalDatasetLoader.load_dataset(
        args.dataset,
        args.data_path,
        args.num_samples
    )
    
    # Baseline evaluation
    print("\nRunning baseline evaluation...")
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
    print("\nRunning EAGLE-3 evaluation...")
    results = evaluate_eagle3(
        eagle_inference,
        prompts,
        args.max_new_tokens,
        args.temperature,
        results,
        args.seed,
        args.visual,
    )
    
    # Calculate metrics
    metrics = calculate_corrected_metrics(results)
    
    # Print final summary
    print(f"\n{colored('='*100, 'cyan')}")
    print(colored("FINAL EVALUATION RESULTS", 'cyan', attrs=['bold']))
    print(colored('='*100, 'cyan'))
    
    print(f"{colored('Dataset:', 'yellow')} {args.dataset}")
    print(f"{colored('Model:', 'yellow')} {args.base_model}")
    print(f"{colored('Samples:', 'yellow')} {len(results)}")
    print(f"{colored('Settings:', 'yellow')} max_tokens={args.max_new_tokens}, temp={args.temperature}")
    
    print(f"\n{colored('Performance Metrics:', 'green')}")
    speedup_color = 'green' if metrics.speedup_ratio > 1.5 else 'yellow' if metrics.speedup_ratio > 1 else 'red'
    print(f"  • Speedup: {colored(f'{metrics.speedup_ratio:.2f}x', speedup_color, attrs=['bold'])}")
    print(f"  • Average accept length (τ): {metrics.avg_accept_len_tau:.2f}")
    print(f"  • Baseline: {metrics.tokens_per_second_baseline:.1f} tok/s ({metrics.latency_ms_baseline:.1f}ms)")
    print(f"  • EAGLE-3: {metrics.tokens_per_second_eagle:.1f} tok/s ({metrics.latency_ms_eagle:.1f}ms)")
    
    print(f"\n{colored('Acceptance Statistics:', 'green')}")
    print(f"  • Total draft tokens: {metrics.total_draft_tokens}")
    print(f"  • Total accepted: {metrics.total_accepted_tokens}")
    accept_rate = metrics.total_accepted_tokens/metrics.total_draft_tokens if metrics.total_draft_tokens > 0 else 0
    accept_color = 'green' if accept_rate > 0.7 else 'yellow' if accept_rate > 0.5 else 'red'
    print(f"  • Overall accept rate: {colored(f'{accept_rate:.1%}', accept_color)}")
    print(f"  • Total cycles: {metrics.total_cycles}")
    
    print(f"\n{colored('n-α Accept Rates:', 'green')}")
    for n, rate in sorted(metrics.n_alpha_rates.items())[:5]:
        rate_color = 'green' if rate > 0.7 else 'yellow' if rate > 0.5 else 'red'
        print(f"  • {n}-α: {colored(f'{rate:.3f}', rate_color)}")
    
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
        'sample_results': []
    }
    
    # Add sample results with draft cycle details
    for r in results[:10]:  # Save first 10 samples with full details
        sample_data = {
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
        }
        
        # Add detailed cycle information if available
        if args.visual and r.draft_cycles:
            sample_data['draft_cycles'] = []
            for cycle in r.draft_cycles[:3]:  # First 3 cycles
                cycle_data = {
                    'cycle_number': cycle.cycle_number,
                    'start_position': cycle.start_position,
                    'draft_time_ms': cycle.draft_time * 1000,
                    'verify_time_ms': cycle.verification_time * 1000,
                    'draft_predictions': [
                        {
                            'token': pred.predicted_token,
                            'probability': pred.probability,
                            'accepted': pred.accepted,
                            'rejection_reason': pred.rejection_reason
                        }
                        for pred in cycle.draft_predictions
                    ]
                }
                sample_data['draft_cycles'].append(cycle_data)
        
        output_data['sample_results'].append(sample_data)
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{colored('Results saved to:', 'yellow')} {args.output_file}")

if __name__ == "__main__":
    main()