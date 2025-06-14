# =============================================================
#  train_eagle_v3.py  --  Complete EAGLE V3 Training Implementation (Fixed)
# =============================================================

import argparse
import os
import math
import random
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from transformers import get_cosine_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    get_linear_schedule_with_warmup, LogitsProcessorList
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config, Qwen3DecoderLayer, Qwen3RMSNorm,
    Qwen3MLP, Qwen3Attention, Qwen3RotaryEmbedding
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import logging
import numpy as np

from preprocess_data import FUSE_LAYERS 

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

from pathlib import Path
import logging

log_dir = Path(__file__).resolve().parent
log_file = log_dir / "train_debug.log"   

logging.basicConfig(
    level=logging.INFO,
    filename=str(log_file),        
    filemode="a",                  
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

debug = logger.debug

# ------------------------------------------------------------------
# 1️⃣  Configuration & Arguments
# ------------------------------------------------------------------

def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _make_ttt_mask(
    batch_size: int,
    seq_len: int, 
    ttt_step: int,
    tree_structure: Optional[List[List[int]]],
    dtype: torch.dtype,
    device: torch.device
):
    """
    Make TTT-aware attention mask for training-time test steps.
    
    For step 0: standard causal mask
    For step 1+: block diagonal mask where each draft token can only attend to its parent and ancestors
    """
    if ttt_step == 0:
        # Standard causal mask for first step
        return _make_causal_mask((batch_size, seq_len), dtype, device)
    
    # For TTT steps, we need a more complex mask
    # This is a simplified version - in practice, you'd use the actual tree structure
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    
    # Block diagonal structure: each token can see its ancestors in the tree
    # This is a placeholder - actual implementation would use tree_structure
    block_size = seq_len // (ttt_step + 1)  # Rough approximation
    
    for i in range(seq_len):
        # Each token can attend to tokens in its block and previous blocks
        block_idx = i // block_size
        start_idx = 0
        end_idx = min((block_idx + 1) * block_size, seq_len)
        mask[i, start_idx:end_idx] = 0
    
    # Apply causal constraint within each block
    mask_cond = torch.arange(seq_len, device=device)
    causal_mask = mask_cond < (mask_cond + 1).view(seq_len, 1)
    mask = torch.where(causal_mask & (mask == 0), torch.tensor(0.0, dtype=dtype, device=device), mask)
    
    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from [bsz, seq_len] to [bsz, 1, tgt_seq_len, src_seq_len].
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


@dataclass
class TrainingConfig:
    # Model
    base_model: str
    data_path: str
    draft_checkpoint: Optional[str] = None

    # Multi-layer draft model
    num_decoder_layers: int = 4
    
    # KL divergence
    use_kl_loss: bool = True
    kl_weight: float = 0.7
    kl_temperature: float = 2.0
    
    # Trace and debug
    enable_trace: bool = False
    trace_output_dir: str = "./traces"
    
    # Data
    max_seq_len: int = 4096
    
    # Training
    output_dir: str = "./draft_v3_ckpt"
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    num_epochs: int = 3
    warmup_steps: int = 2000
    total_steps: int = 100000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # TTT specific
    ttt_steps: int = 2  # Number of training-time test steps
    ttt_step_weights: List[float] = None  # Weights for each TTT step loss
    
    # Loss weights
    token_loss_weight: float = 1.0
    feature_loss_weight: float = 0.0  # Set to 0 for V3
    
    # Data augmentation
    use_noise: bool = True
    noise_type: str = "uniform"  # "uniform" or "gaussian"
    noise_std: float = 0.2
    
    # Logging
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Misc
    seed: int = 42
    mixed_precision: str = "bf16"
    num_workers: int = 16
    wandb_project: str = "eagle-v3"
    
    # Use simple fusion as in paper
    use_simple_fusion: bool = True
    
    def __post_init__(self):
        if self.ttt_step_weights is None:
            # Default: exponentially decaying weights
            self.ttt_step_weights = [1.0 / (i + 1) for i in range(self.ttt_steps)]


def parse_args():
    parser = argparse.ArgumentParser(description='Train EAGLE V3 Draft Model')
    
    # Model
    parser.add_argument('--base_model', type=str, required=True,
                       help='HF name or path of the frozen LLM')
    parser.add_argument('--draft_checkpoint', type=str, default=None,
                       help='Path to resume training from')
    
    parser.add_argument('--num_decoder_layers', type=int, default=4,
                       help='Number of decoder layers in draft model')
    parser.add_argument('--use_kl_loss', action='store_true', default=True,
                       help='Use KL divergence loss')
    parser.add_argument('--kl_weight', type=float, default=0.7,
                       help='Weight for KL divergence loss')
    parser.add_argument('--enable_trace', action='store_true',
                       help='Enable detailed trace output')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                       help='Directory containing preprocessed data')
    parser.add_argument('--max_seq_len', type=int, default=2048)
    
    # Training
    parser.add_argument('--output_dir', type=str, default='./draft_v3_ckpt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--total_steps', type=int, default=400000)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # TTT
    parser.add_argument('--ttt_steps', type=int, default=2)
    
    # Feature fusion
    parser.add_argument('--use_simple_fusion', action='store_true', default=True,
                       help='Use simple linear fusion as in paper')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_project', type=str, default='eagle-v3')
    
    args = parser.parse_args()
    
    # Convert to TrainingConfig
    config = TrainingConfig(
        base_model=args.base_model,
        draft_checkpoint=args.draft_checkpoint,
        data_path=args.data_path,
        max_seq_len=args.max_seq_len,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        weight_decay=args.weight_decay,
        ttt_steps=args.ttt_steps,
        seed=args.seed,
        wandb_project=args.wandb_project,
        num_decoder_layers=args.num_decoder_layers,
        use_kl_loss=args.use_kl_loss,
        kl_weight=args.kl_weight,
        enable_trace=args.enable_trace,
        use_simple_fusion=args.use_simple_fusion,
    )
    
    return config, args


# ------------------------------------------------------------------
# 2️⃣  Model Components
# ------------------------------------------------------------------

class SimpleFusion(nn.Module):
    """Simple linear fusion as described in the paper."""
    
    def __init__(self, hidden_size: int, n_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(n_layers * hidden_size, hidden_size, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, n_layers * hidden_size]"""
        return self.proj(x)


class MultiLayerFeatureFusion(nn.Module):
    """Learnable fusion of features from multiple layers."""
    
    def __init__(self, n_layers: int, hidden_size: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        # Attention-based fusion
        self.layer_attention = nn.Sequential(
            nn.Linear(n_layers * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, n_layers)
        )
        
        debug(f">>> [DEBUG FF] Create MultiLayerFeatureFusion with n_layers={n_layers} hidden_size={hidden_size}")
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(n_layers * hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # Final projection
        self.proj = nn.Linear(n_layers * hidden_size, hidden_size, bias=False)
    
    def forward(self, multi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_features: [batch, seq_len, n_layers * hidden_size]
        Returns:
            fused: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = multi_features.shape
        debug(">>> [DEBUG FF] multi_features.shape =", multi_features.shape)
        
        # Compute attention weights
        attn_logits = self.layer_attention(multi_features)
        attn_weights = F.softmax(attn_logits, dim=-1)  # [batch, seq_len, n_layers]
        
        # Reshape and apply attention
        features = multi_features.view(batch_size, seq_len, self.n_layers, self.hidden_size)
        attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len, n_layers, 1]
        
        # Weighted sum
        weighted = (features * attn_weights).sum(dim=2)  # [batch, seq_len, hidden_size]
        
        # Gated fusion
        gate = self.gate(multi_features)
        projected = self.proj(multi_features)
        
        return gate * weighted + (1 - gate) * projected


class DraftModelV3(nn.Module):
    """EAGLE V3 Draft Model with Multi-layer support and logit scaling."""
    
    def __init__(self, config: Qwen3Config, num_decoder_layers: int = 4, use_simple_fusion: bool = True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_decoder_layers = num_decoder_layers
        
        # Multi-layer feature fusion (3 layers: low, mid, high)
        if use_simple_fusion:
            self.feature_fusion = SimpleFusion(self.hidden_size, n_layers=FUSE_LAYERS)
        else:
            self.feature_fusion = MultiLayerFeatureFusion(FUSE_LAYERS, self.hidden_size)
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, self.hidden_size, config.pad_token_id)
        
        # Rotary Embedding with caching
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = None
        
        # Multiple decoder layers instead of single layer
        self.decoder_layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx=i) 
            for i in range(num_decoder_layers)
        ])
        
        # Output projection
        self.norm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Learnable logit scale for distribution alignment
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        
        # Tree buffers (will be set externally)
        self.tree_buffers = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        is_fused_features: bool = True,
        ttt_step: int = -1,
        tree_structure: Optional[List[List[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        
        debug(f">>> [DEBUG DraftModel.forward] TTT step: {ttt_step}, "
              f"hidden_states shape: {hidden_states.shape}, "
              f"is_fused_features: {is_fused_features}")
        
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Handle input features
        if is_fused_features:
            debug(f">>> [DEBUG] Applying feature fusion to {hidden_states.shape}")
            hidden = self.feature_fusion(hidden_states)
        else:
            debug(f">>> [DEBUG] Skipping feature fusion, using hidden directly")
            hidden = hidden_states
        
        # Add token embeddings if provided
        if input_ids is not None:
            token_embeds = self.embed_tokens(input_ids)
            debug(f">>> [DEBUG] Adding token embeddings: {token_embeds.shape}")
            hidden = hidden + token_embeds
        
        seq_len = hidden.shape[1]
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Create TTT-aware attention mask
        if ttt_step >= 0:
            attention_mask_4d = _make_ttt_mask(
                batch_size, seq_len, ttt_step, tree_structure,
                hidden.dtype, device
            )
        else:
            attention_mask_4d = _make_causal_mask(
                attention_mask.shape, hidden.dtype, device=device
            )
        
        # Handle position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
            debug(f">>> [DEBUG] Created position_ids: {position_ids.shape}")
        
        # Generate position embeddings with caching
        position_embeddings = self.rotary_emb(hidden, position_ids)
        debug(f">>> [DEBUG] Generated position_embeddings: cos={position_embeddings[0].shape}, sin={position_embeddings[1].shape}")
        
        # Pass through multiple decoder layers
        for i, decoder_layer in enumerate(self.decoder_layers):
            hidden = decoder_layer(
                hidden,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )[0]
            debug(f">>> [DEBUG] After decoder layer {i}, hidden shape: {hidden.shape}")
        
        # Output projection
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        
        # Apply learnable logit scale
        logits = logits * self.logit_scale
        
        debug(f">>> [DEBUG] Output logits shape: {logits.shape}, hidden shape: {hidden.shape}")
        debug(f">>> [DEBUG] Logit scale value: {self.logit_scale.item():.4f}")
        
        return logits, hidden


# ------------------------------------------------------------------
# 3️⃣  Data Components
# ------------------------------------------------------------------

class DataAugmentation:
    """Add noise to hidden states for robustness."""
    
    def __init__(self, noise_type: str, std: float):
        self.noise_type = noise_type
        self.std = std
    
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "uniform":
            noise = (torch.rand_like(hidden_states) - 0.5) * self.std
        elif self.noise_type == "gaussian":
            noise = torch.randn_like(hidden_states) * self.std
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
        
        return hidden_states + noise


class EagleDataset(Dataset):
    """Dataset for EAGLE training with pre-extracted features."""
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 2048,
        augmentation: Optional[DataAugmentation] = None
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation

        # Collect all .pt files
        self.files = sorted(self.data_path.rglob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt files found in {data_path}")
        print(f"Found {len(self.files)} data files")

        # Infer hidden_size and validate dimensions
        first_sample = torch.load(self.files[0], map_location="cpu")
        self.base_hidden = first_sample["hidden_states"].size(-1) // FUSE_LAYERS
        self.expected_dim = self.base_hidden * FUSE_LAYERS

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")

        # Dimension validation
        hs = data["hidden_states"]
        if hs.size(-1) != self.expected_dim:
            raise ValueError(
                f"{self.files[idx]} hidden dim {hs.size(-1)} ≠ {self.expected_dim} "
                f"(Expected {FUSE_LAYERS} × {self.base_hidden})"
            )

        # Truncate to max length
        seq_len = min(data["input_ids"].size(0), self.max_seq_len)

        sample = {
            "input_ids": data["input_ids"][:seq_len],
            "hidden_states": hs[:seq_len],
            "attention_mask": data.get("attention_mask", torch.ones(seq_len))[:seq_len],
            "loss_mask": data.get("loss_mask", torch.ones(seq_len))[:seq_len],
        }

        # Apply augmentation if specified
        if self.augmentation:
            sample["hidden_states"] = self.augmentation(sample["hidden_states"])

        return sample


def collate_fn(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    max_len = max(s['input_ids'].size(0) for s in samples)
    batch_size = len(samples)
    
    # Initialize tensors
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    hidden_states = torch.zeros(batch_size, max_len, samples[0]['hidden_states'].size(-1))
    attention_mask = torch.zeros(batch_size, max_len)
    loss_mask = torch.zeros(batch_size, max_len)
    
    # Fill tensors
    for i, sample in enumerate(samples):
        seq_len = sample['input_ids'].size(0)
        input_ids[i, :seq_len] = sample['input_ids']
        hidden_states[i, :seq_len] = sample['hidden_states']
        attention_mask[i, :seq_len] = sample['attention_mask']
        loss_mask[i, :seq_len] = sample['loss_mask']
    
    return {
        'input_ids': input_ids,
        'hidden_states': hidden_states,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
    }


# ------------------------------------------------------------------
# 4️⃣  Training Logic
# ------------------------------------------------------------------

class TrainingTimeTest:
    """Implements training-time test for EAGLE V3."""
    
    def __init__(self, draft_model: DraftModelV3, num_steps: int = 2):
        self.draft_model = draft_model
        self.num_steps = num_steps
        self._cached_teacher_logits = None
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        base_lm_head: nn.Module,
        step_weights: List[float],
        base_model: nn.Module = None,
        kl_weight: float = 0.7,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform multi-step training with TTT and KL divergence.

        Returns:
            Dictionary containing losses and metrics
        """
        debug("\n>>> [DEBUG TTT] Starting TTT forward pass")

        input_ids = batch['input_ids']
        hidden_states = batch['hidden_states']  # Already fused features
        attention_mask = batch['attention_mask']
        loss_mask = batch['loss_mask']

        target_dtype = next(self.draft_model.parameters()).dtype

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        debug(f">>> [DEBUG TTT] Batch size: {batch_size}, Seq len: {seq_len}")
        debug(f">>> [DEBUG TTT] Initial hidden_states shape: {hidden_states.shape}")

        if self.num_steps == 0:
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'step_losses': [],
                'step_accuracies': [],
                'ce_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'kl_loss': torch.tensor(0.0, device=device, requires_grad=True),
            }

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl_loss = torch.tensor(0.0, device=device, requires_grad=True)

        step_losses = []
        step_accuracies = []

        # Shift for next-token prediction
        input_ids_shifted = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]
        hidden_states_shifted = hidden_states[:, :-1].to(target_dtype)
        loss_mask_shifted = loss_mask[:, 1:]
        attention_mask_shifted = attention_mask[:, :-1]

        # Get teacher logits only once (optimization)
        teacher_logits = None
        if base_model is not None and kl_weight > 0:
            with torch.no_grad():
                teacher_outputs = base_model(
                    input_ids=input_ids_shifted,
                    attention_mask=attention_mask_shifted,
                    output_hidden_states=False,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits.detach()
                self._cached_teacher_logits = teacher_logits  # Cache for reuse

        # Current hidden state and position tracking
        current_hidden = None
        cumulative_position_offset = 0

        for step in range(self.num_steps):
            debug(f"\n>>> [DEBUG TTT] Step {step}/{self.num_steps-1}")

            if step == 0:
                # First step: use base model features (fused)
                seq_len_shifted = input_ids_shifted.shape[1]
                position_ids = torch.arange(seq_len_shifted, dtype=torch.long, device=device).unsqueeze(0)

                debug(f">>> [DEBUG TTT] Step 0: Using fused features from base model")
                debug(f">>> [DEBUG TTT] hidden_states_shifted shape: {hidden_states_shifted.shape}")

                logits, current_hidden = self.draft_model(
                    hidden_states_shifted,
                    input_ids=input_ids_shifted,
                    attention_mask=attention_mask_shifted,
                    position_ids=position_ids,
                    is_fused_features=True,
                    ttt_step=step,
                )
            else:
                # Subsequent steps: use previous predictions
                debug(f">>> [DEBUG TTT] Step {step}: Using previous hidden states")

                # Sample from previous logits
                with torch.no_grad():
                    probs = F.softmax(logits, dim=-1)
                    sampled_ids = torch.multinomial(
                        probs.view(-1, probs.size(-1)), 1
                    ).view(batch_size, -1)
                    debug(f">>> [DEBUG TTT] Sampled IDs shape: {sampled_ids.shape}")

                current_seq_len = current_hidden.shape[1]
                
                # Create position IDs with proper offset
                position_ids = torch.arange(
                    cumulative_position_offset,
                    cumulative_position_offset + current_seq_len,
                    dtype=torch.long,
                    device=device
                ).unsqueeze(0)

                current_attention_mask = torch.ones(batch_size, current_seq_len, device=device)

                debug(f">>> [DEBUG TTT] Current hidden shape: {current_hidden.shape}")
                debug(f">>> [DEBUG TTT] Position IDs: {position_ids[0][:10]}...")  # Show first 10

                # Forward pass
                logits, current_hidden = self.draft_model(
                    current_hidden,
                    input_ids=sampled_ids,
                    attention_mask=current_attention_mask,
                    position_ids=position_ids,
                    is_fused_features=False,
                    ttt_step=step,
                )
                
                # Update position offset for next step
                cumulative_position_offset += 1

            # Compute losses
            valid_positions = loss_mask_shifted > 0
            if valid_positions.any():
                # Flatten for loss computation
                logits_flat = logits[valid_positions]
                targets_flat = target_ids[:, :logits.size(1)][valid_positions[:, :logits.size(1)]]

                # Cross-entropy loss
                ce_loss = F.cross_entropy(logits_flat, targets_flat)

                # KL divergence loss (reuse cached teacher logits)
                kl_loss = torch.tensor(0.0, device=device, requires_grad=True)
                if teacher_logits is not None and kl_weight > 0:
                    # Temperature for KL
                    temperature = 2.0

                    # Get teacher logits for valid positions
                    teacher_logits_flat = teacher_logits[:, :logits.size(1)][valid_positions[:, :logits.size(1)]]

                    # Compute KL divergence with temperature scaling
                    log_probs_student = F.log_softmax(logits_flat / temperature, dim=-1)
                    probs_teacher = F.softmax(teacher_logits_flat / temperature, dim=-1)

                    kl_loss = F.kl_div(
                        log_probs_student,
                        probs_teacher,
                        reduction='batchmean'
                    ) * (temperature ** 2)

                # Combined loss
                loss = ce_loss + kl_weight * kl_loss

                # Apply step weight
                weighted_loss = loss * step_weights[step]
                total_loss = total_loss + weighted_loss
                total_ce_loss = total_ce_loss + ce_loss * step_weights[step]
                total_kl_loss = total_kl_loss + kl_loss * step_weights[step]

                step_losses.append(loss.item())

                # Compute accuracy
                with torch.no_grad():
                    predictions = logits_flat.argmax(dim=-1)
                    accuracy = (predictions == targets_flat).float().mean()
                    step_accuracies.append(accuracy.item())

                debug(f">>> [DEBUG TTT] Step {step} - CE Loss: {ce_loss.item():.4f}, "
                      f"KL Loss: {kl_loss.item():.4f}, "
                      f"Total Loss: {loss.item():.4f}, Acc: {accuracy.item():.4f}")

        # Average over steps
        num_valid_steps = len(step_losses) if step_losses else 1
        avg_loss = total_loss / num_valid_steps
        avg_ce_loss = total_ce_loss / num_valid_steps
        avg_kl_loss = total_kl_loss / num_valid_steps

        debug(f">>> [DEBUG TTT] Total steps processed: {len(step_losses)}")
        debug(f">>> [DEBUG TTT] Average loss: {avg_loss.item():.4f}")

        return {
            'loss': avg_loss,
            'ce_loss': avg_ce_loss,
            'kl_loss': avg_kl_loss,
            'step_losses': step_losses,
            'step_accuracies': step_accuracies,
        }


class EagleTrainer:
    """Main trainer for EAGLE V3."""
    
    def __init__(
        self,
        config: TrainingConfig,
        draft_model: DraftModelV3,
        base_model: nn.Module,
        accelerator: Accelerator,
    ):
        self.config = config
        self.draft_model = draft_model
        self.base_model = base_model
        self.accelerator = accelerator
        
        # TTT handler
        self.ttt = TrainingTimeTest(draft_model, config.ttt_steps)
        
        # Metrics tracking
        self.step = 0
        self.epoch = 0
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ):
        """Main training loop."""
        
        # Initialize wandb if available
        if self.accelerator.is_main_process and has_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self._train_epoch(
                train_loader, optimizer, scheduler
            )
            
            # Validation
            if val_loader is not None and epoch % 5 == 0:
                val_metrics = self._validate(val_loader)
                
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}")
                    if has_wandb:
                        wandb.log({
                            'val/loss': val_metrics['loss'],
                            'val/accuracy': val_metrics['accuracy'],
                            'epoch': epoch,
                        })
            
            # Save checkpoint
            if epoch % 10 == 0 or epoch == self.config.num_epochs - 1:
                self._save_checkpoint(epoch)
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ) -> Dict[str, float]:
        """Train one epoch with KL divergence."""

        self.draft_model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch}",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):
            debug(f"\n{'='*60}")
            debug(f">>> [DEBUG] Processing batch {batch_idx}")
            debug(f">>> [DEBUG] Batch hidden_states shape: {batch['hidden_states'].shape}")
            debug(f">>> [DEBUG] Batch input_ids shape: {batch['input_ids'].shape}")

            step_accuracies = []

            # Forward pass
            if self.config.ttt_steps > 0:
                # TTT forward pass with KL
                outputs = self.ttt.forward(
                    batch,
                    self.base_model.lm_head,
                    self.config.ttt_step_weights,
                    base_model=self.base_model if self.config.use_kl_loss else None,
                    kl_weight=self.config.kl_weight if self.config.use_kl_loss else 0.0,
                )
                loss = outputs['loss']
                ce_loss = outputs.get('ce_loss', loss)
                kl_loss = outputs.get('kl_loss', torch.tensor(0.0))
                step_accuracies = outputs.get('step_accuracies', [])
            else:
                # Non-TTT training with KL
                input_ids_shifted = batch['input_ids'][:, :-1]
                hidden_states_shifted = batch['hidden_states'][:, :-1]
                attention_mask_shifted = batch['attention_mask'][:, :-1]
                target_ids = batch['input_ids'][:, 1:]
                loss_mask_shifted = batch['loss_mask'][:, 1:]

                # Draft model forward
                logits, _ = self.draft_model(
                    hidden_states=hidden_states_shifted,
                    input_ids=input_ids_shifted,
                    attention_mask=attention_mask_shifted,
                    is_fused_features=True,
                    ttt_step=-1,  # Indicate non-TTT mode
                )

                # Get teacher logits if using KL
                teacher_logits = None
                if self.config.use_kl_loss and self.base_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.base_model(
                            input_ids=input_ids_shifted,
                            attention_mask=attention_mask_shifted,
                            output_hidden_states=False,
                            use_cache=False,
                        )
                        teacher_logits = teacher_outputs.logits.detach()

                valid_positions = loss_mask_shifted > 0
                if valid_positions.any():
                    logits_flat = logits[valid_positions]
                    targets_flat = target_ids[valid_positions]

                    # CE loss
                    ce_loss = F.cross_entropy(logits_flat, targets_flat)

                    # KL loss
                    kl_loss = torch.tensor(0.0, device=self.accelerator.device)
                    if teacher_logits is not None:
                        teacher_logits_flat = teacher_logits[valid_positions]
                        
                        temperature = self.config.kl_temperature
                        log_probs_student = F.log_softmax(logits_flat / temperature, dim=-1)
                        probs_teacher = F.softmax(teacher_logits_flat / temperature, dim=-1)

                        kl_loss = F.kl_div(
                            log_probs_student,
                            probs_teacher,
                            reduction='batchmean'
                        ) * (temperature ** 2)

                    # Combined loss
                    loss = ce_loss + self.config.kl_weight * kl_loss

                    with torch.no_grad():
                        accuracy = (logits_flat.argmax(-1) == targets_flat).float().mean()
                        step_accuracies.append(accuracy.item())
                else:
                    loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
                    ce_loss = loss
                    kl_loss = torch.tensor(0.0, device=self.accelerator.device)

            # Backward pass
            self.accelerator.backward(loss)

            # Gradient accumulation and optimizer step
            if self.accelerator.sync_gradients:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.draft_model.parameters(),
                    self.config.max_grad_norm
                )
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.step += 1

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_ce_loss += ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss
            epoch_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            if step_accuracies:
                epoch_accuracy += np.mean(step_accuracies)
            num_batches += 1

            # Update progress bar
            if num_batches % self.config.logging_steps == 0:
                avg_loss = epoch_loss / num_batches
                avg_ce = epoch_ce_loss / num_batches
                avg_kl = epoch_kl_loss / num_batches
                avg_acc = epoch_accuracy / num_batches

                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'ce': f"{avg_ce:.4f}",
                    'kl': f"{avg_kl:.4f}",
                    'acc': f"{avg_acc:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # Log to wandb
                if self.accelerator.is_main_process and has_wandb:
                    unwrapped_model = self.accelerator.unwrap_model(self.draft_model)
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/ce_loss': avg_ce,
                        'train/kl_loss': avg_kl,
                        'train/accuracy': avg_acc,
                        'train/learning_rate': scheduler.get_last_lr()[0],
                        'train/logit_scale': unwrapped_model.logit_scale.item(),
                        'train/step': self.step,
                    })

        return {
            'loss': epoch_loss / max(num_batches, 1),
            'ce_loss': epoch_ce_loss / max(num_batches, 1),
            'kl_loss': epoch_kl_loss / max(num_batches, 1),
            'accuracy': epoch_accuracy / max(num_batches, 1),
        }

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        
        self.draft_model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        target_dtype = next(self.accelerator.unwrap_model(self.draft_model).parameters()).dtype
        
        for batch in tqdm(val_loader, desc="Validation", disable=not self.accelerator.is_local_main_process):
            hidden_states_input = batch['hidden_states'][:, :-1].to(target_dtype)
            logits, _ = self.draft_model(
                hidden_states_input,
                input_ids=batch['input_ids'][:, :-1],
                attention_mask=batch['attention_mask'][:, :-1],
                is_fused_features=True,
                ttt_step=-1,  # No TTT during validation
            )
            
            # Compute loss
            targets = batch['input_ids'][:, 1:]
            loss_mask = batch['loss_mask'][:, 1:]
            
            valid_positions = loss_mask > 0
            if valid_positions.any():
                logits_flat = logits[valid_positions[:, :logits.size(1)]]
                targets_flat = targets[:, :logits.size(1)][valid_positions[:, :logits.size(1)]]
                
                loss = F.cross_entropy(logits_flat, targets_flat)
                accuracy = (logits_flat.argmax(-1) == targets_flat).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1),
        }
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""

        if self.accelerator.is_main_process:
            output_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.draft_model)
            torch.save(
                unwrapped_model.state_dict(),
                output_dir / "pytorch_model.bin"
            )

            # Save config
            save_config = unwrapped_model.config.to_dict()

            # Ensure key configuration items exist
            save_config["model_type"] = "qwen3" if "qwen" in self.config.base_model.lower() else save_config.get("model_type", "llama")
            save_config["tie_word_embeddings"] = False
            save_config["num_hidden_layers"] = self.config.num_decoder_layers

            # Save training-specific configuration
            save_config["_eagle_config"] = {
                "fuse_layers": FUSE_LAYERS,
                "base_model": self.config.base_model,
                "training_version": "v3",
                "use_simple_fusion": self.config.use_simple_fusion,
                "num_decoder_layers": self.config.num_decoder_layers,
            }

            with open(output_dir / "config.json", "w") as f:
                json.dump(save_config, f, indent=2)

            print(f"Saved checkpoint to {output_dir}")


# ------------------------------------------------------------------
# 5️⃣  Model Adapters for Different Architectures
# ------------------------------------------------------------------

class BaseModelAdapter:
    """Base adapter for different model architectures."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.config = model.config
    
    @property
    def device(self):
        return next(self.model.parameters()).device
    
    @property
    def dtype(self):
        return next(self.model.parameters()).dtype
    
    def get_fusion_indices(self, model_name: str) -> List[int]:
        """Get the correct layer indices for feature fusion with bounds checking."""
        model_name_lower = model_name.lower()
        num_hidden_layers = self.config.num_hidden_layers
        
        if "qwen" in model_name_lower:
            # Qwen-specific layer selection
            raw_indices = [2, num_hidden_layers // 3, 2 * num_hidden_layers // 3]
        else:
            # Default strategy for Llama-like models
            raw_indices = [num_hidden_layers // 4,
                          num_hidden_layers // 2,
                          num_hidden_layers - 2]
        
        # Remove duplicates and ensure valid indices
        indices = []
        seen = set()
        for idx in raw_indices:
            if idx not in seen and 0 <= idx < num_hidden_layers:
                indices.append(idx)
                seen.add(idx)
        
        # Validate we have the expected number of layers
        if len(indices) != FUSE_LAYERS:
            raise ValueError(
                f"Expected {FUSE_LAYERS} fusion layers, but got {len(indices)} valid indices "
                f"from {raw_indices} with {num_hidden_layers} hidden layers. "
                f"Model may be too small for the current fusion strategy."
            )
        
        return indices
    
    @torch.no_grad()
    def extract_features(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Extract multi-layer features from base model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        
        hidden_states = outputs.hidden_states
        return self.fuse_hidden_states(hidden_states)
    
    def fuse_hidden_states(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """Fuse hidden states from multiple layers."""
        indices = self.get_fusion_indices(self.model.config._name_or_path)
        
        selected = []
        for idx in indices:
            # hidden_states[0] is embeddings, so we need idx+1
            selected.append(hidden_states[idx + 1])
        
        # Concatenate along feature dimension
        fused = torch.cat(selected, dim=-1)
        
        return fused


class Qwen3ModelAdapter(BaseModelAdapter):
    """Adapter for Qwen3 models with specialized fusion strategy."""
    pass  # Fusion logic is now in the base class get_fusion_indices method


# ------------------------------------------------------------------
# 6️⃣  Tree Structure Utilities
# ------------------------------------------------------------------

TOPK = 5
DEFAULT_TREE = [
    [0], [1], [2], [3], [4],
    [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0],
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
]


class TreeNode:
    """Tree node for draft tree structure."""
    def __init__(self, path, parent=None):
        self.path = tuple(path) if isinstance(path, list) else path
        self.parent = parent
        self.children = []
        if parent:
            parent.children.append(self)
    
    def ancestors(self):
        """Get all ancestors of this node."""
        result = []
        current = self.parent
        while current:
            result.append(current)
            current = current.parent
        return result[::-1]  # Return in root-to-parent order


def build_tree_buffers(tree: List[List[int]], device: torch.device) -> Dict[str, List[torch.Tensor]]:
    """Build buffers for tree attention (Fixed implementation)."""
    
    # Build tree structure
    tree = sorted(tree, key=lambda x: (len(x), x))
    root = TreeNode(path=())
    node_map = {(): root}
    
    for path in tree:
        parent_path = tuple(path[:-1])
        if parent_path in node_map:
            node = TreeNode(path=tuple(path), parent=node_map[parent_path])
            node_map[tuple(path)] = node
    
    # Group nodes by depth
    nodes_by_depth = {}
    for path_tuple, node in node_map.items():
        if node.children:  # Only internal nodes
            depth = len(path_tuple)
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)
    
    # Build buffers
    attention_masks = []
    tree_indices = []
    position_ids = []
    repeat_patterns = []
    
    max_depth = max(nodes_by_depth.keys()) if nodes_by_depth else 0
    
    for depth in range(max_depth + 1):
        if depth not in nodes_by_depth:
            continue
        
        parent_nodes = nodes_by_depth[depth]
        
        # Collect all children
        children = []
        for parent in parent_nodes:
            children.extend(parent.children)
        
        if not children:
            continue
        
        num_children = len(children)
        
        # Tree indices (last element of each child's path)
        indices = torch.tensor([child.path[-1] for child in children], dtype=torch.long, device=device)
        tree_indices.append(indices)
        
        # Position IDs (depth + 1 for all children)
        pos_ids = torch.full((num_children,), depth + 1, dtype=torch.long, device=device)
        position_ids.append(pos_ids)
        
        # Repeat pattern (number of children per parent)
        pattern = [len(parent.children) for parent in parent_nodes]
        repeat_patterns.append(pattern)
        
        # Attention mask
        mask = torch.zeros(num_children, num_children, device=device)
        
        # Each child attends to its ancestors' children and siblings
        for i, child in enumerate(children):
            # Attend to self
            mask[i, i] = 1
            
            # Attend to siblings (children of same parent)
            for j, other_child in enumerate(children):
                if child.parent == other_child.parent and j <= i:
                    mask[i, j] = 1
            
            # Attend to ancestors' children
            for ancestor in child.ancestors():
                for j, other_child in enumerate(children):
                    if other_child.parent == ancestor:
                        mask[i, j] = 1
        
        attention_masks.append(mask[None, None])  # Add batch and head dimensions
    
    return {
        'attn': attention_masks,
        'idx': tree_indices,
        'pos': position_ids,
        'rep': repeat_patterns,
    }


# ------------------------------------------------------------------
# 7️⃣  Main Training Script
# ------------------------------------------------------------------

def main():
    # Parse arguments
    config, args = parse_args()
    
    # Set seed
    set_seed(config.seed)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )
    
    if accelerator.is_main_process:
        print(f"Training config: {config}")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16,
        device_map=accelerator.device,
        trust_remote_code=True,
    )
    
    # Freeze base model
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Create model adapter
    if "qwen" in config.base_model.lower():
        adapter = Qwen3ModelAdapter(base_model)
    else:
        adapter = BaseModelAdapter(base_model)
    
    # Initialize draft model
    print("Initializing draft model...")
    base_config = base_model.config
    
    draft_config_dict = base_config.to_dict()
    draft_config_dict['num_hidden_layers'] = config.num_decoder_layers
    draft_config_dict['tie_word_embeddings'] = False
    
    draft_config = base_config.__class__.from_dict(draft_config_dict)
    
    draft_model = DraftModelV3(
        draft_config, 
        num_decoder_layers=config.num_decoder_layers,
        use_simple_fusion=config.use_simple_fusion
    )
    
    # Load checkpoint if provided
    if config.draft_checkpoint:
        print(f"Loading checkpoint from {config.draft_checkpoint}")
        state_dict = torch.load(
            os.path.join(config.draft_checkpoint, "pytorch_model.bin"),
            map_location="cpu"
        )
        draft_model.load_state_dict(state_dict)
    
    # Set tree buffers
    draft_model.tree_buffers = build_tree_buffers(DEFAULT_TREE, accelerator.device)
    
    # Create datasets
    print("Loading datasets...")
    augmentation = None
    if config.use_noise:
        augmentation = DataAugmentation(config.noise_type, config.noise_std)
    
    train_dataset = EagleDataset(
        config.data_path,
        config.max_seq_len,
        augmentation=augmentation
    )
    
    # Split into train/val
    val_size = int(0.05 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        draft_model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare with accelerator
    draft_model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        draft_model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Create trainer
    trainer = EagleTrainer(
        config=config,
        draft_model=draft_model,
        base_model=base_model,
        accelerator=accelerator,
    )
    
    # Train
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    
    # Save final model
    if accelerator.is_main_process:
        final_dir = Path(config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(draft_model)
        torch.save(
            unwrapped_model.state_dict(),
            final_dir / "pytorch_model.bin"
        )
        
        # Save config with proper metadata
        save_config = draft_config.to_dict()
        save_config["_eagle_config"] = {
            "fuse_layers": FUSE_LAYERS,
            "base_model": config.base_model,
            "training_version": "v3",
            "use_simple_fusion": config.use_simple_fusion,
            "num_decoder_layers": config.num_decoder_layers,
        }
        
        if "qwen" in config.base_model.lower():
            save_config["model_type"] = "qwen3"
        
        with open(final_dir / "config.json", "w") as f:
            json.dump(save_config, f, indent=2)
        
        print(f"Training completed! Model saved to {final_dir}")


if __name__ == "__main__":
    main()