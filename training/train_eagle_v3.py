#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_eagle_v3.py - 改进的EAGLE V3训练脚本
使用统一的融合工具和改进的配置管理
"""

import sys
from pathlib import Path
# 添加eagle_v3_refactored到Python路径
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
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
from datetime import datetime
import traceback

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json
import math

# 从统一的模块导入
from eagle_core.fusion_utils import FUSE_LAYERS, get_fusion_indices, fuse_hidden_states

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)
#------------------------------------------------------------------
# Enhanced Logging Setup
#------------------------------------------------------------------

from pathlib import Path
import logging

log_dir = Path(__file__).resolve().parent / "logs"
log_dir.mkdir(exist_ok=True)

# Create timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Main training log
log_file = log_dir / f"train_{timestamp}.log"
# Error log for issues
error_log_file = log_dir / f"errors_{timestamp}.log"
# NaN debug log
nan_log_file = log_dir / f"nan_debug_{timestamp}.log"

# Configure main logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(str(log_file)),
        logging.StreamHandler()  # Also print to console
    ]
)

logger = logging.getLogger(__name__)

# Configure error logger
error_logger = logging.getLogger("error_logger")
error_handler = logging.FileHandler(str(error_log_file))
error_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

# Configure NaN debug logger
nan_logger = logging.getLogger("nan_logger")
nan_handler = logging.FileHandler(str(nan_log_file))
nan_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
nan_logger.addHandler(nan_handler)
nan_logger.setLevel(logging.DEBUG)

debug = logger.debug

#------------------------------------------------------------------
# NaN Detection and Logging Utilities
#------------------------------------------------------------------

def check_tensor_health(tensor: torch.Tensor, name: str, raise_on_nan: bool = False) -> bool:
    """Check if tensor contains NaN/Inf values and log details."""
    if tensor is None:
        return True
        
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        nan_logger.error(f"Unhealthy tensor detected: {name}")
        nan_logger.error(f"  Shape: {tensor.shape}")
        nan_logger.error(f"  Has NaN: {has_nan}")
        nan_logger.error(f"  Has Inf: {has_inf}")
        nan_logger.error(f"  Min: {tensor.min().item() if not has_nan else 'N/A'}")
        nan_logger.error(f"  Max: {tensor.max().item() if not has_nan else 'N/A'}")
        nan_logger.error(f"  Mean: {tensor.mean().item() if not has_nan else 'N/A'}")
        
        # Sample some values
        flat = tensor.flatten()
        sample_size = min(10, flat.size(0))
        nan_logger.error(f"  Sample values: {flat[:sample_size].tolist()}")
        
        if raise_on_nan:
            raise ValueError(f"NaN/Inf detected in {name}")
        return False
    return True

def log_batch_stats(batch: Dict[str, torch.Tensor], prefix: str = ""):
    """Log statistics about a batch for debugging."""
    nan_logger.info(f"{prefix} Batch statistics:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            nan_logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            if value.numel() > 0:
                try:
                    nan_logger.info(f"    min={value.min().item():.6f}, max={value.max().item():.6f}, "
                                   f"mean={value.mean().item():.6f}")
                except:
                    nan_logger.info(f"    Could not compute statistics (likely contains NaN)")

#------------------------------------------------------------------
# 1️⃣  Configuration & Arguments (Enhanced with auto-detection)
#------------------------------------------------------------------

def make_causal_mask(
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
    """
    if ttt_step == 0:
        return make_causal_mask((batch_size, seq_len), dtype, device)

    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    block_size = max(1, seq_len // (ttt_step + 1))  # Ensure non-zero

    for i in range(seq_len):
        block_idx = i // block_size
        start_idx = 0
        end_idx = min((block_idx + 1) * block_size, seq_len)
        mask[i, start_idx:end_idx] = 0

    mask_cond = torch.arange(seq_len, device=device)
    causal_mask = mask_cond < (mask_cond + 1).view(seq_len, 1)
    mask = torch.where(causal_mask & (mask == 0), torch.tensor(0.0, dtype=dtype, device=device), mask)

    return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)


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
    vocab_size: int = 151936
    hidden_size: int = 5120  # Will be auto-detected from data
    data_path: str = ""
    draft_checkpoint: Optional[str] = None

    # Multi-layer draft model
    num_decoder_layers: int = 4

    # KL divergence
    use_kl_loss: bool = True
    kl_weight: float = 0.7
    kl_temperature: float = 2.0
    use_topk_kl: bool = True
    topk_size: int = 50

    # Robustness parameters
    gradient_clip_val: float = 1.0
    eps: float = 1e-8  # Small epsilon for numerical stability
    max_logit_value: float = 30.0  # Clamp logits to prevent overflow
    skip_nan_batches: bool = True  # Skip batches that produce NaN
    nan_debug_mode: bool = True  # Extra logging for NaN debugging
    checkpoint_on_nan: bool = True  # Save checkpoint when NaN detected
    auto_detect_hidden_size: bool = True  # Auto-detect hidden size from data

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
    ttt_steps: int = 2
    ttt_step_weights: List[float] = None

    # Loss weights
    token_loss_weight: float = 1.0
    # EAGLE-3不使用特征回归损失，只关注词元预测，按照论文要求设为0
    feature_loss_weight: float = 0.0  # 设为0表示完全不使用特征回归损失

    # Data augmentation
    use_noise: bool = True
    noise_type: str = "uniform"
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

    scheduler_type: str = "cosine" 

    # Use simple fusion as in paper
    use_simple_fusion: bool = True

    def __post_init__(self):
        if self.ttt_step_weights is None:
            self.ttt_step_weights = [1.0 / (i + 1) for i in range(self.ttt_steps)]


def parse_args():
    parser = argparse.ArgumentParser(description='Train EAGLE V3 Draft Model')

    # Model
    parser.add_argument('--vocab_size', type=int, required=True,
                       help='Vocabulary size of the model')
    parser.add_argument('--hidden_size', type=int, default=None,
                       help='Hidden size of the model (auto-detected if not specified)')
    parser.add_argument('--draft_checkpoint', type=str, default=None,
                       help='Path to resume training from')

    parser.add_argument('--num_decoder_layers', type=int, default=4,
                       help='Number of decoder layers in draft model')
    parser.add_argument('--use_kl_loss', action='store_true', default=False,
                       help='Use KL divergence loss')
    parser.add_argument('--kl_weight', type=float, default=0.7,
                       help='Weight for KL divergence loss')

    parser.add_argument('--scheduler_type', type=str, default='cosine',
                       choices=['linear', 'cosine'],
                       help='Learning rate scheduler type (linear or cosine)')
    
    # Top-k KL arguments
    parser.add_argument('--use_topk_kl', action='store_true', default=True,
                       help='Use top-k KL divergence loss')
    parser.add_argument('--topk_size', type=int, default=50,
                       help='Size of top-k for KL computation')
    
    # Robustness arguments
    parser.add_argument('--skip_nan_batches', action='store_true', default=True,
                       help='Skip batches that produce NaN')
    parser.add_argument('--nan_debug_mode', action='store_true', default=False,
                       help='Enable detailed NaN debugging')
    parser.add_argument('--max_logit_value', type=float, default=30.0,
                       help='Maximum absolute value for logits clamping')
    parser.add_argument('--auto_detect_hidden_size', action='store_true', default=True,
                       help='Auto-detect hidden size from data')
    
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
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size if args.hidden_size is not None else 5120,  # Default value
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
        use_topk_kl=args.use_topk_kl,
        topk_size=args.topk_size,
        enable_trace=args.enable_trace,
        use_simple_fusion=args.use_simple_fusion,
        skip_nan_batches=args.skip_nan_batches,
        nan_debug_mode=args.nan_debug_mode,
        max_logit_value=args.max_logit_value,
        auto_detect_hidden_size=args.auto_detect_hidden_size,
    )

    return config, args


#------------------------------------------------------------------
# 2️⃣  Model Components (Enhanced with auto hidden size)
#------------------------------------------------------------------

class SimpleFusion(nn.Module):
    """Simple fusion module that just does a linear projection to reduce dimensionality."""

    def __init__(self, base_hidden_size: int, n_layers: int = 3):
        super().__init__()
        self.linear = nn.Linear(base_hidden_size * n_layers, base_hidden_size, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with added NaN detection and recovery."""
        # 检测输入是否包含NaN
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_logger.warning("NaN/Inf detected in SimpleFusion input, applying recovery")
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            
        # 正常的前向计算
        fusion_output = self.linear(x)
        
        # 检测输出是否包含NaN
        if torch.isnan(fusion_output).any() or torch.isinf(fusion_output).any():
            nan_logger.error("Unhealthy tensor detected: SimpleFusion output")
            fusion_output = torch.nan_to_num(fusion_output, nan=0.0, posinf=0.0, neginf=0.0)
            
        return fusion_output


class MultiLayerFeatureFusion(nn.Module):
    """Learnable fusion of features from multiple layers."""

    def __init__(self, n_layers: int, base_hidden_size: int):
        super().__init__()
        self.n_layers = n_layers
        self.base_hidden_size = base_hidden_size
        
        # Attention-based fusion
        self.layer_attention = nn.Sequential(
            nn.Linear(n_layers * base_hidden_size, base_hidden_size),
            nn.SiLU(),
            nn.Linear(base_hidden_size, n_layers)
        )
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(n_layers * base_hidden_size, base_hidden_size),
            nn.Sigmoid()
        )
        
        # Final projection
        self.proj = nn.Linear(n_layers * base_hidden_size, base_hidden_size, bias=False)

    def forward(self, multi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_features: [batch, seq_len, n_layers * base_hidden_size]
        Returns:
            fused: [batch, seq_len, base_hidden_size]
        """
        batch_size, seq_len, _ = multi_features.shape
        
        # Compute attention weights with numerical stability
        attn_logits = self.layer_attention(multi_features)
        attn_logits = torch.clamp(attn_logits, min=-10, max=10)  # Prevent overflow
        attn_weights = F.softmax(attn_logits, dim=-1)  # [batch, seq_len, n_layers]
        
        # Reshape and apply attention
        features = multi_features.view(batch_size, seq_len, self.n_layers, self.base_hidden_size)
        attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq_len, n_layers, 1]
        
        # Weighted sum
        weighted = (features * attn_weights).sum(dim=2)  # [batch, seq_len, base_hidden_size]
        
        # Gated fusion
        gate = self.gate(multi_features)
        projected = self.proj(multi_features)
        
        output = gate * weighted + (1 - gate) * projected
        
        # Check health
        if not check_tensor_health(output, "MultiLayerFeatureFusion output"):
            logger.warning("Fusion produced NaN, returning simple average")
            return features.mean(dim=2)
            
        return output


class DraftModelV3(nn.Module):
    """EAGLE V3 Draft Model with Multi-layer support and auto-detected base hidden size."""

    def __init__(self, vocab_size: int, base_hidden_size: int, student_hidden_size: int, 
                 num_decoder_layers: int = 4, use_simple_fusion: bool = True, 
                 max_logit_value: float = 30.0):
        super().__init__()
        self.base_hidden_size = base_hidden_size  # Hidden size from teacher model
        self.student_hidden_size = student_hidden_size  # Hidden size for student model
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers
        self.max_logit_value = max_logit_value
        
        class MinimalConfig(Qwen3Config):
            def __init__(self,
                         hidden_size,
                         vocab_size,
                         num_decoder_layers=4,
                         **kwargs):
                super().__init__(
                    hidden_size=hidden_size,
                    vocab_size=vocab_size,
                    num_hidden_layers=num_decoder_layers,
                    **kwargs
                )
                
        self.config = MinimalConfig(student_hidden_size, vocab_size, num_decoder_layers)
        
        # Multi-layer feature fusion (maps from base_hidden_size to student_hidden_size)
        if use_simple_fusion:
            self.feature_fusion = SimpleFusion(self.base_hidden_size, n_layers=FUSE_LAYERS)
            # If base and student sizes differ, add projection
            if self.base_hidden_size != self.student_hidden_size:
                self.hidden_proj = nn.Linear(self.base_hidden_size, self.student_hidden_size)
            else:
                self.hidden_proj = None
        else:
            self.feature_fusion = MultiLayerFeatureFusion(FUSE_LAYERS, self.base_hidden_size)
            # If base and student sizes differ, add projection
            if self.base_hidden_size != self.student_hidden_size:
                self.hidden_proj = nn.Linear(self.base_hidden_size, self.student_hidden_size)
            else:
                self.hidden_proj = None
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(vocab_size, self.student_hidden_size, self.config.pad_token_id)
        
        # Rotary Embedding
        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)
        
        # Multiple decoder layers
        self.decoder_layers = nn.ModuleList([
            Qwen3DecoderLayer(self.config, layer_idx=i) 
            for i in range(num_decoder_layers)
        ])
        
        # Output projection
        self.norm = Qwen3RMSNorm(self.student_hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.student_hidden_size, self.vocab_size, bias=False)
        
        # Learnable logit scale
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        
        # Tree buffers
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
        """Forward pass for training with enhanced stability."""
        
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Check input health
        check_tensor_health(hidden_states, "DraftModel input hidden_states")
        
        # Handle input features
        if is_fused_features:
            hidden = self.feature_fusion(hidden_states)
            # Apply projection if needed
            if self.hidden_proj is not None:
                hidden = self.hidden_proj(hidden)
        else:
            hidden = hidden_states
        
        # Add token embeddings if provided
        if input_ids is not None:
            token_embeds = self.embed_tokens(input_ids)
            check_tensor_health(token_embeds, "Token embeddings")
            hidden = hidden + token_embeds
        
        # Normalize hidden states to prevent explosion
        hidden = hidden / math.sqrt(self.student_hidden_size)
        
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
            attention_mask_4d = make_causal_mask(
                attention_mask.shape, hidden.dtype, device=device
            )
        
        # Handle position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate position embeddings
        position_embeddings = self.rotary_emb(hidden, position_ids)
        
        # Pass through decoder layers with gradient checkpointing
        for i, decoder_layer in enumerate(self.decoder_layers):
            # Check hidden health before each layer
            if not check_tensor_health(hidden, f"Hidden before layer {i}"):
                logger.warning(f"NaN detected before layer {i}, skipping remaining layers")
                break
                
            hidden = decoder_layer(
                hidden,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )[0]
        
        # Output projection
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        
        # Apply learnable logit scale with clamping
        logit_scale_clamped = torch.clamp(self.logit_scale, min=0.1, max=10.0)
        logits = logits * logit_scale_clamped
        
        # Clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-self.max_logit_value, max=self.max_logit_value)
        
        # Final health check
        if not check_tensor_health(logits, "Final logits"):
            logger.error("NaN in final logits, returning zeros")
            logits = torch.zeros_like(logits)
        
        return logits, hidden


#------------------------------------------------------------------
# 3️⃣  Data Components (Enhanced with field compatibility)
#------------------------------------------------------------------

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
        
        # Clip noise to prevent extreme values
        noise = torch.clamp(noise, min=-2*self.std, max=2*self.std)
        
        return hidden_states + noise


class EagleDataset(Dataset):
    """Dataset for EAGLE training with enhanced validation and field compatibility."""
    
    def __init__(
        self,
        data_path: str,
        vocab_size: int,
        max_seq_len: int = 2048,
        augmentation: Optional[DataAugmentation] = None
    ):
        self.data_path = Path(data_path)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation

        # Collect all .pt files
        self.files = sorted(self.data_path.rglob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt files found in {data_path}")
        logger.info(f"Found {len(self.files)} data files")

        # Validate files and filter out corrupted ones
        self.valid_files = []
        for f in tqdm(self.files, desc="Validating data files"):
            try:
                data = torch.load(f, map_location="cpu")
                if self._validate_sample(data):
                    self.valid_files.append(f)
                else:
                    error_logger.warning(f"Invalid data file: {f}")
            except Exception as e:
                error_logger.error(f"Failed to load {f}: {e}")
        
        if not self.valid_files:
            raise ValueError("No valid data files found!")
        
        logger.info(f"Using {len(self.valid_files)} valid files out of {len(self.files)}")
        self.files = self.valid_files

        # Infer dimensions
        first_sample = torch.load(self.files[0], map_location="cpu")
        hidden_states = first_sample.get("hidden_states")
        if hidden_states is None:
            raise ValueError("No 'hidden_states' found in data files")
            
        # Handle different tensor shapes
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            
        self.base_hidden = hidden_states.size(-1) // FUSE_LAYERS
        self.expected_dim = self.base_hidden * FUSE_LAYERS
        
        logger.info(f"Detected base hidden size: {self.base_hidden}")
        logger.info(f"Expected fused dimension: {self.expected_dim}")

    def _validate_sample(self, data: Dict) -> bool:
        """Validate raw sample data."""
        # Check required fields
        required_fields = ['hidden_states', 'input_ids', 'attention_mask']
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate hidden_states tensor
        hidden_states = data['hidden_states']
        if not isinstance(hidden_states, torch.Tensor):
            logger.warning(f"hidden_states is not a torch tensor: {type(hidden_states)}")
            return False
            
        # Check for NaN values in hidden states
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            logger.warning("NaN or Inf values found in hidden_states")
            return False
            
        # 检查teacher_topk_probs中的NaN/Inf值
        if 'teacher_topk_probs' in data:
            teacher_probs = data['teacher_topk_probs']
            if torch.isnan(teacher_probs).any() or torch.isinf(teacher_probs).any():
                logger.warning("NaN or Inf values found in teacher_topk_probs")
                return False
            
        # Check hidden_states dimensions - expect last dim to be multiple of FUSE_LAYERS
        if hidden_states.size(-1) % FUSE_LAYERS != 0:
            logger.warning(f"Hidden states dim {hidden_states.size(-1)} is not divisible by FUSE_LAYERS={FUSE_LAYERS}")
            # 不直接返回False，因为某些模型可能有特殊情况
            
        # 检查teacher数据的一致性
        if 'teacher_topk_probs' in data and 'teacher_topk_indices' in data:
            topk_probs = data['teacher_topk_probs']
            topk_indices = data['teacher_topk_indices']
            
            # 确保两个张量匹配
            if topk_probs.shape[:-1] != topk_indices.shape[:-1]:
                logger.warning(f"Teacher data shape mismatch: {topk_probs.shape} vs {topk_indices.shape}")
                return False
                
            # 检查概率是否是有效分布
            if topk_probs.min() < 0 or topk_probs.max() > 1.1:  # 允许轻微浮点误差
                logger.warning(f"Invalid probability values: min={topk_probs.min()}, max={topk_probs.max()}")
                return False
                
        return True

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get a single sample."""
        try:
            # Try to load the sample
            path = self.files[idx]
            data = torch.load(path, map_location='cpu')
            
            # 更严格的数据质量验证：增强对fusion_features的检查
            if not self._validate_sample(data):
                logger.warning(f"Invalid sample detected at {path}, using dummy")
                return self._get_dummy_sample()
            
            # Process the loaded sample
            def squeeze_if_needed(t):
                """Remove batch dimension if it exists and is 1."""
                if t.dim() > 1 and t.size(0) == 1:
                    return t.squeeze(0)
                return t
            
            # Extract fields
            hidden_states = data.get('hidden_states')
            input_ids = data.get('input_ids')
            attention_mask = data.get('attention_mask')
            
            # 检查FUSION维度，确保与get_fusion_indices函数的输出匹配
            num_fused_layers = data.get('num_fused_layers', FUSE_LAYERS)
            if num_fused_layers != FUSE_LAYERS:
                logger.warning(f"Fusion layers mismatch in {path}: {num_fused_layers} vs expected {FUSE_LAYERS}")
            
            # 检查并调整维度
            hidden_states = squeeze_if_needed(hidden_states)
            input_ids = squeeze_if_needed(input_ids)
            attention_mask = squeeze_if_needed(attention_mask)
            
            # Take slice if too long
            if self.max_seq_len > 0:
                if hidden_states.size(0) > self.max_seq_len:
                    end_idx = min(self.max_seq_len, hidden_states.size(0))
                    hidden_states = hidden_states[:end_idx]
                    input_ids = input_ids[:end_idx]
                    attention_mask = attention_mask[:end_idx]
                    
                    # Update teacher data if present
                    if 'teacher_topk_probs' in data and 'teacher_topk_indices' in data:
                        data['teacher_topk_probs'] = data['teacher_topk_probs'][:end_idx]
                        data['teacher_topk_indices'] = data['teacher_topk_indices'][:end_idx]
            
            # Apply data augmentation if specified
            if self.augmentation is not None:
                hidden_states = self.augmentation(hidden_states)

            # Create loss mask (non-padding tokens)
            loss_mask = attention_mask.clone()
            
            # Create sample
            sample = {
                'hidden_states': hidden_states,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'teacher_temperature': data.get('teacher_temperature', 1.0)
            }
            
            # Add teacher data if available
            if 'teacher_topk_probs' in data and 'teacher_topk_indices' in data:
                # 将教师模型的topk数据添加到样本中，进行适当的维度调整
                teacher_topk_probs = squeeze_if_needed(data['teacher_topk_probs'])
                teacher_topk_indices = squeeze_if_needed(data['teacher_topk_indices'])
                
                # 检查维度
                if teacher_topk_probs.size(0) != input_ids.size(0):
                    logger.warning(f"Teacher data length mismatch in {path}: {teacher_topk_probs.size(0)} vs {input_ids.size(0)}")
                    # 调整维度以匹配
                    min_len = min(teacher_topk_probs.size(0), input_ids.size(0))
                    teacher_topk_probs = teacher_topk_probs[:min_len]
                    teacher_topk_indices = teacher_topk_indices[:min_len]
                    sample['input_ids'] = sample['input_ids'][:min_len]
                    sample['attention_mask'] = sample['attention_mask'][:min_len]
                    sample['loss_mask'] = sample['loss_mask'][:min_len]
                    sample['hidden_states'] = sample['hidden_states'][:min_len]
                
                sample['teacher_topk_probs'] = teacher_topk_probs
                sample['teacher_topk_indices'] = teacher_topk_indices
                
                # 检查词表ID是否在范围内
                if teacher_topk_indices.max() >= self.vocab_size:
                    logger.warning(f"Teacher token indices out of bounds in {path}: max={teacher_topk_indices.max()}, vocab_size={self.vocab_size}")
                    # 将超出范围的ID替换为UNK或其他安全的ID
                    mask = teacher_topk_indices >= self.vocab_size
                    if mask.any():
                        teacher_topk_indices[mask] = 0  # 替换为UNK或另一个安全值
                        sample['teacher_topk_indices'] = teacher_topk_indices
            
            # 增强对输出样本的验证
            if not self._validate_processed_sample(sample):
                logger.warning(f"Processed sample validation failed for {path}, using dummy")
                return self._get_dummy_sample()
                
            return sample
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {self.files[idx] if idx < len(self.files) else 'unknown'}: {e}")
            return self._get_dummy_sample()

    def _validate_processed_sample(self, sample: Dict) -> bool:
        """Validate processed sample."""
        seq_len = sample["input_ids"].size(0)
        for key in ["hidden_states", "attention_mask", "loss_mask", "teacher_topk_indices", "teacher_topk_probs"]:
            if sample[key].size(0) != seq_len:
                return False
        return True

    def _get_dummy_sample(self):
        """Get a dummy sample for error recovery."""
        seq_len = 10
        return {
            "input_ids": torch.zeros(seq_len, dtype=torch.long),
            "hidden_states": torch.randn(seq_len, self.expected_dim) * 0.02,
            "attention_mask": torch.ones(seq_len),
            "loss_mask": torch.ones(seq_len),
            "teacher_topk_indices": torch.zeros(seq_len, 50, dtype=torch.long),
            "teacher_topk_probs": torch.ones(seq_len, 50) / 50,
            "teacher_temperature": 1.0,
        }
        
    def get_base_hidden_size(self):
        """Return the detected base hidden size."""
        return self.base_hidden


# training/train_eagle_v3.py

def collate_fn(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """Robust collate function with added diagnostics."""
    
    # 过滤无效样本 (您的代码中已有，保持不变)
    valid_samples = []
    for i, sample in enumerate(samples):
        try:
            # 增加一个检查，确保 'hidden_states' 存在且为2D张量
            if 'hidden_states' in sample and isinstance(sample['hidden_states'], torch.Tensor) and sample['hidden_states'].dim() == 2:
                 valid_samples.append(sample)
            else:
                error_logger.warning(f"Skipping sample {i} due to missing, invalid, or non-2D 'hidden_states'.")
        except Exception as e:
            error_logger.warning(f"Skipping sample {i} due to error during validation: {e}")
    
    if not valid_samples:
        error_logger.error("No valid samples in batch! Returning a dummy batch.")
        # 返回一个明确知道形状的虚拟批次
        return {
            'input_ids': torch.zeros(1, 10, dtype=torch.long),
            'hidden_states': torch.randn(1, 10, 15360), # 使用您期望的维度
            'attention_mask': torch.ones(1, 10),
            'loss_mask': torch.ones(1, 10),
            'teacher_topk_indices': torch.zeros(1, 10, 50, dtype=torch.long),
            'teacher_topk_probs': torch.ones(1, 10, 50) / 50,
            'teacher_temperature': 1.0,
        }
    
    samples = valid_samples
    max_len = max(s['input_ids'].size(0) for s in samples)
    batch_size = len(samples)

    # --- 诊断点 1: 检查第一个样本并建立基准维度 ---
    try:
        first_sample_hs = samples[0]['hidden_states']
        hidden_dim = first_sample_hs.size(-1)
        print(f"[DEBUG] collate_fn: Baseline hidden_dim={hidden_dim} from first sample shape {first_sample_hs.shape}")
    except (KeyError, IndexError) as e:
        print(f"[ERROR] collate_fn: Failed to get hidden_dim from the first sample. Error: {e}")
        raise

    # 初始化批次张量
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    hidden_states = torch.zeros(batch_size, max_len, hidden_dim)
    attention_mask = torch.zeros(batch_size, max_len)
    loss_mask = torch.zeros(batch_size, max_len)
    # 假设topk_size已知或从第一个样本获取
    topk_size = samples[0].get('teacher_topk_indices', torch.zeros(1,1,50)).shape[-1]
    teacher_topk_indices = torch.zeros(batch_size, max_len, topk_size, dtype=torch.long)
    teacher_topk_probs = torch.zeros(batch_size, max_len, topk_size)

    # --- 诊断点 2: 在循环中检查每个样本 ---
    for i, sample in enumerate(samples):
        try:
            seq_len = sample['input_ids'].size(0)
            
            # 获取源张量和目标切片
            source_tensor = sample['hidden_states']
            target_slice = hidden_states[i, :seq_len]
            
            # 在赋值前进行形状比较
            if source_tensor.shape != target_slice.shape:
                print(f"\n\n\n!!!!! SHAPE MISMATCH DETECTED in collate_fn !!!!!")
                print(f"  - Batch item index: {i}")
                print(f"  - Target slice shape for assignment: {target_slice.shape}")
                print(f"  - Source tensor 'hidden_states' shape: {source_tensor.shape}")
                print(f"  - This means the hidden_dim of this sample ({source_tensor.shape[-1]}) does not match the batch's hidden_dim ({hidden_dim}).")
                # 为了防止程序崩溃，我们可以跳过这个坏样本，而不是让它崩溃
                print(f"  - SKIPPING this corrupted sample to continue training.")
                continue # 跳过这个样本

            # 如果形状匹配，则正常赋值
            hidden_states[i, :seq_len] = source_tensor
            
            # 处理其他张量
            input_ids[i, :seq_len] = sample['input_ids']
            attention_mask[i, :seq_len] = sample['attention_mask']
            loss_mask[i, :seq_len] = sample['loss_mask']
            if 'teacher_topk_indices' in sample:
                teacher_topk_indices[i, :seq_len] = sample['teacher_topk_indices']
                teacher_topk_probs[i, :seq_len] = sample['teacher_topk_probs']

        except Exception as e:
            print(f"\n\n\n!!!!! EXCEPTION in collate_fn loop !!!!!")
            print(f"  - Batch item index: {i}")
            print(f"  - Error: {e}")
            # 打印出这个出问题的样本的所有信息
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"    - sample['{key}'].shape = {value.shape}")
                else:
                    print(f"    - sample['{key}'] = {value}")
            continue

    # 准备最终返回的批次
    return {
        'input_ids': input_ids,
        'hidden_states': hidden_states,
        'attention_mask': attention_mask,
        'loss_mask': loss_mask,
        'teacher_topk_indices': teacher_topk_indices,
        'teacher_topk_probs': teacher_topk_probs,
        'teacher_temperature': samples[0]['teacher_temperature'],
    }


#------------------------------------------------------------------
# 4️⃣  Training Logic (Fixed dimension issues)
#------------------------------------------------------------------

def compute_ar_loss(logits: torch.Tensor, targets: torch.Tensor, 
                          loss_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute autoregressive loss with numerical stability.
    Note: This function expects already shifted inputs!
    
    Args:
        logits: [batch, seq_len, vocab_size] - predictions for positions 0 to seq_len-1
        targets: [batch, seq_len] - targets for positions 0 to seq_len-1
        loss_mask: [batch, seq_len] - mask for positions 0 to seq_len-1
    """
    # Ensure dimensions match
    assert logits.size(0) == targets.size(0), f"Batch size mismatch: {logits.size(0)} vs {targets.size(0)}"
    assert logits.size(1) == targets.size(1), f"Sequence length mismatch: {logits.size(1)} vs {targets.size(1)}"
    assert logits.size(1) == loss_mask.size(1), f"Loss mask length mismatch: {logits.size(1)} vs {loss_mask.size(1)}"
    
    # Only compute loss on valid positions
    valid_positions = loss_mask > 0
    num_valid = valid_positions.sum()
    
    if num_valid > 0:
        logits_flat = logits[valid_positions]
        targets_flat = targets[valid_positions]
        
        # Add label smoothing for stability
        loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=0.1)
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_logger.error("NaN/Inf in AR loss")
            nan_logger.error(f"Logits stats: min={logits_flat.min()}, max={logits_flat.max()}")
            nan_logger.error(f"Targets unique: {targets_flat.unique()}")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return loss
    else:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def compute_ar_loss_robust(logits: torch.Tensor, targets: torch.Tensor, 
                           loss_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute autoregressive loss with numerical stability and label smoothing.
    Note: This function expects already shifted inputs!
    """
    # 确保维度匹配
    assert logits.size(0) == targets.size(0), f"Batch size mismatch: {logits.size(0)} vs {targets.size(0)}"
    assert logits.size(1) == targets.size(1), f"Sequence length mismatch: {logits.size(1)} vs {targets.size(1)}"
    assert logits.size(1) == loss_mask.size(1), f"Loss mask length mismatch: {logits.size(1)} vs {loss_mask.size(1)}"
    
    # 只在有效位置计算损失
    valid_positions = loss_mask > 0
    num_valid = valid_positions.sum()
    
    if num_valid > 0:
        logits_flat = logits[valid_positions]
        targets_flat = targets[valid_positions]
        
        # 使用带标签平滑的交叉熵以增强稳定性
        loss = F.cross_entropy(logits_flat, targets_flat, label_smoothing=0.1)
        
        if torch.isnan(loss) or torch.isinf(loss):
            nan_logger.error("NaN/Inf in AR loss calculation.")
            nan_logger.error(f"Logits stats: min={logits_flat.min()}, max={logits_flat.max()}")
            nan_logger.error(f"Targets unique: {targets_flat.unique()}")
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        return loss
    else:
        # 如果没有有效的目标，返回0损失
        return torch.tensor(0.0, device=logits.device, requires_grad=True)


def compute_topk_kl_loss_robust(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-9 # 使用一个更小的eps
) -> torch.Tensor:
    """
    Compute KL divergence using top-k teacher probabilities with ultimate robustness.
    If temperature is near zero, it switches to cross-entropy loss against the argmax target.
    Handles the 0 * log(0) = NaN issue using torch.nan_to_num for guaranteed stability.
    """
    # 输入净化：首先确保teacher_topk_probs不含NaN或Inf
    teacher_topk_probs = torch.nan_to_num(teacher_topk_probs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # --- 开始修改 ---
    if temperature < 1e-4:
        # 温度为0或接近0，切换到交叉熵损失，这是最稳健的做法
        nan_logger.debug("Temperature is near zero, switching to Cross-Entropy for KL loss computation.")
        # 取top-1的索引作为正确答案
        hard_targets = teacher_topk_indices[:, :, 0]
        # 使用健壮的交叉熵损失函数
        return compute_ar_loss_robust(student_logits, hard_targets, loss_mask, eps=eps)

    else:
        # 温度为正，计算KL散度
        # 限制温度范围防止除以过小的数
        temperature = max(temperature, 0.1)
        
        # 缩放并限制学生模型的logits以增强稳定性
        student_logits_scaled = student_logits / temperature
        student_logits_scaled = torch.clamp(student_logits_scaled, min=-50, max=50)
        
        # 计算学生模型的对数概率
        student_log_probs = F.log_softmax(student_logits_scaled, dim=-1)
        
        if torch.isnan(student_log_probs).any():
            nan_logger.error("NaN detected in student log probabilities.")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # 收集学生模型在教师模型top-k位置上的对数概率
        student_topk_log_probs = torch.gather(
            student_log_probs,
            dim=-1,
            index=teacher_topk_indices.long()
        )

        # 对教师概率进行归一化，确保它们在k维上和为1
        teacher_topk_probs = F.normalize(teacher_topk_probs, p=1, dim=-1)

        # 安全地计算log(P)，先确保所有概率值都是正数
        teacher_log_probs = torch.log(teacher_topk_probs.clamp_min(eps))

        # 计算KL散度的每个分项
        kl_term = teacher_topk_probs * (teacher_log_probs - student_topk_log_probs)

        # 这是关键一步：将计算中产生的任何NaN值替换为0.0
        # 这完美处理了 0 * -inf 的情况
        kl_topk = torch.nan_to_num(kl_term, nan=0.0)
        
        # 对top-k维度求和，得到每个token的KL散度
        kl_per_token = kl_topk.sum(dim=-1)
        
        # 应用损失掩码
        kl_per_token = kl_per_token * loss_mask
        
        if torch.isnan(kl_per_token).any():
            nan_logger.error("NaN persisted in KL loss even after nan_to_num. This is unexpected. Check inputs.")
            return torch.tensor(0.0, device=student_logits.device, requires_grad=True)

        # 在有效token上求平均
        num_valid_tokens = loss_mask.sum()
        if num_valid_tokens > 0:
            kl_loss = kl_per_token.sum() / num_valid_tokens
        else:
            kl_loss = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        
        # 根据知识蒸馏理论，乘以温度的平方
        kl_loss = kl_loss * (temperature ** 2)
        
        return kl_loss


class TrainingTimeTest:
    """Implements training-time test for EAGLE V3 with robustness."""

    def __init__(self, draft_model: DraftModelV3, num_steps: int = 2, config: TrainingConfig = None):
        self.draft_model = draft_model
        self.num_steps = num_steps
        self.config = config or TrainingConfig()

    @torch.no_grad()          
    def _safe_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Safe sampling with numerical stability."""
        # Clamp logits
        logits = torch.clamp(logits, min=-self.config.max_logit_value, max=self.config.max_logit_value)
        
        # Temperature sampling with stability
        temperature = 1.0
        probs = torch.softmax(logits / temperature, dim=-1)
        
        # Handle numerical issues
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure valid probability distribution
        row_sum = probs.sum(-1, keepdim=True)
        zero_row = row_sum < self.config.eps
        if zero_row.any():
            # Fallback to uniform distribution
            probs = probs + zero_row * (1.0 / probs.size(-1))
        
        # Normalize
        probs = probs / probs.sum(-1, keepdim=True).clamp(min=self.config.eps)
        
        # Sample
        try:
            flat = probs.view(-1, probs.size(-1))
            sampled = torch.multinomial(flat, 1).view(logits.size(0), -1)
            return sampled
        except RuntimeError as e:
            nan_logger.error(f"Sampling failed: {e}")
            # Return argmax as fallback
            return logits.argmax(dim=-1)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        step_weights: List[float],
        kl_weight: float = 0.7,
        kl_temperature: float = 2.0,
        use_topk_kl: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Perform multi-step training with robustness."""
        
        if self.config.nan_debug_mode:
            log_batch_stats(batch, "TTT Input")

        input_ids = batch['input_ids']
        hidden_states = batch['hidden_states']
        attention_mask = batch['attention_mask']
        loss_mask = batch['loss_mask']
        
        # Top-k teacher data
        teacher_topk_indices = batch.get('teacher_topk_indices')
        teacher_topk_probs = batch.get('teacher_topk_probs')
        teacher_temperature = batch.get('teacher_temperature', 1.0)

        target_dtype = next(self.draft_model.parameters()).dtype
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize losses
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl_loss = torch.tensor(0.0, device=device, requires_grad=True)

        step_losses = []
        step_accuracies = []
        
        # Skip if no valid steps
        if self.num_steps == 0 or seq_len <= 1:
            return {
                'loss': total_loss,
                'ce_loss': total_ce_loss,
                'kl_loss': total_kl_loss,
                'step_losses': step_losses,
                'step_accuracies': step_accuracies,
            }

        # Shift for next-token prediction
        input_ids_shifted = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]
        hidden_states_shifted = hidden_states[:, :-1].to(target_dtype)
        loss_mask_shifted = loss_mask[:, 1:]
        attention_mask_shifted = attention_mask[:, :-1]

        # Also shift teacher data
        if teacher_topk_indices is not None:
            teacher_topk_indices_shifted = teacher_topk_indices[:, 1:]
            teacher_topk_probs_shifted = teacher_topk_probs[:, 1:]
        else:
            teacher_topk_indices_shifted = None
            teacher_topk_probs_shifted = None

        # Current state tracking
        current_hidden = None
        current_input_ids = None
        sampled_ids = None
        is_fused = True  # 第一步使用融合特征
        nan_detected = False

        for step in range(self.num_steps):
            try:
                if step == 0:
                    # 第一步：使用原始融合特征作为输入
                    seq_len_shifted = input_ids_shifted.shape[1]
                    position_ids = torch.arange(seq_len_shifted, dtype=torch.long, device=device).unsqueeze(0)
                    current_hidden = hidden_states_shifted
                    current_input_ids = input_ids_shifted
                    current_attention_mask = attention_mask_shifted
                else:
                    # 后续步骤：使用模型自己的输出作为输入（关键修复）
                    is_fused = False
                    # 使用上一步的预测作为当前步的输入ID
                    current_input_ids = sampled_ids
                    # 注意：current_hidden 已经在上一步更新

                # 模型前向传播
                logits, hidden_output = self.draft_model(
                    current_hidden,
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask if step == 0 else None,
                    position_ids=position_ids if step == 0 else None,
                    is_fused_features=is_fused,
                    ttt_step=step,
                )

                # Check for NaN
                if torch.isnan(logits).any():
                    nan_logger.error(f"NaN detected in logits at step {step}")
                    if self.config.skip_nan_batches:
                        nan_detected = True
                        break

                # Compute cross-entropy loss
                ce_loss = compute_ar_loss(logits, target_ids, loss_mask_shifted, eps=self.config.eps)
                
                # Compute KL divergence loss if needed
                kl_loss = torch.tensor(0.0, device=device)
                if kl_weight > 0 and use_topk_kl and teacher_topk_indices_shifted is not None:
                                    kl_loss = compute_topk_kl_loss_robust(
                    logits,
                    teacher_topk_indices_shifted,
                    teacher_topk_probs_shifted,
                    loss_mask_shifted,
                    temperature=kl_temperature,
                    eps=self.config.eps
                )
                
                # 计算加权总损失
                loss = ce_loss + kl_weight * kl_loss
                
                # 更新累计损失
                step_weight = step_weights[step]
                total_loss = total_loss + loss * step_weight
                total_ce_loss = total_ce_loss + ce_loss * step_weight
                total_kl_loss = total_kl_loss + kl_loss * step_weight
                
                # 计算准确率
                with torch.no_grad():
                    correct = (logits.argmax(-1) == target_ids) & loss_mask_shifted.bool()
                    accuracy = correct.sum().float() / max(loss_mask_shifted.sum(), 1)
                    step_losses.append(loss.item())
                    step_accuracies.append(accuracy.item())
                
                # 为下一步准备
                with torch.no_grad():
                    # 从当前步的logits中采样，用于下一步的输入
                    sampled_ids = self._safe_sample(logits)
                    # 关键：使用当前步的隐藏状态输出作为下一步的输入
                    current_hidden = hidden_output.detach()

            except Exception as e:
                nan_logger.error(f"Error in TTT forward step {step}: {e}")
                if self.config.skip_nan_batches:
                    nan_detected = True
                    break
        
        if nan_detected and self.config.skip_nan_batches:
            # 跳过有NaN的批次，返回零损失
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'ce_loss': torch.tensor(0.0, device=device),
                'kl_loss': torch.tensor(0.0, device=device),
                'step_losses': [0.0] * self.num_steps,
                'step_accuracies': [0.0] * self.num_steps,
                'nan_detected': True,
            }
        
        return {
            'loss': total_loss,
            'ce_loss': total_ce_loss,
            'kl_loss': total_kl_loss,
            'step_losses': step_losses,
            'step_accuracies': step_accuracies,
            'nan_detected': nan_detected,
        }


class EagleTrainer:
    """Main trainer for EAGLE V3 with enhanced robustness."""

    def __init__(
        self,
        config: TrainingConfig,
        draft_model: DraftModelV3,
        accelerator: Accelerator,
    ):
        self.config = config
        self.draft_model = draft_model
        self.accelerator = accelerator
        
        # TTT handler
        self.ttt = TrainingTimeTest(draft_model, config.ttt_steps, config)
        
        # Metrics tracking
        self.step = 0
        self.epoch = 0
        self.nan_count = 0
        self.skipped_batches = 0
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
    ):
        """Main training loop with robustness."""
        
        # Initialize wandb
        if self.accelerator.is_main_process and has_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=vars(self.config),
            )
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch}")
            
            # Train epoch
            train_metrics = self._train_epoch(
                train_loader, optimizer, scheduler
            )
            
            # Log epoch summary
            logger.info(f"Epoch {epoch} complete:")
            logger.info(f"  Average loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Average accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"  Skipped batches: {self.skipped_batches}")
            logger.info(f"  NaN occurrences: {self.nan_count}")
            
            # Validation
            if val_loader is not None and epoch % 5 == 0:
                val_metrics = self._validate(val_loader)
                
                if self.accelerator.is_main_process:
                    logger.info(f"Validation - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
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
        """Train one epoch with robustness."""

        self.draft_model.train()
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        epoch_skipped = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch}",
            disable=not self.accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # Forward pass
                if self.config.ttt_steps > 0:
                    outputs = self.ttt.forward(
                        batch,
                        self.config.ttt_step_weights,
                        kl_weight=self.config.kl_weight if self.config.use_kl_loss else 0.0,
                        kl_temperature=self.config.kl_temperature,
                        use_topk_kl=self.config.use_topk_kl,
                    )
                    
                    # Check for NaN
                    if outputs.get('nan_detected', False):
                        self.nan_count += 1
                        epoch_skipped += 1
                        if self.config.skip_nan_batches:
                            logger.warning(f"Skipping batch {batch_idx} due to NaN")
                            optimizer.zero_grad()
                            continue
                    
                    loss = outputs['loss']
                    ce_loss = outputs.get('ce_loss', loss)
                    kl_loss = outputs.get('kl_loss', torch.tensor(0.0))
                    step_accuracies = outputs.get('step_accuracies', [])
                else:
                    # Non-TTT training
                    loss, ce_loss, kl_loss, step_accuracies = self._forward_single_step(batch)
                    
                    if torch.isnan(loss):
                        self.nan_count += 1
                        epoch_skipped += 1
                        if self.config.skip_nan_batches:
                            logger.warning(f"Skipping batch {batch_idx} due to NaN")
                            optimizer.zero_grad()
                            continue

                # Backward pass
                self.accelerator.backward(loss)

                # Check gradients
                grad_norm = self._check_gradients()
                if grad_norm > 100:
                    logger.warning(f"Large gradient norm: {grad_norm:.2f}")

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
                epoch_loss += loss.item() if not torch.isnan(loss) else 0.0
                epoch_ce_loss += ce_loss.item() if isinstance(ce_loss, torch.Tensor) and not torch.isnan(ce_loss) else 0.0
                epoch_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) and not torch.isnan(kl_loss) else 0.0
                if step_accuracies:
                    epoch_accuracy += np.mean(step_accuracies)
                num_batches += 1

                # Update progress bar
                if num_batches % 10 == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    avg_acc = epoch_accuracy / max(num_batches, 1)
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'acc': f"{avg_acc:.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                        'skip': epoch_skipped,
                    })

                # Periodic logging
                if self.step % self.config.logging_steps == 0 and self.accelerator.is_main_process:
                    self._log_metrics(
                        avg_loss=epoch_loss / max(num_batches, 1),
                        avg_ce=epoch_ce_loss / max(num_batches, 1),
                        avg_kl=epoch_kl_loss / max(num_batches, 1),
                        avg_acc=epoch_accuracy / max(num_batches, 1),
                        lr=scheduler.get_last_lr()[0],
                    )

            except Exception as e:
                error_logger.error(f"Error in batch {batch_idx}: {e}")
                error_logger.error(traceback.format_exc())
                epoch_skipped += 1
                optimizer.zero_grad()
                continue

        self.skipped_batches += epoch_skipped

        return {
            'loss': epoch_loss / max(num_batches, 1),
            'ce_loss': epoch_ce_loss / max(num_batches, 1),
            'kl_loss': epoch_kl_loss / max(num_batches, 1),
            'accuracy': epoch_accuracy / max(num_batches, 1),
            'skipped_batches': epoch_skipped,
        }

    def _forward_single_step(self, batch):
        """Single step forward pass (non-TTT)."""
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
            ttt_step=-1,
        )

        valid_positions = loss_mask_shifted > 0
        if valid_positions.any():
            # CE loss - already shifted
            ce_loss = compute_ar_loss(
                logits, 
                target_ids, 
                loss_mask_shifted,
                eps=self.config.eps
            )

            # Top-k KL loss
            kl_loss = torch.tensor(0.0, device=self.accelerator.device)
            if self.config.use_kl_loss and self.config.use_topk_kl:
                teacher_topk_indices_shifted = batch['teacher_topk_indices'][:, 1:]
                teacher_topk_probs_shifted = batch['teacher_topk_probs'][:, 1:]
                
                kl_loss = compute_topk_kl_loss_robust(
                    logits,
                    teacher_topk_indices_shifted,
                    teacher_topk_probs_shifted,
                    loss_mask_shifted,
                    temperature=batch.get('teacher_temperature', 1.0),
                    eps=self.config.eps
                )

            # Combined loss
            loss = ce_loss + self.config.kl_weight * kl_loss

            # Compute accuracy
            with torch.no_grad():
                logits_flat = logits[valid_positions]
                targets_flat = target_ids[valid_positions]
                accuracy = (logits_flat.argmax(-1) == targets_flat).float().mean()
                step_accuracies = [accuracy.item()]
        else:
            loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
            ce_loss = loss
            kl_loss = torch.tensor(0.0, device=self.accelerator.device)
            step_accuracies = [0.0]

        return loss, ce_loss, kl_loss, step_accuracies

    def _check_gradients(self):
        """Check gradient health."""
        total_norm = 0
        for p in self.draft_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
                # Check for NaN
                if torch.isnan(p.grad).any():
                    nan_logger.warning(f"NaN in gradients for parameter of shape {p.shape}")
                    
        total_norm = total_norm ** 0.5
        return total_norm

    def _log_metrics(self, avg_loss, avg_ce, avg_kl, avg_acc, lr):
        """Log metrics to wandb and files."""
        metrics = {
            'train/loss': avg_loss,
            'train/ce_loss': avg_ce,
            'train/kl_loss': avg_kl,
            'train/accuracy': avg_acc,
            'train/learning_rate': lr,
            'train/step': self.step,
            'train/nan_count': self.nan_count,
            'train/skipped_batches': self.skipped_batches,
        }
        
        # Add model-specific metrics
        unwrapped_model = self.accelerator.unwrap_model(self.draft_model)
        metrics['train/logit_scale'] = unwrapped_model.logit_scale.item()
        
        # Log to wandb
        if has_wandb:
            wandb.log(metrics)
        
        # Log to file
        logger.info(f"Step {self.step}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, LR={lr:.2e}")

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop with robustness."""
        
        self.draft_model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        skipped_batches = 0

        target_dtype = next(self.accelerator.unwrap_model(self.draft_model).parameters()).dtype
        
        for batch in tqdm(val_loader, desc="Validation", disable=not self.accelerator.is_local_main_process):
            try:
                hidden_states_input = batch['hidden_states'][:, :-1].to(target_dtype)
                logits, _ = self.draft_model(
                    hidden_states_input,
                    input_ids=batch['input_ids'][:, :-1],
                    attention_mask=batch['attention_mask'][:, :-1],
                    is_fused_features=True,
                    ttt_step=-1,
                )
                
                # Check for NaN
                if torch.isnan(logits).any():
                    skipped_batches += 1
                    continue
                
                # Compute loss - already shifted
                targets = batch['input_ids'][:, 1:]
                loss_mask = batch['loss_mask'][:, 1:]
                
                valid_positions = loss_mask > 0
                if valid_positions.any():
                    loss = compute_ar_loss(
                        logits,
                        targets,
                        loss_mask,
                        eps=self.config.eps
                    )
                    
                    if not torch.isnan(loss):
                        # Compute accuracy
                        logits_flat = logits[valid_positions]
                        targets_flat = targets[valid_positions]
                        accuracy = (logits_flat.argmax(-1) == targets_flat).float().mean()
                        
                        total_loss += loss.item()
                        total_accuracy += accuracy.item()
                        num_batches += 1
                    else:
                        skipped_batches += 1
                        
            except Exception as e:
                error_logger.error(f"Error in validation: {e}")
                skipped_batches += 1
                continue
        
        logger.info(f"Validation complete. Skipped {skipped_batches} batches out of {len(val_loader)}")
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': total_accuracy / max(num_batches, 1),
            'skipped_batches': skipped_batches,
        }

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint with error handling."""

        if self.accelerator.is_main_process:
            try:
                output_dir = Path(self.config.output_dir) / f"checkpoint-{epoch}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Save model
                unwrapped_model = self.accelerator.unwrap_model(self.draft_model)
                torch.save(
                    unwrapped_model.state_dict(),
                    output_dir / "pytorch_model.bin"
                )

                # Save config
                save_config = {
                    "vocab_size": self.config.vocab_size,
                    "hidden_size": self.config.hidden_size,  # This is now the auto-detected size
                    "num_hidden_layers": self.config.num_decoder_layers,
                    "model_type": "eagle_v3",
                    "tie_word_embeddings": False,
                }

                save_config["_eagle_config"] = {
                    "fuse_layers": FUSE_LAYERS,
                    "training_version": "v3",
                    "use_simple_fusion": self.config.use_simple_fusion,
                    "num_decoder_layers": self.config.num_decoder_layers,
                    "use_topk_kl": self.config.use_topk_kl,
                    "topk_size": self.config.topk_size,
                    "nan_count": self.nan_count,
                    "skipped_batches": self.skipped_batches,
                    "base_hidden_size": unwrapped_model.base_hidden_size,
                    "student_hidden_size": unwrapped_model.student_hidden_size,
                }

                with open(output_dir / "config.json", "w") as f:
                    json.dump(save_config, f, indent=2)

                # Save training state
                training_state = {
                    "epoch": epoch,
                    "step": self.step,
                    "nan_count": self.nan_count,
                    "skipped_batches": self.skipped_batches,
                }
                torch.save(training_state, output_dir / "training_state.pt")

                logger.info(f"Saved checkpoint to {output_dir}")
                
            except Exception as e:
                error_logger.error(f"Failed to save checkpoint: {e}")
                error_logger.error(traceback.format_exc())


#------------------------------------------------------------------
# 5️⃣  Tree Structure Utilities
#------------------------------------------------------------------

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
        return result[::-1]


def build_tree_buffers(tree: List[List[int]], device: torch.device) -> Dict[str, List[torch.Tensor]]:
    """Build buffers for tree attention."""

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
        if node.children:
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
        
        # Tree indices
        indices = torch.tensor([child.path[-1] for child in children], dtype=torch.long, device=device)
        tree_indices.append(indices)
        
        # Position IDs
        pos_ids = torch.full((num_children,), depth + 1, dtype=torch.long, device=device)
        position_ids.append(pos_ids)
        
        # Repeat pattern
        pattern = [len(parent.children) for parent in parent_nodes]
        repeat_patterns.append(pattern)
        
        # Attention mask
        mask = torch.zeros(num_children, num_children, device=device)
        
        for i, child in enumerate(children):
            # Attend to self
            mask[i, i] = 1
            
            # Attend to siblings
            for j, other_child in enumerate(children):
                if child.parent == other_child.parent and j <= i:
                    mask[i, j] = 1
            
            # Attend to ancestors' children
            for ancestor in child.ancestors():
                for j, other_child in enumerate(children):
                    if other_child.parent == ancestor:
                        mask[i, j] = 1
        
        attention_masks.append(mask[None, None])

    return {
        'attn': attention_masks,
        'idx': tree_indices,
        'pos': position_ids,
        'rep': repeat_patterns,
    }


#------------------------------------------------------------------
# 6️⃣  Main Training Script
#------------------------------------------------------------------

def main():
    # Parse arguments
    config, args = parse_args()

    # Set seed
    set_seed(config.seed)
    
    # Log configuration
    logger.info("Starting EAGLE V3 training with configuration:")
    logger.info(json.dumps(vars(config), indent=2))

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    if accelerator.is_main_process:
        logger.info(f"Training config: {config}")
        logger.info(f"Using top-k KL loss: {config.use_topk_kl}")
        logger.info(f"Top-k size: {config.topk_size}")
        logger.info(f"Robustness mode: skip_nan={config.skip_nan_batches}, debug={config.nan_debug_mode}")

    # Create datasets
    logger.info("Loading datasets...")
    augmentation = None
    if config.use_noise:
        augmentation = DataAugmentation(config.noise_type, config.noise_std)

    try:
        train_dataset = EagleDataset(
            config.data_path,
            config.vocab_size,
            config.max_seq_len,
            augmentation=augmentation
        )
    except Exception as e:
        error_logger.error(f"Failed to load dataset: {e}")
        raise

    # Auto-detect hidden size from data if enabled
    if config.auto_detect_hidden_size:
        detected_hidden_size = train_dataset.get_base_hidden_size()
        logger.info(f"Auto-detected base hidden size from data: {detected_hidden_size}")
        # Update config with detected size
        config.hidden_size = detected_hidden_size
    else:
        logger.info(f"Using manually specified hidden size: {config.hidden_size}")
        # Verify it matches the data
        if config.hidden_size != train_dataset.get_base_hidden_size():
            logger.warning(f"WARNING: Specified hidden size {config.hidden_size} does not match "
                          f"data hidden size {train_dataset.get_base_hidden_size()}")

    # Initialize draft model with correct dimensions
    logger.info("Initializing draft model...")
    
    # Decide on student hidden size (could be different from base)
    student_hidden_size = config.hidden_size  # By default, use the same size
    
    draft_model = DraftModelV3(
        vocab_size=config.vocab_size,
        base_hidden_size=config.hidden_size,  # From teacher model
        student_hidden_size=student_hidden_size,  # For student model
        num_decoder_layers=config.num_decoder_layers,
        use_simple_fusion=config.use_simple_fusion,
        max_logit_value=config.max_logit_value
    )

    # Load checkpoint if provided
    if config.draft_checkpoint:
        logger.info(f"Loading checkpoint from {config.draft_checkpoint}")
        try:
            state_dict = torch.load(
                os.path.join(config.draft_checkpoint, "pytorch_model.bin"),
                map_location="cpu"
            )
            draft_model.load_state_dict(state_dict)
            
            # Load training state if available
            training_state_path = os.path.join(config.draft_checkpoint, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path)
                logger.info(f"Resumed from epoch {training_state['epoch']}, step {training_state['step']}")
        except Exception as e:
            error_logger.error(f"Failed to load checkpoint: {e}")
            raise

    # Set tree buffers
    draft_model.tree_buffers = build_tree_buffers(DEFAULT_TREE, accelerator.device)

    # Split into train/val
    val_size = int(0.05 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
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
        eps=1e-8,  # Numerical stability
    )

    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps

    scheduler = get_linear_schedule_with_warmup(
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
        accelerator=accelerator,
    )

    # Train
    logger.info("Starting training...")
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        error_logger.error(f"Training failed with error: {e}")
        error_logger.error(traceback.format_exc())
        
        # Save emergency checkpoint
        if accelerator.is_main_process and config.checkpoint_on_nan:
            logger.info("Saving emergency checkpoint...")
            emergency_dir = Path(config.output_dir) / "emergency"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(draft_model)
            torch.save(
                unwrapped_model.state_dict(),
                emergency_dir / "pytorch_model.bin"
            )
            logger.info(f"Emergency checkpoint saved to {emergency_dir}")
        
        raise

    # Save final model
    if accelerator.is_main_process:
        final_dir = Path(config.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        unwrapped_model = accelerator.unwrap_model(draft_model)
        torch.save(
            unwrapped_model.state_dict(),
            final_dir / "pytorch_model.bin"
        )
        
        # Save config
        save_config = {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_decoder_layers,
            "model_type": "eagle_v3",
            "tie_word_embeddings": False,
        }
        
        save_config["_eagle_config"] = {
            "fuse_layers": FUSE_LAYERS,
            "training_version": "v3",
            "use_simple_fusion": config.use_simple_fusion,
            "num_decoder_layers": config.num_decoder_layers,
            "use_topk_kl": config.use_topk_kl,
            "topk_size": config.topk_size,
            "final_nan_count": trainer.nan_count,
            "final_skipped_batches": trainer.skipped_batches,
            "base_hidden_size": unwrapped_model.base_hidden_size,
            "student_hidden_size": unwrapped_model.student_hidden_size,
        }
        
        with open(final_dir / "config.json", "w") as f:
            json.dump(save_config, f, indent=2)
        
        logger.info(f"Training completed! Model saved to {final_dir}")
        logger.info(f"Total NaN occurrences: {trainer.nan_count}")
        logger.info(f"Total skipped batches: {trainer.skipped_batches}")


if __name__ == "__main__":
    main()