#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_utils.py - Unified fusion utilities for EAGLE V3
统一的特征融合工具，确保训练、推理、预处理的一致性
"""

import torch
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# =========================================
# 全局常量 - 唯一来源
# =========================================
FUSE_LAYERS = 3
TOPK_PROBS = 20

# =========================================
# 融合层索引计算
# =========================================
def get_fusion_indices(model_name: str, num_hidden_layers: int) -> List[int]:
    """
    Determines the correct layer indices for feature fusion based on the model type.
    
    Args:
        model_name: Model name or path (e.g., "Qwen3-32B")
        num_hidden_layers: Total number of hidden layers in the model
        
    Returns:
        List of layer indices to fuse
    """
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        logger.info(f"Using Qwen-specific fusion strategy for {model_name}")
        # Qwen模型使用 [2, n/3, 2n/3] 的融合策略
        raw = [2, num_hidden_layers // 3, 2 * num_hidden_layers // 3]
    elif "llama" in model_name_lower or "vicuna" in model_name_lower:
        logger.info(f"Using Llama-like fusion strategy for {model_name}")
        # Llama系列使用 [n/4, n/2, n-2] 的融合策略
        raw = [num_hidden_layers // 4,
               num_hidden_layers // 2,
               num_hidden_layers - 2]
    else:
        # 默认策略：均匀分布
        logger.info(f"Using default uniform fusion strategy for {model_name}")
        raw = [num_hidden_layers // 4,
               num_hidden_layers // 2,
               3 * num_hidden_layers // 4]
    
    # 去重并验证
    indices = list(dict.fromkeys(raw))
    
    # 验证索引合法性
    if max(indices) >= num_hidden_layers:
        raise ValueError(f"Fusion index overflow: {indices} exceeds layer count {num_hidden_layers}")
    
    if len(indices) != FUSE_LAYERS:
        raise ValueError(
            f"Expected {FUSE_LAYERS} layers, but obtained {indices}; "
            "Please check if num_hidden_layers is too small or if the index is out of bounds"
        )
    
    logger.info(f"Selected fusion indices: {indices} from {num_hidden_layers} layers")
    return indices


def fuse_hidden_states(hidden_states_tuple: Tuple[torch.Tensor], indices: List[int]) -> torch.Tensor:
    """
    Fuse hidden states from multiple layers.
    
    Args:
        hidden_states_tuple: Tuple of hidden states from all layers
        indices: List of layer indices to fuse
        
    Returns:
        Fused tensor of shape [batch_size, seq_len, hidden_size * FUSE_LAYERS]
        or [seq_len, hidden_size * FUSE_LAYERS] if batch_size is squeezed
    """
    selected_states = []
    
    for idx in indices:
        # +1 because index 0 is embeddings
        layer_hidden = hidden_states_tuple[idx + 1]
        
        # Handle different shapes
        if layer_hidden.dim() == 3 and layer_hidden.size(0) == 1:
            # Squeeze batch dimension if it's 1
            layer_hidden = layer_hidden.squeeze(0)
        
        selected_states.append(layer_hidden)
    
    # Concatenate along hidden dimension
    fused = torch.cat(selected_states, dim=-1)
    
    # Verify dimension
    expected_dim = selected_states[0].size(-1) * FUSE_LAYERS
    if fused.size(-1) != expected_dim:
        raise ValueError(f"Fused dimension {fused.size(-1)} != expected {expected_dim}")
    
    return fused


def validate_fusion_consistency(
    hidden_states: torch.Tensor,
    base_hidden_size: int,
    model_name: str = None
) -> bool:
    """
    Validate that fused hidden states have correct dimensions.
    
    Args:
        hidden_states: Fused hidden states tensor
        base_hidden_size: Expected base hidden size per layer
        model_name: Optional model name for logging
        
    Returns:
        True if dimensions are correct
    """
    expected_total = base_hidden_size * FUSE_LAYERS
    actual_total = hidden_states.size(-1)
    
    if actual_total != expected_total:
        logger.error(
            f"Fusion dimension mismatch for {model_name or 'model'}: "
            f"expected {expected_total} ({base_hidden_size} × {FUSE_LAYERS}), "
            f"got {actual_total}"
        )
        return False
    
    return True


def extract_topk_probs(logits: torch.Tensor, k: int = TOPK_PROBS, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract top-k probabilities from logits with temperature scaling.
    
    Args:
        logits: Raw logits tensor [..., vocab_size]
        k: Number of top probabilities to keep
        temperature: Temperature for softmax
        
    Returns:
        topk_indices: [..., k] tensor of token indices
        topk_probs: [..., k] tensor of probabilities
    """
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Clamp to prevent overflow
    scaled_logits = torch.clamp(scaled_logits, min=-50, max=50)
    
    # Convert to probabilities
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # Get top-k
    topk_probs, topk_indices = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)
    
    return topk_indices, topk_probs