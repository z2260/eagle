"""
EAGLE V3 核心工具包
"""

from .fusion_utils import (
    FUSE_LAYERS,
    TOPK_PROBS,
    get_fusion_indices,
    fuse_hidden_states,
    validate_fusion_consistency,
)

__all__ = [
    'FUSE_LAYERS',
    'TOPK_PROBS', 
    'get_fusion_indices',
    'fuse_hidden_states',
    'validate_fusion_consistency',
] 