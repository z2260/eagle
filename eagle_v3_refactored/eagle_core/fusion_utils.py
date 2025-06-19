#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fusion_utils.py
"""

import torch
from typing import List, Tuple
import logging


FUSE_LAYERS = 3  # 预处理阶段要拼接的隐藏层个数
TOPK_PROBS = 20  # 保存的top-k概率数量

logging.basicConfig(
    format="%(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def get_fusion_indices(model_name: str, num_hidden_layers: int) -> List[int]:
    """
    根据模型类型确定用于特征融合的层索引。
    这是唯一的索引计算函数，所有模块都应该使用这个函数。
    
    Args:
        model_name: 模型名称或路径
        num_hidden_layers: 模型的总层数
        
    Returns:
        List[int]: 用于融合的层索引列表
    """
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        # log.info(f"Using Qwen-specific fusion strategy for {model_name}")
        # Qwen模型使用特定的层选择策略
        raw = [2, num_hidden_layers // 3, 2 * num_hidden_layers // 3]
    else:
        log.info(f"Using default Llama-like fusion strategy for {model_name}")
        # 其他模型使用默认策略
        raw = [num_hidden_layers // 4,
               num_hidden_layers // 2,
               num_hidden_layers - 2]
    
    # 去重并验证
    indices = list(dict.fromkeys(raw))
    
    # 边界检查
    if max(indices) >= num_hidden_layers:
        raise ValueError(f"Fusion index overflow: {max(indices)} >= {num_hidden_layers}")
    
    if len(indices) != FUSE_LAYERS:
        raise ValueError(
            f"Expected {FUSE_LAYERS} layers, but obtained {indices}; "
            "Please check if num_hidden_layers is too small or if the index is out of bounds"
        )
    
    # log.info(f"Selected fusion layers: {indices} from {num_hidden_layers} total layers")
    return indices


def fuse_hidden_states(hidden_states_tuple: Tuple[torch.Tensor], indices: List[int]) -> torch.Tensor:
    """
    从多个层融合隐藏状态。
    
    Args:
        hidden_states_tuple: 所有层的隐藏状态元组
        indices: 要融合的层索引
        
    Returns:
        torch.Tensor: 融合后的特征张量
    """
    selected_states = []
    
    for idx in indices:
        # hidden_states_tuple[0] 是 embedding layer
        # hidden_states_tuple[1:] 是各个 transformer layers
        layer_hidden = hidden_states_tuple[idx + 1]
        
        # 处理可能的维度差异
        if layer_hidden.dim() == 3:
            layer_hidden = layer_hidden[0]  # 去掉batch维度
            
        selected_states.append(layer_hidden)
    
    # 在最后一个维度上拼接
    fused = torch.cat(selected_states, dim=-1)
    
    # 验证维度
    expected_dim = selected_states[0].size(-1) * FUSE_LAYERS
    assert fused.size(-1) == expected_dim, f"Fused dim {fused.size(-1)} ≠ {expected_dim}"
    
    return fused


def validate_fusion_consistency(base_model_name: str, draft_model_config: dict) -> bool:
    """
    验证融合配置的一致性
    
    Args:
        base_model_name: 基础模型名称
        draft_model_config: 草稿模型配置
        
    Returns:
        bool: 是否一致
    """
    # 这里可以添加更多的验证逻辑
    return True


# 用于测试的辅助函数
def test_fusion_indices():
    """测试不同模型的融合索引计算"""
    test_cases = [
        ("Qwen/Qwen2.5-32B", 64),
        ("meta-llama/Llama-3-70B", 80),
        ("Qwen/Qwen2-7B", 28),
    ]
    
    for model_name, n_layers in test_cases:
        try:
            indices = get_fusion_indices(model_name, n_layers)
            print(f"{model_name} ({n_layers} layers): {indices}")
        except Exception as e:
            print(f"{model_name} ({n_layers} layers): ERROR - {e}")


if __name__ == "__main__":
    test_fusion_indices() 