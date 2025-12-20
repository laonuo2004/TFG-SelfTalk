"""
后端通用工具函数。

提供各模块共享的工具函数，避免代码重复。
"""

from typing import Optional

import torch


def normalize_device(gpu_choice: Optional[str]) -> str:
    """
    将前端的 GPU 选择转换为 PyTorch 可识别的设备字符串。
    
    Args:
        gpu_choice: 前端传入的值，如 "GPU0", "GPU1", "cuda:0", "cpu" 等
    
    Returns:
        str: PyTorch 可识别的设备字符串，如 "cuda", "cuda:0", "cpu"
    
    Examples:
        >>> normalize_device("GPU0")
        "cuda"
        >>> normalize_device("GPU1")
        "cuda:1"
        >>> normalize_device("cuda:0")
        "cuda:0"
        >>> normalize_device("cpu")
        "cpu"
        >>> normalize_device(None)
        "cuda"  # 如果 CUDA 可用，否则 "cpu"
    """
    if not gpu_choice:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    choice = gpu_choice.strip().lower()
    
    # 已经是 cuda:X 格式
    if choice.startswith("cuda"):
        return gpu_choice if torch.cuda.is_available() else "cpu"
    
    # GPU0, GPU1 等格式 -> cuda, cuda:1
    if choice.startswith("gpu"):
        index = "".join(filter(str.isdigit, choice)) or "0"
        if torch.cuda.is_available():
            # GPU0 -> cuda (不带索引更简洁)
            # GPU1, GPU2 -> cuda:1, cuda:2
            return f"cuda:{index}" if index != "0" else "cuda"
        return "cpu"
    
    # cpu
    if choice == "cpu":
        return "cpu"
    
    # 默认
    return "cuda" if torch.cuda.is_available() else "cpu"
