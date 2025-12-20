"""
设备检测工具模块。

提供 GPU/CPU 设备检测功能，供前端显示可用设备列表。
"""

from typing import List, Dict, Any
import torch


def get_available_devices() -> List[Dict[str, Any]]:
    """
    检测当前系统可用的计算设备。
    
    Returns:
        设备列表，每个设备包含 value 和 label 字段，供前端下拉选择。
        
    Example:
        [
            {"value": "cpu", "label": "CPU"},
            {"value": "cuda", "label": "GPU 0: NVIDIA GeForce GTX 1650 (4GB)"},
        ]
    """
    devices = [
        {"value": "cpu", "label": "CPU"}
    ]
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            # 尝试获取显存信息
            try:
                total_memory = torch.cuda.get_device_properties(i).total_memory
                memory_gb = total_memory / (1024 ** 3)
                label = f"GPU {i}: {name} ({memory_gb:.1f}GB)"
            except Exception:
                label = f"GPU {i}: {name}"
            
            # 第一个 GPU 使用 "cuda"，其他使用 "cuda:N"
            value = "cuda" if i == 0 else f"cuda:{i}"
            devices.append({"value": value, "label": label})
    
    return devices


def get_default_device() -> str:
    """
    获取默认计算设备。
    
    优先使用 GPU，没有则使用 CPU。
    
    Returns:
        str: 设备字符串，如 "cuda" 或 "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


# vocaset 数据集中可用的 subject 列表
VOCASET_SUBJECTS = [
    # 默认训练集 (前8个)
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170915_00223_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170912_03278_TA",
    # 默认验证集 (第9-10个)
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170908_03277_TA",
    # 默认测试集 (最后2个)
    "FaceTalk_170809_00138_TA",
    "FaceTalk_170731_00024_TA",
]


def get_subjects_list() -> List[Dict[str, Any]]:
    """
    获取可用的 subject 列表，供前端多选下拉使用。
    
    Returns:
        Subject 列表，每个包含 value, label, group 字段。
        group 字段标识默认分组（train/val/test）。
    """
    subjects = []
    for i, subject in enumerate(VOCASET_SUBJECTS):
        if i < 8:
            group = "train"
        elif i < 10:
            group = "val"
        else:
            group = "test"
        
        subjects.append({
            "value": subject,
            "label": subject,
            "group": group,
        })
    
    return subjects


def get_default_subjects() -> Dict[str, List[str]]:
    """
    获取默认的 subject 分配。
    
    Returns:
        包含 train, val, test 三个列表的字典。
    """
    return {
        "train": VOCASET_SUBJECTS[:8],
        "val": VOCASET_SUBJECTS[8:10],
        "test": VOCASET_SUBJECTS[10:],
    }
