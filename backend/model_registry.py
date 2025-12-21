"""
模型注册表模块。

管理训练好的模型，提供注册、查询、删除功能。
模型信息存储在 models.json 文件中。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 模型注册表存储路径
_REGISTRY_FILE = Path(__file__).resolve().parents[1] / "models.json"

# 项目根目录（用于相对路径转换）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_relative_path(absolute_path: str) -> str:
    """
    将绝对路径转换为相对于项目根目录的相对路径。
    
    Args:
        absolute_path: 绝对路径字符串
    
    Returns:
        相对路径字符串（以 / 为分隔符，跨平台兼容）
    """
    try:
        abs_path = Path(absolute_path).resolve()
        rel_path = abs_path.relative_to(_PROJECT_ROOT)
        return rel_path.as_posix()  # 使用 / 分隔符，跨平台兼容
    except ValueError:
        # 如果无法转换（不在项目目录下），保留原路径
        return absolute_path


def _to_absolute_path(relative_path: str) -> str:
    """
    将相对路径转换为绝对路径。
    
    Args:
        relative_path: 相对于项目根目录的路径
    
    Returns:
        绝对路径字符串
    """
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(relative_path):
        return relative_path
    
    return str((_PROJECT_ROOT / relative_path).resolve())


@dataclass
class ModelInfo:
    """模型信息结构。"""
    
    name: str                    # 模型名称（用户命名或自动生成）
    path: str                    # checkpoint 路径
    created_at: str              # 创建时间
    epochs: int                  # 训练轮数
    dataset: str                 # 数据集名称
    train_subjects: List[str]    # 训练 subjects
    val_subjects: List[str]      # 验证 subjects
    test_subjects: List[str]     # 测试 subjects
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelInfo:
        """从字典创建实例。"""
        return cls(**data)


def _load_registry() -> Dict[str, Any]:
    """加载模型注册表。"""
    if not _REGISTRY_FILE.exists():
        return {"models": []}
    
    try:
        with open(_REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"models": []}


def _save_registry(data: Dict[str, Any]) -> None:
    """保存模型注册表。"""
    with open(_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def list_models() -> List[ModelInfo]:
    """
    获取所有已注册的模型。
    
    Returns:
        模型列表，按创建时间倒序排列。
    """
    registry = _load_registry()
    models = [ModelInfo.from_dict(m) for m in registry.get("models", [])]
    # 按创建时间倒序
    models.sort(key=lambda m: m.created_at, reverse=True)
    return models


def get_model(name: str) -> Optional[ModelInfo]:
    """
    根据名称获取模型信息。
    
    Args:
        name: 模型名称
    
    Returns:
        模型信息，如果不存在则返回 None
    """
    models = list_models()
    for model in models:
        if model.name == name:
            return model
    return None


def register_model(
    path: str,
    epochs: int,
    dataset: str,
    train_subjects: List[str],
    val_subjects: List[str],
    test_subjects: Optional[List[str]] = None,
    name: Optional[str] = None,
) -> ModelInfo:
    """
    注册新训练的模型。
    
    Args:
        path: checkpoint 路径
        epochs: 训练轮数
        dataset: 数据集名称
        train_subjects: 训练 subjects
        val_subjects: 验证 subjects
        test_subjects: 测试 subjects
        name: 模型名称（可选，为空则自动生成）
    
    Returns:
        注册的模型信息
    """
    # 自动生成名称
    if not name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"SelfTalk_{timestamp}"
    
    # 检查名称是否重复
    existing = get_model(name)
    if existing:
        # 附加时间戳使其唯一
        timestamp = datetime.now().strftime("%H%M%S")
        name = f"{name}_{timestamp}"
    
    model = ModelInfo(
        name=name,
        path=_to_relative_path(path),  # 保存为相对路径
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        epochs=epochs,
        dataset=dataset,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        test_subjects=test_subjects or [],
    )
    
    registry = _load_registry()
    registry["models"].append(model.to_dict())
    _save_registry(registry)
    
    return model


def delete_model(name: str) -> bool:
    """
    删除模型注册（不删除实际文件）。
    
    Args:
        name: 模型名称
    
    Returns:
        是否删除成功
    """
    registry = _load_registry()
    models = registry.get("models", [])
    original_len = len(models)
    
    registry["models"] = [m for m in models if m.get("name") != name]
    
    if len(registry["models"]) < original_len:
        _save_registry(registry)
        return True
    return False


def get_models_for_dropdown() -> List[Dict[str, Any]]:
    """
    获取模型列表，供前端下拉选择使用。
    
    Returns:
        模型列表，包含完整信息以供前端展示（路径转换为绝对路径）
    """
    models = list_models()
    result = []
    for model in models:
        result.append({
            "value": model.name,
            "label": f"{model.name} ({model.created_at})",
            "path": _to_absolute_path(model.path),  # 转换为绝对路径
            "created_at": model.created_at,
            "epochs": model.epochs,
            "dataset": model.dataset,
            "train_subjects": model.train_subjects,
            "val_subjects": model.val_subjects,
            "test_subjects": model.test_subjects,
        })
    return result
