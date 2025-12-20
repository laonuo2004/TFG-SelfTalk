"""
SelfTalk 训练 pipeline 骨架。

职责：
    - 接收来自 model_trainer 的 TrainingPayload
    - 组装 main.py 所需的命令/参数
    - 负责日志采集、错误处理、结果回传
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import subprocess
import datetime
import os
import torch

from backend.utils import normalize_device

@dataclass
class SelfTalkTrainConfig:
    """SelfTalk 训练所需的关键参数."""

    dataset: str
    save_dir: Path
    wav_dir: Path
    vertices_dir: Path
    template_file: Path
    max_epoch: int
    feature_dim: int
    vertice_dim: int
    gpu: str
    train_subjects: str
    val_subjects: str
    test_subjects: str
    main_py: Path    # 新增
    root: Path       # 新增


def run_selftalk_training(payload) -> Dict[str, Any]:
    """
    SelfTalk 训练入口。

    Args:
        payload: 来自 `_train_selftalk_pipeline` 的 TrainingPayload。

    Returns:
        Dict[str, Any]: 训练结果摘要，如
            {
                "status": "success",
                "message": "...",
                "artifacts": {"model_dir": "..."}
            }

    TODO:
        1. 根据 payload 构造 SelfTalkTrainConfig
        2. 确认数据与模型目录存在
        3. 组装 main.py 的命令行参数
        4. 启动子进程并流式采集日志
        5. 解析输出目录（vocaset/save/...）
        6. 返回 {status, message, artifacts}
    """
    try:
        config = _build_config(payload)
        _ensure_dataset_ready(config)
        _launch_training_process(config)
        artifacts = _collect_training_artifacts(config)

        return {
            "status": "success",
            "message": "SelfTalk 模型训练完成",
            "artifacts": artifacts
        }

    except Exception as e:
        return {
            "status": "failed",
            "message": str(e),
            "artifacts": {}
        }
    # config = _build_config(payload)
    # _ensure_dataset_ready(config)
    # _launch_training_process(config)
    # return _collect_training_artifacts(config)
    # raise NotImplementedError("SelfTalk 训练核心逻辑待实现")


def _build_config(payload) -> SelfTalkTrainConfig:
    """
    根据 TrainingPayload 生成 SelfTalkTrainConfig。

    Args:
        payload: 训练入参。

    Returns:
        SelfTalkTrainConfig: 便于后续函数使用。
    """
    # TODO:
    #   - 支持 vocaset / BIWI 不同的 vertice_dim / feature_dim
    #   - 从 payload.extra 中读取自定义参数

    # 动态获取项目根目录，避免硬编码路径
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # backend/ -> TFG-SelfTalk/
    ROOT = PROJECT_ROOT / "SelfTalk"
    DATA_ROOT = ROOT / payload.dataset  # .../SelfTalk/vocaset


    # 构造 save 目录
    save_root = DATA_ROOT / "save"
    save_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = save_root / f"selftalk_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练主体目录
    subject = payload.train_subjects

    return SelfTalkTrainConfig(
        dataset=payload.dataset,
        save_dir=save_dir,

        wav_dir=DATA_ROOT / "wav",
        vertices_dir=DATA_ROOT / "vertices_npy",
        template_file=DATA_ROOT / "templates.pkl",

        max_epoch=payload.epochs,
        feature_dim=64,     # 默认值，可根据实际模型调整
        vertice_dim=3,      # 默认值，可根据实际模型调整
        gpu=normalize_device(payload.gpu_choice),
        train_subjects=payload.train_subjects,
        val_subjects=payload.val_subjects,
        test_subjects=payload.test_subjects or "",
        main_py=ROOT / "main.py",
        root=ROOT
    )
    # raise NotImplementedError


def _ensure_dataset_ready(config: SelfTalkTrainConfig) -> None:
    """
    校验数据目录。

    Args:
        config (SelfTalkTrainConfig): 包含所有路径的配置。
    """
    # TODO:
    #   - 检查 Path 是否存在
    #   - 提示缺失文件/目录
    required_paths = [
        config.wav_dir,
        config.vertices_dir,
        config.template_file,
    ]

    for p in required_paths:
        if not p.exists():
            raise FileNotFoundError(f"缺失文件或目录：{p}")
    # raise NotImplementedError


def _launch_training_process(config: SelfTalkTrainConfig) -> None:
    """
    启动 SelfTalk main.py。

    Args:
        config (SelfTalkTrainConfig): 训练配置。
    """
    # TODO:
    #   - 构造命令行数组
    #   - 监听 stdout/stderr
    #   - 对接任务管理器（可选）
    cmd = [
        "python", str(config.main_py),
        "--dataset", config.dataset,
        "--train_subjects", config.train_subjects,
        "--val_subjects", config.val_subjects,
        "--max_epoch", str(config.max_epoch),
        "--save_path", str(config.save_dir),
        "--device", config.gpu,
    ]

    if config.test_subjects:
        cmd.extend(["--test_subjects", config.test_subjects])

    print("\n[SelfTalk] 执行命令：")
    print(" ".join(cmd))
    print("--------------------------------------------------\n")

    process = subprocess.Popen(
        cmd,
        cwd=str(config.root),
    )

    process.wait()
    # raise NotImplementedError


def _collect_training_artifacts(config: SelfTalkTrainConfig) -> Dict[str, Any]:
    """
    汇总训练产物。

    Args:
        config (SelfTalkTrainConfig): 训练配置。

    Returns:
        Dict[str, Any]: 包含模型路径、日志路径等。
    """
    # TODO: 扫描 save_dir 内的最新模型、loss 曲线等
    return {
        "model_dir": str(config.save_dir),
    }
    # raise NotImplementedError