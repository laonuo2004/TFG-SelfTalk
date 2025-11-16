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
    # config = _build_config(payload)
    # _ensure_dataset_ready(config)
    # _launch_training_process(config)
    # return _collect_training_artifacts(config)
    raise NotImplementedError("SelfTalk 训练核心逻辑待实现")


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
    raise NotImplementedError


def _ensure_dataset_ready(config: SelfTalkTrainConfig) -> None:
    """
    校验数据目录。

    Args:
        config (SelfTalkTrainConfig): 包含所有路径的配置。
    """
    # TODO:
    #   - 检查 Path 是否存在
    #   - 提示缺失文件/目录
    raise NotImplementedError


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
    raise NotImplementedError


def _collect_training_artifacts(config: SelfTalkTrainConfig) -> Dict[str, Any]:
    """
    汇总训练产物。

    Args:
        config (SelfTalkTrainConfig): 训练配置。

    Returns:
        Dict[str, Any]: 包含模型路径、日志路径等。
    """
    # TODO: 扫描 save_dir 内的最新模型、loss 曲线等
    raise NotImplementedError

