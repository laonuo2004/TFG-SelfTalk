"""
后端训练调度层。

此文件只负责“流程编排”和“参数规范化”，具体实现由
SyncTalk / SelfTalk 各自的 pipeline 接管。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# NOTE: 延迟导入可避免循环依赖（真实实现时再打开）。
# from backend.selftalk_trainer import run_selftalk_training

TrainingHandler = Callable[["TrainingPayload"], Dict[str, Any]]


@dataclass
class TrainingPayload:
    """统一的训练入参数据结构."""

    model_choice: str
    gpu_choice: str
    epochs: int
    ref_video: Optional[str] = None              # SyncTalk 独有
    dataset: Optional[str] = None                # SelfTalk 独有
    train_subjects: Optional[str] = None         # SelfTalk
    val_subjects: Optional[str] = None
    test_subjects: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def train_model(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练调度入口。

    Args:
        form_data (Dict[str, Any]):
            来自 Flask `request.form` 或 JSON 的原始参数。
            常见字段：
                - `model_choice`: str，"SyncTalk" / "SelfTalk"
                - `gpu_choice`: str，形如 "GPU0"
                - `epoch`: str/int，训练轮数
                - `ref_video`: 上传视频路径（SyncTalk）
                - `dataset`, `train_subjects` 等（SelfTalk）

    Returns:
        Dict[str, Any]: 统一的任务结果。
            建议结构：
                {
                    "status": "success" | "failed",
                    "message": "xxx",
                    "artifacts": {"model_dir": "...", "logs": "..."}
                }
    """
    payload = _normalize_training_payload(form_data)
    handler = _resolve_training_handler(payload.model_choice)
    return handler(payload)


def _normalize_training_payload(form_data: Dict[str, Any]) -> TrainingPayload:
    """
    规范化训练入参。

    Args:
        form_data (Dict[str, Any]): 原始表单字典。

    Returns:
        TrainingPayload: 标准化后的对象，后续 handler 直接使用。

    TODO:
        - 处理文件上传得到的临时路径
        - 将字符串数字转为 int（如 epochs）
        - 检查必填项（SyncTalk 需要 ref_video，自检）
        - 记录日志方便定位问题
    """
    model_choice = form_data.get("model_choice", "SyncTalk")
    gpu_choice = form_data.get("gpu_choice", "GPU0")
    epochs = int(form_data.get("epoch", 100))

    payload = TrainingPayload(
        model_choice=model_choice,
        gpu_choice=gpu_choice,
        epochs=epochs,
        ref_video=form_data.get("ref_video"),
        dataset=form_data.get("dataset", "vocaset"),
        train_subjects=form_data.get("train_subjects"),
        val_subjects=form_data.get("val_subjects"),
        test_subjects=form_data.get("test_subjects"),
        extra={
            "raw_form_data": form_data,
            # TODO: 在这里塞入更多元数据（如用户 id、任务 id）
        },
    )
    return payload


def _resolve_training_handler(model_choice: str) -> TrainingHandler:
    """
    按模型类型选择训练 handler。

    Args:
        model_choice (str): "SyncTalk" / "SelfTalk" / 自定义。

    Returns:
        TrainingHandler: 可调用对象。

    Raises:
        NotImplementedError: 当模型类型尚未支持时。

    TODO:
        - 当 handler 不存在时，返回 4xx 错误给前端
        - 允许通过配置文件扩展自定义模型
    """
    handlers: Dict[str, TrainingHandler] = {
        "SyncTalk": _train_synctalk_pipeline,
        "SelfTalk": _train_selftalk_pipeline,
    }
    if model_choice not in handlers:
        raise NotImplementedError(f"暂不支持的模型类型: {model_choice}")
    return handlers[model_choice]


def _train_synctalk_pipeline(payload: TrainingPayload) -> Dict[str, Any]:
    """
    SyncTalk 训练流程骨架。

    Args:
        payload (TrainingPayload): 经过规范化的数据。

    Returns:
        Dict[str, Any]: 训练状态、模型路径、日志等。

    TODO:
        1. 校验 video_path 是否存在
        2. 组装 shell 命令 (run_synctalk.sh train ...)
        3. 持续读取 stdout/stderr 推送到前端日志面板
        4. 解析训练输出目录，返回模型路径、关键指标
        
    NOTE:
        可以直接照抄之前的 Synctalk 训练逻辑
    """
    # TODO: 实现完整逻辑
    raise NotImplementedError("SyncTalk 训练流程待实现")


def _train_selftalk_pipeline(payload: TrainingPayload) -> Dict[str, Any]:
    """
    SelfTalk 训练流程骨架。

    Args:
        payload (TrainingPayload): SelfTalk 所需的参数集合。

    Returns:
        Dict[str, Any]: 训练结果、模型路径等。

    TODO:
        1. 准备 main.py 需要的参数（dataset、subjects、epochs 等）
        2. 确认数据目录结构（wav、vertices_npy、templates.pkl）
        3. 调用 backend.selftalk_trainer.run_selftalk_training()
        4. 读取输出目录 (vocaset/save/xxx)，返回模型 checkpoint

    NOTE:
        SelfTalk 训练通常耗时较长，需要支持异步/后台任务，
        建议集成任务 id，用于轮询训练状态。
    """
    # TODO: 调用真正的 SelfTalk 训练模块
    # result = run_selftalk_training(payload)
    # return result
    raise NotImplementedError("SelfTalk 训练流程待实现")
