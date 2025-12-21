"""
后端推理调度层。

根据 model_name 选择不同的生成 pipeline（SyncTalk / SelfTalk / ...）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from backend.selftalk_generator import run_selftalk_inference

GenerationHandler = Callable[["GenerationPayload"], Dict[str, Any]]


@dataclass
class GenerationPayload:
    """统一的视频生成入参."""

    model_name: str
    gpu_choice: str
    ref_audio: Optional[str] = None          # 驱动音频（上传路径）
    model_param: Optional[str] = None        # SyncTalk 模型目录
    model_path: Optional[str] = None         # SelfTalk checkpoint
    subject: Optional[str] = None            # SelfTalk 渲染模版
    dataset: Optional[str] = "vocaset"
    output_dir: str = "static/videos"
    extra: Dict[str, Any] = field(default_factory=dict)


def generate_video(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """视频生成调度入口。

    Args:
        form_data (Dict[str, Any]): 来自前端的原始请求数据。

    Returns:
        Dict[str, Any]: handler 返回的统一结构。
    """
    payload = _normalize_generation_payload(form_data)
    handler = _resolve_generation_handler(payload.model_name)
    return handler(payload)


def _normalize_generation_payload(form_data: Dict[str, Any]) -> GenerationPayload:
    """规范化生成入参。

    Args:
        form_data (Dict[str, Any]): 原始表单字段或 JSON。

    Returns:
        GenerationPayload: 标准化后的数据对象。

    TODO:
        - 增加字段校验与错误提示。
        - 通过配置提供默认模型/音频。
    """
    payload = GenerationPayload(
        model_name=form_data.get("model_name", "SyncTalk"),
        gpu_choice=form_data.get("gpu_choice", "GPU0"),
        ref_audio=form_data.get("ref_audio") or form_data.get("audio_path"),
        model_param=form_data.get("model_param"),
        model_path=form_data.get("model_path") or form_data.get("model_param"),
        subject=form_data.get("subject"),
        dataset=form_data.get("dataset", "vocaset"),
        output_dir=form_data.get("output_dir", "static/videos"),
        extra={
            "raw_form_data": form_data,
            # TODO: 记录请求 id、用户 id 等
        },
    )
    return payload


def _resolve_generation_handler(model_name: str) -> GenerationHandler:
    """
    根据模型名称选择推理 handler。

    Args:
        model_name (str): "SyncTalk" / "SelfTalk" / ...

    Returns:
        GenerationHandler: 可调用对象。

    Raises:
        NotImplementedError: 当模型类型尚未实现时。
    """
    handlers: Dict[str, GenerationHandler] = {
        "SyncTalk": _generate_with_synctalk,
        "SelfTalk": _generate_with_selftalk,
    }
    if model_name in handlers:
        return handlers[model_name]

    # 尝试从注册表查找自定义模型
    from backend.model_registry import get_model
    if get_model(model_name):
        return _generate_with_selftalk

    raise NotImplementedError(f"暂不支持的模型: {model_name}")
    return handlers[model_name]


def _generate_with_synctalk(payload: GenerationPayload) -> Dict[str, Any]:
    """SyncTalk 推理流程骨架。

    Args:
        payload (GenerationPayload): 标准化后的 SyncTalk 请求参数。

    Returns:
        Dict[str, Any]: 统一的失败说明，提醒使用 SelfTalk。

    TODO:
        - 若未来仍需兼容 SyncTalk，在此调用 Docker / shell 脚本。
    """
    return {
        "status": "failed",
        "message": "我们不使用 SyncTalk，请选择 SelfTalk。",
        "video_path": "",
    }


def _generate_with_selftalk(payload: GenerationPayload) -> Dict[str, Any]:
    """SelfTalk 推理流程骨架。

    Args:
        payload (GenerationPayload): SelfTalk 所需参数集合。

    Returns:
        Dict[str, Any]: `run_selftalk_inference` 的执行结果。

    TODO:
        - 增加队列/任务管理，避免同步阻塞。
        - 将更多渲染参数暴露为可配置项。
    """
    if not payload.ref_audio:
        return {
            "status": "failed",
            "message": "SelfTalk 推理需要音频路径 ref_audio",
        }

    # 如果 payload 中未包含 path，尝试从注册表补全
    if not payload.model_path:
        from backend.model_registry import get_model
        model_info = get_model(payload.model_name)
        if model_info:
            payload.model_path = model_info.path

    result = run_selftalk_inference(payload)
    return result
