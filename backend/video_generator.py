"""
后端推理调度层。

根据 model_name 选择不同的生成 pipeline（SyncTalk / SelfTalk / ...）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# from backend.selftalk_generator import run_selftalk_inference

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
    """
    视频生成调度入口。

    Args:
        form_data (Dict[str, Any]): 前端传入的参数，常见字段：
            - `model_name`: "SyncTalk" / "SelfTalk"
            - `model_param`: SyncTalk 模型目录
            - `model_path`: SelfTalk checkpoint
            - `ref_audio`: 音频文件路径
            - `subject`: SelfTalk 渲染模板
            - `gpu_choice`: 执行设备

    Returns:
        Dict[str, Any]: 统一的生成结果，如
            {
                "status": "success",
                "video_path": "/static/videos/xxx.mp4",
                "logs": [...]
            }
    """
    payload = _normalize_generation_payload(form_data)
    handler = _resolve_generation_handler(payload.model_name)
    return handler(payload)


def _normalize_generation_payload(form_data: Dict[str, Any]) -> GenerationPayload:
    """
    规范化生成入参。

    Args:
        form_data (Dict[str, Any]): 原始表单字段。

    Returns:
        GenerationPayload: 标准化对象，便于 handler 使用。

    TODO:
        - 处理文件上传路径
        - 支持默认音频/模型的选择
        - 提供参数校验与错误提示
    """
    payload = GenerationPayload(
        model_name=form_data.get("model_name", "SyncTalk"),
        gpu_choice=form_data.get("gpu_choice", "GPU0"),
        ref_audio=form_data.get("ref_audio"),
        model_param=form_data.get("model_param"),
        model_path=form_data.get("model_path"),
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
    if model_name not in handlers:
        raise NotImplementedError(f"暂不支持的模型: {model_name}")
    return handlers[model_name]


def _generate_with_synctalk(payload: GenerationPayload) -> Dict[str, Any]:
    """
    SyncTalk 推理流程骨架。

    Args:
        payload (GenerationPayload): 已校验的参数。

    Returns:
        Dict[str, Any]: 包含 status、video_path、logs 等。

    TODO:
        1. 校验 payload.model_param / payload.ref_audio
        2. 拼接 run_synctalk.sh infer 命令
        3. 监听子进程输出，实时推送给前端
        4. 推理结束后寻找生成的视频，拷贝到 static/videos
        5. 返回 {status, video_path, logs}
    """
    # TODO: 实现与 SyncTalk Docker 的对接
    raise NotImplementedError("SyncTalk 推理流程待实现")


def _generate_with_selftalk(payload: GenerationPayload) -> Dict[str, Any]:
    """
    SelfTalk 推理流程骨架。

    Args:
        payload (GenerationPayload): SelfTalk 所需参数。

    Returns:
        Dict[str, Any]: 包含 status、video_path 等。

    TODO:
        1. 准备 SelfTalk checkpoint、subject 模板、音频
        2. 调用 backend.selftalk_generator.run_selftalk_inference()
        3. 渲染顶点序列 + 合并音频
        4. 返回生成的视频路径

    NOTE:
        SelfTalk 输出为 3D mesh 动画，需要额外渲染步骤；
        推理耗时更长，推荐后台任务 + 轮询进度。
    """
    # TODO: 调用 SelfTalk 推理模块
    # result = run_selftalk_inference(payload)
    # return result
    raise NotImplementedError("SelfTalk 推理流程待实现")
