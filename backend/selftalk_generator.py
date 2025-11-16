"""
SelfTalk 推理 / 渲染 pipeline 骨架。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SelfTalkInferenceConfig:
    """SelfTalk 推理阶段的关键参数."""

    checkpoint_path: Path
    wav_path: Path
    subject: str
    dataset: str
    template_file: Path
    render_template: Path
    output_root: Path
    gpu: str


def run_selftalk_inference(payload) -> Dict[str, Any]:
    """
    SelfTalk 推理主流程。

    Args:
        payload: 来自 video_generator 的 GenerationPayload。

    Returns:
        Dict[str, Any]: 统一响应，如 {status, video_path, logs}。

    TODO:
        1. 构造 SelfTalkInferenceConfig
        2. 调用 model.predict 得到顶点序列
        3. 调用 render.py 逻辑进行渲染
        4. 合并音视频、生成最终 mp4
        5. 返回 {status, video_path, logs}
    """
    # config = _build_config(payload)
    # _validate_resources(config)
    # vertices_path = _run_prediction(config)
    # video_path = _render_with_audio(config, vertices_path)
    # return {"status": "success", "video_path": str(video_path)}
    raise NotImplementedError("SelfTalk 推理逻辑待实现")


def _build_config(payload) -> SelfTalkInferenceConfig:
    """
    构造推理配置。

    Args:
        payload: GenerationPayload。

    Returns:
        SelfTalkInferenceConfig: 含 checkpoint、模板、输出路径等。
    """
    # TODO:
    #   - 处理默认 subject / 模板
    #   - 根据 dataset 切换不同的 vertice_dim / fps
    raise NotImplementedError


def _validate_resources(config: SelfTalkInferenceConfig) -> None:
    """
    检查资源是否齐全。

    Args:
        config (SelfTalkInferenceConfig): 推理配置。
    """
    # TODO: 调用 Path.exists()，并提供友好错误信息
    raise NotImplementedError


def _run_prediction(config: SelfTalkInferenceConfig) -> Path:
    """
    使用 SelfTalk 模型预测顶点序列。

    Args:
        config (SelfTalkInferenceConfig): 推理配置。

    Returns:
        Path: 生成的 `.npy` 文件路径。
    """
    # TODO:
    #   - 加载模型到指定 GPU
    #   - 运行 model.predict
    #   - 把结果写入 output_root/result/*.npy
    raise NotImplementedError


def _render_with_audio(config: SelfTalkInferenceConfig, vertices_path: Path) -> Path:
    """
    渲染视频并与音频合成。

    Args:
        config (SelfTalkInferenceConfig): 推理配置。
        vertices_path (Path): 顶点序列文件。

    Returns:
        Path: 最终生成的视频路径。
    """
    # TODO:
    #   - 调用 render.py 或复用其中函数
    #   - 处理临时 mp4 与最终输出
    raise NotImplementedError

