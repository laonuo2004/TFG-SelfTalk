"""
语音克隆 / TTS 模块骨架。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class TTSRequest:
    """语音克隆请求."""

    text: str                     # AI 回答文本
    reference_audio: Path         # 参考音色
    language: str = "zh"
    speaker_id: Optional[str] = None
    emotion: Optional[str] = None
    output_dir: Path = Path("static/audios")


def clone_voice(request: TTSRequest) -> Dict[str, str]:
    """
    语音克隆统一入口。

    Args:
        request (TTSRequest): 包含文本、参考音频等信息。

    Returns:
        Dict[str, str]: 例如 {"status": "success", "audio_path": "..."}。

    TODO:
        - 选择底层 TTS 引擎（GPT-SoVITS / CosyVoice / F5-TTS ...）
        - 处理模型加载与缓存
        - 输出合成音频文件路径
    """
    # audio_path = _dispatch_engine(request)
    # return {"status": "success", "audio_path": str(audio_path)}
    raise NotImplementedError("TTS 引擎待接入")


def _dispatch_engine(request: TTSRequest) -> Path:
    """
    根据配置选择具体的 TTS 模型。

    Args:
        request (TTSRequest): 语音克隆请求。

    Returns:
        Path: 生成的音频文件路径。
    """
    # TODO:
    #   - 支持多种后端，通过配置切换
    #   - 统一输入/输出格式
    raise NotImplementedError

