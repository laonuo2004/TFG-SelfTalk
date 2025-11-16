"""
人机对话 Pipeline 骨架。

流程概览：
    用户录音 -> 语音识别 -> 大模型生成文本 -> 语音克隆 -> SelfTalk 生成视频
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# from backend import tts_engine, video_generator


@dataclass
class ChatRequest:
    """前端 /chat_system 表单数据."""

    recording_path: Path
    reference_audio: Path
    model_choice: str            # SyncTalk / SelfTalk
    llm_model: str = "glm-4-plus"
    llm_api_key: Optional[str] = None
    gpu_choice: str = "GPU0"
    # TODO: 增加更多可配置参数


def chat_response(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对话系统主入口。

    Args:
        form_data (Dict[str, Any]): 来自 `/chat_system` 的表单字段。

    Returns:
        Dict[str, Any]: 结果字典，包含 status / ai_text / audio_path / video_path。

    流程：
        1. 整理请求参数
        2. 调度语音识别 / LLM / TTS / 视频生成
        3. 汇总返回
    """
    request = _build_chat_request(form_data)

    # 1) 语音识别
    transcript = _run_speech_recognition(request.recording_path)

    # 2) LLM 生成回答文本
    ai_reply = _run_large_language_model(
        transcript=transcript,
        api_key=request.llm_api_key,
        model_name=request.llm_model,
    )

    # 3) 语音克隆
    cloned_audio = _run_tts_clone(
        text=ai_reply,
        reference_audio=request.reference_audio,
    )

    # 4) 视频生成
    video_result = _run_video_generation(
        audio_path=cloned_audio,
        model_choice=request.model_choice,
        gpu_choice=request.gpu_choice,
        # TODO: 传递更多 SelfTalk 所需参数
    )

    return {
        "status": "success",
        "ai_text": ai_reply,
        "audio_path": cloned_audio,
        "video_path": video_result.get("video_path"),
    }


def _build_chat_request(form_data: Dict[str, Any]) -> ChatRequest:
    """
    将表单字段转换为 ChatRequest。

    Args:
        form_data (Dict[str, Any]): 原始表单。

    Returns:
        ChatRequest: 标准化对象。
    """
    # TODO:
    #   - 读取 /save_audio 保存下来的文件路径
    #   - 校验 reference_audio 是否存在
    #   - 提供默认 llm_api_key 的获取方式
    raise NotImplementedError


def _run_speech_recognition(audio_path: Path) -> str:
    """
    语音识别骨架。

    Args:
        audio_path (Path): 录音文件路径。

    Returns:
        str: 转写文本。

    TODO:
        - 调用第三方 ASR（如 Google / 阿里 / 讯飞）
        - 或者离线 wav2vec2 模型
    """
    # TODO: 实现异步/错误处理
    raise NotImplementedError


def _run_large_language_model(transcript: str, api_key: Optional[str], model_name: str) -> str:
    """
    大模型交互骨架。

    Args:
        transcript (str): 用户提问文本。
        api_key (Optional[str]): 调用凭证。
        model_name (str): 使用的模型名。

    Returns:
        str: AI 回复文本。

    TODO:
        - 组织 prompt（含上下文、角色设定）
        - 调用 ZhipuAI / Kimi / OpenAI 等接口
        - 处理错误与重试
    """
    raise NotImplementedError


def _run_tts_clone(text: str, reference_audio: Path) -> str:
    """
    调用 TTS 模块生成“AI 的回答语音”.

    Args:
        text (str): AI 回复文本。
        reference_audio (Path): 参考音色。

    Returns:
        str: 合成音频路径。
    """
    # from backend.tts_engine import TTSRequest, clone_voice
    # request = TTSRequest(text=text, reference_audio=reference_audio)
    # result = clone_voice(request)
    # return result["audio_path"]
    raise NotImplementedError


def _run_video_generation(
    audio_path: str,
    model_choice: str,
    gpu_choice: str,
) -> Dict[str, Any]:
    """
    将克隆后的音频交给视频生成模块。

    Args:
        audio_path (str): TTS 输出音频路径。
        model_choice (str): 使用的模型（SyncTalk / SelfTalk）。
        gpu_choice (str): GPU 配置。

    Returns:
        Dict[str, Any]: video_generator 的返回值。
    """
    # payload = {
    #     "model_name": model_choice,
    #     "ref_audio": audio_path,
    #     "gpu_choice": gpu_choice,
    #     # TODO: SelfTalk 额外参数
    # }
    # return video_generator.generate_video(payload)
    raise NotImplementedError
