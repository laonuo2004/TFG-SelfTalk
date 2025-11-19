"""
人机对话 Pipeline。

流程概览：
    用户录音 -> 语音识别 -> LLM 生成文本 -> 语音克隆 -> SelfTalk 生成视频
"""

from __future__ import annotations

import contextlib
import shutil
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
TTS_OUTPUT_DIR = STATIC_DIR / "audios" / "tts"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ChatRequest:
    """前端 /chat_system 表单数据."""

    recording_path: Path
    reference_audio: Path
    model_choice: str
    model_path: Optional[str]
    subject: Optional[str]
    dataset: str
    gpu_choice: str
    llm_model: str = "glm-4-plus"
    llm_api_key: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def chat_response(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """人机对话主入口。

    Args:
        form_data (Dict[str, Any]): 来自 `/chat_system` 的表单/JSON。

    Returns:
        Dict[str, Any]: 包含语音识别文本、AI 回复、音频/视频路径等。
    """
    request = _build_chat_request(form_data)

    transcript = _run_speech_recognition(request.recording_path)
    ai_reply = _run_large_language_model(
        transcript=transcript,
        api_key=request.llm_api_key,
        model_name=request.llm_model,
    )
    cloned_audio = _run_tts_clone(
        text=ai_reply,
        reference_audio=request.reference_audio,
    )
    video_result = _run_video_generation(
        audio_path=cloned_audio,
        model_choice=request.model_choice,
        gpu_choice=request.gpu_choice,
        model_path=request.model_path,
        subject=request.subject,
        dataset=request.dataset,
    )

    status = video_result.get("status", "success")
    response = {
        "status": status,
        "ai_text": ai_reply,
        "audio_path": cloned_audio,
        "video_path": video_result.get("video_path"),
        "details": {
            "transcript": transcript,
            "video_artifacts": video_result.get("artifacts", {}),
        },
    }
    if status != "success":
        response["message"] = video_result.get("message", "视频生成失败")
    return response


def _build_chat_request(form_data: Dict[str, Any]) -> ChatRequest:
    """将原始字段转换为 ChatRequest。

    Args:
        form_data (Dict[str, Any]): 表单 + JSON。

    Returns:
        ChatRequest: 标准化后的请求对象。

    Raises:
        ValueError/FileNotFoundError: 关键路径缺失或无效时。
    """
    recording_raw = form_data.get("recording_path") or form_data.get("recording")
    if not recording_raw:
        raise ValueError("recording_path 为必填项，请先调用 /save_audio 获取路径。")

    reference_raw = form_data.get("reference_audio") or form_data.get("ref_audio")
    if not reference_raw:
        raise ValueError("reference_audio 为必填项。")

    request = ChatRequest(
        recording_path=_to_abs_path(recording_raw),
        reference_audio=_to_abs_path(reference_raw),
        model_choice=form_data.get("model_name") or form_data.get("model_choice") or "SelfTalk",
        model_path=form_data.get("model_path") or form_data.get("model_param"),
        subject=form_data.get("subject"),
        dataset=form_data.get("dataset", "vocaset"),
        gpu_choice=form_data.get("gpu_choice", "GPU0"),
        llm_model=form_data.get("llm_model", "glm-4-plus"),
        llm_api_key=form_data.get("llm_api_key"),
        extra=form_data,
    )
    return request


def _run_speech_recognition(audio_path: Path) -> str:
    """占位 ASR，将被真实识别替换。

    Args:
        audio_path (Path): 录音文件路径。

    Returns:
        str: 模拟的识别结果。

    TODO:
        - 接入 Whisper 或云端 ASR，返回真实文本。
    """
    duration = _estimate_duration(audio_path)
    return f"用户语音（约 {duration:.1f} 秒）已记录。"


def _run_large_language_model(transcript: str, api_key: Optional[str], model_name: str) -> str:
    """占位 LLM，待替换为真实大模型调用。

    Args:
        transcript (str): 用户问题文本。
        api_key (Optional[str]): 第三方模型所需的 API Key。
        model_name (str): 模型名称。

    Returns:
        str: AI 回复文本。

    TODO:
        - 接入智谱/讯飞/Kimi/OpenAI 等实际接口。
        - 根据上下文构造提示词，支持多轮对话。
    """
    prefix = "AI 回复"
    if model_name:
        prefix = f"{model_name} 回复"
    return f"{prefix}：感谢你的提问。根据语音内容“{transcript}”，我正在使用 SelfTalk 生成回答视频。"


def _run_tts_clone(text: str, reference_audio: Path) -> str:
    """语音克隆占位实现。

    Args:
        text (str): 需要朗读的内容。
        reference_audio (Path): 参考音色文件。

    Returns:
        str: 合成音频路径。

    TODO:
        - 调用真正的 TTS/语音克隆模型（GPT-SoVITS、CosyVoice 等）。
        - 支持多语言、情感控制。
    """
    target = TTS_OUTPUT_DIR / f"tts_{uuid.uuid4().hex}.wav"
    if reference_audio.exists():
        shutil.copy(reference_audio, target)
    else:
        _synthesize_silence(target)
    return str(target.resolve())


def _run_video_generation(
    audio_path: str,
    model_choice: str,
    gpu_choice: str,
    model_path: Optional[str],
    subject: Optional[str],
    dataset: str,
) -> Dict[str, Any]:
    """封装与 video_generator 的交互。

    Args:
        audio_path (str): TTS 输出音频路径。
        model_choice (str): 使用的模型（默认 SelfTalk）。
        gpu_choice (str): GPU 选择。
        model_path (Optional[str]): 模型 checkpoint。
        subject (Optional[str]): 人物模板。
        dataset (str): 数据集名称（vocaset/biwi）。

    Returns:
        Dict[str, Any]: `generate_video` 的返回结果。
    """
    from backend import video_generator

    payload = {
        "model_name": model_choice,
        "ref_audio": audio_path,
        "gpu_choice": gpu_choice,
        "model_path": model_path,
        "subject": subject,
        "dataset": dataset,
    }
    return video_generator.generate_video(payload)


def _to_abs_path(path_str: str) -> Path:
    """将字符串路径解析为项目内的绝对路径。"""
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    normalized = path_str.lstrip("/")
    guess = (BASE_DIR / normalized).resolve()
    if guess.exists():
        return guess
    raise FileNotFoundError(f"无法定位文件：{path_str}")


def _estimate_duration(audio_path: Path) -> float:
    try:
        with contextlib.closing(wave.open(str(audio_path), "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            return frames / float(rate)
    except Exception:
        return 0.0


def _synthesize_silence(target: Path, seconds: int = 1) -> None:
    sample_rate = 16000
    n_frames = sample_rate * seconds
    with wave.open(str(target), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
