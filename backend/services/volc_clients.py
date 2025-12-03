from __future__ import annotations

import base64
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from backend.config import volc_settings

logger = logging.getLogger(__name__)


def _file_to_base64(path: Path) -> str:
    with path.open("rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class VolcSpeechRecognizer:
    """封装火山语音识别 HTTP 接口."""

    def __init__(
        self,
        app_key: str,
        access_key: str,
        resource_id: str,
        endpoint: str,
        model_name: str,
        uid: str,
        timeout: float,
    ) -> None:
        self.app_key = app_key
        self.access_key = access_key
        self.resource_id = resource_id
        self.endpoint = endpoint
        self.model_name = model_name
        self.uid = uid
        self.timeout = timeout

    def transcribe(self, audio_path: str) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"录音文件不存在: {audio_path}")

        audio_payload: Dict[str, Any] = {
            "format": path.suffix.replace(".", "") or "wav",
            "data": _file_to_base64(path),
        }
        request_body = {
            "user": {"uid": self.uid},
            "audio": audio_payload,
            "request": {
                "model_name": self.model_name,
                "enable_punc": True,
                "enable_itn": True,
                "show_utterances": True,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Api-App-Key": self.app_key,
            "X-Api-App-Id": self.app_key,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }

        response = requests.post(
            self.endpoint,
            json=request_body,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        text = self._extract_text(payload)
        if not text:
            raise RuntimeError(f"ASR 未返回有效文本: {payload}")
        return text

    @staticmethod
    def _extract_text(payload: Dict[str, Any]) -> str:
        # 常见结构：payload["text"] 或 payload["stt_result"]["segments"]
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"].strip()
        resp = payload.get("resp")
        if isinstance(resp, dict):
            if isinstance(resp.get("text"), str):
                return resp["text"].strip()
            segments = resp.get("utterances")
            if isinstance(segments, list):
                texts = [
                    seg.get("text", "")
                    for seg in segments
                    if isinstance(seg, dict) and seg.get("text")
                ]
                if texts:
                    return " ".join(texts).strip()
        stt_result = payload.get("stt_result")
        if isinstance(stt_result, dict):
            segments = stt_result.get("segments") or []
            texts = [
                seg.get("text", "")
                for seg in segments
                if isinstance(seg, dict) and seg.get("text")
            ]
            if texts:
                return " ".join(texts).strip()
        return ""


class VolcChatLLM:
    """封装 Ark Chat Completions."""

    def __init__(self, api_key: str, model_id: str, endpoint: str, timeout: float) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.endpoint = endpoint
        self.timeout = timeout

    def chat(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("未配置 ARK_API_KEY，无法调用大模型。")
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            self.endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={
                "model": self.model_id,
                "messages": messages,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM 未返回内容: {data}")
        message = choices[0].get("message") or choices[0].get("delta") or {}
        content = message.get("content")
        if not content:
            raise RuntimeError(f"LLM 返回空内容: {data}")
        return content.strip()


class VolcTTSEngine:
    """封装火山语音合成接口."""

    def __init__(
        self,
        app_key: str,
        access_key: str,
        resource_id: str,
        endpoint: str,
        uid: str,
        speaker: str,
        audio_format: str,
        sample_rate: int,
        timeout: float,
    ) -> None:
        self.app_key = app_key
        self.access_key = access_key
        self.resource_id = resource_id
        self.endpoint = endpoint
        self.uid = uid
        self.speaker = speaker
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.timeout = timeout
        static_root = Path(volc_settings.STATIC_AUDIO_ROOT)
        self.output_dir = _ensure_dir(static_root / "audios" / "tts")

    def synthesize(self, text: str, *, speaker: Optional[str] = None) -> str:
        target_speaker = speaker or self.speaker
        request_body = {
            "user": {"uid": self.uid},
            "req_params": {
                "text": text,
                "speaker": target_speaker,
                "audio_params": {
                    "format": self.audio_format,
                    "sample_rate": self.sample_rate,
                },
            },
        }
        headers = {
            "Content-Type": "application/json",
            "X-Api-App-Key": self.app_key,
            "X-Api-App-Id": self.app_key,
            "X-Api-Access-Key": self.access_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        response = requests.post(
            self.endpoint,
            json=request_body,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        code = data.get("code")
        if code not in (0, 20000000):
            raise RuntimeError(f"TTS 调用失败: {data}")
        audio_b64 = data.get("data")
        if not audio_b64:
            raise RuntimeError(f"TTS 未返回音频数据: {data}")
        audio_bytes = base64.b64decode(audio_b64)
        output_path = self.output_dir / f"tts_{uuid.uuid4().hex}.{self.audio_format}"
        with output_path.open("wb") as fp:
            fp.write(audio_bytes)
        return str(output_path)


# 单例实例，供业务层直接使用
speech_recognizer = VolcSpeechRecognizer(
    app_key=volc_settings.VOLC_APP_ID,
    access_key=volc_settings.VOLC_ACCESS_KEY,
    resource_id=volc_settings.VOLC_RESOURCE_ID_ASR,
    endpoint=volc_settings.VOLC_ASR_ENDPOINT,
    model_name=volc_settings.VOLC_ASR_MODEL_NAME,
    uid=volc_settings.VOLC_UID,
    timeout=volc_settings.HTTP_TIMEOUT,
)

chat_llm = VolcChatLLM(
    api_key=volc_settings.ARK_API_KEY,
    model_id=volc_settings.ARK_MODEL_ID,
    endpoint=volc_settings.ARK_ENDPOINT,
    timeout=volc_settings.HTTP_TIMEOUT,
)

tts_engine = VolcTTSEngine(
    app_key=volc_settings.VOLC_APP_ID,
    access_key=volc_settings.VOLC_ACCESS_KEY,
    resource_id=volc_settings.VOLC_RESOURCE_ID_TTS,
    endpoint=volc_settings.VOLC_TTS_ENDPOINT,
    uid=volc_settings.VOLC_UID,
    speaker=volc_settings.VOLC_TTS_SPEAKER,
    audio_format=volc_settings.VOLC_TTS_FORMAT,
    sample_rate=volc_settings.VOLC_TTS_SAMPLE_RATE,
    timeout=volc_settings.HTTP_TIMEOUT,
)

