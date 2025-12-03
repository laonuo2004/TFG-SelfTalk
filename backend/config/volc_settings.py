"""
火山引擎（Volcengine）相关配置。
"""

from __future__ import annotations

import os

# ---- ASR ----
VOLC_APP_ID: str = os.getenv("VOLC_APP_ID", "1656653832")
VOLC_ACCESS_KEY: str = os.getenv("VOLC_ACCESS_KEY", "Kd5Eo5Dj3YY6tsVfct5FjfIbCeRnjrI8")
VOLC_RESOURCE_ID_ASR: str = os.getenv("VOLC_ASR_RESOURCE_ID", "volc.bigasr.auc_turbo")
VOLC_ASR_ENDPOINT: str = os.getenv(
    "VOLC_ASR_ENDPOINT",
    "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash",
)
VOLC_ASR_MODEL_NAME: str = os.getenv("VOLC_ASR_MODEL_NAME", "bigmodel")
VOLC_UID: str = os.getenv("VOLC_UID", VOLC_APP_ID)

# ---- TTS ----
VOLC_RESOURCE_ID_TTS: str = os.getenv("VOLC_TTS_RESOURCE_ID", "volc.megatts.default")
VOLC_TTS_ENDPOINT: str = os.getenv(
    "VOLC_TTS_ENDPOINT",
    "https://openspeech.bytedance.com/api/v1/tts",
)
VOLC_TTS_SPEAKER: str = os.getenv("VOLC_TTS_SPEAKER", "zh_female_vv_jupiter_bigtts")
VOLC_TTS_FORMAT: str = os.getenv("VOLC_TTS_FORMAT", "mp3")
VOLC_TTS_SAMPLE_RATE: int = int(os.getenv("VOLC_TTS_SAMPLE_RATE", "24000"))

# ---- Ark LLM ----
ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
ARK_MODEL_ID: str = os.getenv("ARK_MODEL_ID", "doubao-1-5-pro-32k-250115")
ARK_ENDPOINT: str = os.getenv(
    "ARK_ENDPOINT",
    "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
)

# ---- 通用 ----
HTTP_TIMEOUT: float = float(os.getenv("VOLC_HTTP_TIMEOUT", "30"))
STATIC_AUDIO_ROOT: str = os.getenv(
    "VOLC_TTS_OUTPUT_DIR",
    str(
        (os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static")))
    ),
)

