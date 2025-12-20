"""
火山引擎（Volcengine）相关配置。
"""

from __future__ import annotations

import os

# ---- ASR (一句话识别 - 流式 WebSocket API) ----
# 获取方式: https://www.volcengine.com/docs/6561/163043
VOLC_APP_ID: str = os.getenv("VOLC_APP_ID", "6555052099")
VOLC_ACCESS_TOKEN: str = os.getenv("VOLC_ACCESS_TOKEN", "nn9JZV__kXGKwcGnasJo0aMC3BJr89Jj")
# Cluster ID: 在控制台开通一句话识别服务后获取
VOLC_ASR_CLUSTER: str = os.getenv("VOLC_ASR_CLUSTER", "volcengine_input_en")
VOLC_ASR_WS_URL: str = os.getenv(
    "VOLC_ASR_WS_URL",
    "wss://openspeech.bytedance.com/api/v2/asr",
)
VOLC_UID: str = os.getenv("VOLC_UID", VOLC_APP_ID)

# ---- Voice Clone (语音克隆) ----
# 文档: https://www.volcengine.com/docs/6561/1305191
VOLC_VOICE_CLONE_UPLOAD_URL: str = os.getenv(
    "VOLC_VOICE_CLONE_UPLOAD_URL",
    "https://openspeech.bytedance.com/api/v1/mega_tts/audio/upload",
)
VOLC_VOICE_CLONE_STATUS_URL: str = os.getenv(
    "VOLC_VOICE_CLONE_STATUS_URL",
    "https://openspeech.bytedance.com/api/v1/mega_tts/status",
)
VOLC_VOICE_CLONE_RESOURCE_ID: str = os.getenv(
    "VOLC_VOICE_CLONE_RESOURCE_ID",
    "seed-icl-2.0",
)
# 语音克隆专用 cluster（在控制台 → 声音复刻 → 服务详情中获取）
# 常见值: volcano_mega, volcano_icl, volcano_megatts
VOLC_VOICE_CLONE_CLUSTER: str = os.getenv("VOLC_VOICE_CLONE_CLUSTER", "volcano_icl")

# ---- TTS (语音合成) ----
VOLC_TTS_CLUSTER: str = os.getenv("VOLC_TTS_CLUSTER", "volcano_tts")
VOLC_TTS_ENDPOINT: str = os.getenv(
    "VOLC_TTS_ENDPOINT",
    "https://openspeech.bytedance.com/api/v1/tts",
)
VOLC_TTS_SPEAKER: str = os.getenv("VOLC_TTS_SPEAKER", "zh_female_vv_jupiter_bigtts")
VOLC_TTS_FORMAT: str = os.getenv("VOLC_TTS_FORMAT", "mp3")
VOLC_TTS_SAMPLE_RATE: int = int(os.getenv("VOLC_TTS_SAMPLE_RATE", "24000"))

# ---- Ark LLM ----
ARK_API_KEY: str = os.getenv("ARK_API_KEY", "e2120650-d36c-430d-b388-0a574cb23afe")
ARK_MODEL_ID: str = os.getenv("ARK_MODEL_ID", "doubao-seed-1-6-flash-250828")
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

