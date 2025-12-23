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
    """封装火山一句话识别 WebSocket 接口.
    
    使用 WebSocket 流式协议，支持直接上传音频文件（无需公网 URL）。
    
    官方文档: https://www.volcengine.com/docs/6561/80816
    """
    
    # WebSocket 二进制协议常量
    PROTOCOL_VERSION = 0b0001
    DEFAULT_HEADER_SIZE = 0b0001
    
    # Message Type
    CLIENT_FULL_REQUEST = 0b0001
    CLIENT_AUDIO_ONLY_REQUEST = 0b0010
    SERVER_FULL_RESPONSE = 0b1001
    SERVER_ACK = 0b1011
    SERVER_ERROR_RESPONSE = 0b1111
    
    # Message Type Specific Flags
    NO_SEQUENCE = 0b0000
    NEG_SEQUENCE = 0b0010
    
    # Message Serialization
    JSON_SERIALIZATION = 0b0001
    NO_SERIALIZATION = 0b0000
    
    # Message Compression
    NO_COMPRESSION = 0b0000
    GZIP_COMPRESSION = 0b0001

    def __init__(
        self,
        appid: str,
        token: str,
        cluster: str = "",
        ws_url: str = "wss://openspeech.bytedance.com/api/v2/asr",
        uid: str = "streaming_asr_client",
        seg_duration: int = 15000,
        timeout: float = 30.0,
        clusters: Optional[List[str]] = None,
    ) -> None:
        """初始化语音识别器.
        
        Args:
            appid: 应用 ID
            token: 访问令牌
            cluster: 单个 cluster ID（已弃用，建议使用 clusters）
            ws_url: WebSocket API 地址
            uid: 用户 ID
            seg_duration: 分片时长（毫秒）
            timeout: 超时时间（秒）
            clusters: cluster ID 列表，按优先级顺序尝试识别。
                     如果为 None，则使用 cluster 参数。
                     典型用法：["volcengine_input", "volcengine_input_en"]
                     表示优先中文识别，失败后回退到英文识别。
        """
        self.appid = appid
        self.token = token
        self.ws_url = ws_url
        self.uid = uid
        self.seg_duration = seg_duration
        self.timeout = timeout
        self.success_code = 1000
        
        # 支持多 cluster 回退识别
        if clusters:
            self.clusters = clusters
        elif cluster:
            self.clusters = [cluster]
        else:
            self.clusters = ["volcengine_input"]  # 默认中文识别
        
        # 保留 cluster 属性以兼容旧代码
        self.cluster = self.clusters[0]

    def _generate_header(
        self,
        message_type: int = None,
        message_type_specific_flags: int = None,
        serial_method: int = None,
        compression_type: int = None,
    ) -> bytearray:
        """生成 WebSocket 二进制协议头"""
        import gzip
        
        if message_type is None:
            message_type = self.CLIENT_FULL_REQUEST
        if message_type_specific_flags is None:
            message_type_specific_flags = self.NO_SEQUENCE
        if serial_method is None:
            serial_method = self.JSON_SERIALIZATION
        if compression_type is None:
            compression_type = self.GZIP_COMPRESSION
            
        header = bytearray()
        header.append((self.PROTOCOL_VERSION << 4) | self.DEFAULT_HEADER_SIZE)
        header.append((message_type << 4) | message_type_specific_flags)
        header.append((serial_method << 4) | compression_type)
        header.append(0x00)  # reserved
        return header

    def _parse_response(self, res: bytes) -> Dict[str, Any]:
        """解析服务端响应"""
        import gzip
        import json
        
        header_size = res[0] & 0x0f
        message_type = res[1] >> 4
        serialization_method = res[2] >> 4
        message_compression = res[2] & 0x0f
        
        payload = res[header_size * 4:]
        result = {}
        payload_msg = None
        
        if message_type == self.SERVER_FULL_RESPONSE:
            payload_size = int.from_bytes(payload[:4], "big", signed=True)
            payload_msg = payload[4:]
        elif message_type == self.SERVER_ACK:
            seq = int.from_bytes(payload[:4], "big", signed=True)
            result['seq'] = seq
            if len(payload) >= 8:
                payload_size = int.from_bytes(payload[4:8], "big", signed=False)
                payload_msg = payload[8:]
        elif message_type == self.SERVER_ERROR_RESPONSE:
            code = int.from_bytes(payload[:4], "big", signed=False)
            result['code'] = code
            payload_size = int.from_bytes(payload[4:8], "big", signed=False)
            payload_msg = payload[8:]
            
        if payload_msg is None:
            return result
            
        if message_compression == self.GZIP_COMPRESSION:
            payload_msg = gzip.decompress(payload_msg)
        if serialization_method == self.JSON_SERIALIZATION:
            payload_msg = json.loads(str(payload_msg, "utf-8"))
        elif serialization_method != self.NO_SERIALIZATION:
            payload_msg = str(payload_msg, "utf-8")
            
        result['payload_msg'] = payload_msg
        return result

    def _construct_request(self, reqid: str, audio_format: str, sample_rate: int, cluster: str = "") -> Dict[str, Any]:
        """构建请求参数.
        
        Args:
            reqid: 请求 ID
            audio_format: 音频格式
            sample_rate: 采样率
            cluster: 使用的 cluster ID，如果为空则使用默认 cluster
        """
        return {
            'app': {
                'appid': self.appid,
                'cluster': cluster or self.cluster,
                'token': self.token,
            },
            'user': {
                'uid': self.uid
            },
            'request': {
                'reqid': reqid,
                'nbest': 1,
                'workflow': 'audio_in,resample,partition,vad,fe,decode,itn,nlu_punctuate',
                'show_language': False,
                'show_utterances': True,
                'result_type': 'full',
                'sequence': 1,
            },
            'audio': {
                'format': audio_format,
                'rate': sample_rate,
                'language': 'zh-CN',
                'bits': 16,
                'channel': 1,
                'codec': 'raw',
            }
        }

    @staticmethod
    def _slice_data(data: bytes, chunk_size: int):
        """分片音频数据"""
        data_len = len(data)
        offset = 0
        while offset + chunk_size < data_len:
            yield data[offset: offset + chunk_size], False
            offset += chunk_size
        else:
            yield data[offset: data_len], True

    def transcribe(self, audio_path: str) -> str:
        """识别音频文件，支持多语言回退.
        
        按照 clusters 列表中的顺序依次尝试识别。典型用法是优先尝试中文识别，
        如果识别失败（例如音频是纯英文），则回退到英文识别。
        
        Args:
            audio_path: 本地音频文件路径
            
        Returns:
            识别出的文本
            
        Raises:
            FileNotFoundError: 音频文件不存在
            RuntimeError: 所有 cluster 都识别失败
        """
        errors = []
        
        for i, cluster in enumerate(self.clusters):
            try:
                logger.info(f"[ASR] 尝试使用 cluster '{cluster}' 进行识别 ({i+1}/{len(self.clusters)})")
                text = self._transcribe_with_cluster(audio_path, cluster)
                logger.info(f"[ASR] 使用 cluster '{cluster}' 识别成功: {text[:50]}...")
                return text
            except Exception as e:
                error_msg = str(e)
                errors.append(f"[{cluster}] {error_msg}")
                logger.warning(f"[ASR] cluster '{cluster}' 识别失败: {error_msg}")
                
                # 如果是最后一个 cluster，不再继续
                if i == len(self.clusters) - 1:
                    break
                    
                logger.info(f"[ASR] 回退到下一个 cluster...")
        
        # 所有 cluster 都失败
        raise RuntimeError(
            f"所有语音识别服务均失败:\n" + "\n".join(f"  - {err}" for err in errors)
        )

    def _transcribe_with_cluster(self, audio_path: str, cluster: str) -> str:
        """使用指定的 cluster 识别音频文件.
        
        Args:
            audio_path: 本地音频文件路径
            cluster: 使用的 cluster ID
            
        Returns:
            识别出的文本
        """
        import asyncio
        import gzip
        import json
        import wave
        from io import BytesIO
        
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"录音文件不存在: {audio_path}")

        # 读取音频文件
        with path.open("rb") as f:
            audio_data = f.read()
        
        # 获取音频格式
        audio_format = path.suffix.replace(".", "") or "wav"
        sample_rate = 16000  # 默认采样率
        nchannels = 1
        sampwidth = 2
        
        # 如果是 wav 文件，尝试获取实际采样率
        if audio_format == "wav":
            try:
                with BytesIO(audio_data) as bio:
                    with wave.open(bio, 'rb') as wf:
                        sample_rate = wf.getframerate()
                        nchannels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
            except Exception as e:
                logger.warning(f"无法解析 wav 文件信息: {e}")

        async def _do_recognize():
            import websockets
            
            reqid = str(uuid.uuid4())
            request_params = self._construct_request(reqid, audio_format, sample_rate, cluster)
            
            # 构建 full client request
            payload_bytes = json.dumps(request_params).encode('utf-8')
            payload_bytes = gzip.compress(payload_bytes)
            
            full_request = bytearray(self._generate_header())
            full_request.extend(len(payload_bytes).to_bytes(4, 'big'))
            full_request.extend(payload_bytes)
            
            # 建立 WebSocket 连接
            headers = {'Authorization': f'Bearer; {self.token}'}
            
            async with websockets.connect(
                self.ws_url, 
                extra_headers=headers, 
                max_size=1000000000
            ) as ws:
                # 发送 full client request
                await ws.send(full_request)
                res = await ws.recv()
                result = self._parse_response(res)
                
                if 'payload_msg' in result and result['payload_msg'].get('code') != self.success_code:
                    raise RuntimeError(f"ASR 初始化失败: {result}")
                
                # 计算分片大小
                if audio_format == "wav":
                    size_per_sec = nchannels * sampwidth * sample_rate
                    segment_size = int(size_per_sec * self.seg_duration / 1000)
                else:
                    segment_size = 10000  # MP3 默认分片大小
                
                # 发送音频分片
                for seq, (chunk, is_last) in enumerate(self._slice_data(audio_data, segment_size), 1):
                    compressed_chunk = gzip.compress(chunk)
                    
                    if is_last:
                        # 最后一包
                        audio_header = self._generate_header(
                            message_type=self.CLIENT_AUDIO_ONLY_REQUEST,
                            message_type_specific_flags=self.NEG_SEQUENCE,
                        )
                    else:
                        audio_header = self._generate_header(
                            message_type=self.CLIENT_AUDIO_ONLY_REQUEST,
                        )
                    
                    audio_request = bytearray(audio_header)
                    audio_request.extend(len(compressed_chunk).to_bytes(4, 'big'))
                    audio_request.extend(compressed_chunk)
                    
                    await ws.send(audio_request)
                    res = await ws.recv()
                    result = self._parse_response(res)
                    
                    if 'payload_msg' in result and result['payload_msg'].get('code') != self.success_code:
                        raise RuntimeError(f"ASR 识别失败: {result}")
                
                return result
        
        # 运行异步识别
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(_do_recognize())
        
        # 提取识别文本
        text = self._extract_text(result)
        if not text:
            raise RuntimeError(f"ASR 未返回有效文本: {result}")
        
        return text

    @staticmethod
    def _extract_text(result: Dict[str, Any]) -> str:
        """从响应中提取识别文本"""
        payload = result.get('payload_msg', {})
        
        if not isinstance(payload, dict):
            return ""
        
        # 1. 优先尝试直接的 text 字段
        direct_text = payload.get('text')
        if isinstance(direct_text, str) and direct_text.strip():
            return direct_text.strip()
        
        # 2. 尝试 result 字段
        res = payload.get('result')
        if res:
            # result 是字典，提取其中的 text
            if isinstance(res, dict):
                text = res.get('text', '')
                if isinstance(text, str) and text.strip():
                    return text.strip()
            # result 是字符串
            elif isinstance(res, str):
                return res.strip()
            # result 是列表
            elif isinstance(res, list):
                texts = []
                for item in res:
                    if isinstance(item, dict):
                        t = item.get('text', '')
                        if t:
                            texts.append(str(t))
                    elif isinstance(item, str):
                        texts.append(item)
                if texts:
                    return ' '.join(texts).strip()
        
        # 3. 尝试 utterances 字段
        utterances = payload.get('utterances', [])
        if utterances and isinstance(utterances, list):
            texts = [u.get('text', '') for u in utterances if isinstance(u, dict) and u.get('text')]
            if texts:
                return ' '.join(texts).strip()
        
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


# class VolcTTSEngine:
#     """封装火山语音合成接口."""

#     def __init__(
#         self,
#         app_key: str,
#         access_key: str,
#         resource_id: str,
#         endpoint: str,
#         uid: str,
#         speaker: str,
#         audio_format: str,
#         sample_rate: int,
#         timeout: float,
#     ) -> None:
#         self.app_key = app_key
#         self.access_key = access_key
#         self.resource_id = resource_id
#         self.endpoint = endpoint
#         self.uid = uid
#         self.speaker = speaker
#         self.audio_format = audio_format
#         self.sample_rate = sample_rate
#         self.timeout = timeout
#         static_root = Path(volc_settings.STATIC_AUDIO_ROOT)
#         self.output_dir = _ensure_dir(static_root / "audios" / "tts")

#     def synthesize(self, text: str, *, speaker: Optional[str] = None) -> str:
#         target_speaker = speaker or self.speaker
#         request_body = {
#             "user": {"uid": self.uid},
#             "req_params": {
#                 "text": text,
#                 "speaker": target_speaker,
#                 "audio_params": {
#                     "format": self.audio_format,
#                     "sample_rate": self.sample_rate,
#                 },
#             },
#         }
#         headers = {
#             "Content-Type": "application/json",
#             "X-Api-App-Key": self.app_key,
#             "X-Api-App-Id": self.app_key,
#             "X-Api-Access-Key": self.access_key,
#             "X-Api-Resource-Id": self.resource_id,
#             "X-Api-Request-Id": str(uuid.uuid4()),
#             "X-Api-Sequence": "-1",
#         }
#         response = requests.post(
#             self.endpoint,
#             json=request_body,
#             headers=headers,
#             timeout=self.timeout,
#         )
#         response.raise_for_status()
#         data = response.json()
#         code = data.get("code")
#         if code not in (0, 20000000):
#             raise RuntimeError(f"TTS 调用失败: {data}")
#         audio_b64 = data.get("data")
#         if not audio_b64:
#             raise RuntimeError(f"TTS 未返回音频数据: {data}")
#         audio_bytes = base64.b64decode(audio_b64)
#         output_path = self.output_dir / f"tts_{uuid.uuid4().hex}.{self.audio_format}"
#         with output_path.open("wb") as fp:
#             fp.write(audio_bytes)
#         return str(output_path)


class VolcVoiceCloner:
    """封装火山引擎语音克隆接口.
    
    工作流程:
    1. upload_and_train(): 上传参考音频并训练音色
    2. get_status(): 查询训练状态
    3. synthesize(): 使用克隆音色合成语音
    
    注意:
    - 每个账号提供 10 个预训练模型 (speaker_id)
    - 每个模型最多微调 10 次，超过会返回错误码 1123
    - 系统会自动切换到下一个 speaker_id
    
    文档: https://www.volcengine.com/docs/6561/1305191
    """
    
    # 火山引擎提供的 10 个预训练 Speaker ID
    AVAILABLE_SPEAKER_IDS = [
        "S_7fL4WBaO1",
        "S_DJm4WBaO1",
        "S_PTS3WBaO1",
        "S_PzyYVBaO1",
        "S_FofYVBaO1",
        "S_XDwXVBaO1",
        "S_PtXWVBaO1",
        "S_FiEWVBaO1",
        "S_nXLVVBaO1",
        "S_XvoVVBaO1",
    ]
    
    # 微调次数耗尽错误码
    QUOTA_EXHAUSTED_ERROR = 1123

    def __init__(
        self,
        appid: str,
        token: str,
        upload_url: str,
        status_url: str,
        tts_url: str,
        tts_cluster: str,
        resource_id: str,
        audio_format: str = "mp3",
        sample_rate: int = 24000,
        timeout: float = 60.0,
    ) -> None:
        self.appid = appid
        self.token = token
        self.upload_url = upload_url
        self.status_url = status_url
        self.tts_url = tts_url
        self.tts_cluster = tts_cluster
        self.resource_id = resource_id
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.timeout = timeout
        static_root = Path(volc_settings.STATIC_AUDIO_ROOT)
        self.output_dir = _ensure_dir(static_root / "audios" / "cloned")
        
        # 当前使用的 speaker_id 索引
        self._current_speaker_index = 0

    def _build_auth_headers(self) -> Dict[str, str]:
        """构建认证请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer;{self.token}",
            "Resource-Id": self.resource_id,
        }

    def upload_and_train(self, audio_path: str, speaker_id: str) -> Dict[str, Any]:
        """上传参考音频并训练音色.
        
        Args:
            audio_path: 参考音频文件路径
            speaker_id: 自定义的唯一音色 ID
            
        Returns:
            API 响应字典
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"参考音频不存在: {audio_path}")
        
        # 读取并编码音频
        audio_data = _file_to_base64(path)
        audio_format = path.suffix.replace(".", "") or "wav"
        
        request_body = {
            "appid": self.appid,
            "speaker_id": speaker_id,
            "audios": [{"audio_bytes": audio_data, "audio_format": audio_format}],
            "source": 2,  # 用户上传
            "language": 0,  # 自动检测
            "model_type": 4,  # ICL 2.0 模型
        }
        
        logger.info(f"[VoiceClone] 上传参考音频: {path.name}, speaker_id={speaker_id}")
        response = requests.post(
            self.upload_url,
            json=request_body,
            headers=self._build_auth_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"[VoiceClone] 上传响应: {result}")
        return result

    def get_status(self, speaker_id: str) -> Dict[str, Any]:
        """查询音色训练状态.
        
        Args:
            speaker_id: 音色 ID
            
        Returns:
            状态信息字典，包含 status 字段:
            - 0: 准备中
            - 1: 训练中
            - 2: 训练完成
            - 3: 训练失败
        """
        request_body = {
            "appid": self.appid,
            "speaker_id": speaker_id,
        }
        
        response = requests.post(
            self.status_url,
            json=request_body,
            headers=self._build_auth_headers(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_training(self, speaker_id: str, max_wait: int = 60, poll_interval: float = 2.0) -> bool:
        """等待音色训练完成.
        
        Args:
            speaker_id: 音色 ID
            max_wait: 最大等待时间（秒）
            poll_interval: 轮询间隔（秒）
            
        Returns:
            是否训练成功
        """
        import time
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status = self.get_status(speaker_id)
            state = status.get("status", -1)
            
            if state == 2:  # 训练完成
                logger.info(f"[VoiceClone] 音色训练完成: {speaker_id}")
                return True
            elif state == 3:  # 训练失败
                logger.error(f"[VoiceClone] 音色训练失败: {status}")
                return False
            else:
                logger.debug(f"[VoiceClone] 训练中... state={state}")
                time.sleep(poll_interval)
        
        logger.warning(f"[VoiceClone] 训练超时: {speaker_id}")
        return False

    def synthesize(self, text: str, speaker_id: str) -> str:
        """使用克隆音色合成语音.
        
        Args:
            text: 要合成的文本
            speaker_id: 已训练的音色 ID
            
        Returns:
            生成的音频文件路径
        """
        request_body = {
            "app": {
                "appid": self.appid,
                "token": self.token,
                "cluster": self.tts_cluster,
            },
            "user": {
                "uid": self.appid,
            },
            "audio": {
                "voice_type": speaker_id,  # 使用克隆的音色 ID
                "encoding": self.audio_format,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson",
            },
        }
        
        headers = {"Authorization": f"Bearer;{self.token}"}
        
        logger.info(f"[VoiceClone] 合成语音: text={text[:30]}..., speaker_id={speaker_id}")
        response = requests.post(
            self.tts_url,
            json=request_body,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        if "data" not in data:
            raise RuntimeError(f"语音克隆合成失败: {data}")
        
        audio_bytes = base64.b64decode(data["data"])
        output_path = self.output_dir / f"cloned_{uuid.uuid4().hex}.{self.audio_format}"
        with output_path.open("wb") as fp:
            fp.write(audio_bytes)
        
        logger.info(f"[VoiceClone] 合成成功: {output_path}")
        return str(output_path)

    def clone_and_synthesize(self, text: str, reference_audio: str) -> str:
        """一站式语音克隆: 上传音频 + 训练 + 合成.
        
        使用预定义的 speaker_id 进行微调。如果当前 speaker_id 的微调次数
        已耗尽（错误码 1123），会自动切换到下一个 speaker_id 重试。
        
        Args:
            text: 要合成的文本
            reference_audio: 参考音频文件路径
            
        Returns:
            生成的音频文件路径
            
        Raises:
            RuntimeError: 所有 speaker_id 的微调次数都已耗尽
        """
        # 遍历所有可用的 speaker_id
        start_index = self._current_speaker_index
        tried_count = 0
        
        while tried_count < len(self.AVAILABLE_SPEAKER_IDS):
            speaker_id = self.AVAILABLE_SPEAKER_IDS[self._current_speaker_index]
            logger.info(f"[VoiceClone] 尝试使用 speaker_id: {speaker_id} (索引: {self._current_speaker_index})")
            
            try:
                # 1. 上传并训练
                result = self.upload_and_train(reference_audio, speaker_id)
                base_response = result.get("BaseResp")
                # 检查是否返回微调次数耗尽错误
                error_code = base_response.get("code") or base_response.get("error_code") or base_response.get("StatusCode")
                if error_code == self.QUOTA_EXHAUSTED_ERROR:
                    logger.warning(f"[VoiceClone] speaker_id {speaker_id} 微调次数已耗尽，切换到下一个")
                    self._switch_to_next_speaker()
                    tried_count += 1
                    continue
                
                # 2. 等待训练完成
                if not self.wait_for_training(speaker_id, max_wait=30):
                    raise RuntimeError("语音克隆训练超时或失败")
                
                # 3. 合成语音
                return self.synthesize(text, speaker_id)
                
            except requests.HTTPError as e:
                # 检查 HTTP 响应中的错误码
                try:
                    error_data = e.response.json()
                    base_response = error_data.get("BaseResp")
                    error_code = base_response.get("code") or base_response.get("error_code") or base_response.get("StatusCode")
                    if error_code == self.QUOTA_EXHAUSTED_ERROR:
                        logger.warning(f"[VoiceClone] speaker_id {speaker_id} 微调次数已耗尽，切换到下一个")
                        self._switch_to_next_speaker()
                        tried_count += 1
                        continue
                except Exception:
                    pass
                logger.error(f"[VoiceClone] HTTP 错误: {e}")
                raise
                
            except Exception as e:
                logger.error(f"[VoiceClone] 克隆失败: {e}")
                raise
        
        # 所有 speaker_id 都已耗尽
        raise RuntimeError(
            f"所有 {len(self.AVAILABLE_SPEAKER_IDS)} 个 speaker_id 的微调次数都已耗尽，"
            "请在火山引擎控制台购买更多微调次数"
        )

    def _switch_to_next_speaker(self) -> None:
        """切换到下一个 speaker_id"""
        self._current_speaker_index = (self._current_speaker_index + 1) % len(self.AVAILABLE_SPEAKER_IDS)
        logger.info(f"[VoiceClone] 已切换到 speaker_id 索引: {self._current_speaker_index}")


# 单例实例，供业务层直接使用
# 配置中英文识别回退：优先中文识别，失败后回退到英文识别
speech_recognizer = VolcSpeechRecognizer(
    appid=volc_settings.VOLC_APP_ID,
    token=volc_settings.VOLC_ACCESS_TOKEN,
    ws_url=volc_settings.VOLC_ASR_WS_URL,
    uid=volc_settings.VOLC_UID,
    timeout=volc_settings.HTTP_TIMEOUT,
    clusters=[
        volc_settings.VOLC_ASR_CLUSTER_ZH,  # 优先中文识别
        volc_settings.VOLC_ASR_CLUSTER_EN,  # 回退到英文识别
    ],
)

chat_llm = VolcChatLLM(
    api_key=volc_settings.ARK_API_KEY,
    model_id=volc_settings.ARK_MODEL_ID,
    endpoint=volc_settings.ARK_ENDPOINT,
    timeout=volc_settings.HTTP_TIMEOUT,
)

voice_cloner = VolcVoiceCloner(
    appid=volc_settings.VOLC_APP_ID,
    token=volc_settings.VOLC_ACCESS_TOKEN,
    upload_url=volc_settings.VOLC_VOICE_CLONE_UPLOAD_URL,
    status_url=volc_settings.VOLC_VOICE_CLONE_STATUS_URL,
    tts_url=volc_settings.VOLC_TTS_ENDPOINT,
    tts_cluster=volc_settings.VOLC_VOICE_CLONE_CLUSTER,  # 使用语音克隆专用 cluster
    resource_id=volc_settings.VOLC_VOICE_CLONE_RESOURCE_ID,
    audio_format=volc_settings.VOLC_TTS_FORMAT,
    sample_rate=volc_settings.VOLC_TTS_SAMPLE_RATE,
    timeout=volc_settings.HTTP_TIMEOUT,
)
