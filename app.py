from flask import Flask, jsonify, render_template, request
from pathlib import Path
from typing import Any, Dict
from werkzeug.utils import secure_filename
import datetime
import uuid
import asyncio
import threading
import json
import os
import sys

import websockets
from websockets.server import WebSocketServerProtocol
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "python3.7"))
import config
from realtime_dialog_client import RealtimeDialogClient

app = Flask(__name__)

# ================== 文件路径相关 ==================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"
AUDIO_UPLOAD_DIR = UPLOAD_DIR / "audios"
REC_AUDIO_DIR = STATIC_DIR / "audios" / "recordings"
STATIC_VIDEO_DIR = STATIC_DIR / "videos"

for _dir in (
    STATIC_DIR,
    UPLOAD_DIR,
    VIDEO_UPLOAD_DIR,
    AUDIO_UPLOAD_DIR,
    REC_AUDIO_DIR,
    STATIC_VIDEO_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)


def _build_payload_from_request() -> Dict[str, Any]:
    """合并请求参数为统一字典：form > JSON > query。"""
    data: Dict[str, Any] = {}
    if request.form:
        data.update(request.form.to_dict())
    json_payload = request.get_json(silent=True)
    if json_payload:
        data.update(json_payload)
    if request.args:
        for key, value in request.args.items():
            data.setdefault(key, value)
    return data


def _save_uploaded_file(file_storage, target_dir: Path, prefix: str) -> str:
    """保存上传文件并返回绝对路径。"""
    if not file_storage or not file_storage.filename:
        return ""
    filename = secure_filename(file_storage.filename)
    extension = Path(filename).suffix or ".bin"
    unique_name = f"{prefix}_{uuid.uuid4().hex}{extension}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / unique_name
    file_storage.save(target_path)
    return str(target_path.resolve())


def _relative_static_url(path: Path) -> str:
    """将绝对路径转换为以项目根目录为基准的 URL。"""
    try:
        return "/" + path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def _json_error(message: str, status_code: int = 400):
    """统一失败响应。"""
    response = jsonify({"status": "failed", "message": message})
    response.status_code = status_code
    return response


def _json_success(payload: Dict[str, Any]):
    """统一成功响应。"""
    response = jsonify(payload)
    response.status_code = 200
    return response


# ================== WebSocket 中继（火山 RealtimeDialog） ==================


class DialogSession:
    """
    一条浏览器 WebSocket 连接对应一个火山对话会话。
    """

    def __init__(
        self,
        ws_config: Dict[str, Any],
        output_audio_format: str = "pcm_s16le",
        mod: str = "audio",
        recv_timeout: int = 10,
        expect_sample_rate: int = 16000,
    ):
        self.session_id = str(uuid.uuid4())
        self.output_audio_format = output_audio_format
        self.mod = mod
        self.recv_timeout = recv_timeout
        self.expect_sample_rate = expect_sample_rate

        self.recording_enabled = False
        # 20ms 一帧：16k * 0.02 * 2 bytes = 640 bytes（小于此丢掉，防止碎片太小）
        self.min_chunk_bytes = 640

        # 火山实时对话客户端（沿用你的现有实现）
        # 支持一个可选的本地模拟模式（用于没有上游凭证时本地调试）
        self.mock_upstream = bool(os.environ.get("MOCK_UPSTREAM", "") in ("1", "true", "yes"))
        if not self.mock_upstream:
            self.client = RealtimeDialogClient(
                config=ws_config,
                session_id=self.session_id,
                output_audio_format=output_audio_format,
                mod=mod,
                recv_timeout=recv_timeout,
            )
        else:
            self.client = None

    def _generate_tone_pcm(self, freq: float = 440.0, duration: float = 0.4, sample_rate: int = 16000):
        """生成单声道 16-bit PCM 的简单正弦波（little endian）。用于本地 mock 回放测试。"""
        import math, array

        total = int(sample_rate * duration)
        arr = array.array('h')
        for i in range(total):
            t = i / sample_rate
            v = int(0.5 * 32767 * math.sin(2 * math.pi * freq * t))
            arr.append(v)
        return arr.tobytes()

    async def _send_json(self, websocket: WebSocketServerProtocol, obj: Dict[str, Any]):
        try:
            await websocket.send(json.dumps(obj, ensure_ascii=False))
        except Exception as e:
            print(f"[DialogSession] send_json error: {e}")

    async def handle_server_response(
        self, response: Dict[str, Any], websocket: WebSocketServerProtocol
    ):
        """
        处理火山侧返回：
        - SERVER_ACK：payload_msg 为 PCM 字节，直接二进制发给前端
        - SERVER_FULL_RESPONSE：事件/文本，转成 JSON 文本下发
        - SERVER_ERROR：错误转成 JSON
        """
        if not response:
            return

        msg_type = response.get("message_type")

        if msg_type == "SERVER_ACK" and isinstance(
            response.get("payload_msg"), (bytes, bytearray)
        ):
            await websocket.send(response["payload_msg"])

        elif msg_type == "SERVER_FULL_RESPONSE":
            await self._send_json(
                websocket,
                {
                    "type": "event",
                    "event": response.get("event"),
                    "payload": response.get("payload_msg"),
                },
            )

        elif msg_type == "SERVER_ERROR":
            await self._send_json(
                websocket,
                {"type": "error", "message": str(response.get("payload_msg"))},
            )
            raise Exception(f"服务器错误：{response.get('payload_msg')}")

    async def run(self, websocket: WebSocketServerProtocol):
        print(f"[DialogSession] start session_id={self.session_id}")
        try:
            # 1) 建连火山
            await self.client.connect()

            # 2) 可选：打个招呼
            try:
                if hasattr(self.client, "say_hello"):
                    await self.client.say_hello()
            except Exception as e:
                print(f"[DialogSession] say_hello failed: {e}")

            # 前端→模型
            async def frontend_to_model():
                try:
                    async for message in websocket:
                        # 二进制：音频帧
                        if isinstance(message, (bytes, bytearray)):
                            if (
                                self.recording_enabled
                                and message
                                and len(message) >= self.min_chunk_bytes
                            ):
                                # 如果处于 mock 模式，直接回送一段合成的 PCM 用于前端播放测试
                                if self.mock_upstream:
                                    try:
                                        pcm = self._generate_tone_pcm()
                                        await websocket.send(pcm)
                                    except Exception as e:
                                        print(f"[DialogSession] mock send pcm error: {e}")
                                else:
                                    await self.client.task_request(bytes(message))
                            continue

                        # 字符串：控制/文本
                        text = (message or "").strip()
                        obj = None
                        try:
                            obj = json.loads(text)
                        except Exception:
                            pass

                        if isinstance(obj, dict) and obj.get("type") == "control":
                            op = obj.get("op")
                            if op == "start":
                                self.recording_enabled = True
                                await self._send_json(
                                    websocket,
                                    {"type": "control_ack", "op": "start"},
                                )
                            elif op == "stop":
                                self.recording_enabled = False
                                await self._send_json(
                                    websocket,
                                    {"type": "control_ack", "op": "stop"},
                                )
                            elif op == "text":
                                q = (obj.get("text") or "").strip()
                                if q:
                                    await self.client.chat_text_query(q)
                            else:
                                await self._send_json(
                                    websocket,
                                    {
                                        "type": "warn",
                                        "message": f"未知操作: {op}",
                                    },
                                )
                            continue

                        # 兜底：当作纯文本 query
                        if text:
                            await self.client.chat_text_query(text)

                except websockets.exceptions.ConnectionClosed:
                    print(
                        f"[DialogSession] websocket closed by frontend (send), session={self.session_id}"
                    )
                except Exception as e:
                    print(f"[DialogSession] frontend_to_model error: {e}")

            # 模型→前端
            async def model_to_frontend():
                try:
                    # 如果为 mock 模式，周期性不从上游读取，而是发送模拟文本事件（覆盖真实交互）
                    if self.mock_upstream:
                        while True:
                            await asyncio.sleep(0.6)
                            # 发送一条模拟文本事件
                            try:
                                await self._send_json(
                                    websocket,
                                    {
                                        "type": "event",
                                        "event": "mock",
                                        "payload": "这是来自本地模拟的回复",
                                    },
                                )
                            except Exception as e:
                                print(f"[DialogSession] mock send event error: {e}")
                                break
                    else:
                        while True:
                            response = await self.client.receive_server_response()
                            await self.handle_server_response(response, websocket)
                except websockets.exceptions.ConnectionClosed:
                    print(
                        f"[DialogSession] websocket closed by frontend (recv), session={self.session_id}"
                    )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"[DialogSession] model_to_frontend error: {e}")

            # 并发跑
            task_send = asyncio.create_task(frontend_to_model())
            task_recv = asyncio.create_task(model_to_frontend())
            done, pending = await asyncio.wait(
                [task_send, task_recv], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()

        finally:
            print(
                f"[DialogSession] closing RealtimeDialog session {self.session_id}"
            )
            # 顺序关闭
            # 顺序关闭（仅在 client.ws 存在或方法内部安全时调用）
            try:
                ws_attr = getattr(self.client, 'ws', None)
            except Exception:
                ws_attr = None

            if ws_attr:
                for fn in (
                    self.client.finish_session,
                    self.client.finish_connection,
                    self.client.close,
                ):
                    try:
                        await fn()
                    except Exception as e:
                        print(f"[DialogSession] close error: {e}")
            else:
                # client 未成功建立到上游的 WebSocket，尝试仅调用 close()（防守式）
                try:
                    if hasattr(self.client, 'close'):
                        await self.client.close()
                except Exception as e:
                    print(f"[DialogSession] close error (no ws): {e}")


# WebSocket 入口
async def ws_handler(websocket: WebSocketServerProtocol, path: str):
    print(f"[WS] new frontend connection, path={path}")
    session = DialogSession(
        ws_config=config.ws_connect_config,
        output_audio_format="pcm_s16le",
        mod="audio",
        recv_timeout=10,
        expect_sample_rate=16000,
    )
    try:
        await session.run(websocket)
    except Exception as e:
        # 打印完整堆栈到服务器日志，便于排查
        tb = traceback.format_exc()
        print(f"[WS] session error: {e}\n{tb}")
        # 尝试发送结构化错误信息给前端，提醒原因
        try:
            err_obj = {"type": "error", "message": f"Server internal error: {str(e)}"}
            await websocket.send(json.dumps(err_obj, ensure_ascii=False))
        except Exception:
            pass
        try:
            # 1011 表示服务器内部错误（Internal Error）
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        print(f"[WS] frontend connection closed, path={path}")


async def ws_main():
    """
    启动 WebSocket 中继服务器：
    - 监听 0.0.0.0:6006（可按需改端口，对应 AutoDL 映射）
    """
    host, port = "0.0.0.0", 6006
    server = await websockets.serve(
        ws_handler,
        host,
        port,
        max_size=4 * 1024 * 1024,
        ping_interval=20,
        ping_timeout=20,
    )
    print(f"[WS] relay server listening on {host}:{port}")
    await server.wait_closed()


def run_ws_server():
    """在当前线程中阻塞运行 WebSocket 中继"""
    asyncio.run(ws_main())


# ================== Flask 路由 ==================


# 首页
@app.route("/")
def index():
    return render_template("index.html")


# 模型训练页面 & 表单提交
@app.route("/model_training", methods=["GET", "POST"])
def model_training():
    if request.method == "POST":
        from backend.model_trainer import train_model

        payload = _build_payload_from_request()
        uploaded_video = request.files.get("ref_video_file")
        if uploaded_video:
            payload["ref_video"] = _save_uploaded_file(
                uploaded_video, VIDEO_UPLOAD_DIR, "refvideo"
            )

        try:
            result = train_model(payload)
        except Exception as exc:
            return _json_error(f"训练任务触发异常：{exc}", 500)
        return _json_success(result)

    return render_template("model_training.html")


# 视频生成页面 & 表单提交
@app.route("/video_generation", methods=["GET", "POST"])
def video_generation():
    if request.method == "POST":
        from backend.video_generator import generate_video

        payload = _build_payload_from_request()
        audio_upload = request.files.get("ref_audio_file") or request.files.get(
            "audio_file"
        )
        if audio_upload:
            payload["ref_audio"] = _save_uploaded_file(
                audio_upload, AUDIO_UPLOAD_DIR, "refaudio"
            )

        try:
            result = generate_video(payload)
        except Exception as exc:
            return _json_error(f"生成任务触发异常：{exc}", 500)
        return _json_success(result)

    return render_template("video_generation.html")


# 人机对话页面 & 表单提交
@app.route("/chat_system", methods=["GET", "POST"])
def chat_system():
    if request.method == "POST":
        from backend.chat_engine import chat_response

        payload = _build_payload_from_request()
        reference_upload = request.files.get("reference_audio")
        if reference_upload:
            payload["reference_audio"] = _save_uploaded_file(
                reference_upload, AUDIO_UPLOAD_DIR, "reference"
            )

        try:
            result = chat_response(payload)
        except Exception as exc:
            return _json_error(f"对话流程触发异常：{exc}", 500)
        return _json_success(result)

    return render_template("chat_system.html")


# 录音上传接口（前端会用 fetch 调用）
@app.route("/save_audio", methods=["POST"])
def save_audio():
    audio_blob = request.files.get("audio")
    if not audio_blob:
        return _json_error("未接收到音频数据")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(audio_blob.filename or f"recording_{timestamp}.wav")
    if not filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm")):
        filename = f"{filename}.wav"

    target_path = REC_AUDIO_DIR / f"{timestamp}_{uuid.uuid4().hex}_{filename}"
    REC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_blob.save(target_path)

    return _json_success(
        {
            "status": "success",
            "filepath": str(target_path.resolve()),
            "url": _relative_static_url(target_path),
        }
    )


@app.route("/about")
def about():
    return render_template("about.html")


# 新增路由：直接渲染模板文件 `实时对话.html`
@app.route("/实时对话.html")
def realtime_html():
    return render_template("实时对话.html")


# ================== 启动入口 ==================

if __name__ == "__main__":
    # 1) 启动 WebSocket 中继（后台线程）
    ws_thread = threading.Thread(
        target=run_ws_server,
        name="ws-relay-thread",
        daemon=True,
    )
    ws_thread.start()
    print("[MAIN] WebSocket relay started in background thread")

    # 2) 启动 Flask（用 AutoDL 映射的另一个端口，比如 6008）
    #    use_reloader=False 防止 debug 重载器起第二个进程导致 WS 端口冲突
    app.run(host="0.0.0.0", port=6008, debug=True, use_reloader=False)
