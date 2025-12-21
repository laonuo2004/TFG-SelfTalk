from flask import Flask, jsonify, render_template, request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from werkzeug.utils import secure_filename
import datetime
import uuid
import asyncio
import threading
import json
import os
import sys
import base64
import traceback

from backend.training_task_manager import task_manager, TaskStatus

import websockets
from websockets.server import WebSocketServerProtocol

# ============ 你自己的依赖 ============
BASE_DIR_STR = os.path.dirname(os.path.abspath(__file__))
PYTHON37_PATH = os.path.join(BASE_DIR_STR, "python3.7")
if PYTHON37_PATH not in sys.path:
    sys.path.insert(0, PYTHON37_PATH)

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


# ================== Flask 路由 ==================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model_training", methods=["GET", "POST"])
def model_training():
    if request.method == "POST":
        from backend.model_trainer import start_training_async, train_model

        payload_dict = _build_payload_from_request()
        uploaded_video = request.files.get("ref_video_file")
        if uploaded_video:
            payload_dict["ref_video"] = _save_uploaded_file(
                uploaded_video, VIDEO_UPLOAD_DIR, "refvideo"
            )

        # 尝试启动异步训练
        try:
            result = start_training_async(payload_dict)
            return _json_success(result)
        except Exception as exc:
            # 如果异步启动失败，回退到同步模式
            traceback.print_exc()
            return _json_error(f"训练任务触发异常：{exc}", 500)

    return render_template("model_training.html")


@app.route("/video_generation", methods=["GET", "POST"])
def video_generation():
    if request.method == "POST":
        from backend.video_generator import generate_video

        payload = _build_payload_from_request()
        audio_upload = request.files.get("ref_audio_file") or request.files.get("audio_file")
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


# ================== API 路由（供前端调用）==================

@app.route("/api/training/status/<task_id>", methods=["GET"])
def api_training_status(task_id: str):
    """获取训练任务状态和日志。"""
    since_index = request.args.get("since", 0, type=int)
    result = task_manager.get_logs_since(task_id, since_index)
    return jsonify(result)


@app.route("/api/devices", methods=["GET"])
def api_devices():
    """获取可用的计算设备列表。"""
    from backend.device_utils import get_available_devices, get_default_device
    
    devices = get_available_devices()
    default_device = get_default_device()
    
    return _json_success({
        "status": "success",
        "devices": devices,
        "default": default_device,
    })


@app.route("/api/subjects", methods=["GET"])
def api_subjects():
    """获取可用的 subject 列表。"""
    from backend.device_utils import get_subjects_list, get_default_subjects
    
    subjects = get_subjects_list()
    defaults = get_default_subjects()
    
    return _json_success({
        "status": "success",
        "subjects": subjects,
        "defaults": defaults,
    })


@app.route("/api/models", methods=["GET", "POST", "DELETE"])
def api_models():
    """模型管理 API。"""
    from backend.model_registry import (
        list_models, register_model, delete_model, get_models_for_dropdown
    )
    
    if request.method == "GET":
        # 获取模型列表
        models = get_models_for_dropdown()
        return _json_success({
            "status": "success",
            "models": models,
        })
    
    elif request.method == "POST":
        # 注册新模型
        payload = _build_payload_from_request()
        
        required_fields = ["path", "epochs", "dataset", "train_subjects", "val_subjects"]
        for field in required_fields:
            if field not in payload:
                return _json_error(f"缺少必填字段: {field}")
        
        try:
            # 处理 subjects 字段（可能是逗号分隔的字符串或列表）
            train_subjects = payload["train_subjects"]
            val_subjects = payload["val_subjects"]
            test_subjects = payload.get("test_subjects", [])
            
            if isinstance(train_subjects, str):
                train_subjects = [s.strip() for s in train_subjects.split(",") if s.strip()]
            if isinstance(val_subjects, str):
                val_subjects = [s.strip() for s in val_subjects.split(",") if s.strip()]
            if isinstance(test_subjects, str):
                test_subjects = [s.strip() for s in test_subjects.split(",") if s.strip()]
            
            model = register_model(
                path=payload["path"],
                epochs=int(payload["epochs"]),
                dataset=payload["dataset"],
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                name=payload.get("name"),  # 可选
            )
            
            return _json_success({
                "status": "success",
                "message": "模型注册成功",
                "model": model.to_dict(),
            })
        except Exception as exc:
            return _json_error(f"模型注册失败: {exc}")
    
    elif request.method == "DELETE":
        # 删除模型
        payload = _build_payload_from_request()
        name = payload.get("name")
        if not name:
            return _json_error("缺少模型名称")
        
        success = delete_model(name)
        if success:
            return _json_success({"status": "success", "message": "模型已删除"})
        else:
            return _json_error("模型不存在")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/realtime_talk.html")
def realtime_html():
    return render_template("realtime_talk.html")


# ================== 本地 WebSocket 网关 ==================

WS_HOST = "0.0.0.0"
WS_PORT = 8765

# 打开它会把上游原始包也透传到前端（调试用，默认关掉避免刷屏）
DEBUG_UPSTREAM = False

# --------- 工具：安全 JSON 化（仅用于 debug）---------
def _jsonable(obj: Any) -> Any:
    """把上游 dict 安全 JSON 化（bytes -> base64）。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
        return {"__bytes_b64__": base64.b64encode(b).decode("utf-8"), "__len__": len(b)}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


# --------- 工具：从上游数据里“尽量”抽取音频 bytes ---------
_AUDIO_KEYS = {"audio", "pcm", "tts", "tts_audio", "output_audio", "audio_bytes", "audio_data"}

def _find_first_bytes(obj: Any, depth: int = 0, max_depth: int = 6) -> Optional[bytes]:
    if depth > max_depth:
        return None
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj)
    if isinstance(obj, dict):
        # 先按 key 优先找
        for k in obj.keys():
            if k in _AUDIO_KEYS:
                v = obj.get(k)
                if isinstance(v, (bytes, bytearray)):
                    return bytes(v)
        # 再递归找
        for v in obj.values():
            b = _find_first_bytes(v, depth + 1, max_depth)
            if b:
                return b
    if isinstance(obj, (list, tuple)):
        for v in obj:
            b = _find_first_bytes(v, depth + 1, max_depth)
            if b:
                return b
    return None


# --------- 工具：从上游数据里“尽量”抽取文本 ---------
_TEXT_KEYS = {"text", "transcript", "asr", "result", "content", "answer", "message"}
_PARTIAL_HINT_KEYS = {"partial", "is_partial", "final", "is_final", "finished", "end"}

def _find_first_text(obj: Any, depth: int = 0, max_depth: int = 6) -> Optional[str]:
    if depth > max_depth:
        return None
    if isinstance(obj, str) and obj.strip():
        return obj
    if isinstance(obj, dict):
        # 优先看常见 key
        for k in obj.keys():
            if k in _TEXT_KEYS:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        # 递归
        for v in obj.values():
            t = _find_first_text(v, depth + 1, max_depth)
            if t:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _find_first_text(v, depth + 1, max_depth)
            if t:
                return t
    return None


def _guess_is_final(obj: Any) -> Optional[bool]:
    """尽量判断是不是 final（不保证准确）"""
    if not isinstance(obj, dict):
        return None
    for k in ("is_final", "final", "finished", "end"):
        v = obj.get(k)
        if isinstance(v, bool):
            return v
    # 有些协议会用数字/字符串
    for k in ("event", "status", "type"):
        v = obj.get(k)
        if isinstance(v, (int, str)):
            s = str(v).lower()
            if "final" in s or "end" in s or s in {"2", "completed"}:
                return True
    return None


async def ws_handler(websocket: WebSocketServerProtocol, path: str = ""):
    """
    浏览器 -> 本地WS -> RealtimeDialogClient -> VOLC
    - 浏览器发送 JSON: {"type":"start"|"stop"|"text"}
    - 浏览器发送 二进制: PCM S16LE 16k mono chunk
    - 服务端发送 JSON: {type:"log"|"error"|"partial"|"asr"|"assistant"}
    - 服务端发送 二进制: TTS PCM 16k s16le mono
    """
    session_id = uuid.uuid4().hex
    client = RealtimeDialogClient(
        config=config.ws_connect_config,
        session_id=session_id,
        output_audio_format="pcm_s16le",
        mod="audio",
        recv_timeout=30,
    )

    async def send_json(obj: dict):
        await websocket.send(json.dumps(obj, ensure_ascii=False))

    recv_task = None
    try:
        await send_json({"type": "log", "message": f"ws connected, session_id={session_id}"})

        # 连接上游
        await client.connect()
        await send_json({"type": "log", "message": "upstream connected"})

        # 后台接收上游消息：抽取文本/音频 -> 发给前端
        async def upstream_loop():
            last_partial = ""
            while True:
                data = await client.receive_server_response()

                # 1) 如果找到音频 bytes，直接二进制推给前端播放
                audio_bytes = _find_first_bytes(data)
                if audio_bytes:
                    try:
                        await websocket.send(audio_bytes)
                    except Exception:
                        # 前端断开就退出
                        break

                # 2) 抽取文本，尽量区分 partial / final
                text = _find_first_text(data)
                is_final = None
                if isinstance(data, dict):
                    is_final = _guess_is_final(data)

                if text:
                    # 避免 partial 同一句疯狂重复刷
                    if is_final is False:
                        if text != last_partial:
                            last_partial = text
                            await send_json({"type": "partial", "text": text})
                    elif is_final is True:
                        last_partial = ""
                        await send_json({"type": "asr", "text": text})
                    else:
                        # 不确定 final：先当 partial（更符合实时体验）
                        if text != last_partial:
                            last_partial = text
                            await send_json({"type": "partial", "text": text})

                # 3) 调试：需要时再透传原始包（默认关闭，避免刷屏）
                if DEBUG_UPSTREAM:
                    await send_json({"type": "upstream", "data": _jsonable(data)})

        recv_task = asyncio.create_task(upstream_loop())

        # 处理来自浏览器的输入（音频/控制/文本）
        async for msg in websocket:
            if isinstance(msg, (bytes, bytearray)):
                # 浏览器发来的二进制音频 chunk
                await client.task_request(bytes(msg))
                continue

            # JSON 控制/文本
            try:
                payload = json.loads(msg)
            except Exception:
                await send_json({"type": "error", "message": f"invalid json: {str(msg)[:200]}"})
                continue

            t = payload.get("type")
            if t == "start":
                await send_json({"type": "log", "message": "recording start"})
            elif t == "stop":
                await send_json({"type": "log", "message": "recording stop"})
            elif t == "text":
                text = payload.get("text", "")
                await send_json({"type": "log", "message": f"text -> upstream: {text[:50]}"})
                await client.chat_text_query(text)
            else:
                await send_json({"type": "log", "message": f"unknown: {payload}"})

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        traceback.print_exc()
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            if recv_task:
                recv_task.cancel()
        except Exception:
            pass
        try:
            await client.close()
        except Exception:
            pass


async def _run_ws_server():
    async with websockets.serve(
        ws_handler,
        WS_HOST,
        WS_PORT,
        max_size=10 * 1024 * 1024,
        ping_interval=None,
    ):
        print(f"[WS] listening on ws://{WS_HOST}:{WS_PORT}")
        await asyncio.Future()


def run_ws_server():
    """在后台线程运行 WS server（独立 event loop），不影响 Flask。"""
    asyncio.run(_run_ws_server())


# ================== 启动入口 ==================
if __name__ == "__main__":
    ws_thread = threading.Thread(target=run_ws_server, name="ws-gateway", daemon=True)
    ws_thread.start()
    print("[MAIN] WS gateway started")

    app.run(host="0.0.0.0", port=6009, debug=True, use_reloader=False)
