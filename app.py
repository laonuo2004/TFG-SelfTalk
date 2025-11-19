from flask import Flask, jsonify, render_template, request
from pathlib import Path
from typing import Any, Dict
from werkzeug.utils import secure_filename
import datetime
import uuid

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
VIDEO_UPLOAD_DIR = UPLOAD_DIR / "videos"
AUDIO_UPLOAD_DIR = UPLOAD_DIR / "audios"
REC_AUDIO_DIR = STATIC_DIR / "audios" / "recordings"
STATIC_VIDEO_DIR = STATIC_DIR / "videos"

for _dir in (STATIC_DIR, UPLOAD_DIR, VIDEO_UPLOAD_DIR, AUDIO_UPLOAD_DIR, REC_AUDIO_DIR, STATIC_VIDEO_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _build_payload_from_request() -> Dict[str, Any]:
    """合并请求参数为统一字典。

    Args:
        None

    Returns:
        Dict[str, Any]: 汇总后的参数，优先级 form > JSON > query。
    """
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
    """保存上传文件并返回绝对路径。

    Args:
        file_storage (FileStorage): Flask 提供的文件对象。
        target_dir (Path): 存储目录。
        prefix (str): 文件名前缀（用于区分用途）。

    Returns:
        str: 保存后的绝对路径；若未上传则为空字符串。
    """
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
    """将绝对路径转换为以项目根目录为基准的 URL。

    Args:
        path (Path): 目标文件绝对路径。

    Returns:
        str: 以 `/` 开头的相对 URL；若无法相对化则返回原路径。
    """
    try:
        return "/" + path.resolve().relative_to(BASE_DIR).as_posix()
    except ValueError:
        return str(path)


def _json_error(message: str, status_code: int = 400):
    """生成统一的失败响应。

    Args:
        message (str): 错误描述。
        status_code (int, optional): HTTP 状态码。默认为 400。

    Returns:
        Response: Flask JSON 响应对象。
    """
    response = jsonify({"status": "failed", "message": message})
    response.status_code = status_code
    return response


def _json_success(payload: Dict[str, Any]):
    """生成统一的成功响应。

    Args:
        payload (Dict[str, Any]): 需要序列化的键值对。

    Returns:
        Response: Flask JSON 响应对象。
    """
    response = jsonify(payload)
    response.status_code = 200
    return response


# 首页
@app.route('/')
def index():
    return render_template('index.html')


# 模型训练页面 & 表单提交
@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        from backend.model_trainer import train_model

        payload = _build_payload_from_request()
        uploaded_video = request.files.get('ref_video_file')
        if uploaded_video:
            payload['ref_video'] = _save_uploaded_file(uploaded_video, VIDEO_UPLOAD_DIR, 'refvideo')

        try:
            result = train_model(payload)
        except Exception as exc:
            return _json_error(f"训练任务触发异常：{exc}", 500)
        return _json_success(result)

    return render_template('model_training.html')


# 视频生成页面 & 表单提交
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        from backend.video_generator import generate_video

        payload = _build_payload_from_request()
        audio_upload = request.files.get('ref_audio_file') or request.files.get('audio_file')
        if audio_upload:
            payload['ref_audio'] = _save_uploaded_file(audio_upload, AUDIO_UPLOAD_DIR, 'refaudio')

        try:
            result = generate_video(payload)
        except Exception as exc:
            return _json_error(f"生成任务触发异常：{exc}", 500)
        return _json_success(result)

    return render_template('video_generation.html')


# 人机对话页面 & 表单提交
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        from backend.chat_engine import chat_response

        payload = _build_payload_from_request()
        reference_upload = request.files.get('reference_audio')
        if reference_upload:
            payload['reference_audio'] = _save_uploaded_file(reference_upload, AUDIO_UPLOAD_DIR, 'reference')

        try:
            result = chat_response(payload)
        except Exception as exc:
            return _json_error(f"对话流程触发异常：{exc}", 500)
        return _json_success(result)

    return render_template('chat_system.html')


# 录音上传接口（前端会用 fetch 调用）
@app.route('/save_audio', methods=['POST'])
def save_audio():
    audio_blob = request.files.get('audio')
    if not audio_blob:
        return _json_error("未接收到音频数据")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(audio_blob.filename or f"recording_{timestamp}.wav")
    if not filename.lower().endswith(('.wav', '.mp3', '.m4a')):
        filename = f"{filename}.wav"

    target_path = REC_AUDIO_DIR / f"{timestamp}_{uuid.uuid4().hex}_{filename}"
    REC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_blob.save(target_path)

    return _json_success({
        "status": "success",
        "filepath": str(target_path.resolve()),
        "url": _relative_static_url(target_path),
    })


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    # 监听所有网卡，方便 VS Code 端口转发
    app.run(host='0.0.0.0', port=5001, debug=True)
