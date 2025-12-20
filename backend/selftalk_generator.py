"""
SelfTalk 推理 / 渲染 pipeline 实现。
"""

from __future__ import annotations

import datetime
import os
import pickle
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import torch
from psbody.mesh import Mesh
from transformers import Wav2Vec2Processor

# 将 SelfTalk 目录添加到 Python 路径开头，确保优先使用本地的 wav2vec.py
# 而不是 conda 环境中安装的 wav2vec 包
_SELFTALK_DIR = Path(__file__).resolve().parents[1] / "SelfTalk"
if str(_SELFTALK_DIR) not in sys.path:
    sys.path.insert(0, str(_SELFTALK_DIR))

from backend.utils import normalize_device
from SelfTalk import SelfTalk
from demo_voca import render_sequence_meshes

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


@dataclass
class SelfTalkInferenceConfig:
    """SelfTalk 推理阶段的关键参数."""

    repo_root: Path
    selftalk_root: Path
    dataset: str
    dataset_root: Path
    checkpoint_path: Path
    wav_path: Path
    subject: str
    template_file: Path
    render_template: Path
    output_root: Path
    gpu: str
    feature_dim: int
    vertice_dim: int
    period: int
    fps: int


def run_selftalk_inference(payload) -> Dict[str, Any]:
    """SelfTalk 推理主流程。

    Args:
        payload: `backend.video_generator.GenerationPayload` 实例。

    Returns:
        Dict[str, Any]: 包含 status、video_path、artifacts 或错误信息。

    TODO:
        - 支持异步/任务队列，避免长时间占用 Flask worker。
        - 将日志输出回传给前端，便于展示推理进度。
    """
    try:
        config = _build_config(payload)
        _validate_resources(config)
        vertices_path = _run_prediction(config)
        video_path = _render_with_audio(config, vertices_path)
        web_path = "/" + video_path.relative_to(config.repo_root).as_posix()

        return {
            "status": "success",
            "video_path": web_path,
            "artifacts": {
                "video_abspath": str(video_path),
                "vertices": str(vertices_path),
                "checkpoint": str(config.checkpoint_path),
                "subject": config.subject,
            },
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": str(exc),
            "error_type": exc.__class__.__name__,
        }


def _build_config(payload) -> SelfTalkInferenceConfig:
    """根据 payload 构造 SelfTalkInferenceConfig。

    Args:
        payload: GenerationPayload，含用户输入的原始参数。

    Returns:
        SelfTalkInferenceConfig: 推理所需的完整配置。

    Raises:
        ValueError/FileNotFoundError: 当关键输入缺失或路径不存在时。
    """
    repo_root = Path(__file__).resolve().parents[1]
    selftalk_root = repo_root / "SelfTalk"
    dataset = (payload.dataset or "vocaset").lower()
    dataset_root = selftalk_root / dataset

    wav_path = _resolve_path(payload.ref_audio, repo_root)
    if wav_path is None:
        raise ValueError("SelfTalk 推理必须提供音频路径 ref_audio")

    checkpoint_path = _resolve_checkpoint_path(
        candidate=payload.model_path,
        dataset_root=dataset_root,
    )

    subject = payload.subject or "FaceTalk_170908_03277_TA"
    template_file = dataset_root / "templates.pkl"
    render_template = dataset_root / "templates" / "FLAME_sample.ply"

    output_base = payload.output_dir or "static/videos"
    output_root = (repo_root / output_base).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"selftalk_{timestamp}_{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    feature_dim, vertice_dim, period, fps = _dataset_defaults(dataset)
    device = normalize_device(payload.gpu_choice)

    return SelfTalkInferenceConfig(
        repo_root=repo_root,
        selftalk_root=selftalk_root,
        dataset=dataset,
        dataset_root=dataset_root,
        checkpoint_path=checkpoint_path,
        wav_path=wav_path,
        subject=subject,
        template_file=template_file,
        render_template=render_template,
        output_root=run_dir,
        gpu=device,
        feature_dim=feature_dim,
        vertice_dim=vertice_dim,
        period=period,
        fps=fps,
    )

def _dataset_defaults(dataset: str) -> Tuple[int, int, int, int]:
    """返回 dataset 对应的 feature_dim / vertice_dim / period / fps。"""
    if dataset.lower() == "biwi":
        return 1024, 23370 * 3, 25, 25
    return 512, 5023 * 3, 30, 30


def _resolve_path(path_str: Optional[str], base_dir: Path) -> Optional[Path]:
    """将可能的相对路径解析为绝对 Path。"""
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    normalized = path_str.lstrip("/")
    guess = (base_dir / normalized).resolve()
    if guess.exists():
        return guess
    raise FileNotFoundError(f"找不到文件：{path_str}")


def _resolve_checkpoint_path(candidate: Optional[str], dataset_root: Path) -> Path:
    """定位 checkpoint 路径，若传入目录则自动选择最新模型。"""
    if candidate:
        resolved = Path(candidate)
        if not resolved.is_absolute():
            resolved = (dataset_root / candidate).resolve()
        if resolved.is_dir():
            latest = _latest_checkpoint(resolved)
            if latest:
                return latest
        if resolved.suffix != ".pth":
            maybe = resolved.with_suffix(".pth")
            if maybe.exists():
                resolved = maybe
        if not resolved.exists():
            raise FileNotFoundError(f"未找到模型文件：{candidate}")
        return resolved

    save_dir = dataset_root / "save"
    latest = _latest_checkpoint(save_dir)
    if latest:
        return latest
    raise FileNotFoundError(f"在 {save_dir} 下没有找到可用的模型 checkpoint")


def _latest_checkpoint(directory: Path) -> Optional[Path]:
    """返回目录下最近修改的 .pth 文件。"""
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("**/*.pth"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _validate_resources(config: SelfTalkInferenceConfig) -> None:
    """确保推理所需的文件/目录齐全。"""
    required_paths = [
        config.dataset_root,
        config.template_file,
        config.render_template,
        config.checkpoint_path,
        config.wav_path,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件或目录：{path}")


def _prepare_model(config: SelfTalkInferenceConfig) -> SelfTalk:
    """实例化 SelfTalk 模型并加载 checkpoint。"""
    device = torch.device(config.gpu if config.gpu.startswith("cuda") and torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(
        dataset=config.dataset,
        period=config.period,
        feature_dim=config.feature_dim,
        vertice_dim=config.vertice_dim,
        device=str(device),
    )
    model = SelfTalk(args)
    state_dict = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _load_template(config: SelfTalkInferenceConfig, device: torch.device) -> torch.Tensor:
    """读取 templates.pkl 中的 subject 模板。"""
    with open(config.template_file, "rb") as fin:
        templates = pickle.load(fin, encoding="latin1")
    if config.subject not in templates:
        raise KeyError(f"templates.pkl 中不存在 subject：{config.subject}")
    template = templates[config.subject]
    template = template.reshape(1, -1)
    tensor = torch.FloatTensor(template).to(device)
    return tensor


def _run_prediction(config: SelfTalkInferenceConfig) -> Path:
    """运行 SelfTalk 推理并保存顶点序列。

    Args:
        config (SelfTalkInferenceConfig): 推理配置。

    Returns:
        Path: 生成的 `.npy` 文件路径。

    TODO:
        - 缓存 processor / 模型权重，避免重复加载。
        - 将中间日志写入文件方便排查。
    """
    model = _prepare_model(config)
    device = next(model.parameters()).device
    template_tensor = _load_template(config, device)

    speech_array, _ = librosa.load(str(config.wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    audio_feature = processor(speech_array, sampling_rate=16000).input_values
    audio_feature = np.reshape(np.squeeze(audio_feature), (1, -1))
    audio_tensor = torch.FloatTensor(audio_feature).to(device)

    with torch.no_grad():
        prediction, _, _ = model.predict(audio_tensor, template_tensor)

    vertices = prediction.squeeze().detach().cpu().numpy()
    vertices = vertices.reshape(vertices.shape[0], -1, 3)
    npy_path = config.output_root / f"{config.wav_path.stem}.npy"
    np.save(npy_path, vertices)
    return npy_path


def _render_with_audio(config: SelfTalkInferenceConfig, vertices_path: Path) -> Path:
    """渲染网格并合成音频。

    Args:
        config (SelfTalkInferenceConfig): 推理配置。
        vertices_path (Path): `_run_prediction` 输出的顶点文件。

    Returns:
        Path: 最终生成的视频路径。

    TODO:
        - 支持自定义纹理/背景，提升渲染观感。
        - 增加异常捕获，若渲染失败可返回 npy 供再次尝试。
    """
    predicted_vertices = np.load(vertices_path)
    template = Mesh(filename=str(config.render_template))
    render_sequence_meshes(
        str(config.wav_path),
        predicted_vertices,
        template,
        str(config.output_root),
        uv_template_fname="",
        texture_img_fname="",
    )
    return config.output_root / "video.mp4"

