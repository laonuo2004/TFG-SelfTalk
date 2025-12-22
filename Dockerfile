# =============================================================================
# TFG-SelfTalk Dockerfile
# 基于 NVIDIA CUDA 11.3 的 GPU 加速镜像
# =============================================================================

FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 防止交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libboost-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    portaudio19-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 3.8 为默认
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装 PyTorch (CUDA 11.3)
# 先固定 typing-extensions 版本以兼容 Python 3.8
RUN pip install --no-cache-dir typing-extensions==4.7.1 && \
    pip install --no-cache-dir \
    torch==1.12.1+cu113 \
    torchvision==0.13.1+cu113 \
    torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 安装其他 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 克隆并安装 MPI-IS/mesh
RUN git clone https://github.com/MPI-IS/mesh.git /tmp/mesh \
    && cd /tmp/mesh \
    && pip install --no-cache-dir . \
    && rm -rf /tmp/mesh

# 复制项目文件
COPY . .

# 创建必要目录
RUN mkdir -p SelfTalk/vocaset

# 预下载 Hugging Face 模型（使用国内镜像源）
# 这样容器启动时无需访问 Hugging Face
ENV HF_ENDPOINT=https://hf-mirror.com
RUN python -c "from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer; \
    Wav2Vec2Model.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english'); \
    Wav2Vec2ForCTC.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english'); \
    Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english'); \
    Wav2Vec2CTCTokenizer.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english'); \
    Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')"

# 设置环境变量
ENV PYOPENGL_PLATFORM=osmesa
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000
EXPOSE 8765

# 创建数据卷挂载点
VOLUME ["/app/SelfTalk/vocaset/wav", "/app/SelfTalk/vocaset/vertices_npy"]

# 启动命令
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "5000"]
