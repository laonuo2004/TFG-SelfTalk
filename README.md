# TFG-SelfTalk

基于 SelfTalk 的说话人脸生成对话系统 (Talking Face Generation System)

<p align="center">
  <img src="SelfTalk/media/SelfTalk.png" width="80%" />
</p>

## 📋 项目概述

本项目基于 [SelfTalk](https://github.com/psyai-net/SelfTalk_release) 实现，提供了一个完整的 Web 界面，支持：

- 🎓 **模型训练**：在线训练 SelfTalk 模型，实时查看训练日志
- 🎬 **视频生成**：根据语音生成 3D 说话人脸动画视频
- 💬 **实时对话**：与 AI 进行语音对话并生成虚拟人视频

---

## 🚀 快速开始

### 系统要求

- **操作系统**：Linux / WSL2 (Windows Subsystem for Linux)
- **GPU**：NVIDIA GPU (支持 CUDA 11.3)
- **内存**：建议 16GB+
- **磁盘空间**：建议 20GB+ (包含数据集和模型)

---

## 📦 手动部署

### 步骤 1: 安装系统依赖

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libboost-dev libgl1-mesa-glx libosmesa6-dev portaudio19-dev
```

### 步骤 2: 克隆项目

```bash
git clone https://github.com/laonuo2004/TFG-SelfTalk.git
cd TFG-SelfTalk
```

### 步骤 3: 创建 Conda 环境

```bash
conda create -n selftalk python=3.8.8
conda activate selftalk
```

### 步骤 4: 安装 PyTorch (CUDA 11.3)

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 步骤 5: 安装其他依赖

```bash
pip install -r requirements.txt
```

### 步骤 6: 安装 MPI-IS/mesh

```bash
# 克隆 mesh 库
git clone https://github.com/MPI-IS/mesh.git
cd mesh
pip install .
cd ..
```

### 步骤 7: 下载预训练模型(可选，用于快速测试视频生成与实时对话功能，否则需要自行训练模型)

1. 从 [Google Drive](https://drive.google.com/file/d/1iwxw4snYndoip2u2Iwe7h-rfPhVJRm2U/view?usp=sharing) 下载 `vocaset.pth`
2. 将文件放置到 `SelfTalk/vocaset/vocaset.pth`

### 步骤 8: 准备 VOCASET 数据集

1. 下载 [FLAME_sample.ply](https://github.com/TimoBolkart/voca/tree/master/template)，存放到 `SelfTalk/vocaset/templates/` 目录
2. 从 [VOCA 官网](https://voca.is.tue.mpg.de/) 申请并下载数据集，包括 Template Meshes 与 Training Data
3. 将 `templates.zip` 解压到 `SelfTalk/vocaset/templates/` 目录
4. 将以下文件放入 `SelfTalk/vocaset/` 目录：
   - `data_verts.npy`
   - `raw_audio_fixed.pkl`
   - `templates.pkl`
   - `subj_seq_to_idx.pkl`
5. 处理数据：
   ```bash
   cd SelfTalk/vocaset
   python process_voca_data.py
   cd ../..
   ```

### 步骤 9: 启动应用

```bash
python app.py
```

访问 http://localhost:6009 即可使用。

---

## 🐳 Docker 部署

### 使用 Docker Compose (推荐)

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

### 使用 Dockerfile

```bash
# 构建镜像
docker build -t tfg-selftalk .

# 运行容器
docker run -d \
  --gpus all \
  -p 6009:6009 \
  -p 8765:8765 \
  -v ./SelfTalk/vocaset:/app/SelfTalk/vocaset \
  tfg-selftalk
```

---

## 📖 使用说明

> **注意**：首次使用请确保能够访问 Hugging Face 的模型仓库。

### 模型训练

1. 访问「模型训练」页面
2. 选择训练设备 (GPU/CPU)
3. 配置训练参数 (Epochs、训练集、验证集等)
4. 点击「开始训练」
5. 实时查看训练日志和进度

### 视频生成

1. 访问「视频生成」页面
2. 选择已训练的模型
3. 上传音频文件 (.wav 格式)
4. 选择目标人物 (Subject)
5. 点击「生成视频」

### 人机对话

1. 访问「人机对话」页面
2. 点击麦克风开始录音
3. 与 AI 进行语音对话
4. 系统自动生成虚拟人视频回复

### 实时对话

1. 访问「实时对话」页面
2. 系统会自动连接语音对话服务（需要麦克风权限）
3. 直接与 AI 进行实时语音交流
4. AI 会以语音形式实时回复

---

## 📁 项目结构

```
TFG-SelfTalk/
├── app.py                  # Flask 主应用
├── requirements.txt        # Python 依赖
├── models.json             # 已注册模型列表
├── SelfTalk/               # SelfTalk 核心代码
│   ├── main.py             # 训练入口
│   ├── demo_voca.py        # 推理 Demo
│   └── vocaset/            # 数据集目录
│       ├── vocaset.pth     # 预训练模型
│       ├── wav/            # 音频文件
│       ├── vertices_npy/   # 顶点数据
│       └── save/           # 训练模型保存目录
├── backend/                # 后端逻辑
│   ├── model_trainer.py    # 训练调度
│   ├── selftalk_trainer.py # SelfTalk 训练
│   ├── selftalk_generator.py # 视频生成
│   └── model_registry.py   # 模型注册管理
├── templates/              # HTML 模板
└── static/                 # 静态资源
```

---

## 📄 许可证

本项目基于 [SelfTalk](https://github.com/psyai-net/SelfTalk_release) 开发，遵循 CC-BY-NC 4.0 许可证。

## 🙏 致谢

- [SelfTalk](https://github.com/psyai-net/SelfTalk_release) - 核心算法
- [VOCASET](https://voca.is.tue.mpg.de/) - 数据集
- [MPI-IS/mesh](https://github.com/MPI-IS/mesh) - 网格处理库