# Dockerfile
# NVIDIA Canary-1B-v2 语音识别 API 服务镜像
#
# 基于 PyTorch 官方 CUDA 镜像, 集成 NeMo 框架和 FastAPI 服务

# 基础镜像: PyTorch 2.1 + CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
# - ffmpeg: 音频格式转换
# - libsndfile1: 音频文件读取
# - sox: 音频处理工具
# - curl: 健康检查
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 注意: NeMo 安装需要较长时间, 请耐心等待
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制应用源码
COPY src/ ./src/

# 创建模型存储目录
RUN mkdir -p /data/model

# 暴露 API 端口
EXPOSE 8909

# 设置默认环境变量
ENV MODEL_PATH=/data/model \
    MODEL_TIMEOUT_SEC=300 \
    MODEL_NAME=nvidia/canary-1b-v2 \
    USE_FP16=true \
    API_PORT=8909 \
    LOG_LEVEL=INFO

# 启动命令
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8909"]
