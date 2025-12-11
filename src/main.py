# -*- coding: utf-8 -*-
"""
FastAPI 主应用模块

提供 OpenAI Whisper 兼容的 REST API 接口:
- POST /v1/audio/transcriptions - 音频转录
- GET /health - 健康检查
- GET /status - 模型状态
- POST /model/load - 预加载模型
- POST /model/unload - 卸载模型
"""

import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from .model_manager import get_model_manager, shutdown_model_manager
from .engine import get_transcription_engine


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging():
    """
    配置日志系统
    
    使用 loguru 库实现结构化日志, 支持:
    - 控制台彩色输出
    - 文件日志轮转
    - 日志级别配置
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )
    
    logger.info(f"日志系统已初始化, 级别: {log_level}")


# ============================================================================
# 应用生命周期管理
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    启动时:
    - 初始化日志
    - 创建模型管理器 (但不加载模型, 实现懒加载)
    
    关闭时:
    - 卸载模型
    - 释放 GPU 资源
    """
    # 启动时执行
    setup_logging()
    logger.info("=== Canary ASR API 服务启动 ===")
    logger.info(f"模型路径: {os.getenv('MODEL_PATH', '/data/model')}")
    logger.info(f"超时时间: {os.getenv('MODEL_TIMEOUT_SEC', '300')}秒")
    logger.info(f"FP16 模式: {os.getenv('USE_FP16', 'true')}")
    
    # 初始化模型管理器 (仅创建实例, 不加载模型)
    _ = get_model_manager()
    logger.info("模型管理器已就绪 (懒加载模式, 首次请求时加载模型)")
    
    yield
    
    # 关闭时执行
    logger.info("=== 正在关闭 API 服务 ===")
    shutdown_model_manager()
    logger.info("API 服务已关闭")


# ============================================================================
# FastAPI 应用实例
# ============================================================================

app = FastAPI(
    title="Canary ASR API",
    description="""
## NVIDIA Canary-1B-v2 语音识别 API

兼容 OpenAI Whisper API 的语音转录服务, 基于 NVIDIA Canary-1B-v2 模型。

### 特性
- 🚀 **懒加载**: 首次请求时才加载模型, 节省资源
- ⏱️ **自动卸载**: 闲置超时后自动释放 GPU 显存
- 🎯 **高精度**: 支持 25 种欧洲语言的转录和翻译
- 📝 **多格式**: 支持 text/json/srt/vtt/verbose_json 输出格式

### 支持的语言
en, de, fr, es, it, pt, nl, pl, ru, uk, cs, sk, bg, hr, da, fi, el, hu, ro, sv, et, lv, lt, sl, mt
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# 数据模型
# ============================================================================

class TranscriptionResponse(BaseModel):
    """转录响应模型 (JSON 格式)"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


class VerboseTranscriptionResponse(BaseModel):
    """详细转录响应模型 (verbose_json 格式)"""
    task: str = "transcribe"
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: list = []
    words: Optional[list] = None


class ModelStatusResponse(BaseModel):
    """模型状态响应"""
    model_loaded: bool
    model_name: str
    model_path: str
    usage_count: int
    idle_seconds: Optional[float] = None
    timeout_seconds: int
    use_fp16: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_allocated_mb: Optional[float] = None
    gpu_memory_reserved_mb: Optional[float] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    message: str


class OperationResponse(BaseModel):
    """操作响应"""
    success: bool
    message: str


# ============================================================================
# API 路由
# ============================================================================

@app.get("/", response_class=PlainTextResponse)
async def root():
    """
    根路由 - 返回服务信息
    """
    return "Canary ASR API - 兼容 OpenAI Whisper API 的语音识别服务"


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    健康检查端点
    
    用于 Docker 健康检查和负载均衡器探测
    """
    return HealthResponse(
        status="healthy",
        message="服务运行正常"
    )


@app.get("/status", response_model=ModelStatusResponse)
async def get_status():
    """
    获取模型状态
    
    返回模型加载状态、GPU 使用情况等信息
    """
    manager = get_model_manager()
    status = manager.get_status()
    return ModelStatusResponse(**status)


@app.post("/model/load", response_model=OperationResponse)
async def load_model():
    """
    预加载模型
    
    手动触发模型加载, 用于预热服务
    """
    try:
        manager = get_model_manager()
        success = manager.force_load()
        
        if success:
            return OperationResponse(
                success=True,
                message="模型加载成功"
            )
        else:
            return OperationResponse(
                success=False,
                message="模型加载失败"
            )
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {e}")


@app.post("/model/unload", response_model=OperationResponse)
async def unload_model():
    """
    卸载模型
    
    手动释放 GPU 显存
    """
    try:
        manager = get_model_manager()
        success = manager.force_unload()
        
        if success:
            return OperationResponse(
                success=True,
                message="模型已卸载, 显存已释放"
            )
        else:
            return OperationResponse(
                success=False,
                message="无法卸载模型 (可能有请求正在处理)"
            )
    except Exception as e:
        logger.error(f"卸载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {e}")


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(..., description="要转录的音频文件"),
    model: str = Form(default="canary-1b-v2", description="模型名称 (兼容参数, 实际使用 Canary)"),
    language: Optional[str] = Form(default=None, description="音频语言代码, 如 'en', 'zh'"),
    response_format: str = Form(default="json", description="响应格式: text, json, srt, vtt, verbose_json"),
    temperature: Optional[float] = Form(default=None, description="采样温度 (兼容参数, 暂不使用)"),
    timestamp_granularities: Optional[str] = Form(default=None, description="时间戳粒度 (兼容参数)"),
):
    """
    音频转录 API
    
    兼容 OpenAI Whisper API 的 /v1/audio/transcriptions 端点
    
    ## 请求参数
    
    - **file**: 音频文件 (支持 wav, flac, mp3, m4a 等格式)
    - **model**: 模型名称 (兼容参数, 实际使用 Canary-1B-v2)
    - **language**: 音频语言代码, 如 'en', 'de', 'fr' 等
    - **response_format**: 响应格式
        - `text`: 纯文本
        - `json`: JSON 格式 (默认)
        - `srt`: SRT 字幕格式
        - `vtt`: WebVTT 字幕格式
        - `verbose_json`: 详细 JSON (包含时间戳)
    
    ## 响应
    
    根据 response_format 返回不同格式的转录结果
    
    ## 示例
    
    ```python
    import requests
    
    url = "http://localhost:8000/v1/audio/transcriptions"
    files = {"file": open("audio.wav", "rb")}
    data = {"language": "en", "response_format": "json"}
    
    response = requests.post(url, files=files, data=data)
    print(response.json())
    ```
    """
    # 验证响应格式
    valid_formats = {"text", "json", "srt", "vtt", "verbose_json"}
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的响应格式: {response_format}, 支持: {valid_formats}"
        )
    
    # 验证文件类型
    if file.content_type:
        allowed_types = {
            "audio/wav", "audio/wave", "audio/x-wav",
            "audio/flac", "audio/x-flac",
            "audio/mpeg", "audio/mp3",
            "audio/mp4", "audio/m4a", "audio/x-m4a",
            "audio/ogg", "audio/webm",
            "application/octet-stream",  # 允许未知类型
        }
        # 放宽类型检查, 允许更多格式
        logger.debug(f"文件类型: {file.content_type}")
    
    try:
        # 读取文件内容
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        
        logger.info(
            f"收到转录请求 - 文件: {file.filename}, "
            f"大小: {len(audio_bytes)} bytes, "
            f"语言: {language}, 格式: {response_format}"
        )
        
        # 获取转录引擎并执行转录
        engine = get_transcription_engine()
        result = engine.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            language=language,
            response_format=response_format,
            timestamps=True,
        )
        
        # 根据格式返回响应
        if response_format in {"text", "srt", "vtt"}:
            return PlainTextResponse(content=result, media_type="text/plain")
        else:
            return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"转录失败: {e}")
        raise HTTPException(status_code=500, detail=f"转录失败: {e}")


@app.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile = File(..., description="要翻译的音频文件"),
    model: str = Form(default="canary-1b-v2", description="模型名称"),
    response_format: str = Form(default="json", description="响应格式"),
    temperature: Optional[float] = Form(default=None, description="采样温度 (兼容参数)"),
):
    """
    音频翻译 API (翻译为英语)
    
    兼容 OpenAI Whisper API 的 /v1/audio/translations 端点
    将任意支持的语言翻译为英语
    
    ## 请求参数
    
    - **file**: 音频文件
    - **model**: 模型名称 (兼容参数)
    - **response_format**: 响应格式 (text, json, srt, vtt, verbose_json)
    
    ## 响应
    
    翻译后的英语文本
    """
    valid_formats = {"text", "json", "srt", "vtt", "verbose_json"}
    if response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的响应格式: {response_format}"
        )
    
    try:
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="上传的文件为空")
        
        logger.info(
            f"收到翻译请求 - 文件: {file.filename}, "
            f"大小: {len(audio_bytes)} bytes, 格式: {response_format}"
        )
        
        # 翻译任务: 源语言设为英语 (会自动检测), 目标语言设为英语
        engine = get_transcription_engine()
        result = engine.transcribe_bytes(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            language="en",  # Canary 会自动检测源语言
            response_format=response_format,
            timestamps=True,
            target_language="en",  # 翻译到英语
        )
        
        if response_format in {"text", "srt", "vtt"}:
            return PlainTextResponse(content=result, media_type="text/plain")
        else:
            return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"翻译失败: {e}")
        raise HTTPException(status_code=500, detail=f"翻译失败: {e}")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,  # 单 worker, 避免多进程加载模型
    )
