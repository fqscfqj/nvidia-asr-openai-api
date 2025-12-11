# -*- coding: utf-8 -*-
"""
推理引擎模块

实现核心功能:
- 音频转录 (ASR)
- 语音翻译 (AST)
- 时间戳提取
- 多种输出格式支持
"""

import os
import tempfile
from typing import Optional, Dict, Any, List, Union
from pathlib import Path

from loguru import logger

from .model_manager import get_model_manager
from .utils import (
    segments_to_srt,
    segments_to_vtt,
    build_json_response,
    build_verbose_json_response,
    normalize_language_code,
    save_audio_to_temp,
    cleanup_temp_file,
    get_audio_duration,
    convert_audio_to_wav,
)


class TranscriptionEngine:
    """
    转录引擎类
    
    封装 Canary 模型的推理逻辑, 提供:
    - 音频文件转录
    - 多种输出格式 (text, json, srt, vtt, verbose_json)
    - 时间戳提取
    
    使用示例:
        engine = TranscriptionEngine()
        result = engine.transcribe(
            audio_path="audio.wav",
            language="en",
            response_format="json"
        )
    """
    
    # 支持的响应格式
    SUPPORTED_FORMATS = {"text", "json", "srt", "vtt", "verbose_json"}
    
    def __init__(self):
        """初始化转录引擎"""
        self.model_manager = get_model_manager()
        logger.info("转录引擎初始化完成")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        response_format: str = "json",
        timestamps: bool = True,
        target_language: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        执行音频转录
        
        Args:
            audio_path: 音频文件路径
            language: 源语言代码 (如 'en', 'zh'), 默认自动检测
            response_format: 响应格式, 支持 text/json/srt/vtt/verbose_json
            timestamps: 是否提取时间戳
            target_language: 目标语言 (用于翻译任务), 默认与源语言相同
            
        Returns:
            根据 response_format 返回不同格式的结果:
            - text: 纯文本字符串
            - json: {"text": "..."} 字典
            - srt: SRT 格式字幕字符串
            - vtt: VTT 格式字幕字符串
            - verbose_json: 包含详细信息的字典
            
        Raises:
            ValueError: 不支持的响应格式
            RuntimeError: 转录失败
        """
        # 验证响应格式
        if response_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的响应格式: {response_format}, "
                f"支持的格式: {self.SUPPORTED_FORMATS}"
            )
        
        # 标准化语言代码
        source_lang = normalize_language_code(language)
        target_lang = normalize_language_code(target_language) if target_language else source_lang
        
        # 确定是否需要时间戳 (SRT/VTT 格式必须启用)
        need_timestamps = timestamps or response_format in {"srt", "vtt", "verbose_json"}
        
        logger.info(
            f"开始转录 - 文件: {audio_path}, "
            f"源语言: {source_lang}, 目标语言: {target_lang}, "
            f"格式: {response_format}, 时间戳: {need_timestamps}"
        )
        
        try:
            # 获取音频时长
            duration = get_audio_duration(audio_path)
            
            # 使用模型进行推理
            with self.model_manager.get_model() as model:
                # 调用 Canary 模型的 transcribe 方法
                output = model.transcribe(
                    [audio_path],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    timestamps=need_timestamps,
                )
                
                # 提取转录结果
                if output and len(output) > 0:
                    result = output[0]
                    text = result.text if hasattr(result, 'text') else str(result)
                    
                    # 提取时间戳信息
                    segments = []
                    words = []
                    
                    if hasattr(result, 'timestamp') and result.timestamp:
                        if 'segment' in result.timestamp:
                            segments = result.timestamp['segment']
                        if 'word' in result.timestamp:
                            words = result.timestamp['word']
                else:
                    text = ""
                    segments = []
                    words = []
            
            logger.info(f"转录完成 - 文本长度: {len(text)}, 分段数: {len(segments)}")
            
            # 根据格式生成响应
            return self._format_response(
                text=text,
                segments=segments,
                words=words,
                language=source_lang,
                duration=duration,
                response_format=response_format
            )
            
        except Exception as e:
            logger.error(f"转录失败: {e}")
            raise RuntimeError(f"转录失败: {e}") from e
    
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = None,
        response_format: str = "json",
        timestamps: bool = True,
        target_language: Optional[str] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        从字节数据执行音频转录
        
        用于处理 API 上传的文件数据
        
        Args:
            audio_bytes: 音频文件的字节内容
            filename: 原始文件名 (用于确定格式)
            language: 源语言代码
            response_format: 响应格式
            timestamps: 是否提取时间戳
            target_language: 目标语言
            
        Returns:
            转录结果 (格式取决于 response_format)
        """
        temp_path = None
        converted_path = None
        
        try:
            # 从文件名获取后缀
            suffix = Path(filename).suffix.lower() or ".wav"
            
            # 保存到临时文件
            temp_path = save_audio_to_temp(audio_bytes, suffix=suffix)
            
            # 如果不是 WAV 格式, 需要转换
            if suffix not in {".wav", ".flac"}:
                logger.info(f"将 {suffix} 格式转换为 WAV...")
                converted_path = convert_audio_to_wav(temp_path)
                audio_path = converted_path
            else:
                audio_path = temp_path
            
            # 执行转录
            result = self.transcribe(
                audio_path=audio_path,
                language=language,
                response_format=response_format,
                timestamps=timestamps,
                target_language=target_language,
            )
            
            return result
            
        finally:
            # 清理临时文件
            if temp_path:
                cleanup_temp_file(temp_path)
            if converted_path and converted_path != temp_path:
                cleanup_temp_file(converted_path)
    
    def _format_response(
        self,
        text: str,
        segments: List[Dict[str, Any]],
        words: List[Dict[str, Any]],
        language: str,
        duration: Optional[float],
        response_format: str,
    ) -> Union[str, Dict[str, Any]]:
        """
        根据指定格式生成响应
        
        Args:
            text: 完整转录文本
            segments: 分段时间戳列表
            words: 词级时间戳列表
            language: 语言代码
            duration: 音频时长
            response_format: 响应格式
            
        Returns:
            格式化的响应
        """
        if response_format == "text":
            return text
        
        elif response_format == "json":
            return build_json_response(
                text=text,
                segments=segments,
                language=language,
                duration=duration
            )
        
        elif response_format == "srt":
            if not segments:
                # 没有时间戳信息, 创建单个分段
                segments = [{"start": 0, "end": duration or 0, "segment": text}]
            return segments_to_srt(segments)
        
        elif response_format == "vtt":
            if not segments:
                segments = [{"start": 0, "end": duration or 0, "segment": text}]
            return segments_to_vtt(segments)
        
        elif response_format == "verbose_json":
            return build_verbose_json_response(
                text=text,
                segments=segments,
                language=language,
                duration=duration,
                words=words
            )
        
        else:
            # 默认返回 JSON 格式
            return build_json_response(text=text)


class TranslationEngine:
    """
    翻译引擎类
    
    封装 Canary 模型的语音翻译功能
    支持从任意源语言翻译到英语, 或从英语翻译到其他语言
    """
    
    def __init__(self):
        """初始化翻译引擎"""
        self.model_manager = get_model_manager()
        logger.info("翻译引擎初始化完成")
    
    def translate(
        self,
        audio_path: str,
        source_language: str,
        target_language: str,
        response_format: str = "json",
    ) -> Union[str, Dict[str, Any]]:
        """
        执行语音翻译
        
        Args:
            audio_path: 音频文件路径
            source_language: 源语言代码
            target_language: 目标语言代码
            response_format: 响应格式
            
        Returns:
            翻译结果
        """
        # 使用转录引擎, 设置不同的源和目标语言
        engine = TranscriptionEngine()
        return engine.transcribe(
            audio_path=audio_path,
            language=source_language,
            target_language=target_language,
            response_format=response_format,
            timestamps=True,
        )


# 全局引擎实例
_transcription_engine: Optional[TranscriptionEngine] = None


def get_transcription_engine() -> TranscriptionEngine:
    """
    获取转录引擎单例实例
    
    Returns:
        TranscriptionEngine 实例
    """
    global _transcription_engine
    
    if _transcription_engine is None:
        _transcription_engine = TranscriptionEngine()
    
    return _transcription_engine
