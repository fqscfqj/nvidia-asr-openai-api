# -*- coding: utf-8 -*-
"""
工具函数模块

提供音频处理和字幕格式转换的辅助功能:
- SRT 字幕格式生成
- VTT 字幕格式生成
- 时间戳格式化
- 音频文件预处理
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

from loguru import logger


def format_timestamp_srt(seconds: float) -> str:
    """
    将秒数转换为 SRT 格式时间戳
    
    SRT 格式: HH:MM:SS,mmm (注意逗号分隔毫秒)
    
    Args:
        seconds: 时间秒数
        
    Returns:
        SRT 格式的时间字符串, 如 "00:01:23,456"
    """
    if seconds < 0:
        seconds = 0
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """
    将秒数转换为 VTT 格式时间戳
    
    VTT 格式: HH:MM:SS.mmm (注意点号分隔毫秒)
    
    Args:
        seconds: 时间秒数
        
    Returns:
        VTT 格式的时间字符串, 如 "00:01:23.456"
    """
    if seconds < 0:
        seconds = 0
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """
    将分段时间戳数据转换为 SRT 字幕格式
    
    SRT 格式说明:
    - 每个字幕块包含: 序号、时间轴、字幕文本
    - 字幕块之间用空行分隔
    
    Args:
        segments: 分段列表, 每个分段包含 'start', 'end', 'segment'/'text' 字段
        
    Returns:
        SRT 格式的字幕字符串
        
    示例输入:
        [{"start": 0.0, "end": 2.5, "segment": "你好世界"}]
        
    示例输出:
        1
        00:00:00,000 --> 00:00:02,500
        你好世界
    """
    if not segments:
        return ""
    
    srt_lines = []
    
    for idx, segment in enumerate(segments, start=1):
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        # Canary 模型返回的是 'segment' 字段, 也兼容 'text' 字段
        text = segment.get('segment', segment.get('text', ''))
        
        # 格式化时间戳
        start_str = format_timestamp_srt(start_time)
        end_str = format_timestamp_srt(end_time)
        
        # 构建 SRT 字幕块
        srt_block = f"{idx}\n{start_str} --> {end_str}\n{text}"
        srt_lines.append(srt_block)
    
    # 用双换行符连接各字幕块
    return "\n\n".join(srt_lines)


def segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    """
    将分段时间戳数据转换为 WebVTT 字幕格式
    
    VTT 格式说明:
    - 文件必须以 "WEBVTT" 开头
    - 时间戳使用点号分隔毫秒
    - 可选的序号行
    
    Args:
        segments: 分段列表, 每个分段包含 'start', 'end', 'segment'/'text' 字段
        
    Returns:
        VTT 格式的字幕字符串
        
    示例输出:
        WEBVTT
        
        00:00:00.000 --> 00:00:02.500
        你好世界
    """
    if not segments:
        return "WEBVTT\n"
    
    vtt_lines = ["WEBVTT", ""]  # VTT 头部和空行
    
    for segment in segments:
        start_time = segment.get('start', 0)
        end_time = segment.get('end', 0)
        text = segment.get('segment', segment.get('text', ''))
        
        # 格式化时间戳
        start_str = format_timestamp_vtt(start_time)
        end_str = format_timestamp_vtt(end_time)
        
        # 构建 VTT 字幕块 (不包含序号)
        vtt_block = f"{start_str} --> {end_str}\n{text}"
        vtt_lines.append(vtt_block)
        vtt_lines.append("")  # 空行分隔
    
    return "\n".join(vtt_lines)


def build_json_response(
    text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = None,
    duration: Optional[float] = None
) -> Dict[str, Any]:
    """
    构建 OpenAI Whisper 兼容的 JSON 响应格式
    
    Args:
        text: 完整的转录文本
        segments: 可选的分段时间戳列表
        language: 检测到的语言代码
        duration: 音频时长(秒)
        
    Returns:
        符合 OpenAI Whisper API 格式的字典
    """
    response = {"text": text}
    
    if language:
        response["language"] = language
    
    if duration is not None:
        response["duration"] = duration
    
    return response


def build_verbose_json_response(
    text: str,
    segments: List[Dict[str, Any]],
    language: Optional[str] = None,
    duration: Optional[float] = None,
    words: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    构建 OpenAI Whisper 兼容的详细 JSON 响应格式 (verbose_json)
    
    包含完整的分段信息和可选的词级时间戳
    
    Args:
        text: 完整的转录文本
        segments: 分段时间戳列表
        language: 检测到的语言代码
        duration: 音频时长(秒)
        words: 可选的词级时间戳列表
        
    Returns:
        符合 OpenAI Whisper API verbose_json 格式的字典
    """
    # 转换 Canary 格式的 segments 为 OpenAI 格式
    formatted_segments = []
    for idx, seg in enumerate(segments):
        formatted_seg = {
            "id": idx,
            "start": seg.get('start', 0),
            "end": seg.get('end', 0),
            "text": seg.get('segment', seg.get('text', '')),
        }
        formatted_segments.append(formatted_seg)
    
    response = {
        "task": "transcribe",
        "text": text,
        "segments": formatted_segments,
    }
    
    if language:
        response["language"] = language
    
    if duration is not None:
        response["duration"] = duration
    
    if words:
        # 转换词级时间戳格式
        formatted_words = []
        for w in words:
            formatted_words.append({
                "word": w.get('word', w.get('segment', '')),
                "start": w.get('start', 0),
                "end": w.get('end', 0),
            })
        response["words"] = formatted_words
    
    return response


def save_audio_to_temp(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    将音频字节数据保存到临时文件
    
    Args:
        audio_bytes: 音频文件的字节内容
        suffix: 文件后缀名, 默认 ".wav"
        
    Returns:
        临时文件的完整路径
    """
    # 创建临时文件, 不自动删除
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, audio_bytes)
    finally:
        os.close(fd)
    
    logger.debug(f"音频已保存到临时文件: {temp_path}")
    return temp_path


def cleanup_temp_file(file_path: str) -> None:
    """
    清理临时文件
    
    Args:
        file_path: 要删除的文件路径
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"已清理临时文件: {file_path}")
    except Exception as e:
        logger.warning(f"清理临时文件失败: {file_path}, 错误: {e}")


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    获取音频文件时长
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        音频时长(秒), 失败返回 None
    """
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        logger.warning(f"获取音频时长失败: {e}")
        return None


def convert_audio_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """
    将音频文件转换为 WAV 格式 (16kHz, 单声道)
    
    Canary 模型要求输入为 16kHz 单声道音频
    
    Args:
        input_path: 输入音频文件路径
        output_path: 输出文件路径, 为空则自动生成临时文件
        
    Returns:
        转换后的 WAV 文件路径
    """
    try:
        from pydub import AudioSegment
        
        # 加载音频
        audio = AudioSegment.from_file(input_path)
        
        # 转换为单声道, 16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # 确定输出路径
        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
        
        # 导出为 WAV
        audio.export(output_path, format="wav")
        logger.debug(f"音频已转换为 WAV: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"音频转换失败: {e}")
        raise


# 语言代码映射: OpenAI Whisper 语言代码 -> Canary 语言代码
LANGUAGE_CODE_MAP = {
    # Canary 支持的 25 种欧洲语言
    "en": "en",      # English
    "english": "en",
    "de": "de",      # German
    "german": "de",
    "fr": "fr",      # French
    "french": "fr",
    "es": "es",      # Spanish
    "spanish": "es",
    "it": "it",      # Italian
    "italian": "it",
    "pt": "pt",      # Portuguese
    "portuguese": "pt",
    "nl": "nl",      # Dutch
    "dutch": "nl",
    "pl": "pl",      # Polish
    "polish": "pl",
    "ru": "ru",      # Russian
    "russian": "ru",
    "uk": "uk",      # Ukrainian
    "ukrainian": "uk",
    "cs": "cs",      # Czech
    "czech": "cs",
    "sk": "sk",      # Slovak
    "slovak": "sk",
    "bg": "bg",      # Bulgarian
    "bulgarian": "bg",
    "hr": "hr",      # Croatian
    "croatian": "hr",
    "da": "da",      # Danish
    "danish": "da",
    "fi": "fi",      # Finnish
    "finnish": "fi",
    "el": "el",      # Greek
    "greek": "el",
    "hu": "hu",      # Hungarian
    "hungarian": "hu",
    "ro": "ro",      # Romanian
    "romanian": "ro",
    "sv": "sv",      # Swedish
    "swedish": "sv",
    "et": "et",      # Estonian
    "estonian": "et",
    "lv": "lv",      # Latvian
    "latvian": "lv",
    "lt": "lt",      # Lithuanian
    "lithuanian": "lt",
    "sl": "sl",      # Slovenian
    "slovenian": "sl",
    "mt": "mt",      # Maltese
    "maltese": "mt",
}


def normalize_language_code(language: Optional[str]) -> str:
    """
    标准化语言代码为 Canary 支持的格式
    
    Args:
        language: 输入的语言代码或语言名称
        
    Returns:
        Canary 支持的语言代码, 默认返回 "en"
    """
    if not language:
        return "en"
    
    lang_lower = language.lower().strip()
    
    # 查找映射
    if lang_lower in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[lang_lower]
    
    # 尝试使用前两个字符作为语言代码
    if len(lang_lower) >= 2:
        short_code = lang_lower[:2]
        if short_code in LANGUAGE_CODE_MAP:
            return LANGUAGE_CODE_MAP[short_code]
    
    # 默认返回英语
    logger.warning(f"未知语言代码 '{language}', 使用默认值 'en'")
    return "en"
