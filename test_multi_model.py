#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模型测试脚本

测试 Canary-1B-v2 模型的转录功能
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8909"

def test_canary_model():
    """测试 Canary-1B-v2 模型"""
    print("=" * 60)
    print("测试 Canary-1B-v2 模型")
    print("=" * 60)
    
    audio_file = Path("test/OSR_us_000_0059_8k.wav")
    
    if not audio_file.exists():
        print(f"错误: 音频文件不存在: {audio_file}")
        return
    
    url = f"{BASE_URL}/v1/audio/transcriptions"
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        data = {
            "model": "canary-1b-v2",
            "language": "en",
            "response_format": "json"
        }
        
        print(f"\n请求 URL: {url}")
        print(f"模型: canary-1b-v2")
        print(f"音频文件: {audio_file}")
        print(f"文件大小: {audio_file.stat().st_size} bytes")
        print("\n发送请求...")
        
        response = requests.post(url, files=files, data=data)
        
        print(f"\n响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n转录结果:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"\n错误: {response.text}")

def test_invalid_model():
    """测试无效的模型名称"""
    print("\n" + "=" * 60)
    print("测试无效的模型名称 (应该返回错误)")
    print("=" * 60)
    
    audio_file = Path("test/OSR_us_000_0059_8k.wav")
    url = f"{BASE_URL}/v1/audio/transcriptions"
    
    with open(audio_file, "rb") as f:
        files = {"file": f}
        data = {
            "model": "invalid-model-name",
            "language": "en",
            "response_format": "json"
        }
        
        print(f"\n请求 URL: {url}")
        print(f"模型: invalid-model-name")
        print("\n发送请求...")
        
        response = requests.post(url, files=files, data=data)
        
        print(f"\n响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")

def check_model_status():
    """检查模型状态"""
    print("\n" + "=" * 60)
    print("检查模型状态")
    print("=" * 60)
    
    url = f"{BASE_URL}/status"
    response = requests.get(url)
    
    print(f"\n响应状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n模型状态:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n错误: {response.text}")

if __name__ == "__main__":
    # 测试 Canary 模型
    test_canary_model()
    
    # 测试无效模型
    test_invalid_model()
    
    # 检查模型状态
    check_model_status()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
