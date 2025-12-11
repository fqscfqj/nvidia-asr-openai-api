#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型路径结构
"""

import os
from pathlib import Path

models_dir = Path("models")

print("=" * 60)
print("Models 目录结构检查")
print("=" * 60)

if models_dir.exists():
    print(f"\n模型目录: {models_dir.absolute()}")
    print(f"目录内容:")
    
    # 列出所有文件和子目录
    for item in sorted(models_dir.iterdir()):
        if item.is_file():
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  📄 {item.name} ({size_mb:.2f} MB)")
        elif item.is_dir():
            print(f"  📁 {item.name}/")
            # 列出子目录内容
            for sub_item in sorted(item.iterdir()):
                if sub_item.is_file():
                    size_mb = sub_item.stat().st_size / (1024 * 1024)
                    print(f"      📄 {sub_item.name} ({size_mb:.2f} MB)")
    
    # 检查是否有不必要的子目录
    subdirs = [d for d in models_dir.iterdir() if d.is_dir()]
    if subdirs:
        print(f"\n⚠️  警告: 发现 {len(subdirs)} 个子目录，这些可能是旧的模型目录:")
        for d in subdirs:
            print(f"  - {d.name}/")
        print("\n建议: 可以删除这些子目录，模型应该直接存放在 models/ 根目录中")
    else:
        print(f"\n✅ 正确: 没有子目录，所有模型文件都直接存放在 models/ 根目录中")
    
    # 检查 .nemo 文件
    nemo_files = list(models_dir.glob("*.nemo"))
    if nemo_files:
        print(f"\n✅ 找到 {len(nemo_files)} 个 .nemo 模型文件:")
        for f in nemo_files:
            print(f"  - {f.name}")
    else:
        print(f"\n⚠️  未找到 .nemo 文件")
        
else:
    print(f"\n❌ 模型目录不存在: {models_dir.absolute()}")

print("\n" + "=" * 60)
