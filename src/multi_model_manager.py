# -*- coding: utf-8 -*-
"""
多模型管理器模块

支持同时管理多个 ASR 模型，每个模型独立进行懒加载和自动卸载
"""

import os
import time
import threading
from typing import Dict, Optional, Any, List
from pathlib import Path
from contextlib import contextmanager

import torch
from loguru import logger

from .model_manager import ModelManager


class MultiModelManager:
    """
    多模型管理器
    
    管理多个 ASR 模型的生命周期，支持：
    - 按需加载指定模型
    - 每个模型独立的超时管理
    - 线程安全的多模型访问
    
    支持的模型：
    - nvidia/canary-1b-v2
    - nvidia/parakeet-tdt-0.6b-v3
    """
    
    # 模型配置映射
    MODEL_CONFIGS = {
        "canary-1b-v2": {
            "hf_name": "nvidia/canary-1b-v2",
            "nemo_filename": "canary-1b-v2.nemo",
            "description": "Canary 1B v2 多语言 ASR 模型"
        },
        "parakeet-tdt-0.6b-v3": {
            "hf_name": "nvidia/parakeet-tdt-0.6b-v3",
            "nemo_filename": "parakeet-tdt-0.6b-v3.nemo",
            "description": "Parakeet TDT 0.6B v3 ASR 模型"
        }
    }
    
    def __init__(
        self,
        models_base_path: Optional[str] = None,
        enabled_models: Optional[list] = None,
        timeout_sec: int = 300,
        use_fp16: bool = True
    ):
        """
        初始化多模型管理器
        
        Args:
            models_base_path: 模型基础存储路径
            enabled_models: 启用的模型列表，如 ["canary-1b-v2", "parakeet-tdt-0.6b-v3"]
            timeout_sec: 模型闲置超时时间
            use_fp16: 是否使用 FP16 半精度
        """
        # 基础路径配置
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            "models"
        )
        self.models_base_path = models_base_path or os.getenv("MODEL_PATH", default_path)
        self.models_base_path = os.path.abspath(self.models_base_path)
        
        # 读取启用的模型列表
        if enabled_models is None:
            # 从环境变量读取，格式：ENABLED_MODELS=canary-1b-v2,parakeet-tdt-0.6b-v3
            env_models = os.getenv("ENABLED_MODELS", "canary-1b-v2")
            enabled_models = [m.strip() for m in env_models.split(",")]
        
        self.enabled_models = enabled_models
        self.timeout_sec = timeout_sec or int(os.getenv("MODEL_TIMEOUT_SEC", "300"))
        self.use_fp16 = use_fp16 if use_fp16 is not None else os.getenv("USE_FP16", "true").lower() == "true"
        
        # 模型管理器字典：{model_name: ModelManager}
        self._managers: Dict[str, ModelManager] = {}
        
        # 全局锁
        self._lock = threading.RLock()
        
        # 初始化启用的模型管理器
        self._initialize_managers()
        
        logger.info(
            f"多模型管理器初始化完成 - "
            f"基础路径: {self.models_base_path}, "
            f"启用模型: {', '.join(self.enabled_models)}"
        )
    
    def _initialize_managers(self):
        """初始化各个模型的管理器"""
        for model_name in self.enabled_models:
            if model_name not in self.MODEL_CONFIGS:
                logger.warning(f"未知的模型名称: {model_name}, 跳过")
                continue
            
            config = self.MODEL_CONFIGS[model_name]
            
            manager = ModelManager(
                model_path=self.models_base_path,
                model_name=config["hf_name"],
                nemo_filename=config["nemo_filename"],
                timeout_sec=self.timeout_sec,
                use_fp16=self.use_fp16
            )
            
            self._managers[model_name] = manager
            logger.info(f"已初始化模型管理器: {model_name} -> {config['description']}")
    
    def get_model_manager(self, model_name: str) -> Optional[ModelManager]:
        """
        获取指定模型的管理器
        
        Args:
            model_name: 模型名称，如 "canary-1b-v2" 或 "parakeet-tdt-0.6b-v3"
        
        Returns:
            模型管理器实例，如果模型未启用则返回 None
        """
        # 标准化模型名称
        model_name = self._normalize_model_name(model_name)
        
        if model_name not in self._managers:
            logger.warning(
                f"模型 {model_name} 未启用。"
                f"可用模型: {', '.join(self._managers.keys())}"
            )
            return None
        
        return self._managers[model_name]
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        标准化模型名称
        
        支持多种输入格式：
        - "canary-1b-v2" -> "canary-1b-v2"
        - "nvidia/canary-1b-v2" -> "canary-1b-v2"
        - "canary-1b" -> "canary-1b-v2"
        """
        # 移除 nvidia/ 前缀
        if model_name.startswith("nvidia/"):
            model_name = model_name[7:]
        
        # 标准化常见别名
        alias_map = {
            "canary": "canary-1b-v2",
            "canary-1b": "canary-1b-v2",
            "parakeet": "parakeet-tdt-0.6b-v3",
            "parakeet-tdt": "parakeet-tdt-0.6b-v3",
        }
        
        return alias_map.get(model_name, model_name)
    
    @contextmanager
    def get_model(self, model_name: str):
        """
        获取指定模型的上下文管理器
        
        Args:
            model_name: 模型名称
        
        Yields:
            模型实例
        
        Example:
            with multi_manager.get_model("canary-1b-v2") as model:
                result = model.transcribe(...)
        """
        manager = self.get_model_manager(model_name)
        if manager is None:
            raise ValueError(
                f"模型 {model_name} 不可用。"
                f"可用模型: {', '.join(self._managers.keys())}"
            )
        
        with manager.get_model() as model:
            yield model
    
    def load_model(self, model_name: str) -> bool:
        """
        预加载指定模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            加载成功返回 True
        """
        manager = self.get_model_manager(model_name)
        if manager is None:
            return False
        
        return manager.load_model()
    
    def unload_model(self, model_name: str) -> bool:
        """
        卸载指定模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            卸载成功返回 True
        """
        manager = self.get_model_manager(model_name)
        if manager is None:
            return False
        
        return manager.unload_model()
    
    def get_enabled_models(self) -> List[str]:
        """
        获取当前启用的模型列表
        
        Returns:
            启用的模型名称列表
        """
        return list(self._managers.keys())
    
    def get_status(self, model_name: Optional[str] = None) -> dict:
        """
        获取模型状态
        
        Args:
            model_name: 模型名称，如果为 None 则返回所有模型状态
        
        Returns:
            状态信息字典
        """
        if model_name:
            manager = self.get_model_manager(model_name)
            if manager is None:
                return {"error": f"模型 {model_name} 不可用"}
            return {
                "model_name": model_name,
                **manager.get_status()
            }
        
        # 返回所有模型状态
        all_status = {
            "enabled_models": list(self._managers.keys()),
            "models": {}
        }
        
        for name, manager in self._managers.items():
            all_status["models"][name] = manager.get_status()
        
        return all_status
    
    def shutdown(self):
        """关闭所有模型管理器"""
        logger.info("正在关闭多模型管理器...")
        
        for name, manager in self._managers.items():
            logger.info(f"关闭模型: {name}")
            manager.shutdown()
        
        self._managers.clear()
        logger.info("多模型管理器已关闭")


# 全局实例
_multi_manager: Optional[MultiModelManager] = None
_manager_lock = threading.Lock()


def get_multi_model_manager() -> MultiModelManager:
    """
    获取多模型管理器单例
    
    Returns:
        MultiModelManager 实例
    """
    global _multi_manager
    
    if _multi_manager is None:
        with _manager_lock:
            if _multi_manager is None:
                _multi_manager = MultiModelManager()
    
    return _multi_manager


def shutdown_multi_model_manager():
    """关闭多模型管理器"""
    global _multi_manager
    
    if _multi_manager is not None:
        _multi_manager.shutdown()
        _multi_manager = None
