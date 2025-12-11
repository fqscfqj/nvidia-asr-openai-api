# -*- coding: utf-8 -*-
"""
模型管理器模块

实现核心功能:
- 懒加载 (Lazy Loading): 首次请求时才加载模型
- 自动卸载 (Auto-Unload): 超时未使用时释放显存
- 线程安全: 确保并发请求安全
- 智能下载: 优先使用本地模型, 否则从 HuggingFace 下载
"""

import os
import time
import threading
from typing import Optional, Any
from pathlib import Path
from contextlib import contextmanager

import torch
from loguru import logger


class ModelManager:
    """
    模型管理器类
    
    负责模型的生命周期管理, 包括:
    - 懒加载模型到 GPU
    - 自动在超时后卸载模型释放显存
    - 线程安全的模型访问
    
    属性:
        model: ASR 模型实例
        model_path: 模型存储路径
        model_name: HuggingFace 模型名称
        timeout_sec: 模型闲置超时时间(秒)
        use_fp16: 是否使用 FP16 半精度
        
    使用示例:
        manager = ModelManager()
        with manager.get_model() as model:
            output = model.transcribe(['audio.wav'], source_lang='en', target_lang='en')
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "nvidia/canary-1b-v2",
        timeout_sec: int = 300,
        use_fp16: bool = True
    ):
        """
        初始化模型管理器
        
        Args:
            model_path: 模型本地存储路径, 默认从环境变量 MODEL_PATH 读取
            model_name: HuggingFace 模型名称
            timeout_sec: 模型闲置超时时间(秒), 默认从环境变量读取
            use_fp16: 是否使用 FP16 半精度推理
        """
        # 从环境变量读取配置, 参数优先
        # 默认使用项目根目录下的 models 文件夹
        default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.model_path = model_path or os.getenv("MODEL_PATH", default_path)
        # 确保是绝对路径
        self.model_path = os.path.abspath(self.model_path)
        self.model_name = model_name or os.getenv("MODEL_NAME", "nvidia/canary-1b-v2")
        self.timeout_sec = timeout_sec or int(os.getenv("MODEL_TIMEOUT_SEC", "300"))
        self.use_fp16 = use_fp16 if use_fp16 is not None else os.getenv("USE_FP16", "true").lower() == "true"
        
        # 模型实例
        self._model: Optional[Any] = None
        
        # 线程锁: 保护模型加载和卸载操作
        self._lock = threading.RLock()
        
        # 使用计数器: 当前有多少请求正在使用模型
        self._usage_count = 0
        
        # 最后使用时间戳
        self._last_used_time: float = 0
        
        # 超时监控线程
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop_event = threading.Event()
        
        # 模型加载状态
        self._is_loading = False
        
        logger.info(f"模型管理器初始化完成 - 模型路径: {self.model_path}, 超时时间: {self.timeout_sec}秒")
    
    def _check_local_model_exists(self) -> bool:
        """
        检查本地是否已存在模型文件
        
        Returns:
            如果本地存在模型文件返回 True
        """
        model_dir = Path(self.model_path)
        
        # 检查目录是否存在且不为空
        if model_dir.exists() and model_dir.is_dir():
            # 检查是否有 .nemo 文件或模型配置文件
            nemo_files = list(model_dir.glob("*.nemo"))
            config_files = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.json"))
            
            if nemo_files or config_files:
                logger.info(f"检测到本地模型文件: {model_dir}")
                return True
        
        return False
    
    def _get_model_source(self) -> str:
        """
        确定模型加载源
        
        Returns:
            本地路径或 HuggingFace 模型名称
        """
        if self._check_local_model_exists():
            # 查找本地 .nemo 文件
            nemo_files = list(Path(self.model_path).glob("*.nemo"))
            if nemo_files:
                return str(nemo_files[0])
            return self.model_path
        
        # 使用 HuggingFace 模型, 下载到指定路径
        logger.info(f"本地未找到模型, 将从 HuggingFace 下载: {self.model_name}")
        return self.model_name
    
    def _load_model(self) -> Any:
        """
        加载 ASR 模型到 GPU
        
        实现智能加载逻辑:
        1. 优先使用本地模型文件
        2. 本地不存在则从 HuggingFace 下载
        3. 使用 FP16 半精度优化显存占用
        
        Returns:
            加载好的 ASR 模型实例
        """
        logger.info("开始加载 Canary ASR 模型...")
        start_time = time.time()
        
        try:
            # 延迟导入 NeMo, 避免启动时占用资源
            from nemo.collections.asr.models import ASRModel
            
            # 确定模型源
            model_source = self._get_model_source()
            
            # 设置模型缓存目录
            os.environ["NEMO_CACHE_DIR"] = self.model_path
            
            # 加载模型
            if model_source.endswith(".nemo"):
                # 从本地 .nemo 文件加载
                logger.info(f"从本地文件加载模型: {model_source}")
                model = ASRModel.restore_from(model_source)
            else:
                # 从 HuggingFace 下载并加载
                logger.info(f"从 HuggingFace 下载模型: {model_source}")
                model = ASRModel.from_pretrained(model_name=model_source)
                
                # 保存模型到本地供后续使用
                save_path = os.path.join(self.model_path, "canary-1b-v2.nemo")
                os.makedirs(self.model_path, exist_ok=True)
                try:
                    model.save_to(save_path)
                    logger.info(f"模型已保存到本地: {save_path}")
                except Exception as e:
                    logger.warning(f"保存模型到本地失败: {e}")
            
            # 移动模型到 GPU (如果可用)
            if torch.cuda.is_available():
                model = model.cuda()
                logger.info(f"模型已移动到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("CUDA 不可用, 使用 CPU 推理 (速度较慢)")
            
            # 设置为评估模式
            model.eval()
            
            # 使用 FP16 半精度 (减少显存占用)
            if self.use_fp16 and torch.cuda.is_available():
                model = model.half()
                logger.info("已启用 FP16 半精度推理")
            
            elapsed = time.time() - start_time
            logger.info(f"模型加载完成, 耗时: {elapsed:.2f}秒")
            
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _unload_model(self) -> None:
        """
        卸载模型并释放显存
        
        注意: 只有在没有请求使用模型时才能卸载
        """
        with self._lock:
            if self._model is None:
                return
            
            if self._usage_count > 0:
                logger.warning(f"有 {self._usage_count} 个请求正在使用模型, 跳过卸载")
                return
            
            logger.info("开始卸载模型并释放显存...")
            
            try:
                # 删除模型引用
                del self._model
                self._model = None
                
                # 清理 GPU 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                logger.info("模型已卸载, 显存已释放")
                
            except Exception as e:
                logger.error(f"卸载模型时出错: {e}")
    
    def _monitor_timeout(self) -> None:
        """
        后台监控线程: 检查模型是否超时未使用
        
        运行逻辑:
        1. 每隔一定时间检查一次
        2. 如果超过 timeout_sec 未使用且无正在处理的请求, 则卸载模型
        """
        check_interval = min(30, self.timeout_sec // 2)  # 检查间隔
        
        logger.info(f"超时监控线程已启动, 检查间隔: {check_interval}秒")
        
        while not self._monitor_stop_event.is_set():
            # 等待一段时间
            self._monitor_stop_event.wait(check_interval)
            
            if self._monitor_stop_event.is_set():
                break
            
            # 检查是否需要卸载
            with self._lock:
                if self._model is None:
                    continue
                
                if self._usage_count > 0:
                    continue
                
                elapsed = time.time() - self._last_used_time
                if elapsed >= self.timeout_sec:
                    logger.info(f"模型闲置超过 {self.timeout_sec}秒, 准备卸载...")
                    self._unload_model()
        
        logger.info("超时监控线程已停止")
    
    def _start_monitor(self) -> None:
        """启动超时监控线程"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_stop_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor_timeout,
                daemon=True,
                name="ModelTimeoutMonitor"
            )
            self._monitor_thread.start()
    
    def _stop_monitor(self) -> None:
        """停止超时监控线程"""
        if self._monitor_thread is not None:
            self._monitor_stop_event.set()
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
    
    def ensure_model_loaded(self) -> Any:
        """
        确保模型已加载 (内部方法)
        
        Returns:
            加载好的模型实例
        """
        with self._lock:
            if self._model is None:
                if self._is_loading:
                    # 其他线程正在加载, 等待
                    logger.debug("等待其他线程完成模型加载...")
                else:
                    self._is_loading = True
                    try:
                        self._model = self._load_model()
                        self._start_monitor()
                    finally:
                        self._is_loading = False
            
            return self._model
    
    @contextmanager
    def get_model(self):
        """
        获取模型的上下文管理器
        
        使用上下文管理器确保:
        1. 模型在使用时不会被卸载
        2. 正确更新使用计数和时间戳
        
        使用示例:
            with manager.get_model() as model:
                output = model.transcribe(...)
        
        Yields:
            ASR 模型实例
        """
        model = None
        try:
            with self._lock:
                # 确保模型已加载
                model = self.ensure_model_loaded()
                
                # 增加使用计数
                self._usage_count += 1
                self._last_used_time = time.time()
            
            yield model
            
        finally:
            with self._lock:
                # 减少使用计数
                self._usage_count = max(0, self._usage_count - 1)
                self._last_used_time = time.time()
    
    def get_status(self) -> dict:
        """
        获取模型管理器状态
        
        Returns:
            包含模型状态信息的字典
        """
        with self._lock:
            is_loaded = self._model is not None
            idle_time = time.time() - self._last_used_time if self._last_used_time > 0 else 0
            
            status = {
                "model_loaded": is_loaded,
                "model_name": self.model_name,
                "model_path": self.model_path,
                "usage_count": self._usage_count,
                "idle_seconds": round(idle_time, 2) if is_loaded else None,
                "timeout_seconds": self.timeout_sec,
                "use_fp16": self.use_fp16,
                "gpu_available": torch.cuda.is_available(),
            }
            
            if torch.cuda.is_available():
                status["gpu_name"] = torch.cuda.get_device_name(0)
                status["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated(0) / 1024 / 1024, 2
                )
                status["gpu_memory_reserved_mb"] = round(
                    torch.cuda.memory_reserved(0) / 1024 / 1024, 2
                )
            
            return status
    
    def force_unload(self) -> bool:
        """
        强制卸载模型 (用于手动释放资源)
        
        Returns:
            是否成功卸载
        """
        with self._lock:
            if self._usage_count > 0:
                logger.warning("有请求正在处理, 无法强制卸载")
                return False
            
            self._unload_model()
            return True
    
    def force_load(self) -> bool:
        """
        强制预加载模型
        
        Returns:
            是否成功加载
        """
        try:
            self.ensure_model_loaded()
            return True
        except Exception as e:
            logger.error(f"强制加载模型失败: {e}")
            return False
    
    def shutdown(self) -> None:
        """
        关闭模型管理器, 释放所有资源
        """
        logger.info("正在关闭模型管理器...")
        
        # 停止监控线程
        self._stop_monitor()
        
        # 卸载模型
        with self._lock:
            if self._model is not None:
                self._unload_model()
        
        logger.info("模型管理器已关闭")


# 全局单例实例
_model_manager: Optional[ModelManager] = None
_instance_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """
    获取模型管理器单例实例
    
    Returns:
        ModelManager 实例
    """
    global _model_manager
    
    if _model_manager is None:
        with _instance_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    
    return _model_manager


def shutdown_model_manager() -> None:
    """
    关闭模型管理器单例
    """
    global _model_manager
    
    with _instance_lock:
        if _model_manager is not None:
            _model_manager.shutdown()
            _model_manager = None
