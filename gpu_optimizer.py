"""
RTX 4090 GPU优化配置模块
针对Hunyuan3D-Part项目的专用GPU加速优化
"""

import torch
import torch.nn as nn
import os
import time
import psutil
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple
import subprocess
import gc

class RTX4090Optimizer:
    """RTX 4090 专用优化配置类"""

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.total_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0

        # RTX 4090 专用优化配置
        self.config = {
            # Flash Attention配置
            "enable_flash_attention": True,
            "flash_attention_dropout": 0.0,

            # 混合精度配置
            "enable_mixed_precision": True,
            "autocast_dtype": torch.float16,
            "grad_scaler": True,

            # 编译优化配置
            "enable_compile": True,
            "compile_mode": "reduce-overhead",  # max-autotune, reduce-overhead, default
            "compile_dynamic": True,

            # 内存管理配置
            "memory_fraction": 0.9,  # 使用90%显存
            "empty_cache_threshold": 0.8,  # 80%时清理缓存
            "enable_memory_pool": True,

            # CUDA优化配置
            "enable_cudnn_benchmark": True,
            "enable_deterministic": False,  # 性能优先
            "cuda_launch_blocking": False,

            # 并行配置
            "num_workers": min(8, os.cpu_count()),
            "pin_memory": True,
            "non_blocking": True,

            # 3D模型处理专用配置
            "max_points_per_batch": 100000,  # 点云批处理大小
            "mesh_chunk_size": 50000,        # 网格分块大小
            "enable_mesh_optimization": True,
        }

        self.logger = self._setup_logger()
        self.performance_stats = {}
        self._initialize_gpu()

    def _setup_logger(self) -> logging.Logger:
        """设置性能监控日志"""
        logger = logging.getLogger("RTX4090Optimizer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_gpu(self):
        """初始化GPU设置"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，将使用CPU模式")
            return

        try:
            # 设置内存分配策略
            if self.config["memory_fraction"] < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.config["memory_fraction"]
                )

            # 启用cuDNN基准模式
            if self.config["enable_cudnn_benchmark"]:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True

            # 设置确定性行为
            if self.config["enable_deterministic"]:
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)

            # 设置CUDA启动阻塞
            os.environ["CUDA_LAUNCH_BLOCKING"] = str(
                int(self.config["cuda_launch_blocking"])
            )

            # 优化CUDA内存分配
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

            self.logger.info(f"GPU初始化完成: {self.gpu_name}")
            self.logger.info(f"总显存: {self.total_memory / 1024**3:.1f}GB")
            self.logger.info(f"可用显存分配: {self.config['memory_fraction']*100:.0f}%")

        except Exception as e:
            self.logger.error(f"GPU初始化失败: {e}")

    @contextmanager
    def optimized_inference(self):
        """优化推理上下文管理器"""
        if not torch.cuda.is_available():
            yield
            return

        # 记录开始状态
        start_memory = torch.cuda.memory_allocated()
        start_time = time.time()

        try:
            # 清理显存
            self.clear_memory_if_needed()

            # 设置推理模式
            with torch.inference_mode():
                if self.config["enable_mixed_precision"]:
                    with torch.autocast(
                        device_type='cuda',
                        dtype=self.config["autocast_dtype"],
                        enabled=True
                    ):
                        yield
                else:
                    yield

        finally:
            # 记录性能统计
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated()

            self.performance_stats.update({
                "inference_time": end_time - start_time,
                "memory_used": (end_memory - start_memory) / 1024**2,  # MB
                "peak_memory": torch.cuda.max_memory_allocated() / 1024**2,  # MB
            })

            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()

            # 根据需要清理内存
            self.clear_memory_if_needed()

    def clear_memory_if_needed(self):
        """智能内存清理"""
        if not torch.cuda.is_available():
            return

        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        total_memory = torch.cuda.get_device_properties(0).total_memory

        # 计算内存使用率
        usage_ratio = memory_reserved / total_memory

        if usage_ratio > self.config["empty_cache_threshold"]:
            self.logger.info(f"显存使用率 {usage_ratio:.1%}，执行内存清理")

            # 强制垃圾回收
            gc.collect()

            # 清理CUDA缓存
            torch.cuda.empty_cache()

            # 同步CUDA操作
            torch.cuda.synchronize()

            new_usage = torch.cuda.memory_reserved() / total_memory
            self.logger.info(f"内存清理完成，使用率: {new_usage:.1%}")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """优化模型配置"""
        if model is None:
            return model

        try:
            # 设置评估模式
            model.eval()

            # 移动到GPU
            if torch.cuda.is_available():
                model = model.to(self.device)

            # 编译优化
            if self.config["enable_compile"] and hasattr(torch, 'compile'):
                self.logger.info("启用torch.compile优化...")
                model = torch.compile(
                    model,
                    mode=self.config["compile_mode"],
                    dynamic=self.config["compile_dynamic"]
                )

            self.logger.info("模型优化完成")
            return model

        except Exception as e:
            self.logger.error(f"模型优化失败: {e}")
            return model

    def get_optimal_batch_size(self, model_size_mb: int) -> int:
        """计算最优批处理大小"""
        if not torch.cuda.is_available():
            return 1

        available_memory = self.total_memory * self.config["memory_fraction"]
        model_memory = model_size_mb * 1024 * 1024

        # 预留内存用于激活和梯度
        usable_memory = available_memory - model_memory * 3

        # 估算单样本内存需求（基于点云大小）
        points_per_sample = self.config["max_points_per_batch"]
        memory_per_sample = points_per_sample * 4 * 6  # float32 * (xyz + rgb)

        batch_size = max(1, int(usable_memory / memory_per_sample))

        self.logger.info(f"计算得出最优批处理大小: {batch_size}")
        return batch_size

    def benchmark_performance(self, model: nn.Module, input_data: torch.Tensor,
                            num_runs: int = 10) -> Dict[str, float]:
        """性能基准测试"""
        if not torch.cuda.is_available():
            return {"error": "CUDA不可用"}

        model.eval()
        results = []

        # 预热
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_data)

        torch.cuda.synchronize()

        # 基准测试
        with torch.no_grad():
            for i in range(num_runs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                with self.optimized_inference():
                    output = model(input_data)
                end.record()

                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)  # ms
                results.append(elapsed_time)

        # 统计结果
        avg_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)

        benchmark_results = {
            "average_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "throughput_fps": 1000.0 / avg_time,
            "memory_usage_mb": torch.cuda.max_memory_allocated() / 1024**2,
        }

        self.logger.info(f"性能基准测试完成: {benchmark_results}")
        return benchmark_results

    def get_gpu_status(self) -> Dict[str, Any]:
        """获取GPU状态信息"""
        if not torch.cuda.is_available():
            return {"error": "CUDA不可用"}

        # GPU基本信息
        props = torch.cuda.get_device_properties(0)

        # 内存信息
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        memory_total = props.total_memory

        # NVIDIA-SMI信息
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            smi_data = result.stdout.strip().split(',')
            gpu_util = float(smi_data[0])
            memory_used_mb = float(smi_data[1])
            temperature = float(smi_data[2])
            power_draw = float(smi_data[3])
        except:
            gpu_util = memory_used_mb = temperature = power_draw = 0

        return {
            "gpu_name": props.name,
            "driver_version": torch.version.cuda,
            "memory_allocated_mb": memory_allocated / 1024**2,
            "memory_reserved_mb": memory_reserved / 1024**2,
            "memory_total_mb": memory_total / 1024**2,
            "memory_usage_percent": (memory_reserved / memory_total) * 100,
            "gpu_utilization_percent": gpu_util,
            "temperature_celsius": temperature,
            "power_draw_watts": power_draw,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
        }

    def log_performance_summary(self):
        """记录性能摘要"""
        if self.performance_stats:
            self.logger.info("=== 性能统计摘要 ===")
            for key, value in self.performance_stats.items():
                if isinstance(value, float):
                    self.logger.info(f"{key}: {value:.2f}")
                else:
                    self.logger.info(f"{key}: {value}")

            gpu_status = self.get_gpu_status()
            self.logger.info(f"GPU利用率: {gpu_status.get('gpu_utilization_percent', 0):.1f}%")
            self.logger.info(f"显存使用: {gpu_status.get('memory_usage_percent', 0):.1f}%")
            self.logger.info(f"GPU温度: {gpu_status.get('temperature_celsius', 0):.1f}°C")


class FlashAttentionOptimizer:
    """Flash Attention优化模块"""

    def __init__(self):
        self.available = self._check_flash_attention()

    def _check_flash_attention(self) -> bool:
        """检查Flash Attention可用性"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False

    def optimize_attention(self, query: torch.Tensor, key: torch.Tensor,
                          value: torch.Tensor, dropout_p: float = 0.0) -> torch.Tensor:
        """优化的注意力计算"""
        if self.available:
            try:
                from flash_attn import flash_attn_func
                return flash_attn_func(query, key, value, dropout_p=dropout_p)
            except Exception as e:
                # 回退到标准实现
                return self._standard_attention(query, key, value, dropout_p)
        else:
            return self._standard_attention(query, key, value, dropout_p)

    def _standard_attention(self, query: torch.Tensor, key: torch.Tensor,
                           value: torch.Tensor, dropout_p: float = 0.0) -> torch.Tensor:
        """标准注意力实现"""
        scale = 1.0 / (query.size(-1) ** 0.5)
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        if dropout_p > 0.0:
            attn_weights = torch.dropout(attn_weights, dropout_p, train=False)

        return torch.matmul(attn_weights, value)


# 全局优化器实例
_global_optimizer = None

def get_optimizer() -> RTX4090Optimizer:
    """获取全局优化器实例"""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = RTX4090Optimizer()
    return _global_optimizer

def optimize_for_rtx4090(model: nn.Module) -> nn.Module:
    """RTX 4090优化快捷函数"""
    optimizer = get_optimizer()
    return optimizer.optimize_model(model)

@contextmanager
def gpu_inference_mode():
    """GPU推理模式上下文管理器"""
    optimizer = get_optimizer()
    with optimizer.optimized_inference():
        yield