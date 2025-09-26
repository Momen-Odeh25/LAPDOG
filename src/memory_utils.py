# memory_utils.py - Memory optimization utilities for Colab
"""
Memory management utilities specifically designed for Google Colab environment.
Includes gradient checkpointing, memory monitoring, and optimization strategies.
"""

import gc
import torch
import logging
from typing import Optional, Dict, Any
import psutil

logger = logging.getLogger(__name__)


class ColabMemoryManager:
    """Memory management utility for Google Colab environment."""
    
    def __init__(self, max_memory_gb: float = 12.0):
        self.max_memory_gb = max_memory_gb
        self.memory_threshold = max_memory_gb * 0.8  # 80% threshold
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3
            stats['gpu_free'] = (torch.cuda.get_device_properties(0).total_memory - 
                                torch.cuda.memory_reserved()) / 1024**3
        
        # CPU memory
        memory = psutil.virtual_memory()
        stats['cpu_used'] = memory.used / 1024**3
        stats['cpu_available'] = memory.available / 1024**3
        stats['cpu_percent'] = memory.percent
        
        return stats
    
    def log_memory_stats(self):
        """Log current memory statistics."""
        stats = self.get_memory_stats()
        
        if 'gpu_allocated' in stats:
            logger.info(f"GPU Memory - Allocated: {stats['gpu_allocated']:.2f}GB, "
                       f"Reserved: {stats['gpu_reserved']:.2f}GB, "
                       f"Free: {stats['gpu_free']:.2f}GB")
        
        logger.info(f"CPU Memory - Used: {stats['cpu_used']:.2f}GB, "
                   f"Available: {stats['cpu_available']:.2f}GB, "
                   f"Usage: {stats['cpu_percent']:.1f}%")
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Memory cleanup completed")
    
    def check_memory_threshold(self) -> bool:
        """Check if memory usage exceeds threshold."""
        stats = self.get_memory_stats()
        
        if 'gpu_allocated' in stats and stats['gpu_allocated'] > self.memory_threshold:
            logger.warning(f"GPU memory usage ({stats['gpu_allocated']:.2f}GB) "
                          f"exceeds threshold ({self.memory_threshold:.2f}GB)")
            return True
        
        return False
    
    def auto_cleanup_if_needed(self):
        """Automatically cleanup memory if threshold is exceeded."""
        if self.check_memory_threshold():
            logger.info("Auto-cleaning memory due to high usage")
            self.cleanup_memory()


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing for memory efficiency."""
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    elif hasattr(model, 'reader') and hasattr(model.reader, 'gradient_checkpointing_enable'):
        model.reader.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for reader")
    else:
        logger.warning("Gradient checkpointing not available for this model")


def optimize_batch_size(initial_batch_size: int, memory_manager: ColabMemoryManager) -> int:
    """Dynamically optimize batch size based on available memory."""
    stats = memory_manager.get_memory_stats()
    
    if 'gpu_allocated' in stats:
        gpu_usage_ratio = stats['gpu_allocated'] / memory_manager.max_memory_gb
        
        if gpu_usage_ratio > 0.8:
            # Reduce batch size if memory usage is high
            new_batch_size = max(1, initial_batch_size // 2)
            logger.info(f"Reducing batch size from {initial_batch_size} to {new_batch_size} "
                       f"due to high memory usage")
            return new_batch_size
        elif gpu_usage_ratio < 0.5 and initial_batch_size < 4:
            # Increase batch size if memory usage is low
            new_batch_size = initial_batch_size * 2
            logger.info(f"Increasing batch size from {initial_batch_size} to {new_batch_size} "
                       f"due to low memory usage")
            return new_batch_size
    
    return initial_batch_size


def setup_mixed_precision_training():
    """Setup mixed precision training for memory efficiency."""
    try:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        logger.info("Mixed precision training enabled")
        return scaler, autocast
    except ImportError:
        logger.warning("Mixed precision training not available")
        return None, None


class MemoryEfficientDataLoader:
    """Memory-efficient data loader for Colab."""
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.memory_manager = ColabMemoryManager()
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            # Check memory before loading batch
            self.memory_manager.auto_cleanup_if_needed()
            
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            
            yield batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def apply_model_optimizations(model, config):
    """Apply various model optimizations for Colab."""
    optimizations_applied = []
    
    # Enable gradient checkpointing
    enable_gradient_checkpointing(model)
    optimizations_applied.append("gradient_checkpointing")
    
    # Apply mixed precision if specified
    if hasattr(config, 'USE_MIXED_PRECISION') and config.USE_MIXED_PRECISION == "fp16":
        if hasattr(model, 'half'):
            model = model.half()
            optimizations_applied.append("fp16")
    
    # Enable memory efficient attention if available
    if hasattr(model, 'config') and hasattr(model.config, 'use_memory_efficient_attention'):
        model.config.use_memory_efficient_attention = True
        optimizations_applied.append("memory_efficient_attention")
    
    logger.info(f"Applied optimizations: {', '.join(optimizations_applied)}")
    return model


def monitor_training_memory(step: int, memory_manager: ColabMemoryManager, 
                          log_frequency: int = 10):
    """Monitor memory usage during training."""
    if step % log_frequency == 0:
        memory_manager.log_memory_stats()
        memory_manager.auto_cleanup_if_needed()


def estimate_model_memory(model) -> Dict[str, float]:
    """Estimate memory usage of a model."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'parameters_mb': param_size / 1024**2,
        'buffers_mb': buffer_size / 1024**2,
        'total_mb': total_size / 1024**2,
        'total_gb': total_size / 1024**3
    }


class AdaptiveBatchSizer:
    """Adaptive batch size management for Colab."""
    
    def __init__(self, initial_batch_size: int = 1, max_batch_size: int = 8):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = 1
        self.memory_manager = ColabMemoryManager()
        self.oom_count = 0
    
    def adjust_batch_size(self, oom_occurred: bool = False):
        """Adjust batch size based on memory conditions."""
        if oom_occurred:
            self.oom_count += 1
            # Reduce batch size more aggressively after OOM
            self.current_batch_size = max(self.min_batch_size, 
                                        self.current_batch_size // 2)
            logger.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")
        else:
            # Try to increase batch size if memory allows
            stats = self.memory_manager.get_memory_stats()
            if ('gpu_allocated' in stats and 
                stats['gpu_allocated'] < self.memory_manager.memory_threshold * 0.6 and
                self.current_batch_size < self.max_batch_size):
                
                self.current_batch_size = min(self.max_batch_size, 
                                            self.current_batch_size + 1)
                logger.info(f"Increasing batch size to {self.current_batch_size}")
        
        return self.current_batch_size