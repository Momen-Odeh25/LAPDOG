# model_io_colab.py - Modified model I/O for Colab environment
"""
Modified model loading and saving functions optimized for Google Colab.
This replaces the original model_io.py with Colab-specific optimizations.
"""

import errno
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
import gc

import torch
import transformers

from src.lapdog import Lapdog
from src.retrievers import Contriever, DualEncoderRetriever, UntiedDualEncoderRetriever
from src.gemma_model import load_gemma_model, GemmaForRetrievalAugmentedGeneration
from src.colab_config import ColabConfig
from src.util import cast_to_precision, set_dropout, set_optim
from src import dist_utils

Number = Union[float, int]
logger = logging.getLogger(__name__)


def get_checkpoint_path(opt):
    """Get checkpoint path optimized for Colab (Google Drive)."""
    checkpoint_path = Path(ColabConfig.CHECKPOINT_DIR) / opt.name
    return checkpoint_path


def create_checkpoint_directories(opt):
    """Create checkpoint directories in Google Drive."""
    checkpoint_path = get_checkpoint_path(opt)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Create data directory
    os.makedirs(ColabConfig.DATA_DIR, exist_ok=True)
    
    # Create cache directory
    os.makedirs(ColabConfig.CACHE_DIR, exist_ok=True)
    
    if hasattr(opt, 'save_index_path') and opt.save_index_path:
        os.makedirs(opt.save_index_path, exist_ok=True)
    
    return checkpoint_path, getattr(opt, 'save_index_path', None)


def load_lightweight_retriever(opt):
    """
    Load a lightweight retriever optimized for Colab.
    Uses CPU-based retrieval to save GPU memory.
    """
    if getattr(opt, 'use_file_passages', False):
        return None, None
    
    try:
        # Use lighter sentence transformer model
        from sentence_transformers import SentenceTransformer
        
        retriever_model = SentenceTransformer(
            ColabConfig.RETRIEVER_MODEL,
            device='cpu' if ColabConfig.USE_CPU_RETRIEVER else 'cuda'
        )
        
        # Create simple tokenizer
        retriever_tokenizer = transformers.AutoTokenizer.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        logger.info(f"Loaded lightweight retriever: {ColabConfig.RETRIEVER_MODEL}")
        return retriever_model, retriever_tokenizer
        
    except Exception as e:
        logger.error(f"Error loading retriever: {e}")
        # Fallback to no retriever
        return None, None


def load_gemma_reader(opt):
    """
    Load Gemma model optimized for Colab environment.
    """
    try:
        # Try different Gemma models in order of preference
        models_to_try = [ColabConfig.READER_MODEL] + ColabConfig.READER_MODEL_ALTERNATIVES
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load {model_name}...")
                
                model, tokenizer = load_gemma_model(
                    model_name=model_name,
                    use_quantization=True
                )
                
                logger.info(f"Successfully loaded {model_name}")
                return model, tokenizer
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        raise Exception("All model loading attempts failed")
        
    except Exception as e:
        logger.error(f"Error loading Gemma reader: {e}")
        raise


def optimize_model_for_colab(model):
    """
    Apply Colab-specific optimizations to the model.
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Enable mixed precision if supported
    if hasattr(model, 'half') and ColabConfig.USE_MIXED_PRECISION == "fp16":
        model = model.half()
        logger.info("Converted model to half precision")
    
    # Clear unused memory
    gc.collect()
    torch.cuda.empty_cache()
    
    return model


def load_reader(opt):
    """Load reader model (Gemma 3) optimized for Colab."""
    # Check if using causal LM (Gemma)
    if hasattr(opt, 'reader_causallm') and opt.reader_causallm:
        return load_gemma_reader(opt)
    
    # Default to Gemma if no specific reader specified
    return load_gemma_reader(opt)


def _set_reader_config(model, opt):
    """Set reader configuration for Colab environment."""
    if hasattr(model, 'config'):
        config = model.config
        config.n_context = getattr(opt, 'n_context', ColabConfig.N_CONTEXT)
        config.bsz = getattr(opt, 'per_gpu_batch_size', ColabConfig.BATCH_SIZE_TRAIN)


def _cast_and_optimize_for_colab(model, opt):
    """Cast model and apply Colab optimizations."""
    model = optimize_model_for_colab(model)
    
    if hasattr(opt, 'device'):
        model = model.to(opt.device)
    
    return model


def init_lapdog_model_colab(opt, eval_only=False):
    """
    Initialize LAPDOG model optimized for Colab environment.
    """
    logger.info("Initializing LAPDOG model for Colab...")
    
    # Load reader (Gemma)
    reader, reader_tokenizer = load_reader(opt)
    
    # Load retriever (lightweight)
    retriever, retriever_tokenizer = load_lightweight_retriever(opt)
    
    # Create LAPDOG model
    model = Lapdog(opt, reader, retriever, reader_tokenizer, retriever_tokenizer)
    
    # Apply Colab optimizations
    model = _cast_and_optimize_for_colab(model, opt)
    _set_reader_config(model, opt)
    
    if eval_only:
        return model, None, None, None, None, opt, 0
    
    # Set up optimizers with Colab-specific settings
    optimizer, scheduler, retr_optimizer, retr_scheduler = set_optim(opt, model)
    
    return model, optimizer, scheduler, retr_optimizer, retr_scheduler, opt, 0


def save_model_checkpoint_colab(model, optimizer, scheduler, step, opt, checkpoint_path):
    """
    Save model checkpoint optimized for Colab (saves to Google Drive).
    """
    save_path = os.path.join(checkpoint_path, f"checkpoint_step_{step}.pth")
    
    # Prepare checkpoint data
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'opt': opt,
    }
    
    # Add optimizer states if available
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")
    
    # Also save as latest checkpoint
    latest_path = os.path.join(checkpoint_path, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)
    
    # Clean up old checkpoints to save space
    cleanup_old_checkpoints(checkpoint_path, keep_last=3)


def cleanup_old_checkpoints(checkpoint_path, keep_last=3):
    """Remove old checkpoints to save Google Drive space."""
    try:
        checkpoint_files = []
        for file in os.listdir(checkpoint_path):
            if file.startswith("checkpoint_step_") and file.endswith(".pth"):
                step_num = int(file.split("_")[-1].split(".")[0])
                checkpoint_files.append((step_num, file))
        
        # Sort by step number and keep only the last N
        checkpoint_files.sort(key=lambda x: x[0])
        
        if len(checkpoint_files) > keep_last:
            for step_num, filename in checkpoint_files[:-keep_last]:
                file_path = os.path.join(checkpoint_path, filename)
                os.remove(file_path)
                logger.info(f"Removed old checkpoint: {filename}")
                
    except Exception as e:
        logger.warning(f"Error cleaning up checkpoints: {e}")


def load_checkpoint_colab(checkpoint_path, opt):
    """Load checkpoint optimized for Colab environment."""
    latest_checkpoint = os.path.join(checkpoint_path, "latest_checkpoint.pth")
    
    if os.path.exists(latest_checkpoint):
        checkpoint_file = latest_checkpoint
    else:
        # Find latest numbered checkpoint
        checkpoint_files = []
        for file in os.listdir(checkpoint_path):
            if file.startswith("checkpoint_step_") and file.endswith(".pth"):
                step_num = int(file.split("_")[-1].split(".")[0])
                checkpoint_files.append((step_num, file))
        
        if not checkpoint_files:
            return None
        
        checkpoint_files.sort(key=lambda x: x[0])
        checkpoint_file = os.path.join(checkpoint_path, checkpoint_files[-1][1])
    
    logger.info(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    return checkpoint


def monitor_memory_usage():
    """Monitor and log memory usage for Colab."""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        
        logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
        
        # Warning if memory usage is high
        if memory_allocated > ColabConfig.MAX_MEMORY_GB * 0.8:
            logger.warning("High GPU memory usage detected. Consider reducing batch size.")
    
    # CPU memory monitoring
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"CPU Memory Usage: {memory_percent:.1f}%")
    except ImportError:
        pass