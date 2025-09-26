# Colab Configuration for LAPDOG
# This file contains optimized configuration settings for Google Colab

class ColabConfig:
    """Configuration class optimized for Google Colab environment"""
    
    # Memory and compute settings
    MAX_MEMORY_GB = 12  # Conservative estimate for Colab
    BATCH_SIZE_TRAIN = 1  # Start with minimal batch size
    BATCH_SIZE_EVAL = 2
    GRADIENT_ACCUMULATION_STEPS = 8  # Simulate larger batches
    
    # Model settings for Gemma 3
    READER_MODEL = "google/gemma-2b"  # Start with 2B model
    READER_MODEL_ALTERNATIVES = [
        "google/gemma-2b-it",  # Instruction-tuned version
        "microsoft/DialoGPT-small",  # Fallback option
        "microsoft/DialoGPT-medium"
    ]
    
    # Retriever settings (lighter alternatives)
    RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight
    USE_CPU_RETRIEVER = True  # Use CPU for retrieval to save GPU memory
    
    # Training settings
    MAX_STEPS = 1000  # Reduced for Colab
    EVAL_FREQ = 100
    SAVE_FREQ = 200
    WARMUP_STEPS = 50
    LEARNING_RATE = 5e-5
    
    # Memory optimization flags
    USE_GRADIENT_CHECKPOINTING = True
    USE_8BIT_OPTIMIZER = True
    USE_MIXED_PRECISION = "fp16"
    
    # Data settings
    MAX_CONTEXT_LENGTH = 128  # Reduced from original
    MAX_TARGET_LENGTH = 64
    N_CONTEXT = 5  # Reduced number of retrieved passages
    
    # Paths (will be adjusted for Colab)
    CHECKPOINT_DIR = "/content/drive/MyDrive/lapdog_checkpoints"
    DATA_DIR = "/content/lapdog_data"
    CACHE_DIR = "/content/drive/MyDrive/huggingface_cache"