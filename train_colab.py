# train_colab.py - Modified training script for Google Colab
"""
Training script optimized for Google Colab environment.
Includes memory management, checkpoint saving to Google Drive, and progress monitoring.
"""

import os
import time
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.cuda
import logging
from tqdm.auto import tqdm
import wandb

# Colab-specific imports
from src.colab_config import ColabConfig
from src.model_io_colab import (
    init_lapdog_model_colab, 
    save_model_checkpoint_colab,
    monitor_memory_usage,
    create_checkpoint_directories
)
from src.memory_utils import (
    ColabMemoryManager, 
    AdaptiveBatchSizer,
    monitor_training_memory,
    setup_mixed_precision_training
)
from src.evaluation import evaluate
from src import dist_utils, util
from src.tasks import get_task
from src.options import get_options

# Set environment variables for Colab
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


class ColabTrainer:
    """Training manager optimized for Google Colab."""
    
    def __init__(self, opt):
        self.opt = opt
        self.memory_manager = ColabMemoryManager(ColabConfig.MAX_MEMORY_GB)
        self.batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=ColabConfig.BATCH_SIZE_TRAIN,
            max_batch_size=8
        )
        
        # Setup mixed precision
        self.scaler, self.autocast = setup_mixed_precision_training()
        
        # Initialize wandb for experiment tracking
        if hasattr(opt, 'use_wandb') and opt.use_wandb:
            wandb.init(
                project="lapdog-colab",
                name=opt.name,
                config=vars(opt)
            )
    
    def setup_model_and_data(self):
        """Setup model and data loaders."""
        logger.info("Setting up model and data for Colab training...")
        
        # Create checkpoint directories
        self.checkpoint_path, _ = create_checkpoint_directories(self.opt)
        
        # Initialize model
        (self.model, self.optimizer, self.scheduler, 
         self.retr_optimizer, self.retr_scheduler, 
         self.opt_checkpoint, self.step) = init_lapdog_model_colab(self.opt)
        
        # Setup task and data
        task = get_task(self.opt, self.model.reader_tokenizer)
        self.train_examples = task.load_data(self.opt, split='train')
        self.eval_examples = task.load_data(self.opt, split='valid')
        
        logger.info(f"Loaded {len(self.train_examples)} training examples")
        logger.info(f"Loaded {len(self.eval_examples)} validation examples")
        
        # Log initial memory usage
        self.memory_manager.log_memory_stats()
    
    def train_step(self, batch):
        """Single training step with memory optimization."""
        try:
            self.model.train()
            
            # Adjust batch size if needed
            current_batch_size = self.batch_sizer.current_batch_size
            
            # Use automatic mixed precision if available
            if self.autocast is not None:
                with self.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            return loss.item()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM detected during training step")
                self.memory_manager.cleanup_memory()
                self.batch_sizer.adjust_batch_size(oom_occurred=True)
                return None
            else:
                raise e
    
    def evaluate_model(self):
        """Evaluate model with memory optimization."""
        logger.info("Starting evaluation...")
        
        self.model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.eval_examples[:100], desc="Evaluating"):  # Limit eval size
                try:
                    if self.autocast is not None:
                        with self.autocast():
                            outputs = self.model(**batch)
                            loss = outputs.loss
                    else:
                        outputs = self.model(**batch)
                        loss = outputs.loss
                    
                    eval_losses.append(loss.item())
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("OOM during evaluation, skipping batch")
                        self.memory_manager.cleanup_memory()
                        continue
                    else:
                        raise e
        
        avg_eval_loss = np.mean(eval_losses) if eval_losses else float('inf')
        logger.info(f"Evaluation loss: {avg_eval_loss:.4f}")
        
        return avg_eval_loss
    
    def save_checkpoint(self):
        """Save checkpoint to Google Drive."""
        save_model_checkpoint_colab(
            self.model, 
            self.optimizer, 
            self.scheduler,
            self.step,
            self.opt,
            self.checkpoint_path
        )
    
    def train(self):
        """Main training loop optimized for Colab."""
        logger.info("Starting Colab-optimized training...")
        
        # Setup
        self.setup_model_and_data()
        
        # Training statistics
        run_stats = util.WeightedAvgStats()
        best_eval_loss = float('inf')
        
        # Progress bar
        pbar = tqdm(range(self.step, ColabConfig.MAX_STEPS), 
                   desc="Training", 
                   initial=self.step, 
                   total=ColabConfig.MAX_STEPS)
        
        for step in pbar:
            self.step = step
            
            # Memory monitoring
            monitor_training_memory(step, self.memory_manager)
            
            # Training step
            # Note: In a real implementation, you'd iterate over actual batches
            # This is a simplified version for demonstration
            batch = self.get_next_batch()  # You'd implement this method
            
            loss = self.train_step(batch)
            
            if loss is not None:
                run_stats.update("train_loss", loss)
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'batch_size': self.batch_sizer.current_batch_size,
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                
                # Log to wandb
                if hasattr(self.opt, 'use_wandb') and self.opt.use_wandb:
                    wandb.log({
                        'train_loss': loss,
                        'step': step,
                        'batch_size': self.batch_sizer.current_batch_size,
                        'gpu_memory_gb': torch.cuda.memory_allocated()/1024**3 if torch.cuda.is_available() else 0
                    })
            
            # Evaluation
            if step % ColabConfig.EVAL_FREQ == 0 and step > 0:
                eval_loss = self.evaluate_model()
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    logger.info(f"New best evaluation loss: {best_eval_loss:.4f}")
                
                if hasattr(self.opt, 'use_wandb') and self.opt.use_wandb:
                    wandb.log({'eval_loss': eval_loss, 'step': step})
            
            # Save checkpoint
            if step % ColabConfig.SAVE_FREQ == 0 and step > 0:
                self.save_checkpoint()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Final save
        self.save_checkpoint()
        logger.info("Training completed!")
        
        if hasattr(self.opt, 'use_wandb') and self.opt.use_wandb:
            wandb.finish()
    
    def get_next_batch(self):
        """Get next training batch (placeholder - implement based on your data loader)."""
        # This is a placeholder method
        # In reality, you'd implement proper data loading here
        pass


def setup_colab_environment():
    """Setup environment variables and paths for Colab."""
    # Mount Google Drive (this would be done in the notebook)
    print("Please ensure Google Drive is mounted at /content/drive/")
    
    # Set cache directories
    os.environ['TRANSFORMERS_CACHE'] = ColabConfig.CACHE_DIR
    os.environ['HF_HOME'] = ColabConfig.CACHE_DIR
    
    # Create directories
    os.makedirs(ColabConfig.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ColabConfig.DATA_DIR, exist_ok=True)
    os.makedirs(ColabConfig.CACHE_DIR, exist_ok=True)


def get_colab_optimized_options():
    """Get options optimized for Colab environment."""
    # This would be integrated with the existing options system
    opt = type('Options', (), {})()
    
    # Basic settings
    opt.name = "lapdog_colab_experiment"
    opt.checkpoint_dir = ColabConfig.CHECKPOINT_DIR
    opt.per_gpu_batch_size = ColabConfig.BATCH_SIZE_TRAIN
    opt.per_gpu_eval_batch_size = ColabConfig.BATCH_SIZE_EVAL
    opt.total_steps = ColabConfig.MAX_STEPS
    opt.eval_freq = ColabConfig.EVAL_FREQ
    opt.save_freq = ColabConfig.SAVE_FREQ
    opt.precision = ColabConfig.USE_MIXED_PRECISION
    
    # Model settings
    opt.reader_causallm = ColabConfig.READER_MODEL
    opt.n_context = ColabConfig.N_CONTEXT
    opt.text_maxlength = ColabConfig.MAX_CONTEXT_LENGTH
    opt.target_maxlength = ColabConfig.MAX_TARGET_LENGTH
    
    # Memory optimization settings
    opt.gradient_accumulation_steps = ColabConfig.GRADIENT_ACCUMULATION_STEPS
    opt.use_gradient_checkpointing = ColabConfig.USE_GRADIENT_CHECKPOINTING
    
    # Device settings
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.world_size = 1
    opt.global_rank = 0
    opt.is_main = True
    
    # Additional settings
    opt.use_wandb = True
    opt.seed = 42
    
    return opt


def main():
    """Main training function for Colab."""
    # Setup environment
    setup_colab_environment()
    
    # Get options
    opt = get_colab_optimized_options()
    
    # Initialize trainer
    trainer = ColabTrainer(opt)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()