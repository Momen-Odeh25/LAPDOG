# Fixed training loop for Colab notebook
import torch
import numpy as np
from tqdm.auto import tqdm
import time

def create_working_training_loop(model, tokenizer, config):
    """Create a training loop that actually works with real model outputs."""
    
    # Sample training data
    training_examples = [
        "Persona: I love hiking. Human: What do you do for fun? Assistant: I absolutely love spending time outdoors hiking!",
        "Persona: I'm a chef. Human: What's your specialty? Assistant: I specialize in traditional Italian pasta dishes.",
        "Persona: I study CS. Human: What language do you recommend? Assistant: Python is perfect for beginners!",
        "Persona: I work in a library. Human: What's your favorite part? Assistant: I love helping people discover new books.",
        "Persona: I run marathons. Human: How do you stay motivated? Assistant: Setting small daily goals keeps me going!"
    ]
    
    # Determine device from model
    device = next(model.parameters()).device
    print(f"🔧 Using device: {device}")
    
    def get_training_batch():
        """Get a real training batch."""
        # Randomly sample an example
        example = np.random.choice(training_examples)
        
        # Tokenize
        tokens = tokenizer(
            example,
            truncation=True,
            max_length=config.MAX_CONTEXT_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = tokens['input_ids'].clone()
        
        # Move to same device as model
        tokens = {k: v.to(device) for k, v in tokens.items()}
        labels = labels.to(device)
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'labels': labels
        }
    
    def train_step(batch, use_amp=False, scaler=None):
        """Perform one training step with real model forward pass."""
        model.train()
        
        # Forward pass - this creates a real loss that works with GradScaler
        if use_amp and scaler:
            with torch.amp.autocast(device.type):
                outputs = model(**batch)
                loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.loss
        
        return loss
    
    def evaluate_step(use_amp=False, scaler=None):
        """Simple evaluation step."""
        model.eval()
        eval_losses = []
        
        with torch.no_grad():
            for _ in range(5):  # Evaluate on 5 batches
                batch = get_training_batch()
                if use_amp and scaler:
                    with torch.amp.autocast(device.type):
                        outputs = model(**batch)
                        loss = outputs.loss
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                eval_losses.append(loss.item())
        
        return np.mean(eval_losses)
    
    return get_training_batch, train_step, evaluate_step

def run_fixed_training(model, tokenizer, config, max_steps=100):
    """Run the fixed training loop."""
    
    print("🚀 Starting fixed training loop...")
    
    # Setup training components
    get_batch, train_step, evaluate_step = create_working_training_loop(model, tokenizer, config)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    try:
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.WARMUP_STEPS,
            num_training_steps=max_steps
        )
    except ImportError:
        # Fallback to simple scheduler
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=config.WARMUP_STEPS)
    
    # Setup mixed precision correctly
    device = next(model.parameters()).device
    scaler = None
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler(device.type)
    
    # Training statistics
    training_losses = []
    eval_losses = []
    best_eval_loss = float('inf')
    
    try:
        pbar = tqdm(range(max_steps), desc="Training")
        
        for step in pbar:
            try:
                # Get real batch
                batch = get_batch()
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Training step with proper mixed precision
                if use_amp and scaler:
                    with torch.amp.autocast(device.type):
                        loss = train_step(batch, use_amp, scaler)
                    
                    # Proper gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = train_step(batch, use_amp, scaler)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # Track loss
                loss_value = loss.item()
                training_losses.append(loss_value)
                
                # Update progress bar
                avg_loss = np.mean(training_losses[-10:]) if len(training_losses) >= 10 else np.mean(training_losses)
                pbar.set_postfix({
                    'loss': f"{loss_value:.4f}",
                    'avg_loss': f"{avg_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'gpu_mem': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                
                # Periodic evaluation
                if step > 0 and step % 25 == 0:
                    print(f"\n📊 Evaluating at step {step}...")
                    eval_loss = evaluate_step(use_amp, scaler)
                    eval_losses.append(eval_loss)
                    
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        print(f"   🌟 New best eval loss: {best_eval_loss:.4f}")
                    
                    print(f"   Current eval loss: {eval_loss:.4f}")
                
                # Memory cleanup
                if step % 50 == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n⚠️ OOM at step {step}, cleaning up...")
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    print(f"\n🏁 Training completed!")
    print(f"   Final training loss: {training_losses[-1]:.4f}")
    print(f"   Best evaluation loss: {best_eval_loss:.4f}")
    print(f"   Total steps completed: {len(training_losses)}")
    
    return {
        'training_losses': training_losses,
        'eval_losses': eval_losses,
        'best_eval_loss': best_eval_loss
    }

# Simple usage example:
"""
# In your notebook, replace the problematic training loop with:

# Load model and tokenizer (assuming already done)
# model, tokenizer = load_dialogue_model()

# Run fixed training
results = run_fixed_training(model, tokenizer, ColabConfig, max_steps=100)

# Plot results if needed
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(results['training_losses'])
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

if results['eval_losses']:
    plt.subplot(1, 2, 2)
    plt.plot(results['eval_losses'])
    plt.title('Evaluation Loss')
    plt.xlabel('Evaluation')
    plt.ylabel('Loss')

plt.tight_layout()
plt.show()
"""