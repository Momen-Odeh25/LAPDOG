# simple_colab_setup.py - Simplified setup for Colab to avoid import issues
"""
Simplified setup script for LAPDOG-Gemma on Google Colab.
This bypasses complex imports that might cause compatibility issues.
"""

import os
import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import gc

logger = logging.getLogger(__name__)


class SimpleColabConfig:
    """Simple configuration for Colab environment."""
    
    # Memory and compute settings
    MAX_MEMORY_GB = 12
    BATCH_SIZE_TRAIN = 1
    BATCH_SIZE_EVAL = 2
    GRADIENT_ACCUMULATION_STEPS = 8
    
    # Model settings
    READER_MODEL = "microsoft/DialoGPT-small"  # Start with most compatible model
    READER_MODEL_ALTERNATIVES = [
        "microsoft/DialoGPT-medium",
        "gpt2",
        "distilgpt2"
    ]
    
    # Training settings
    MAX_STEPS = 500  # Reduced for quick testing
    EVAL_FREQ = 50
    SAVE_FREQ = 100
    WARMUP_STEPS = 25
    LEARNING_RATE = 5e-5
    
    # Memory optimization flags
    USE_GRADIENT_CHECKPOINTING = True
    USE_MIXED_PRECISION = "fp16"
    
    # Data settings
    MAX_CONTEXT_LENGTH = 256
    MAX_TARGET_LENGTH = 64
    N_CONTEXT = 3
    
    # Paths
    CHECKPOINT_DIR = "/content/drive/MyDrive/lapdog_checkpoints"
    DATA_DIR = "/content/lapdog_data"
    CACHE_DIR = "/content/drive/MyDrive/huggingface_cache"


def simple_memory_cleanup():
    """Simple memory cleanup function."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_info():
    """Get basic memory information."""
    info = {}
    
    if torch.cuda.is_available():
        info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
        info['gpu_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['gpu_free_gb'] = info['gpu_total_gb'] - info['gpu_allocated_gb']
    else:
        info['gpu_allocated_gb'] = 0
        info['gpu_total_gb'] = 0
        info['gpu_free_gb'] = 0
    
    return info


def load_simple_dialogue_model(model_name: str = None):
    """Load a simple dialogue model that's guaranteed to work."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    if model_name is None:
        model_name = SimpleColabConfig.READER_MODEL
    
    models_to_try = [model_name] + SimpleColabConfig.READER_MODEL_ALTERNATIVES
    
    for model_candidate in models_to_try:
        try:
            print(f"🔄 Attempting to load {model_candidate}...")
            
            # Try with quantization first for memory efficiency
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_candidate,
                    quantization_config=quantization_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"✅ Loaded {model_candidate} with 4-bit quantization")
                
            except Exception as quant_error:
                print(f"⚠️  Quantization failed, trying without: {quant_error}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_candidate,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                print(f"✅ Loaded {model_candidate} without quantization")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_candidate)
            
            # Add special tokens if needed
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Estimate memory usage
            param_count = sum(p.numel() for p in model.parameters())
            memory_gb = param_count * 2 / 1024**3  # Approximate for FP16
            
            print(f"📊 Model loaded successfully!")
            print(f"   Parameters: {param_count:,}")
            print(f"   Estimated memory: {memory_gb:.2f} GB")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load {model_candidate}: {e}")
            continue
    
    raise Exception("All model loading attempts failed!")


def create_simple_retriever(stories: List[str]):
    """Create a simple keyword-based retriever."""
    
    class SimpleRetriever:
        def __init__(self, story_corpus):
            self.stories = story_corpus
        
        def retrieve(self, query: str, k: int = 5) -> List[str]:
            """Simple retrieval based on keyword overlap."""
            query_words = set(query.lower().split())
            scored_stories = []
            
            for story in self.stories:
                story_words = set(story.lower().split())
                overlap = len(query_words.intersection(story_words))
                if overlap > 0:
                    scored_stories.append((overlap, story))
            
            # Sort by score and return top k
            scored_stories.sort(key=lambda x: x[0], reverse=True)
            return [story for _, story in scored_stories[:k]]
    
    return SimpleRetriever(stories)


def format_dialogue_input(persona: str, context: str, retrieved_stories: List[str] = None) -> str:
    """Format input for dialogue generation."""
    parts = []
    
    if persona:
        parts.append(f"Persona: {persona}")
    
    if retrieved_stories:
        parts.append("Background:")
        for story in retrieved_stories[:2]:  # Limit to avoid length issues
            parts.append(f"- {story[:100]}...")
    
    if context:
        parts.append(f"Context: {context}")
    
    parts.append("Response:")
    
    return " ".join(parts)


def generate_response(model, tokenizer, persona: str, context: str, 
                     retriever=None, max_length: int = 50) -> str:
    """Generate a response using the model."""
    
    # Retrieve relevant stories if retriever available
    retrieved_stories = []
    if retriever:
        query = f"{persona} {context}"
        retrieved_stories = retriever.retrieve(query, k=2)
    
    # Format input
    input_text = format_dialogue_input(persona, context, retrieved_stories)
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=SimpleColabConfig.MAX_CONTEXT_LENGTH,
        padding=False
    )
    
    # Generate
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(input_text):].strip()
    
    return response


def setup_simple_training_data():
    """Create simple training data for demonstration."""
    
    # Create directories
    os.makedirs(SimpleColabConfig.DATA_DIR, exist_ok=True)
    os.makedirs(f"{SimpleColabConfig.DATA_DIR}/convai2", exist_ok=True)
    os.makedirs(f"{SimpleColabConfig.DATA_DIR}/corpora/story", exist_ok=True)
    
    # Sample dialogue data
    sample_dialogues = [
        {
            "question": "persona: I love hiking and nature. context: What do you like to do for fun?",
            "answers": ["I absolutely love spending time outdoors! Hiking through mountain trails is my favorite way to relax and connect with nature."]
        },
        {
            "question": "persona: I'm a chef who specializes in Italian cuisine. context: Can you recommend a good restaurant?",
            "answers": ["As a chef myself, I'd recommend trying authentic Italian places that make their pasta fresh daily. The difference in taste is incredible!"]
        },
        {
            "question": "persona: I'm a student studying computer science. context: What programming language should I learn first?",
            "answers": ["As a CS student, I'd definitely recommend starting with Python! It's beginner-friendly but powerful enough for real projects."]
        },
        {
            "question": "persona: I work as a librarian and love reading. context: What's your favorite book genre?",
            "answers": ["Working in a library has exposed me to so many genres! I particularly enjoy historical fiction and mystery novels."]
        },
        {
            "question": "persona: I'm a fitness enthusiast who runs marathons. context: How do you stay motivated to exercise?",
            "answers": ["Training for marathons taught me that consistency is key. I set small daily goals and celebrate each milestone!"]
        }
    ]
    
    # Sample stories for retrieval
    sample_stories = [
        {
            "id": "story_1",
            "title": "Mountain Adventure",
            "text": "Sarah packed her backpack early in the morning. The mountain trail stretched ahead, promising beautiful views and fresh air. She loved these solo hiking trips that helped her clear her mind."
        },
        {
            "id": "story_2", 
            "title": "The Chef's Secret",
            "text": "Marco had been perfecting his pasta recipe for years. His grandmother's techniques, passed down through generations, were the secret to his restaurant's success."
        },
        {
            "id": "story_3",
            "title": "Late Night Coding",
            "text": "Emma stared at her computer screen, debugging her first Python program. The satisfaction of solving each error made the long hours worth it."
        },
        {
            "id": "story_4",
            "title": "Library Discovery", 
            "text": "Among the dusty shelves, Tom found an old mystery novel that changed his perspective on literature. The quiet library had become his sanctuary."
        },
        {
            "id": "story_5",
            "title": "Marathon Morning",
            "text": "At 5 AM, Lisa laced up her running shoes. The empty streets and cool morning air provided the perfect conditions for her training run."
        }
    ]
    
    # Save training data
    train_path = f"{SimpleColabConfig.DATA_DIR}/convai2/train.jsonl"
    with open(train_path, 'w') as f:
        for dialogue in sample_dialogues * 20:  # Repeat for more training data
            f.write(json.dumps(dialogue) + '\n')
    
    # Save validation data  
    valid_path = f"{SimpleColabConfig.DATA_DIR}/convai2/valid.jsonl"
    with open(valid_path, 'w') as f:
        for dialogue in sample_dialogues:
            f.write(json.dumps(dialogue) + '\n')
    
    # Save stories
    story_path = f"{SimpleColabConfig.DATA_DIR}/corpora/story/story.jsonl"
    with open(story_path, 'w') as f:
        for story in sample_stories * 10:  # Repeat for more retrieval options
            f.write(json.dumps(story) + '\n')
    
    print(f"✅ Created sample data:")
    print(f"   Training examples: {len(sample_dialogues * 20)}")
    print(f"   Validation examples: {len(sample_dialogues)}")
    print(f"   Stories for retrieval: {len(sample_stories * 10)}")


def test_simple_setup():
    """Test the simple setup to make sure everything works."""
    print("🧪 Testing simple LAPDOG setup...")
    
    try:
        # Setup data
        setup_simple_training_data()
        
        # Load model
        model, tokenizer = load_simple_dialogue_model()
        
        # Load stories and create retriever
        stories = []
        story_path = f"{SimpleColabConfig.DATA_DIR}/corpora/story/story.jsonl"
        with open(story_path, 'r') as f:
            for line in f:
                story_data = json.loads(line)
                stories.append(story_data['text'])
        
        retriever = create_simple_retriever(stories[:20])  # Use subset
        
        # Test generation
        test_cases = [
            ("I love hiking", "What's your favorite outdoor activity?"),
            ("I'm a chef", "What's your specialty?"),
            ("I study computer science", "What programming language do you recommend?")
        ]
        
        print("\n🎯 Testing dialogue generation:")
        for persona, context in test_cases:
            response = generate_response(model, tokenizer, persona, context, retriever)
            print(f"   Persona: {persona}")
            print(f"   Context: {context}")
            print(f"   Response: {response}")
            print()
        
        print("✅ Simple setup test completed successfully!")
        return model, tokenizer, retriever
        
    except Exception as e:
        print(f"❌ Simple setup test failed: {e}")
        raise


def create_simple_training_loop(model, tokenizer, training_data, config=None):
    """Create a simple training loop that works with the model."""
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    
    if config is None:
        config = SimpleColabConfig()
    
    class SimpleDialogueDataset(Dataset):
        def __init__(self, data, tokenizer, max_length=256):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            example = self.data[idx]
            
            # Format the dialogue
            if isinstance(example, dict) and 'question' in example and 'answers' in example:
                # ConvAI2 format
                input_text = example['question']
                target_text = example['answers'][0] if example['answers'] else ""
            else:
                # Simple format 
                input_text = f"Persona: {example.get('persona', '')} Context: {example.get('context', '')}"
                target_text = example.get('response', '')
            
            # Combine input and target
            full_text = f"{input_text} {target_text}"
            
            # Tokenize
            tokens = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': tokens['input_ids'].squeeze(),
                'attention_mask': tokens['attention_mask'].squeeze(),
                'labels': tokens['input_ids'].squeeze().clone()
            }
    
    # Create dataset and dataloader
    dataset = SimpleDialogueDataset(training_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE_TRAIN, shuffle=True)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.WARMUP_STEPS,
        num_training_steps=config.MAX_STEPS
    )
    
    # Setup mixed precision if available
    scaler = None
    use_amp = torch.cuda.is_available() and config.USE_MIXED_PRECISION
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    def train_step(batch):
        """Single training step."""
        model.train()
        
        # Move batch to device
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        if use_amp and scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        return loss.item()
    
    def evaluate_model(eval_data, num_batches=5):
        """Simple evaluation."""
        model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                if torch.cuda.is_available():
                    batch = {k: v.cuda() for k, v in batch.items()}
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(**batch)
                        loss = outputs.loss
                else:
                    outputs = model(**batch)
                    loss = outputs.loss
                
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else float('inf')
    
    return {
        'train_step': train_step,
        'evaluate': evaluate_model,
        'dataloader': dataloader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'scaler': scaler
    }


def run_simple_training(model, tokenizer, max_steps=100):
    """Run a simple training demonstration."""
    print("🚀 Setting up simple training demonstration...")
    
    # Create sample training data
    sample_data = [
        {
            "persona": "I love hiking",
            "context": "What do you do for fun?",
            "response": "I love spending time outdoors hiking and exploring nature trails!"
        },
        {
            "persona": "I'm a chef",
            "context": "What's your favorite dish?", 
            "response": "I absolutely love making fresh pasta from scratch using traditional techniques."
        },
        {
            "persona": "I study computer science",
            "context": "What programming language do you recommend?",
            "response": "Python is great for beginners - it's versatile and has a gentle learning curve."
        }
    ] * 20  # Repeat for more training data
    
    # Setup training
    training_components = create_simple_training_loop(model, tokenizer, sample_data)
    train_step = training_components['train_step']
    evaluate = training_components['evaluate']
    dataloader = training_components['dataloader']
    
    print(f"📊 Training dataset size: {len(sample_data)}")
    print(f"🔄 Training for {max_steps} steps...")
    
    # Training loop
    import time
    from tqdm.auto import tqdm
    
    best_loss = float('inf')
    losses = []
    
    try:
        pbar = tqdm(range(max_steps), desc="Training")
        
        for step in pbar:
            # Get a batch
            try:
                batch = next(iter(dataloader))
                loss = train_step(batch)
                losses.append(loss)
                
                # Update progress
                avg_loss = sum(losses[-10:]) / min(len(losses), 10)  # Last 10 steps average
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'avg_loss': f"{avg_loss:.4f}",
                    'best': f"{best_loss:.4f}"
                })
                
                # Periodic evaluation
                if step > 0 and step % 20 == 0:
                    eval_loss = evaluate(sample_data[:5])  # Quick eval
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        print(f"\n🌟 New best loss: {best_loss:.4f}")
                
                # Memory cleanup
                if step % 25 == 0:
                    simple_memory_cleanup()
                
            except Exception as e:
                print(f"\n⚠️  Error at step {step}: {e}")
                break
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
    
    print(f"\n✅ Training completed! Best loss: {best_loss:.4f}")
    return losses, best_loss


if __name__ == "__main__":
    test_simple_setup()