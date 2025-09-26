# gemma_model.py - Gemma 3 integration for LAPDOG
"""
This module provides Gemma 3 model integration for LAPDOG framework.
It replaces the T5-based FiD model with Gemma 3 while maintaining compatibility.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GemmaForCausalLM,
    GemmaTokenizer
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)


class GemmaForRetrievalAugmentedGeneration(nn.Module):
    """
    Gemma 3 model adapted for retrieval-augmented dialogue generation.
    This class mimics the FiD interface while using Gemma 3 as the backbone.
    """
    
    def __init__(self, config, model_name="google/gemma-2b"):
        super().__init__()
        self.config = config
        self.model_name = model_name
        
        # Load Gemma model with quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configure LoRA for parameter-efficient training
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # Low rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass that handles retrieval-augmented input format.
        Processes multiple retrieved passages and generates responses.
        """
        # Handle multi-passage input (similar to FiD)
        if len(input_ids.shape) == 3:  # [batch, n_passages, seq_len]
            batch_size, n_passages, seq_len = input_ids.shape
            # Flatten for processing
            input_ids = input_ids.view(-1, seq_len)
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1, seq_len)
            if labels is not None:
                labels = labels.view(-1, seq_len)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """Generation method compatible with LAPDOG's evaluation pipeline."""
        # Handle multi-passage input for generation
        if len(input_ids.shape) == 3:
            batch_size, n_passages, seq_len = input_ids.shape
            # For generation, we typically use the first passage or concatenate
            input_ids = input_ids[:, 0, :]  # Use first passage
            if attention_mask is not None:
                attention_mask = attention_mask[:, 0, :]
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )


def load_gemma_model(model_name="google/gemma-2b", use_quantization=True):
    """
    Load Gemma model for LAPDOG with appropriate configuration.
    
    Args:
        model_name: Hugging Face model name
        use_quantization: Whether to use 4-bit quantization
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<bos>",
        }
        
        num_added_tokens = tokenizer.add_special_tokens(special_tokens)
        
        # Load model
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Resize embeddings if tokens were added
        if num_added_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
        
        logger.info(f"Successfully loaded Gemma model: {model_name}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading Gemma model: {e}")
        # Fallback to smaller model
        logger.info("Falling back to DialoGPT-small...")
        return load_fallback_model()


def load_fallback_model():
    """Load a fallback model if Gemma fails to load."""
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def create_retrieval_prompt(persona, context, retrieved_passages):
    """
    Create a formatted prompt for retrieval-augmented generation.
    
    Args:
        persona: Persona description
        context: Dialogue context
        retrieved_passages: List of retrieved story passages
    
    Returns:
        Formatted prompt string
    """
    prompt_parts = []
    
    # Add persona information
    prompt_parts.append(f"Persona: {persona}")
    
    # Add retrieved passages as context
    if retrieved_passages:
        prompt_parts.append("Relevant background:")
        for i, passage in enumerate(retrieved_passages[:3]):  # Limit to top 3
            prompt_parts.append(f"- {passage}")
    
    # Add dialogue context
    prompt_parts.append(f"Context: {context}")
    
    # Add generation prompt
    prompt_parts.append("Response:")
    
    return "\n".join(prompt_parts)


def format_training_example(example, retrieved_passages, tokenizer, max_length=512):
    """
    Format a training example for Gemma 3 model.
    
    Args:
        example: Training example with persona, context, and response
        retrieved_passages: Retrieved story passages
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
    
    Returns:
        Tokenized input and labels
    """
    # Extract components
    persona = example.get('persona', '')
    context = example.get('context', '')
    response = example.get('response', '')
    
    # Create prompt
    prompt = create_retrieval_prompt(persona, context, retrieved_passages)
    full_text = f"{prompt} {response}"
    
    # Tokenize
    inputs = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create labels (mask prompt tokens)
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt"
    )
    
    labels = inputs.input_ids.clone()
    prompt_length = prompt_tokens.input_ids.shape[1]
    labels[:, :prompt_length] = -100  # Ignore prompt tokens in loss
    
    return {
        'input_ids': inputs.input_ids,
        'attention_mask': inputs.attention_mask,
        'labels': labels
    }