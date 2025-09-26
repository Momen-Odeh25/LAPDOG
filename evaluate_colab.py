# evaluate_colab.py - Evaluation script optimized for Google Colab
"""
Evaluation script for LAPDOG-Gemma model in Colab environment.
Includes lightweight metrics computation and comparison with baselines.
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from tqdm.auto import tqdm
import torch

# Evaluation metrics
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

from src.colab_config import ColabConfig
from src.gemma_model import load_gemma_model
from src.data_utils_colab import ColabRetriever, load_story_corpus
from src.memory_utils import ColabMemoryManager

logger = logging.getLogger(__name__)


class ColabEvaluator:
    """Lightweight evaluator for LAPDOG-Gemma in Colab environment."""
    
    def __init__(self, model, tokenizer, retriever=None):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.rouge = Rouge()
        self.memory_manager = ColabMemoryManager()
        
        # Initialize smoothing function for BLEU
        self.bleu_smoothing = SmoothingFunction()
    
    def generate_response(self, persona: str, context: str, 
                         max_length: int = 50) -> tuple:
        """Generate response for given persona and context."""
        
        # Retrieve relevant passages if retriever available
        retrieved_passages = []
        if self.retriever:
            query = f"{persona} {context}"
            retrieved_passages = self.retriever.retrieve(query, k=3)
        
        # Format input
        input_parts = []
        if persona:
            input_parts.append(f"Persona: {persona}")
        
        if retrieved_passages:
            input_parts.append("Background:")
            for passage in retrieved_passages:
                input_parts.append(f"- {passage[:100]}...")  # Truncate
        
        input_parts.extend([f"Context: {context}", "Response:"])
        input_text = " ".join(input_parts)
        
        # Tokenize and generate
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=ColabConfig.MAX_CONTEXT_LENGTH
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response[len(input_text):].strip()
        
        return response, retrieved_passages
    
    def compute_rouge_scores(self, predictions: List[str], 
                           references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores for predictions vs references."""
        
        if not predictions or not references:
            return {}
        
        try:
            # Filter out empty predictions/references
            valid_pairs = [
                (pred, ref) for pred, ref in zip(predictions, references)
                if pred.strip() and ref.strip()
            ]
            
            if not valid_pairs:
                return {}
            
            predictions, references = zip(*valid_pairs)
            
            # Compute ROUGE scores
            scores = self.rouge.get_scores(list(predictions), list(references), avg=True)
            
            return {
                'rouge-1': scores['rouge-1']['f'],
                'rouge-2': scores['rouge-2']['f'],
                'rouge-l': scores['rouge-l']['f']
            }
            
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            return {}
    
    def compute_bleu_scores(self, predictions: List[str], 
                          references: List[str]) -> Dict[str, float]:
        """Compute BLEU scores for predictions vs references."""
        
        if not predictions or not references:
            return {}
        
        try:
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                if not pred.strip() or not ref.strip():
                    continue
                
                # Tokenize
                pred_tokens = pred.lower().split()
                ref_tokens = [ref.lower().split()]  # BLEU expects list of references
                
                # Compute BLEU score
                bleu = sentence_bleu(
                    ref_tokens, 
                    pred_tokens,
                    smoothing_function=self.bleu_smoothing.method1
                )
                bleu_scores.append(bleu)
            
            if bleu_scores:
                return {
                    'bleu': np.mean(bleu_scores),
                    'bleu_std': np.std(bleu_scores)
                }
            
        except Exception as e:
            logger.error(f"Error computing BLEU scores: {e}")
        
        return {}
    
    def compute_length_metrics(self, predictions: List[str], 
                             references: List[str]) -> Dict[str, float]:
        """Compute length-based metrics."""
        
        pred_lengths = [len(pred.split()) for pred in predictions if pred.strip()]
        ref_lengths = [len(ref.split()) for ref in references if ref.strip()]
        
        metrics = {}
        
        if pred_lengths:
            metrics.update({
                'avg_pred_length': np.mean(pred_lengths),
                'std_pred_length': np.std(pred_lengths),
                'max_pred_length': np.max(pred_lengths),
                'min_pred_length': np.min(pred_lengths)
            })
        
        if ref_lengths:
            metrics.update({
                'avg_ref_length': np.mean(ref_lengths),
                'std_ref_length': np.std(ref_lengths)
            })
        
        if pred_lengths and ref_lengths:
            # Length ratio
            length_ratios = [p/r for p, r in zip(pred_lengths, ref_lengths) if r > 0]
            if length_ratios:
                metrics['avg_length_ratio'] = np.mean(length_ratios)
        
        return metrics
    
    def evaluate_on_dataset(self, dataset_path: str, 
                          max_examples: int = 100) -> Dict[str, Any]:
        """Evaluate model on a dataset."""
        
        logger.info(f"Evaluating on {dataset_path} (max {max_examples} examples)")
        
        # Load examples
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break
                examples.append(json.loads(line.strip()))
        
        predictions = []
        references = []
        personas = []
        contexts = []
        
        # Generate predictions
        for example in tqdm(examples, desc="Generating responses"):
            try:
                # Parse example
                question = example.get('question', '')
                target_answer = example.get('answers', [''])[0]
                
                # Extract persona and context
                if 'persona:' in question and 'context:' in question:
                    parts = question.split('context:')
                    persona = parts[0].replace('persona:', '').strip()
                    context = parts[1].strip() if len(parts) > 1 else ''
                else:
                    persona = ''
                    context = question
                
                # Generate response
                response, _ = self.generate_response(persona, context)
                
                # Store results
                predictions.append(response)
                references.append(target_answer)
                personas.append(persona)
                contexts.append(context)
                
                # Memory cleanup periodically
                if len(predictions) % 20 == 0:
                    self.memory_manager.auto_cleanup_if_needed()
                
            except Exception as e:
                logger.error(f"Error processing example: {e}")
                continue
        
        # Compute metrics
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.compute_rouge_scores(predictions, references)
        metrics.update(rouge_scores)
        
        # BLEU scores
        bleu_scores = self.compute_bleu_scores(predictions, references)
        metrics.update(bleu_scores)
        
        # Length metrics
        length_metrics = self.compute_length_metrics(predictions, references)
        metrics.update(length_metrics)
        
        # Additional statistics
        metrics.update({
            'num_examples': len(predictions),
            'num_successful': len([p for p in predictions if p.strip()]),
            'success_rate': len([p for p in predictions if p.strip()]) / len(predictions) if predictions else 0
        })
        
        return {
            'metrics': metrics,
            'examples': {
                'predictions': predictions[:10],  # Sample for inspection
                'references': references[:10],
                'personas': personas[:10],
                'contexts': contexts[:10]
            }
        }


def load_baseline_responses(dataset_path: str) -> List[str]:
    """Load baseline responses (simple template-based)."""
    
    baseline_responses = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            question = example.get('question', '')
            
            # Simple template-based baseline
            if 'persona:' in question:
                baseline = "That's interesting! I can relate to that."
            else:
                baseline = "I see what you mean."
                
            baseline_responses.append(baseline)
    
    return baseline_responses


def compare_with_baseline(evaluator: ColabEvaluator, 
                         dataset_path: str,
                         max_examples: int = 50) -> Dict[str, Any]:
    """Compare model performance with simple baseline."""
    
    logger.info("Comparing with baseline model...")
    
    # Load examples
    examples = []
    references = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            example = json.loads(line.strip())
            examples.append(example)
            references.append(example.get('answers', [''])[0])
    
    # Get model predictions
    model_predictions = []
    baseline_predictions = []
    
    for example in tqdm(examples, desc="Generating comparisons"):
        # Model prediction
        question = example.get('question', '')
        
        # Extract persona and context
        if 'persona:' in question and 'context:' in question:
            parts = question.split('context:')
            persona = parts[0].replace('persona:', '').strip()
            context = parts[1].strip() if len(parts) > 1 else ''
        else:
            persona = ''
            context = question
        
        model_response, _ = evaluator.generate_response(persona, context)
        model_predictions.append(model_response)
        
        # Baseline prediction (simple template)
        if persona:
            baseline = f"Based on my experience, I would say {context.lower().replace('?', '.')} is quite common."
        else:
            baseline = "That's an interesting point to consider."
        baseline_predictions.append(baseline)
    
    # Compute metrics for both
    model_rouge = evaluator.compute_rouge_scores(model_predictions, references)
    baseline_rouge = evaluator.compute_rouge_scores(baseline_predictions, references)
    
    model_bleu = evaluator.compute_bleu_scores(model_predictions, references)
    baseline_bleu = evaluator.compute_bleu_scores(baseline_predictions, references)
    
    comparison = {
        'model_metrics': {**model_rouge, **model_bleu},
        'baseline_metrics': {**baseline_rouge, **baseline_bleu},
        'improvements': {}
    }
    
    # Calculate improvements
    for metric in model_rouge.keys():
        if metric in baseline_rouge:
            improvement = model_rouge[metric] - baseline_rouge[metric]
            comparison['improvements'][metric] = improvement
    
    return comparison


def main():
    """Main evaluation function."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🧪 Starting LAPDOG-Gemma Evaluation on Colab")
    
    # Load model and tokenizer
    print("📥 Loading model...")
    try:
        model, tokenizer = load_gemma_model(
            model_name=ColabConfig.READER_MODEL,
            use_quantization=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load retriever
    print("📚 Loading retriever...")
    story_corpus = load_story_corpus(max_stories=200)
    retriever = ColabRetriever(story_corpus)
    
    # Initialize evaluator
    evaluator = ColabEvaluator(model, tokenizer, retriever)
    
    # Evaluate on validation set
    print("🔄 Evaluating on validation set...")
    val_results = evaluator.evaluate_on_dataset(
        '/content/lapdog_data/convai2/valid.jsonl',
        max_examples=50  # Small sample for Colab
    )
    
    print("\n📊 Evaluation Results:")
    for metric, value in val_results['metrics'].items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    # Compare with baseline
    print("\n🔄 Comparing with baseline...")
    comparison = compare_with_baseline(
        evaluator,
        '/content/lapdog_data/convai2/valid.jsonl',
        max_examples=30
    )
    
    print("\n📈 Model vs Baseline Comparison:")
    print("Model Performance:")
    for metric, value in comparison['model_metrics'].items():
        print(f"   {metric}: {value:.4f}")
    
    print("Baseline Performance:")
    for metric, value in comparison['baseline_metrics'].items():
        print(f"   {metric}: {value:.4f}")
    
    print("Improvements:")
    for metric, improvement in comparison['improvements'].items():
        print(f"   {metric}: {improvement:+.4f}")
    
    # Save results
    results_path = '/content/drive/MyDrive/lapdog_checkpoints/evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'validation_results': val_results,
            'baseline_comparison': comparison
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()