# data_utils_colab.py - Optimized data loading for Google Colab
"""
Data loading and preprocessing utilities optimized for Google Colab environment.
Includes streaming datasets, memory-efficient preprocessing, and automatic downloads.
"""

import os
import json
import logging
from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path
import gdown
import zipfile
import requests
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.colab_config import ColabConfig
from src.memory_utils import MemoryEfficientDataLoader

logger = logging.getLogger(__name__)


class ColabDataDownloader:
    """Utility class for downloading and managing data in Colab."""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or ColabConfig.DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_convai2_data(self):
        """Setup ConvAI2 dataset for Colab (use local data if available)."""
        # First check if data exists locally (in workspace)
        local_data_dir = os.path.join(os.getcwd(), 'data', 'convai2')
        colab_data_dir = os.path.join(self.data_dir, 'convai2')
        os.makedirs(colab_data_dir, exist_ok=True)
        
        files_to_check = ['train.jsonl', 'valid.jsonl']
        
        for filename in files_to_check:
            local_filepath = os.path.join(local_data_dir, filename)
            colab_filepath = os.path.join(colab_data_dir, filename)
            
            if os.path.exists(local_filepath):
                logger.info(f"Found local {filename}, copying to data directory...")
                try:
                    import shutil
                    shutil.copy2(local_filepath, colab_filepath)
                    logger.info(f"Successfully copied {filename}")
                except Exception as e:
                    logger.error(f"Failed to copy {filename}: {e}")
            elif not os.path.exists(colab_filepath):
                logger.warning(f"{filename} not found locally or in data directory")
                # Create a minimal sample file for testing
                self._create_sample_data(colab_filepath, filename)
    
    def download_story_corpus(self):
        """Setup story corpus data (use local data if available)."""
        # Check for local story data first
        local_story_dir = os.path.join(os.getcwd(), 'data', 'corpora', 'story')
        colab_story_dir = os.path.join(self.data_dir, 'corpora', 'story')
        os.makedirs(colab_story_dir, exist_ok=True)
        
        local_story_file = os.path.join(local_story_dir, 'story.jsonl')
        colab_story_file = os.path.join(colab_story_dir, 'story.jsonl')
        
        if os.path.exists(local_story_file):
            logger.info("Found local story corpus, copying...")
            try:
                import shutil
                shutil.copy2(local_story_file, colab_story_file)
                logger.info("Successfully copied story corpus")
            except Exception as e:
                logger.error(f"Failed to copy story corpus: {e}")
                self._create_sample_story_data(colab_story_file)
        elif not os.path.exists(colab_story_file):
            logger.info("Creating sample story corpus...")
            self._create_sample_story_data(colab_story_file)
    
    def _download_file(self, url: str, filepath: str):
        """Download file from URL with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _create_sample_data(self, filepath: str, filename: str):
        """Create sample data for demonstration."""
        logger.info(f"Creating sample {filename} for demonstration...")
        
        if 'train' in filename:
            sample_size = 1000
        else:
            sample_size = 100
        
        sample_data = []
        for i in range(sample_size):
            example = {
                "question": f"persona: I like sample persona {i}. context: Sample context {i}?",
                "answers": [f"Sample response {i}."]
            }
            sample_data.append(example)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for example in sample_data:
                f.write(json.dumps(example) + '\n')
    
    def _create_sample_story_data(self, filepath: str):
        """Create sample story data."""
        sample_stories = [
            {
                "id": f"story_{i}",
                "title": f"Sample Story {i}",
                "text": f"This is a sample story number {i}. It contains some narrative content that could be used for retrieval in dialogue systems."
            }
            for i in range(500)  # Create 500 sample stories
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for story in sample_stories:
                f.write(json.dumps(story) + '\n')


class StreamingJSONLDataset(Dataset):
    """Memory-efficient streaming dataset for JSONL files."""
    
    def __init__(self, filepath: str, max_examples: Optional[int] = None):
        self.filepath = filepath
        self.max_examples = max_examples
        self._length = None
        self._examples_cache = {}
    
    def __len__(self):
        if self._length is None:
            self._length = self._count_lines()
            if self.max_examples is not None:
                self._length = min(self._length, self.max_examples)
        return self._length
    
    def __getitem__(self, idx):
        if idx in self._examples_cache:
            return self._examples_cache[idx]
        
        example = self._read_example_at_index(idx)
        
        # Cache recently accessed examples (limit cache size)
        if len(self._examples_cache) < 100:
            self._examples_cache[idx] = example
        
        return example
    
    def _count_lines(self):
        count = 0
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count
    
    def _read_example_at_index(self, idx):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == idx:
                    return json.loads(line.strip())
                if self.max_examples and i >= self.max_examples:
                    break
        raise IndexError(f"Index {idx} out of range")


class ColabDataProcessor:
    """Data processor optimized for Colab environment."""
    
    def __init__(self, tokenizer, max_context_length: int = 128):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
    
    def process_dialogue_example(self, example: Dict[str, Any], 
                                retrieved_passages: List[str] = None) -> Dict[str, torch.Tensor]:
        """Process a single dialogue example."""
        # Extract components
        question = example.get('question', '')
        answers = example.get('answers', [''])
        answer = answers[0] if answers else ''
        
        # Parse persona and context from question
        if 'persona:' in question and 'context:' in question:
            parts = question.split('context:')
            persona = parts[0].replace('persona:', '').strip()
            context = parts[1].strip() if len(parts) > 1 else ''
        else:
            persona = ''
            context = question
        
        # Create input text with retrieved passages
        input_parts = []
        
        if persona:
            input_parts.append(f"Persona: {persona}")
        
        if retrieved_passages:
            input_parts.append("Background:")
            for passage in retrieved_passages[:3]:  # Limit to top 3
                input_parts.append(f"- {passage}")
        
        if context:
            input_parts.append(f"Context: {context}")
        
        input_parts.append("Response:")
        
        input_text = " ".join(input_parts)
        full_text = f"{input_text} {answer}"
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_context_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels for generation task
        input_only = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_context_length,
            padding=False,
            return_tensors="pt"
        )
        
        labels = inputs.input_ids.clone()
        input_length = input_only.input_ids.shape[1]
        labels[:, :input_length] = -100  # Ignore input tokens in loss
        
        return {
            'input_ids': inputs.input_ids.squeeze(0),
            'attention_mask': inputs.attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }
    
    def create_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Create a batch from processed examples."""
        batch = {}
        
        for key in examples[0].keys():
            batch[key] = torch.stack([ex[key] for ex in examples])
        
        return batch


class ColabDataLoader:
    """Memory-efficient data loader for Colab."""
    
    def __init__(self, dataset_path: str, tokenizer, batch_size: int = 1, 
                 max_examples: Optional[int] = None):
        self.dataset = StreamingJSONLDataset(dataset_path, max_examples)
        self.processor = ColabDataProcessor(tokenizer)
        self.batch_size = batch_size
        
    def __iter__(self):
        """Iterate over batches."""
        batch_examples = []
        
        for i in range(len(self.dataset)):
            example = self.dataset[i]
            processed_example = self.processor.process_dialogue_example(example)
            batch_examples.append(processed_example)
            
            if len(batch_examples) == self.batch_size:
                yield self.processor.create_batch(batch_examples)
                batch_examples = []
        
        # Yield remaining examples
        if batch_examples:
            yield self.processor.create_batch(batch_examples)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def setup_colab_data():
    """Setup data for Colab environment (or use local data if available)."""
    logger.info("Setting up data...")
    
    # Check if we're in Colab or local environment
    in_colab = os.path.exists('/content') and not os.path.exists('C:\\')
    
    if in_colab:
        # In Colab: use the original data directory
        data_dir = ColabConfig.DATA_DIR
    else:
        # Local: use the existing data directory
        data_dir = os.path.join(os.getcwd(), 'data')
        logger.info(f"Running locally, using data directory: {data_dir}")
    
    # Initialize downloader with the appropriate directory
    downloader = ColabDataDownloader(data_dir)
    
    # Setup datasets
    downloader.download_convai2_data()
    downloader.download_story_corpus()
    
    logger.info("Data setup complete!")


def get_colab_data_loaders(tokenizer, batch_size: int = 1):
    """Get data loaders optimized for Colab."""
    # Determine data directory based on environment
    in_colab = os.path.exists('/content') and not os.path.exists('C:\\')
    if in_colab:
        data_dir = ColabConfig.DATA_DIR
    else:
        data_dir = os.path.join(os.getcwd(), 'data')
    
    # Training data loader
    train_path = os.path.join(data_dir, 'convai2', 'train.jsonl')
    train_loader = ColabDataLoader(
        train_path, 
        tokenizer, 
        batch_size=batch_size,
        max_examples=1000  # Limit for Colab
    )
    
    # Validation data loader
    valid_path = os.path.join(data_dir, 'convai2', 'valid.jsonl')
    valid_loader = ColabDataLoader(
        valid_path, 
        tokenizer, 
        batch_size=batch_size,
        max_examples=100  # Smaller validation set
    )
    
    return train_loader, valid_loader


def load_story_corpus(max_stories: int = 500):
    """Load story corpus for retrieval."""
    # Determine data directory based on environment
    in_colab = os.path.exists('/content') and not os.path.exists('C:\\')
    if in_colab:
        data_dir = ColabConfig.DATA_DIR
    else:
        data_dir = os.path.join(os.getcwd(), 'data')
    
    story_path = os.path.join(data_dir, 'corpora', 'story', 'story.jsonl')
    
    stories = []
    with open(story_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_stories:
                break
            story = json.loads(line.strip())
            stories.append(story['text'])
    
    logger.info(f"Loaded {len(stories)} stories for retrieval")
    return stories


class ColabRetriever:
    """Simple retriever for Colab environment."""
    
    def __init__(self, story_corpus: List[str]):
        self.stories = story_corpus
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Simple retrieval based on keyword overlap."""
        # This is a very simple retrieval mechanism
        # In practice, you'd use more sophisticated methods
        
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


def create_colab_dataset_info():
    """Create dataset information file for Colab."""
    info = {
        "name": "LAPDOG-Colab",
        "description": "LAPDOG dataset optimized for Google Colab",
        "version": "1.0",
        "splits": {
            "train": "convai2/train.jsonl",
            "valid": "convai2/valid.jsonl"
        },
        "story_corpus": "corpora/story/story.jsonl",
        "preprocessing": {
            "max_context_length": ColabConfig.MAX_CONTEXT_LENGTH,
            "max_target_length": ColabConfig.MAX_TARGET_LENGTH,
            "n_context": ColabConfig.N_CONTEXT
        }
    }
    
    info_path = os.path.join(ColabConfig.DATA_DIR, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Created dataset info at {info_path}")


if __name__ == "__main__":
    # Setup data when run directly
    setup_colab_data()
    create_colab_dataset_info()