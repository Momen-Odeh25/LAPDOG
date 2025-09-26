#!/bin/bash
# setup_colab.sh - Environment setup script for Google Colab

echo "Setting up LAPDOG for Google Colab..."

# Install system dependencies if needed
apt-get update -qq
apt-get install -y git wget

# Install Python dependencies
pip install -r requirements_colab.txt

# Set up environment variables for Colab
export TOKENIZERS_PARALLELISM=false  # Avoid warnings in Colab
export TRANSFORMERS_CACHE=/content/drive/MyDrive/huggingface_cache
export HF_HOME=/content/drive/MyDrive/huggingface_cache

# Create necessary directories
mkdir -p /content/drive/MyDrive/lapdog_checkpoints
mkdir -p /content/drive/MyDrive/huggingface_cache
mkdir -p /content/lapdog_data

echo "Environment setup complete!"
echo "Remember to mount Google Drive and adjust paths accordingly."