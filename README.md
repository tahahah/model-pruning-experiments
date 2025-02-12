# Model Pruning Experiments

This repository contains experiments with pruning different deep learning models:

1. **Vision Transformer (ViT) Pruning**
   - Fine-grained magnitude-based pruning of ViT-B/32
   - Analysis of accuracy vs sparsity trade-offs

2. **DC-AE (Deep Compression AutoEncoder) Pruning**
   - Pruning experiments on the DC-AE model from MIT-Han-Lab
   - Visual analysis of reconstruction quality at different sparsity levels
   - Parameter reduction analysis

## Quick Start in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/model-pruning-experiments/blob/main/notebooks/pruning_experiments.ipynb)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run ViT pruning experiments:
```bash
python vit_pruning_analysis.py
```

3. Run DC-AE pruning experiments:
```bash
python vae_pruning_analysis.py
```

## Directory Structure

- `images/` - Test images for VAE reconstruction
- `models/` - Model implementations and pruning utilities
- `notebooks/` - Jupyter notebooks for interactive experiments
- `output/` - Generated outputs and analysis reports

## Results

Results will be saved in the `output` directory:
- Pruning analysis reports in markdown format
- Reconstructed images at different sparsity levels
- Model statistics and performance metrics

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- torch-pruning
- efficientvit (for DC-AE experiments)
- pandas (for analysis)
