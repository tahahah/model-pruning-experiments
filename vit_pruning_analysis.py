#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision.models as models
import torch_pruning as tp
import pandas as pd
from typing import Union, List
import copy

def analyze_weight_distribution(model):
    """Analyze weight distribution and sparsity statistics"""
    total_params = 0
    total_nonzero = 0
    stats = []
    
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Skip 1D tensors like biases
            nonzero = torch.count_nonzero(param).item()
            total = param.numel()
            total_params += total
            total_nonzero += nonzero
            stats.append({
                'Layer': name,
                'Size': total,
                'NonZero': nonzero,
                'Sparsity(%)': (1 - nonzero/total)*100
            })
    
    print("\nModel Sparsity Analysis:")
    print(f"Total Parameters: {total_params:,}")
    print(f"Non-zero Parameters: {total_nonzero:,} ({total_nonzero/total_params:.1%})")
    print(f"Zero Parameters: {total_params - total_nonzero:,} ({1 - total_nonzero/total_params:.1%})")
    
    df = pd.DataFrame(stats)
    print("\nTop 10 Most Sparse Layers:")
    print(df.sort_values('Sparsity(%)', ascending=False).head(10).to_string())
    return total_params, total_nonzero

def fine_grained_prune(model, sparsity):
    """Magnitude-based fine-grained pruning"""
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            tensor = module.weight.data
            threshold = torch.quantile(tensor.abs(), sparsity)
            mask = torch.abs(tensor) > threshold
            module.weight.data *= mask
    return model

def evaluate(model, val_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

def main():
    # Load pretrained ViT model
    print("Loading pretrained ViT-B/32 (most efficient ViT variant)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model with pretrained weights
    weights = models.ViT_B_32_Weights.DEFAULT
    model = models.vit_b_32(weights=weights).to(device)
    model.eval()
    
    # Create a smaller synthetic dataset for faster testing
    preprocess = weights.transforms()
    dataset = torch.utils.data.TensorDataset(
        preprocess(torch.randn(50, 3, 224, 224)),  # 50 random images instead of 100
        torch.randint(0, 1000, (50,))   # 50 random labels
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0)  # Smaller batch size, single worker
    
    # Analyze original model
    print("\nAnalyzing original model...")
    orig_params, orig_nonzero = analyze_weight_distribution(model)
    orig_acc = evaluate(model, val_loader, device)
    print(f"\nOriginal Model Accuracy: {orig_acc:.2f}%")
    
    # Fine-grained pruning analysis with fewer test points
    print("\nPerforming fine-grained pruning...")
    sparsities = [0.5, 0.7, 0.9]  # Reduced test points
    for sparsity in sparsities:
        pruned_model = fine_grained_prune(model, sparsity)
        pruned_model.to(device)
        acc = evaluate(pruned_model, val_loader, device)
        params, nonzero = analyze_weight_distribution(pruned_model)
        print(f"\nFine-grained pruning (sparsity={sparsity:.1f}):")
        print(f"Accuracy: {acc:.2f}%")
        print(f"Parameter reduction: {1 - nonzero/orig_nonzero:.1%}")

if __name__ == '__main__':
    main()
