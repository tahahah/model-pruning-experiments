#!/usr/bin/env python
import torch
import torch.nn as nn
import torch_pruning as tp
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.apps.utils.image import DMCrop
import os
import pandas as pd
from copy import deepcopy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

def fine_grained_prune(model: nn.Module, sparsity: float) -> nn.Module:
    """Prune model weights using magnitude-based pruning"""
    # Create copy of original model
    pruned_model = deepcopy(model)
    
    # Iterate through all model parameters
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Process each layer separately to avoid memory issues
            tensor = module.weight.data
            
            # Convert to CPU for memory-efficient calculation
            tensor_cpu = tensor.cpu().numpy().flatten()
            
            # Calculate threshold using numpy's percentile with limited memory
            threshold = np.percentile(
                np.abs(tensor_cpu), 
                sparsity * 100,
                method='lower',
                interpolation='lower'
            )
            
            # Create mask on GPU if available
            mask = torch.tensor(np.abs(tensor_cpu) > threshold, device=tensor.device)
            mask = mask.reshape(tensor.shape)
            
            # Apply mask
            module.weight.data = tensor * mask.float()
            
            # Clean up temporary variables
            del tensor_cpu, mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return pruned_model

def process_image(model, image_path, output_path, device):
    """Process a single image through the VAE"""
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),  # Convert to RGB first
        DMCrop(512),  # resolution
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    image = Image.open(image_path)
    x = transform(image)[None].to(device)
    
    with torch.no_grad():
        latent = model.encode(x)
        y = model.decode(latent)
        
    # Save output
    save_image(y * 0.5 + 0.5, output_path)
    
    # Create comparison plot
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    original_img = x[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(1, 2, 2)
    reconstructed_img = y[0].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    plt.imshow(reconstructed_img)
    plt.title(f"Reconstructed (Sparsity: {model.sparsity if hasattr(model, 'sparsity') else 0:.1%})")
    plt.axis('off')
    
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {e}")
    plt.close()
    
    return y

def main():
    # Load pretrained DC-AE model
    print("Loading DC-AE model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model with CUDA-optimized configuration
    model = DCAE_HF.from_pretrained("mit-han-lab/dc-ae-f64c128-in-1.0", use_cuda=True)
    
    # Move model to device before pruning
    model = model.to(device).eval()
    
    # Warm up CUDA and Triton kernels
    if torch.cuda.is_available():
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 512, 512).to(device)
            _ = model(dummy_input)
    
    # Create output directories
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze original model
    print("\nAnalyzing original model...")
    orig_params, orig_nonzero = analyze_weight_distribution(model)
    
    # Process images with different sparsity levels
    sparsity_levels = [0.0, 0.5, 0.7, 0.9]
    image_paths = list(Path("images").glob("*.png"))
    
    # Create markdown report
    with open("output/pruning_report.md", "w") as f:
        f.write("# VAE Pruning Analysis Report\n\n")
        f.write("## Model Information\n")
        f.write(f"- Total Parameters: {orig_params:,}\n")
        f.write(f"- Non-zero Parameters: {orig_nonzero:,}\n\n")
        
        f.write("## Image Reconstruction Analysis\n\n")
        
        for img_path in image_paths:
            f.write(f"### Image: {img_path.name}\n\n")
            f.write("| Sparsity | Original vs Reconstructed | Parameter Reduction |\n")
            f.write("|----------|--------------------------|--------------------|\n")
            
            for sparsity in sparsity_levels:
                print(f"\nProcessing {img_path.name} with sparsity {sparsity:.1f}")
                
                # Get model for this sparsity level
                if sparsity == 0.0:
                    current_model = model
                else:
                    current_model = fine_grained_prune(model, sparsity)
                    current_model = current_model.to(device)
                
                # Process image
                output_name = f"{img_path.stem}_sparsity_{sparsity:.1f}.png"
                output_path = output_dir / output_name
                process_image(current_model, img_path, output_path, device)
                
                # Get parameter stats
                _, nonzero = analyze_weight_distribution(current_model)
                param_reduction = 1 - nonzero/orig_nonzero if sparsity > 0 else 0
                
                # Add to markdown
                f.write(f"| {sparsity:.1%} | ![{output_name}](./output/{output_name}) | {param_reduction:.1%} |\n")
            
            f.write("\n")

if __name__ == "__main__":
    main()
