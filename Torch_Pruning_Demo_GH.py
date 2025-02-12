#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torch_pruning as tp
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import math
import pandas as pd

# Define available models (only classification models here)
model_dict = {
    'resnet50': torchvision.models.resnet50,
    'resnet18': torchvision.models.resnet18,
    'convnext': torchvision.models.convnext_base,
    'vgg_19_bn': torchvision.models.vgg19_bn,
    'regnet_x_1_6gf': torchvision.models.regnet_x_1_6gf,
    'efficientnet_b4': torchvision.models.efficientnet_b4,
    'densenet121': torchvision.models.densenet121,
    'vit_b_32': torchvision.models.vit_b_32,
    'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
}

def load_model(model_name):
    """Load a model with pretrained weights."""
    print(f"Loading model '{model_name}' ...")
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found. Available models: {list(model_dict.keys())}")
    
    model = model_dict[model_name](pretrained=True)
    print("Model loaded successfully!\n")
    
    # Display model summary and stats
    print("Model architecture (summary):")
    print(model)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_input)
    print("\nBaseline statistics:")
    print(f"MACs: {base_macs/1e9:.2f} G, Params: {base_nparams/1e6:.2f} M")
    
    return model

def analyze_weight_distribution(model, save_plot=None):
    """Analyze and display weight distribution statistics."""
    # Group parameters by a heuristic
    groups = {}
    sparsity_stats = []
    total_params = 0
    total_non_zero = 0

    for name, param in model.named_parameters():
        parts = name.split('.')
        group_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups.setdefault(group_key, []).append(param.view(-1))
        
        # Calculate sparsity statistics
        non_zero = torch.count_nonzero(param).item()
        total = param.numel()
        total_params += total
        total_non_zero += non_zero
        sparsity_stats.append({
            'Layer': name,
            'Size': total,
            'Non-Zero': non_zero,
            'Sparsity (%)': (1 - non_zero/total)*100
        })

    # Print summary statistics
    print(f"""
Weight Distribution Analysis:
---------------------------
Total Parameters: {total_params:,}
Non-Zero Parameters: {total_non_zero:,} ({total_non_zero/total_params:.1%})
Zero Parameters: {total_params - total_non_zero:,} ({(1 - total_non_zero/total_params):.1%})

Layer-wise Statistics (Top 20 Most Sparse):
""")

    # Display top sparse layers
    df = pd.DataFrame(sparsity_stats)
    print(df.sort_values('Sparsity (%)', ascending=False).head(20).to_string())

    # Merge and analyze groups
    merged_groups = {}
    threshold = 1000
    
    for key, tensors in groups.items():
        merged = torch.cat(tensors)
        if merged.numel() < threshold:
            merged_groups.setdefault("others", []).append(merged)
        else:
            merged_groups[key] = merged

    if "others" in merged_groups and isinstance(merged_groups["others"], list):
        merged_groups["others"] = torch.cat(merged_groups["others"])

    print("\nParameter Group Analysis:")
    for group_name, tensor in merged_groups.items():
        group_non_zero = torch.count_nonzero(tensor).item()
        print(f"- {group_name}:")
        print(f"  Parameters: {tensor.numel():,}")
        print(f"  Non-Zero: {group_non_zero:,} ({group_non_zero/tensor.numel():.1%})")
        print(f"  Sparsity: {100*(1 - group_non_zero/tensor.numel()):.1f}%")

    if save_plot:
        # Plot histograms
        num_groups = len(merged_groups)
        cols = min(3, num_groups)
        rows = math.ceil(num_groups / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        
        if num_groups == 1:
            axs = [axs]
        elif rows > 1:
            axs = axs.flatten()
        else:
            axs = list(axs)

        for ax, (group, data) in zip(axs, merged_groups.items()):
            data_np = data.detach().numpy()
            ax.hist(data_np, bins=500, color="#1f77b4", edgecolor="black")
            ax.set_title(f"{group} ({data.numel()} weights)")
            ax.set_xlabel("Weight Value")
            ax.set_ylabel("Frequency")

        # Remove extra subplots if any
        for ax in axs[len(merged_groups):]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig(save_plot)
        plt.close()
        print(f"\nWeight distribution plot saved to: {save_plot}")

def run_pruning(model, pruning_ratio, iterative_steps=1, global_pruning=False):
    """Run the pruning process on the model."""
    if not 0 <= pruning_ratio <= 1:
        raise ValueError("Pruning ratio must be between 0 and 1")
    
    print(f"\nPruning Configuration:")
    print(f"- Pruning Ratio: {pruning_ratio:.2f}")
    print(f"- Iterative Steps: {iterative_steps}")
    print(f"- Global Pruning: {'Yes' if global_pruning else 'No'}")
    
    dummy_input = torch.randn(1, 3, 224, 224)
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, dummy_input)
    
    # Create Pruner
    example_inputs = dummy_input
    importance = tp.importance.MagnitudeImportance(p=2)
    ignored_layers = []
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance,
        iterative_steps=iterative_steps,
        ch_sparsity=pruning_ratio,
        global_pruning=global_pruning,
        ignored_layers=ignored_layers,
    )
    
    # Run pruning
    for i in range(iterative_steps):
        pruner.step()
        macs, nparams = tp.utils.count_ops_and_params(model, dummy_input)
        print(f"\nIteration {i + 1}/{iterative_steps}:")
        print(f"MACs: {macs/1e9:.2f}G ({macs/base_macs:.2%})")
        print(f"Params: {nparams/1e6:.2f}M ({nparams/base_nparams:.2%})")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='PyTorch Model Pruning Demo')
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=list(model_dict.keys()),
                        help='Model architecture to use')
    parser.add_argument('--pruning-ratio', type=float, default=0.2,
                        help='Pruning ratio (0.0 to 1.0)')
    parser.add_argument('--iterative-steps', type=int, default=1,
                        help='Number of iterative pruning steps')
    parser.add_argument('--global-pruning', action='store_true',
                        help='Enable global pruning')
    parser.add_argument('--save-plot', type=str,
                        help='Save weight distribution plot to file')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze weight distribution without pruning')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Analyze initial weight distribution
    print("\nAnalyzing initial weight distribution...")
    analyze_weight_distribution(model, args.save_plot)
    
    if not args.analyze_only:
        # Run pruning
        model = run_pruning(
            model,
            args.pruning_ratio,
            args.iterative_steps,
            args.global_pruning
        )
        
        # Analyze final weight distribution
        print("\nAnalyzing final weight distribution...")
        if args.save_plot:
            plot_name = args.save_plot.rsplit('.', 1)
            final_plot = f"{plot_name[0]}_final.{plot_name[1]}"
        else:
            final_plot = None
        analyze_weight_distribution(model, final_plot)

if __name__ == '__main__':
    main()