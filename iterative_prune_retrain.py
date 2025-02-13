#!/usr/bin/env python
import os
import sys
import torch
import argparse
import yaml
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from vae_pruning_analysis import fine_grained_prune, analyze_weight_distribution
from dcaecore.train_dc_ae import (
    load_config, 
    create_run_config, 
    setup_logger,
    set_random_seed,
    setup_dist_env,
    get_dist_size,
    get_dist_rank
)
from efficientvit.ae_model_zoo import DCAE_HF
from dcaecore.trainer import DCAETrainer
from dcaecore.pacman_dataset_copy import SimplePacmanDatasetProvider, PacmanDatasetProviderConfig

def get_user_input(prompt: str) -> bool:
    """Get yes/no input from user"""
    while True:
        response = input(f"{prompt} (yes/no): ").lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        print("Please answer 'yes' or 'no'")

def save_model(model: torch.nn.Module, save_dir: str, iteration: int):
    """Save model with iteration number"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"model_pruned_iter_{iteration}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def visualize_reconstructions(model, sample_batch, save_dir: str, step: str, iteration: int, device):
    """Visualize and save reconstructions for a batch of images"""
    model.eval()
    with torch.no_grad():
        x = sample_batch["data"].to(device)
        # Get reconstructions
        latent = model.encode(x)
        y = model.decode(latent)
        
        # Create a grid of original vs reconstructed images
        n_samples = min(4, x.size(0))  # Show up to 4 samples
        plt.figure(figsize=(12, 3*n_samples))
        
        for idx in range(n_samples):
            # Original image
            plt.subplot(n_samples, 2, 2*idx + 1)
            original_img = x[idx].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            plt.imshow(original_img)
            plt.title(f"Original Image {idx+1}")
            plt.axis('off')
            
            # Reconstructed image
            plt.subplot(n_samples, 2, 2*idx + 2)
            reconstructed_img = y[idx].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5
            plt.imshow(reconstructed_img)
            plt.title(f"Reconstructed {idx+1}")
            plt.axis('off')
        
        plt.suptitle(f"Iteration {iteration} - {step}")
        plt.tight_layout()
        
        # Save the comparison plot
        save_path = os.path.join(save_dir, f"reconstructions_iter_{iteration}_{step}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved reconstruction comparison to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to initial model or HuggingFace model name (e.g., "username/model-name")')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--gpu", type=str, help="Comma-separated list of GPU IDs to use")
    args = parser.parse_args()

    # Setup distributed environment and seed
    device = setup_dist_env(args.gpu)
    set_random_seed(args.seed)

    # Load config and setup
    config = load_config(args.config)
    logger = setup_logger(args.save_dir)
    
    # Initialize model and load weights
    model = DCAE_HF.from_pretrained(args.model_path)
    model = model.to(device)
    
    # Setup dataset for visualization
    data_provider_cfg = config.get('data_provider', {})
    data_cfg = PacmanDatasetProviderConfig(
        name=data_provider_cfg.get('name', 'SimplePacmanDatasetProvider'),
        train_dataset=data_provider_cfg.get('train_dataset', 'Tahahah/PacmanDataset_3'),
        val_dataset=data_provider_cfg.get('val_dataset', 'Tahahah/PacmanDataset_3'),
        train_split=data_provider_cfg.get('train_split', 'train'),
        val_split=data_provider_cfg.get('val_split', 'train'),
        image_size=data_provider_cfg.get('image_size', 512),
        verification_mode=data_provider_cfg.get('verification_mode', 'no_checks'),
        streaming=data_provider_cfg.get('streaming', True),
        batch_size=data_provider_cfg.get('batch_size', 16),
        n_worker=data_provider_cfg.get('num_workers', 4),
        val_steps=data_provider_cfg.get('val_steps', 100)
    )

    # Initialize data provider with distributed info
    data_cfg.num_replicas = get_dist_size()
    data_cfg.rank = get_dist_rank()
    dataset_provider = SimplePacmanDatasetProvider(data_cfg)
    
    # Get a sample batch for visualization
    sample_batch = next(iter(dataset_provider.valid))

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"\n=== Starting iteration {iteration} ===")

        # Analyze current model
        logger.info("Current model statistics:")
        total_params, total_nonzero = analyze_weight_distribution(model)

        # Visualize reconstructions before pruning
        visualize_reconstructions(model, sample_batch, args.save_dir, "before_pruning", iteration, device)

        # Prune model by 50%
        logger.info("Pruning model by 50%...")
        pruned_model = fine_grained_prune(model, sparsity=0.5)
        
        # Save pruned model
        save_model(pruned_model, args.save_dir, iteration)

        # Analyze pruned model
        logger.info("Pruned model statistics:")
        analyze_weight_distribution(pruned_model)

        # Visualize reconstructions after pruning
        visualize_reconstructions(pruned_model, sample_batch, args.save_dir, "after_pruning", iteration, device)

        # Ask if user wants to retrain
        if get_user_input("Do you want to retrain the pruned model?"):
            logger.info("Starting retraining...")
            
            # Setup training
            run_config = create_run_config(config)
            trainer = DCAETrainer(
                model=pruned_model,
                run_config=run_config,
                dataset_provider=dataset_provider,
                device=device
            )
            
            # Train model
            trainer.train()
            
            # Visualize reconstructions after retraining
            visualize_reconstructions(pruned_model, sample_batch, args.save_dir, "after_retraining", iteration, device)
            
            model = pruned_model  # Update reference for next iteration
        else:
            model = pruned_model

        # Ask if user wants to continue
        if not get_user_input("Do you want to continue with another round of pruning?"):
            break

    logger.info("Finished iterative pruning and retraining")

if __name__ == "__main__":
    main()
