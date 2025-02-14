import os
import logging
import traceback
import copy
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from dcaecore.models import DCAE_HF
from dcaecore.data import PacmanDataProvider
import yaml
import itertools
import torchvision
import torchvision.transforms as transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from PIL import Image
from dcaecore.train_dc_ae import setup_dist_env, set_random_seed, get_dist_size, get_dist_rank
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.models.efficientvit.dc_ae import DCAE
from dcaecore.trainer import DCAETrainer
from dcaecore.pacman_dataset_copy import SimplePacmanDatasetProvider, PacmanDatasetProviderConfig
from vae_pruning_analysis import fine_grained_prune, analyze_weight_distribution, plot_weight_distributions

class ModelManager:
    """
    ModelManager handles all model-related operations including initialization,
    pruning, training, and evaluation. It serves as a bridge between the UI
    and the dcaecore functionality.
    """
    
    def __init__(self, config_path: str, save_dir: str):
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.save_dir = save_dir
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Setup device and seed
        self.device = setup_dist_env("0") # TODO: Make configurable to multiple gpus
        set_random_seed(42)  # TODO: Make configurable
        
        # Models
        self.original_model = None
        self.equipped_model = None
        self.experimental_model = None
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        
        # Dataset
        self.dataset_provider = None
        self.sample_batch = None
        
        os.makedirs(save_dir, exist_ok=True)

    def _setup_dataset_if_needed(self):
        """Setup dataset provider if not already set up"""
        if self.dataset_provider is None:
            self.logger.info("Setting up dataset provider...")
            data_provider_cfg = self.config.get('data_provider', {})
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
            
            # Set up distributed training info
            data_cfg.num_replicas = get_dist_size()
            data_cfg.rank = get_dist_rank()
            
            self.dataset_provider = SimplePacmanDatasetProvider(data_cfg)
            self.sample_batch = next(iter(self.dataset_provider.valid))
            self.logger.info("Dataset provider setup complete")

    def load_initial_model(self, model_path_or_name: str) -> Dict[str, Any]:
        """
        Load initial model and set it as both original and equipped model.
        
        Args:
            model_path_or_name: Path to model or HuggingFace model name
            
        Returns:
            Dict containing model metrics
        """
        try:
            self.logger.info(f"Loading initial model from {model_path_or_name}")
            
            # Clear any existing models from memory
            if self.original_model is not None:
                del self.original_model
            if self.equipped_model is not None:
                del self.equipped_model
            if self.experimental_model is not None:
                del self.experimental_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Initialize model
            self.logger.info("Initializing model...")
            self.original_model = DCAE_HF.from_pretrained(model_path_or_name)
            
            # Move model to device and set to eval mode
            self.logger.info(f"Moving model to device: {self.device}")
            self.original_model = self.original_model.to(self.device)
            self.original_model.eval()
            
            # Set as equipped model
            self.logger.info("Creating equipped model copy...")
            self.equipped_model = copy.deepcopy(self.original_model)
            self.equipped_model.eval()
            
            # Get initial metrics and save visualizations
            self.logger.info("Computing metrics and saving visualizations...")
            metrics = self._get_model_metrics(self.original_model)
            self._save_reconstructions(self.original_model, "initial")
            self._save_weight_distribution(self.original_model, "initial")
            
            if torch.cuda.is_available():
                self.logger.info(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error loading initial model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def create_experimental_model(self) -> None:
        """Create a new experimental model from the currently equipped model."""
        try:
            self.logger.info("Creating new experimental model from equipped model")
            
            # Clear any existing experimental model
            if self.experimental_model is not None:
                del self.experimental_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create new experimental model
            self.experimental_model = copy.deepcopy(self.equipped_model)
            self.experimental_model.eval()
            
            if torch.cuda.is_available():
                self.logger.info(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                
        except Exception as e:
            self.logger.error(f"Error creating experimental model: {str(e)}\n{traceback.format_exc()}")
            raise

    def prune_experimental_model(self, sparsity: float) -> Dict[str, Any]:
        """
        Prune the experimental model to the specified sparsity.
        
        Args:
            sparsity: Target sparsity ratio (0.0 to 1.0)
            
        Returns:
            Dict containing updated model metrics
        """
        try:
            if self.experimental_model is None:
                self.create_experimental_model()
                
            self.logger.info(f"Pruning experimental model to {sparsity*100}% sparsity")
            
            # Prune the model
            self.experimental_model = fine_grained_prune(self.experimental_model, sparsity)
            
            # Get metrics after pruning
            metrics = self._get_model_metrics(self.experimental_model)
            
            # Save visualizations
            self._save_reconstructions(self.experimental_model, "after_pruning")
            self._save_weight_distribution(self.experimental_model, "after_pruning")
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error pruning experimental model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def train_experimental_model(self, epochs: int, steps_per_epoch: int) -> Dict[str, Any]:
        """
        Train the experimental model for specified epochs.
        
        Args:
            epochs: Number of epochs to train
            steps_per_epoch: Number of steps per epoch
            
        Returns:
            Dict containing updated model metrics
        """
        try:
            if self.experimental_model is None:
                raise ValueError("No experimental model to train")
                
            self.logger.info(f"Training experimental model: epochs={epochs}, steps={steps_per_epoch}")
            
            # Initialize trainer
            trainer = DCAETrainer(
                path=self.save_dir,
                model=self.experimental_model,
                data_provider=self.dataset_provider
            )
            
            # Update training config
            trainer.config.run_config.n_epochs = epochs
            trainer.config.run_config.steps_per_epoch = steps_per_epoch
            
            # Setup trainer with safe defaults
            trainer.prep_for_training(
                run_config=self.config.get('run_config', {}),
                ema_decay=None,  # EMA is handled by EfficientViT's trainer
                amp=None  # AMP is handled by EfficientViT's trainer
            )
            self.logger.info("Trainer setup complete")
            self.logger.info("Starting training...")

            # Train the model
            trainer.train()
            
            # Save visualizations
            self._save_reconstructions(self.experimental_model, "after_training")
            self._save_weight_distribution(self.experimental_model, "after_training")
            
            return self._get_model_metrics(self.experimental_model)
        except Exception as e:
            self.logger.error(f"Error training experimental model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def equip_experimental_model(self) -> Dict[str, Any]:
        """
        Promote the experimental model to equipped status.
        
        Returns:
            Dict containing updated model metrics
        """
        try:
            if self.experimental_model is None:
                raise ValueError("No experimental model to equip")
                
            self.logger.info("Equipping experimental model")
            
            # Clear current equipped model
            if self.equipped_model is not None:
                del self.equipped_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Update equipped model
            self.equipped_model = copy.deepcopy(self.experimental_model)
            self.equipped_model.eval()
            
            # Clear experimental model
            del self.experimental_model
            self.experimental_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Save visualizations of newly equipped model
            metrics = self._get_model_metrics(self.equipped_model)
            self._save_reconstructions(self.equipped_model, "equipped")
            self._save_weight_distribution(self.equipped_model, "equipped")
            
            if torch.cuda.is_available():
                self.logger.info(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error equipping experimental model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _calculate_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate reconstruction and perceptual losses"""
        try:
            def normalize_for_lpips(x):
                return (x.clamp(-1, 1) + 1) / 2
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, original)
            
            # Perceptual loss (LPIPS)
            images_norm = normalize_for_lpips(original)
            recon_norm = normalize_for_lpips(reconstructed)
            perceptual_loss = self.lpips(images_norm, recon_norm)
            
            # Total loss
            total_loss = recon_loss + 0.1 * perceptual_loss
            
            return {
                "loss": total_loss,
                "recon_loss": recon_loss,
                "perceptual_loss": perceptual_loss,
            }
        except Exception as e:
            self.logger.error(f"Error calculating loss: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _save_reconstructions(self, model: DCAE, step: str):
        """Save reconstruction visualizations"""
        try:
            model.eval()
            with torch.no_grad():
                x = self.sample_batch['data'].to(self.device)
                # Get reconstructions
                latent = model.encode(x)
                y = model.decode(latent)
                
                # Calculate reconstruction loss
                loss = F.mse_loss(y, x).item()
                
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
                
                plt.suptitle(f"{step.capitalize()} Model - MSE Loss: {loss:.4f}")
                plt.tight_layout()
                
                # Save the comparison plot
                save_path = os.path.join(self.save_dir, f"reconstructions_{step}.png")
                plt.savefig(save_path)
                plt.close()
                
                # Clear memory
                del x, y, latent
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                self.logger.info(f"Saved reconstruction comparison to {save_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving reconstructions: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_weight_distribution(self, model: DCAE, step: str):
        """Save weight distribution plots"""
        try:
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
            
            # Create weight distribution plot
            plt.figure(figsize=(10, 6))
            weights = []
            for name, param in model.named_parameters():
                if param.dim() > 1:  # Skip 1D tensors like biases
                    weights.extend(param.data.cpu().numpy().flatten())
            
            plt.hist(weights, bins=100, density=True, alpha=0.7)
            plt.title(f'Weight Distribution - {step} (Sparsity: {1 - total_nonzero/total_params:.1%})')
            plt.xlabel('Weight Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            save_path = os.path.join(self.save_dir, f"weight_dist_{step}.png")
            plt.savefig(save_path)
            plt.close()
            
            # Clear memory
            del weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info(f"Saved weight distribution to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving weight distribution: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _get_model_metrics(self, model: DCAE) -> Dict[str, Any]:
        """Get model metrics including sparsity and reconstruction loss"""
        try:
            # Setup dataset if needed (only when computing metrics)
            self._setup_dataset_if_needed()
            
            metrics = {}
            
            # Calculate sparsity
            total_params = 0
            zero_params = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    total_params += param.numel()
                    zero_params += (param.data.abs() < 1e-6).sum().item()
            metrics['sparsity'] = zero_params / total_params if total_params > 0 else 0
            
            # Calculate reconstruction loss
            model.eval()
            with torch.no_grad():
                images = self.sample_batch['data'].to(self.device)
                latent = model.encode(images)
                reconstructions = model.decode(latent)
                loss = F.mse_loss(reconstructions, images)
                metrics['reconstruction_loss'] = loss.item()
                
                # Clear memory
                del images, latent, reconstructions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return metrics
        except Exception as e:
            self.logger.error(f"Error computing model metrics: {str(e)}\n{traceback.format_exc()}")
            raise
