import copy
import logging
import os
from typing import Dict, Optional, Tuple, Any
import itertools
import yaml

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
from PIL import Image

from dcaecore.train_dc_ae import setup_dist_env, set_random_seed, get_dist_size, get_dist_rank
from diffusers import AutoencoderDC
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
        """
        Initialize the ModelManager with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
            save_dir: Directory to save models and visualizations
        """
        self.config = yaml.safe_load(config_path)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Setup device and seed
        self.device = setup_dist_env("0") # TODO: Make configurable to multiple gpus
        set_random_seed(42)  # TODO: Make configurable
        
        # Model states
        self.original_model: Optional[AutoencoderDC] = None
        self.equipped_model: Optional[AutoencoderDC] = None
        self.experimental_model: Optional[AutoencoderDC] = None
        
        # Initialize dataset
        self._setup_dataset()
        
        # Get a random sample batch for visualization
        random_index = torch.randint(0, 100, (1,)).item()
        self.sample_batch = next(itertools.islice(self.dataset_provider.valid, random_index, None))
        
        # Initialize LPIPS for loss calculation
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        
    def _setup_dataset(self):
        """Initialize the dataset provider with configuration"""
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
        
    def load_initial_model(self, model_path_or_name: str) -> Dict[str, Any]:
        """
        Load the initial model and set it as both original and equipped model.
        
        Args:
            model_path_or_name: Path to model or HuggingFace model name
            
        Returns:
            Dict containing model metrics
        """
        self.logger.info(f"Loading initial model from {model_path_or_name}")
        
        # Initialize model
        self.original_model = AutoencoderDC.from_pretrained(model_path_or_name)
        self.original_model.to(self.device)
        
        # Set as equipped model
        self.equipped_model = copy.deepcopy(self.original_model)
        
        # Get initial metrics and save visualizations
        metrics = self._get_model_metrics(self.original_model)
        self._save_reconstructions(self.original_model, "initial")
        self._save_weight_distribution(self.original_model, "initial")
        
        return metrics
    
    def create_experimental_model(self) -> None:
        """Create a new experimental model from the currently equipped model."""
        self.logger.info("Creating new experimental model from equipped model")
        self.experimental_model = copy.deepcopy(self.equipped_model)
        
    def prune_experimental_model(self, sparsity: float) -> Dict[str, Any]:
        """
        Prune the experimental model to the specified sparsity.
        
        Args:
            sparsity: Target sparsity ratio (0.0 to 1.0)
            
        Returns:
            Dict containing updated model metrics
        """
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
    
    def train_experimental_model(self, epochs: int, steps_per_epoch: int) -> Dict[str, Any]:
        """
        Train the experimental model for specified epochs.
        
        Args:
            epochs: Number of epochs to train
            steps_per_epoch: Number of steps per epoch
            
        Returns:
            Dict containing updated model metrics
        """
        if self.experimental_model is None:
            raise ValueError("No experimental model to train")
            
        self.logger.info(f"Training experimental model: epochs={epochs}, steps={steps_per_epoch}")
        
        # Initialize trainer
        trainer = DCAETrainer(
            model=self.experimental_model,
            config=self.config,
            data_provider=self.dataset_provider
        )
        
        # Update training config
        trainer.config.run_config.n_epochs = epochs
        trainer.config.run_config.steps_per_epoch = steps_per_epoch
        
        # Train the model
        trainer.train()
        
        # Save visualizations
        self._save_reconstructions(self.experimental_model, "after_training")
        self._save_weight_distribution(self.experimental_model, "after_training")
        
        return self._get_model_metrics(self.experimental_model)
    
    def equip_experimental_model(self) -> Dict[str, Any]:
        """
        Promote the experimental model to equipped status.
        
        Returns:
            Dict containing updated model metrics
        """
        if self.experimental_model is None:
            raise ValueError("No experimental model to equip")
            
        self.logger.info("Equipping experimental model")
        
        # Update equipped model
        self.equipped_model = copy.deepcopy(self.experimental_model)
        
        # Clear experimental model
        self.experimental_model = None
        
        # Save visualizations of newly equipped model
        metrics = self._get_model_metrics(self.equipped_model)
        self._save_reconstructions(self.equipped_model, "equipped")
        self._save_weight_distribution(self.equipped_model, "equipped")
        
        return metrics
    
    def _calculate_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate reconstruction and perceptual losses"""
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
    
    def _save_reconstructions(self, model: AutoencoderDC, step: str) -> None:
        """Save reconstruction visualizations"""
        model.eval()
        with torch.no_grad():
            x = self.sample_batch["data"].to(self.device)
            # Get reconstructions
            latent = model.encode(x)
            y = model.decode(latent)
            
            loss_dict = self._calculate_loss(x, y)
            
            # Create visualization
            n_samples = min(4, x.size(0))
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
            
            plt.suptitle(f"{step} - Loss: {loss_dict['loss']:.4f}")
            plt.tight_layout()
            
            # Save the plot
            save_path = os.path.join(self.save_dir, f"reconstructions_{step}.png")
            plt.savefig(save_path)
            plt.close()
    
    def _save_weight_distribution(self, model: AutoencoderDC, step: str) -> None:
        """Save weight distribution plots"""
        save_path = os.path.join(self.save_dir, f"weight_dist_{step}.png")
        plot_weight_distributions(model, save_path, 0.0)  # sparsity param not used for plotting
    
    def _get_model_metrics(self, model: AutoencoderDC) -> Dict[str, Any]:
        """Get comprehensive metrics for a model"""
        metrics = {}
        
        # Get parameter counts and distribution
        total_params, nonzero_params = analyze_weight_distribution(model)
        
        metrics.update({
            "total_params": total_params,
            "nonzero_params": nonzero_params,
            "sparsity_ratio": 1.0 - (nonzero_params / total_params)
        })
        
        # Get VRAM usage
        if torch.cuda.is_available():
            metrics["vram_usage"] = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return metrics
