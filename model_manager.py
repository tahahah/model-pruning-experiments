import copy
import logging
import os
from typing import Dict, Optional, Tuple, Any
import itertools
import yaml
import traceback

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
import matplotlib.pyplot as plt
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
        try:
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # Initialize logger
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Setup device and seed
            self.device = setup_dist_env("0") # TODO: Make configurable to multiple gpus
            set_random_seed(42)  # TODO: Make configurable
            
            # Model states
            self.original_model: Optional[DCAE] = None
            self.equipped_model: Optional[DCAE] = None
            self.experimental_model: Optional[DCAE] = None
            
            # Initialize dataset
            self._setup_dataset()
            
            # Get a random sample batch for visualization
            random_index = torch.randint(0, 100, (1,)).item()
            try:
                self.sample_batch = next(itertools.islice(self.dataset_provider.valid, random_index, None))
                if not isinstance(self.sample_batch, dict) or 'data' not in self.sample_batch:
                    self.logger.error("Invalid sample batch format. Expected dict with 'data' key.")
                    # Create a default sample batch with random data
                    self.sample_batch = {
                        'data': torch.randn(1, 3, 512, 512)  # B=1, C=3, H=512, W=512
                    }
            except Exception as e:
                self.logger.error(f"Error getting sample batch: {str(e)}\n{traceback.format_exc()}")
                # Create a default sample batch with random data
                self.sample_batch = {
                    'data': torch.randn(1, 3, 512, 512)  # B=1, C=3, H=512, W=512
                }
            
            # Initialize LPIPS for loss calculation
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        except Exception as e:
            self.logger.error(f"Error initializing ModelManager: {str(e)}\n{traceback.format_exc()}")
            raise
        
    def _setup_dataset(self):
        """Initialize the dataset provider with configuration"""
        try:
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
        except Exception as e:
            self.logger.error(f"Error setting up dataset: {str(e)}\n{traceback.format_exc()}")
            raise
        
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
            # Get sample batch - assuming it's a dictionary with 'data' key
            images = self.sample_batch['data'].to(self.device)
            
            # Ensure model is in eval mode
            model.eval()
            
            # Get reconstructions
            with torch.no_grad():
                try:
                    self.logger.info("Computing latent representation...")
                    latent = model.encode(images)
                    self.logger.info("Decoding reconstruction...")
                    reconstructions = model.decode(latent)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger.error("CUDA out of memory during reconstruction")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError("GPU out of memory. Try reducing batch size or image size.")
                    raise
            
            # Move tensors to CPU before visualization
            images = images.cpu()
            reconstructions = reconstructions.cpu()
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # Changed to 2x1 since we only have 1 image
            fig.suptitle(f"Original vs Reconstructed Images - {step}")
            
            # Original
            orig = images[0].permute(1, 2, 0).numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min())  # Normalize to [0,1]
            axes[0].imshow(orig)
            axes[0].axis('off')
            axes[0].set_title('Original')
            
            # Reconstruction
            recon = reconstructions[0].permute(1, 2, 0).numpy()
            recon = (recon - recon.min()) / (recon.max() - recon.min())  # Normalize to [0,1]
            axes[1].imshow(recon)
            axes[1].axis('off')
            axes[1].set_title('Reconstructed')
            
            # Save figure
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"reconstructions_{step}.png")
            plt.savefig(save_path)
            plt.close()
            
            # Clear memory
            del images, reconstructions, latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            self.logger.error(f"Error saving reconstructions: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _save_weight_distribution(self, model: DCAE, step: str):
        """Save weight distribution plots"""
        try:
            weights = []
            for name, param in model.named_parameters():
                if 'weight' in name:
                    weights.extend(param.data.cpu().numpy().flatten())
            
            plt.figure(figsize=(10, 6))
            plt.hist(weights, bins=100, density=True)
            plt.title(f'Weight Distribution - {step}')
            plt.xlabel('Weight Value')
            plt.ylabel('Density')
            
            # Save figure
            save_path = os.path.join(self.save_dir, f"weight_dist_{step}.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error saving weight distribution: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _get_model_metrics(self, model: DCAE) -> Dict[str, Any]:
        """Get comprehensive metrics for a model"""
        try:
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
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}\n{traceback.format_exc()}")
            raise
