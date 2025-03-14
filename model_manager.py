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
import yaml
import itertools
import torchvision
import torchvision.transforms as transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from PIL import Image
from dcaecore.train_dc_ae import setup_dist_env, set_random_seed, get_dist_size, get_dist_rank
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.models.efficientvit.dc_ae import DCAE
from dcaecore.trainer import DCAETrainer, DCAERunConfig
from dcaecore.pacman_dataset_copy import SimplePacmanDatasetProvider, PacmanDatasetProviderConfig
from vae_pruning_analysis import fine_grained_prune, analyze_weight_distribution, plot_weight_distributions
import huggingface_hub
from dotenv import load_dotenv
import time
from diffusers import AutoencoderTiny

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
        self.active_model = None  # Tracks which model is currently in VRAM
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
                    
            random_index = torch.randint(0, 100, (1,)).item()
            self.sample_batch = next(itertools.islice(self.dataset_provider.valid, random_index, None))
            self.logger.info("Dataset provider setup complete")

    def _ensure_model_on_device(self, model_name: str) -> None:
        """Ensure the specified model is loaded on device, moving other models to CPU if needed"""
        if self.active_model == model_name:
            return
            
        # First move current active model to CPU if exists
        if self.active_model is not None:
            current_model = getattr(self, self.active_model)
            if current_model is not None:
                current_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Now move requested model to device
        target_model = getattr(self, model_name)
        if target_model is not None:
            target_model.to(self.device)
            self.active_model = model_name
            
    def load_initial_model(self, model_path_or_name: str) -> Dict[str, Any]:
        """
        Load initial model and set it as both original and equipped model.
        
        Args:
            model_path_or_name: Path to model or HuggingFace model name. For AutoencoderTiny,
                              use format "tiny:<model_name>" e.g. "tiny:madebyollin/taesd"
            
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
            if model_path_or_name.startswith("tiny:"):
                # Load AutoencoderTiny model
                model_name = model_path_or_name[5:]  # Remove "tiny:" prefix
                base_model = AutoencoderTiny.from_pretrained(model_name)
                self.original_model = AutoencoderTinyWrapper(base_model)
            else:
                # Load DCAE model
                self.original_model = DCAE_HF.from_pretrained(model_path_or_name)
            
            # Move model to device and set to eval mode
            self.logger.info(f"Moving model to device: {self.device}")
            self._ensure_model_on_device("original_model")
            self.original_model.eval()
            
            # Move original model to CPU before creating equipped copy
            self.original_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Set as equipped model (creates copy on CPU)
            self.logger.info("Creating equipped model copy...")
            self.equipped_model = copy.deepcopy(self.original_model)
            self.equipped_model.eval()
            
            # Move equipped model to device and make it active
            self._ensure_model_on_device("equipped_model")
            
            # Get initial metrics and save visualizations
            self.logger.info("Computing metrics and saving visualizations...")
            metrics = self._get_model_metrics(self.original_model, save_reconstructions=True, step="initial")
            
            # Save initial visualizations
            self._save_weight_distribution(self.original_model, "initial")
            
            # Copy initial visualizations to equipped filenames
            import shutil
            recon_src = os.path.join(self.save_dir, "reconstruction_initial.png")
            recon_dst = os.path.join(self.save_dir, "reconstruction_equipped.png")
            weight_src = os.path.join(self.save_dir, "weight_dist_initial.png")
            weight_dst = os.path.join(self.save_dir, "weight_dist_equipped.png")
            
            shutil.copy2(recon_src, recon_dst)
            shutil.copy2(weight_src, weight_dst)
            self.logger.info("Copied initial visualizations to equipped filenames")
            
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
            
            # Ensure equipped model is on device for deepcopy
            self._ensure_model_on_device("equipped_model")
            
            # Create new experimental model
            self.experimental_model = copy.deepcopy(self.equipped_model)
            self.experimental_model.eval()
            
            # Move equipped model back to CPU and make experimental model active
            self.equipped_model.cpu()
            self.active_model = "experimental_model"
            
            if torch.cuda.is_available():
                self.logger.info(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
                
        except Exception as e:
            self.logger.error(f"Error creating experimental model: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def create_run_config(self, config: dict) -> DCAERunConfig:
        """Create RunConfig from configuration dictionary."""
        # Get model config for loss weights
        model_cfg = config.get('model', {})
        run_cfg = config.get('run_config', {})
        
        run_config = DCAERunConfig(
            # Required RunConfig parameters
            n_epochs=run_cfg['n_epochs'],
            init_lr=run_cfg['init_lr'],
            warmup_epochs=run_cfg['warmup_epochs'],
            warmup_lr=run_cfg['warmup_lr'],
            lr_schedule_name=run_cfg['lr_schedule_name'],
            lr_schedule_param=run_cfg['lr_schedule_param'],
            optimizer_name=run_cfg['optimizer_name'],
            optimizer_params=run_cfg['optimizer_params'],
            weight_decay=run_cfg['weight_decay'],
            no_wd_keys=run_cfg['no_wd_keys'],
            grad_clip=run_cfg['grad_clip'],
            reset_bn=run_cfg['reset_bn'],
            reset_bn_size=run_cfg['reset_bn_size'],
            reset_bn_batch_size=run_cfg['reset_bn_batch_size'],
            eval_image_size=run_cfg['eval_image_size'],
            
            # DCAE specific parameters
            reconstruction_weight=model_cfg.get('reconstruction_weight', 1.0),
            perceptual_weight=model_cfg.get('perceptual_weight', 0.1)
        )
        
        return run_config

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
            
            # Move model to CPU for pruning to avoid duplicate GPU memory usage
            self.experimental_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Prune the model (creates a new model instance)
            pruned_model = fine_grained_prune(self.experimental_model, sparsity)
            
            # Clean up old model
            del self.experimental_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Update reference and mark as active
            self.experimental_model = pruned_model
            self.active_model = "experimental_model"
            
            # Move to device for metrics computation
            self.experimental_model.to(self.device)
            
            # Get metrics after pruning
            metrics = self._get_model_metrics(self.experimental_model, save_reconstructions=True, step="after_pruning")
            self._save_weight_distribution(self.experimental_model, "after_pruning")
            
            # Move back to CPU to free up memory
            self.experimental_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
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
            
            # Move model to CPU before trainer initialization
            self.experimental_model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(f"Initial VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # Configure run config first
            run_config = self.create_run_config(self.config)
            run_config.steps_per_epoch = steps_per_epoch
            run_config.n_epochs = epochs
            
            # Initialize trainer with minimal logging
            trainer = DCAETrainer(
                path=self.save_dir,
                model=self.experimental_model,
                data_provider=self.dataset_provider
            )
            
            # Configure trainer with minimal logging
            trainer.write_train_log = False  # Reduce memory from logging
            trainer.write_val_log = False
            trainer.log_interval = 0  # Disable image logging during training
            
            # Setup trainer with optimized defaults
            trainer.prep_for_training(
                run_config=run_config,
                ema_decay=None,
                amp=True  # Enable AMP for memory efficiency
            )
            
            # Train model
            trainer.train()
            
            # Move model back to CPU and clear GPU memory
            self.experimental_model.cpu()
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info(f"Final VRAM usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            
            # Get metrics after training
            metrics = self._get_model_metrics(self.experimental_model, save_reconstructions=True, step="after_training")
            self._save_weight_distribution(self.experimental_model, "after_training")
            
            return metrics
            
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
            self.logger.info("Promoting experimental model to equipped status")
            
            if self.experimental_model is None:
                raise ValueError("No experimental model exists to equip")
            
            # Clear current equipped model if it exists
            if self.equipped_model is not None:
                del self.equipped_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Ensure experimental model is on device
            self._ensure_model_on_device("experimental_model")
            
            # Update equipped model
            self.equipped_model = copy.deepcopy(self.experimental_model)
            self.equipped_model.eval()
            
            # Clear experimental model
            del self.experimental_model
            self.experimental_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Make equipped model active
            self.active_model = "equipped_model"
            
            # Save visualizations of newly equipped model
            metrics = self._get_model_metrics(self.equipped_model, save_reconstructions=True, step="equipped")
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

    def _get_model_metrics(self, model: DCAE, save_reconstructions: bool = False, step: str = "") -> Dict[str, Any]:
        """Get model metrics including sparsity and reconstruction loss
        
        Args:
            model: The model to evaluate
            save_reconstructions: If True, save reconstruction visualizations
            step: Step identifier for saving reconstructions (e.g. "initial", "after_training")
            
        Returns:
            Dict containing model metrics
        """
        try:
            # Setup dataset if not already done
            self._setup_dataset_if_needed()
            
            # Find which model this is and ensure it's on device
            model_name = None
            if model is self.original_model:
                model_name = "original_model"
            elif model is self.equipped_model:
                model_name = "equipped_model"
            elif model is self.experimental_model:
                model_name = "experimental_model"
            
            if model_name is None:
                raise ValueError("_get_model_metrics called with untracked model")
                
            self._ensure_model_on_device(model_name)
            
            # Get model parameters and sparsity
            total_params, nonzero_params = analyze_weight_distribution(model)
            sparsity_ratio = 1 - (nonzero_params / total_params)
            
            # Calculate memory usage
            param_memory = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
            buffer_memory = sum(b.nelement() * b.element_size() for b in model.buffers()) / (1024 * 1024)  # MB
            if torch.cuda.is_available():
                vram_usage = f"{param_memory + buffer_memory:.1f} MB"
                torch.cuda.empty_cache()  # Clear unused memory
            else:
                vram_usage = "N/A (CPU)"
            
            # Do inference once and measure everything
            images = self.sample_batch['data'].to(self.device)
            start_time = time.time()
            with torch.no_grad():
                latent = model.encode(images)
                reconstructed = model.decode(latent)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            # Calculate losses
            loss_dict = self._calculate_loss(images, reconstructed)
            loss = loss_dict['loss']
            recon_loss = loss_dict['recon_loss']
            perceptual_loss = loss_dict['perceptual_loss']
            
            # Save reconstructions if requested
            if save_reconstructions and step:
                self._save_reconstruction_batch(images, reconstructed, step, loss)
            
            # Clean up
            del images, latent, reconstructed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return {
                "total_params": f"{total_params:,} weights",
                "nonzero_params": f"{nonzero_params:,} weights",
                "sparsity_ratio": f"{sparsity_ratio:.1%}",
                "vram_usage": vram_usage,
                "latency": f"{latency:.1f} ms",
                "reconstruction_loss": f"{recon_loss:.4f} MSE",
                "perceptual_loss": f"{perceptual_loss:.4f} LPIPS"
            }
        except Exception as e:
            self.logger.error(f"Error computing model metrics: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_reconstruction_batch(self, original: torch.Tensor, reconstructed: torch.Tensor, step: str, loss = -1):
        """Save reconstruction visualizations for a batch"""
        try:
            # Save reconstruction comparison
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original[0].cpu().permute(1, 2, 0))
            axes[0].set_title("Original")
            axes[0].axis("off")
            axes[1].imshow(reconstructed[0].cpu().permute(1, 2, 0))
            axes[1].set_title("Reconstructed")
            axes[1].axis("off")
            plt.suptitle(f"Reconstruction Comparison - {step} | Weighted Loss: {loss:.4f}")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"reconstruction_{step}.png"))
            plt.close()
        except Exception as e:
            self.logger.error(f"Error saving reconstructions: {str(e)}\n{traceback.format_exc()}")
            raise

    def _save_weight_distribution(self, model: DCAE, step: str):
        """Save weight distribution plots in a memory-efficient way"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Define bins once for consistency
            bins = np.linspace(-0.5, 0.5, 100)
            
            # Initialize histogram arrays
            hist_counts = np.zeros_like(bins[:-1], dtype=np.float64)
            total_weights = 0
            total_nonzero = 0
            
            # Process weights layer by layer
            for name, param in model.named_parameters():
                if param.dim() > 1:  # Skip 1D tensors like biases
                    # Get layer weights and update counts
                    weights = param.data.cpu().numpy()
                    total_weights += weights.size
                    total_nonzero += torch.count_nonzero(param).item()
                    
                    # Update histogram counts
                    hist, _ = np.histogram(weights.ravel(), bins=bins, density=False)
                    hist_counts += hist
                    
                    # Clear memory
                    del weights
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Normalize histogram
            if total_weights > 0:
                hist_counts = hist_counts / total_weights
            
            # Plot histogram
            plt.stairs(hist_counts, bins, alpha=0.7)
            sparsity = 1 - (total_nonzero / total_weights) if total_weights > 0 else 0
            plt.title(f'Weight Distribution - {step} (Sparsity: {sparsity:.1%})')
            plt.xlabel('Weight Value')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
            
            # Save figure
            save_path = os.path.join(self.save_dir, f"weight_dist_{step}.png")
            plt.savefig(save_path)
            plt.close()
            
            self.logger.info(f"Saved weight distribution to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving weight distribution: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def upload_to_huggingface(self) -> bool:
        """Upload equipped model to HuggingFace Hub."""
        if self.equipped_model is None:
            self.logger.error("No equipped model to upload")
            return False

        load_dotenv()  # Load environment variables from .env file
        
        if "HF_TOKEN" not in os.environ:
            self.logger.error("HF_TOKEN not found in environment variables")
            return False
            
        try:
            # Save model to temporary file
            file_path = os.path.join(self.save_dir, "equipped_model.pth")
            torch.save(self.equipped_model.state_dict(), file_path)
            
            huggingface_hub.login(token=os.environ["HF_TOKEN"])
            repo_id = "Tahahah/PrunedPacmanDCAE"
            
            try:
                huggingface_hub.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"checkpoints/{os.path.basename(file_path)}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                self.logger.info(f"Uploaded checkpoint to HuggingFace: {repo_id}")
                return True
            except huggingface_hub.utils.RepositoryNotFoundError:
                huggingface_hub.create_repo(repo_id, repo_type="model")
                huggingface_hub.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"checkpoints/{os.path.basename(file_path)}",
                    repo_id=repo_id,
                    repo_type="model"
                )
                self.logger.info(f"Created repo and uploaded checkpoint to HuggingFace: {repo_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to upload checkpoint to HuggingFace: {str(e)}\n{traceback.format_exc()}")
            return False

    def move_model_to_device(self, model_name: str, device: str) -> None:
        """Move a specific model to the specified device."""
        try:
            if model_name not in ["original_model", "equipped_model", "experimental_model"]:
                raise ValueError(f"Invalid model name: {model_name}")
                
            model = getattr(self, model_name)
            if model is None:
                self.logger.warning(f"{model_name} does not exist")
                return
                
            if device == "cuda" and not torch.cuda.is_available():
                self.logger.warning("CUDA not available, using CPU instead")
                device = "cpu"
                
            model.to(device)
            if device == "cuda":
                # Ensure the encoder and decoder are on the right device for AutoencoderTiny
                if isinstance(model, AutoencoderTinyWrapper):
                    model.model.encoder.to(device)
                    model.model.decoder.to(device)
            self.active_model = model_name
            self.logger.info(f"Moved {model_name} to {device}")
            
        except Exception as e:
            self.logger.error(f"Error moving model to device: {str(e)}\n{traceback.format_exc()}")
            raise
    
class AutoencoderTinyWrapper(nn.Module):
    """Wrapper for AutoencoderTiny to make it compatible with our interface"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def encode(self, x):
        return self.model.encoder(x)
        
    def decode(self, x):
        return self.model.decoder(x).clamp(0, 1)
        
    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)
        
    def to(self, device):
        self.model.to(device)
        return self
        
    def cpu(self):
        self.model.cpu()
        return self
        
    def eval(self):
        self.model.eval()
        return self
        
    def parameters(self):
        return self.model.parameters()
        
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix=prefix, recurse=recurse)
        
    def buffers(self):
        return self.model.buffers()
