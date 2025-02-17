from efficientvit.apps.trainer import Trainer
from efficientvit.apps.trainer.run_config import RunConfig
from efficientvit.apps.utils import dist_barrier, is_master
from efficientvit.apps.utils.metric import AverageMeter
from efficientvit.models.efficientvit.dc_ae import DCAE
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from efficientvit.apps.metrics.psnr.psnr import PSNRStats, PSNRStatsConfig
import torch
import torch.nn.functional as F
from typing import Any, Dict
import os
from tqdm import tqdm
import torchvision
from contextlib import nullcontext
import logging

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

class DCAERunConfig(RunConfig):
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 perceptual_weight: float = 0.1,
                 save_interval: int = 5,
                 eval_interval: int = 1,
                 log_interval: int = 100,
                 steps_per_epoch: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        # Model specific parameters
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        
        # Training intervals
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.steps_per_epoch = steps_per_epoch

def log_gpu_memory(msg=""):
    """Log GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n[GPU Memory] {msg}")
        print(f"  Allocated: {allocated:.1f}MB")
        print(f"  Reserved:  {reserved:.1f}MB")
        print(f"  Peak:      {max_allocated:.1f}MB")

class DCAETrainer(Trainer):
    def __init__(self, path: str, model: DCAE, data_provider):
        super().__init__(path, model, data_provider)
        # Initialize metrics (but keep on CPU initially)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0))
        self.psnr_stats = PSNRStats(PSNRStatsConfig())
        
        # Convert to half precision to save memory
        if torch.cuda.is_available():
            self.lpips = self.lpips.half().cuda()
            self.ssim = self.ssim.half().cuda()
        
        # Initialize best validation score
        self.best_val = float('inf')
        log_gpu_memory("After trainer init")

    def normalize_for_lpips(self, x):
        """Normalize tensor to [0,1] range for LPIPS"""
        if x.dtype != self.lpips.dtype:
            x = x.to(self.lpips.dtype)
        return (x.clamp(-1, 1) + 1) / 2

    def log_images(self, images, reconstructed, prefix="train", step=0):
        """Helper function to log original and reconstructed images"""
        if wandb_available and wandb.run is not None:
            # Convert images from [-1,1] to [0,1] range
            images = (images + 1) / 2
            reconstructed = (reconstructed + 1) / 2
            
            # Create a grid of original and reconstructed images side by side
            n_samples = min(4, images.size(0))  # Log up to 4 images
            image_grid = []
            for i in range(n_samples):
                pair = torch.cat([images[i:i+1], reconstructed[i:i+1]], dim=-1)
                image_grid.append(pair)
            image_grid = torch.cat(image_grid, dim=-2)  # Stack vertically
            
            wandb.log({
                f"{prefix}/reconstructions": wandb.Image(
                    image_grid, 
                    caption=f"Left: Original, Right: Reconstructed (Step {step})"
                )
            })

    def _validate(self, model, data_loader, epoch) -> Dict[str, Any]:
        model.eval()
        val_loss = AverageMeter(is_distributed=False)
        val_recon_loss = AverageMeter(is_distributed=False)
        val_perceptual_loss = AverageMeter(is_distributed=False)
        val_psnr = AverageMeter(is_distributed=False)
        val_ssim = AverageMeter(is_distributed=False)
        val_lpips = AverageMeter(is_distributed=False)
        
        with torch.no_grad():
            # Limit validation steps
            max_val_steps = self.data_provider.cfg.val_steps
            val_iterator = iter(data_loader)
            
            with tqdm(total=max_val_steps, desc=f"Validation Epoch #{epoch}") as t:
                for batch_id in range(max_val_steps):
                    try:
                        feed_dict = next(val_iterator)
                    except StopIteration:
                        break
                        
                    images = feed_dict["data"].cuda()
                    
                    # Forward pass
                    with torch.amp.autocast(device_type='cuda') if self.enable_amp else nullcontext():
                        # Get the actual model from DDP wrapper if needed
                        model_unwrapped = model.module if hasattr(model, 'module') else model
                        
                        encoded = model_unwrapped.encode(images)
                        reconstructed = model_unwrapped.decode(encoded)
                        
                        # Calculate losses
                        recon_loss = F.mse_loss(reconstructed, images)
                        
                        # Normalize images for LPIPS
                        images_norm = self.normalize_for_lpips(images)
                        recon_norm = self.normalize_for_lpips(reconstructed)
                        perceptual_loss = self.lpips(images_norm, recon_norm)
                        
                        total_loss = (self.run_config.reconstruction_weight * recon_loss + 
                                    self.run_config.perceptual_weight * perceptual_loss)
                        
                        # Calculate metrics
                        images_uint8 = (255 * ((images + 1) / 2) + 0.5).clamp(0, 255).to(torch.uint8)
                        recon_uint8 = (255 * ((reconstructed + 1) / 2) + 0.5).clamp(0, 255).to(torch.uint8)
                        
                        ssim_val = self.ssim(images_uint8, recon_uint8)
                        self.psnr_stats.add_data(images_uint8, recon_uint8)
                        psnr_val = self.psnr_stats.compute()
                        
                    # Update metrics
                    val_loss.update(total_loss.item(), images.size(0))
                    val_recon_loss.update(recon_loss.item(), images.size(0))
                    val_perceptual_loss.update(perceptual_loss.item(), images.size(0))
                    val_psnr.update(psnr_val, images.size(0))
                    val_ssim.update(ssim_val.item(), images.size(0))
                    val_lpips.update(perceptual_loss.item(), images.size(0))
                    
                    # Log validation metrics
                    self.write_metric(
                        {
                            "val/loss": total_loss.item(),
                            "val/recon_loss": recon_loss.item(),
                            "val/perceptual_loss": perceptual_loss.item(),
                            "val/psnr": psnr_val,
                            "val/ssim": ssim_val.item(),
                            "val/lpips": perceptual_loss.item(),
                            "val/epoch": epoch,
                        },
                        "val",
                    )
                    
                    # Log validation images for first batch
                    if batch_id == 0:
                        self.log_images(
                            images, 
                            reconstructed,
                            prefix="val",
                            step=epoch
                        )
                    
                    # Update progress bar
                    t.set_postfix({
                        'loss': val_loss.avg,
                        'recon_loss': val_recon_loss.avg,
                        'perceptual_loss': val_perceptual_loss.avg,
                        'psnr': val_psnr.avg,
                        'ssim': val_ssim.avg,
                        'lpips': val_lpips.avg
                    })
                    t.update()
                
                    images.cpu()
                    feed_dict["data"].cpu()
                    del encoded
                    del reconstructed
        
        metrics = {
            "val/loss": val_loss.avg,
            "val/recon_loss": val_recon_loss.avg,
            "val/perceptual_loss": val_perceptual_loss.avg,
            "val/psnr": val_psnr.avg,
            "val/ssim": val_ssim.avg,
            "val/lpips": val_lpips.avg
        }
        
        return metrics
        
    def run_step(self, feed_dict):
        images = feed_dict["data"]
        log_gpu_memory("Before forward pass")
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16) if self.enable_amp else nullcontext():
            # Get the actual model from DDP wrapper if needed
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Forward pass through DCAE in chunks if needed
            batch_size = images.size(0)
            if batch_size > 2 and torch.cuda.get_device_properties(0).total_memory < 4 * 1024**3:  # Less than 4GB
                # Process in chunks of 2 to save memory
                encoded_chunks = []
                for i in range(0, batch_size, 2):
                    chunk = images[i:i+2].cuda()
                    with torch.no_grad():
                        encoded_chunks.append(model.encode(chunk).cpu())
                    del chunk
                    torch.cuda.empty_cache()
                encoded = torch.cat(encoded_chunks, dim=0).cuda()
                del encoded_chunks
            else:
                encoded = model.encode(images)
            
            # Decode
            reconstructed = model.decode(encoded)
            del encoded
            log_gpu_memory("After decode")
            
            # Move images to same device and dtype as reconstructed
            if images.device != reconstructed.device or images.dtype != reconstructed.dtype:
                images = images.to(device=reconstructed.device, dtype=reconstructed.dtype)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, images)
            
            # Compute LPIPS with memory optimization
            with torch.cuda.amp.autocast():  # Use FP16 for LPIPS
                images_norm = self.normalize_for_lpips(images)
                recon_norm = self.normalize_for_lpips(reconstructed)
                perceptual_loss = self.lpips(images_norm, recon_norm)
                del images_norm, recon_norm
            log_gpu_memory("After LPIPS")
            
            # Total loss
            total_loss = (self.run_config.reconstruction_weight * recon_loss + 
                         self.run_config.perceptual_weight * perceptual_loss)
            
        result = {
            "loss": total_loss,  # Keep on GPU for backward
            "recon_loss": recon_loss.detach().cpu(),
            "perceptual_loss": perceptual_loss.detach().cpu(),
            "reconstructed": reconstructed.detach().cpu() if self.run_config.log_interval > 0 else None
        }
        
        # Cleanup
        del images, reconstructed
        torch.cuda.empty_cache()
        log_gpu_memory("After cleanup")
        
        return result
        
    def _train_one_epoch(self, epoch):
        train_loss = AverageMeter(is_distributed=False)
        train_recon_loss = AverageMeter(is_distributed=False)
        train_perceptual_loss = AverageMeter(is_distributed=False)
        
        with tqdm(total=self.run_config.steps_per_epoch, desc=f"Training Epoch #{epoch}") as t:
            for step, feed_dict in enumerate(self.data_provider.train):
                if step >= self.run_config.steps_per_epoch:
                    break
                    
                log_gpu_memory(f"Start of step {step}")
                feed_dict = self.before_step(feed_dict)
                self.optimizer.zero_grad()
                
                output_dict = self.run_step(feed_dict)
                log_gpu_memory(f"After run_step {step}")
                
                # Scale loss and backward
                self.scaler.scale(output_dict["loss"]).backward()
                log_gpu_memory(f"After backward {step}")
                
                # Update metrics
                train_loss.update(output_dict["loss"].item(), feed_dict["data"].size(0))
                train_recon_loss.update(output_dict["recon_loss"], feed_dict["data"].size(0))
                train_perceptual_loss.update(output_dict["perceptual_loss"], feed_dict["data"].size(0))
                
                # Log training metrics and images
                if step % self.run_config.log_interval == 0:
                    self.write_metric(
                        {
                            "train/loss": output_dict["loss"].item(),
                            "train/recon_loss": output_dict["recon_loss"],
                            "train/perceptual_loss": output_dict["perceptual_loss"],
                            "train/lr": self.optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                            "train/step": step + epoch * self.run_config.steps_per_epoch,
                        },
                        "train",
                    )
                    
                    self.log_images(
                        feed_dict["data"].cpu(), 
                        output_dict["reconstructed"],
                        prefix="train",
                        step=step + epoch * self.run_config.steps_per_epoch
                    )
                
                # Optimizer step
                self.after_step()
                del output_dict["loss"]  # Free GPU tensor
                
                # Move feed_dict to CPU and clear CUDA cache
                if feed_dict is not None:
                    for key in feed_dict:
                        if isinstance(feed_dict[key], torch.Tensor):
                            feed_dict[key] = feed_dict[key].cpu()
                torch.cuda.empty_cache()
                log_gpu_memory(f"End of step {step}")
                
                # Update progress bar
                t.set_postfix({
                    'loss': train_loss.avg,
                    'recon_loss': train_recon_loss.avg,
                    'perceptual_loss': train_perceptual_loss.avg,
                    'lr': self.optimizer.param_groups[0]["lr"]
                })
                t.update()
        
        metrics = {
            "train/loss": train_loss.avg,
            "train/recon_loss": train_recon_loss.avg,
            "train/perceptual_loss": train_perceptual_loss.avg,
            "train/lr": self.optimizer.param_groups[0]["lr"]
        }
        
        return metrics
        
    def train(self):
        for epoch in range(self.start_epoch, self.run_config.n_epochs):
            train_info = self.train_one_epoch(epoch)
            
            # Run validation if needed
            if (epoch + 1) % self.run_config.eval_interval == 0:
                val_info = self.validate(epoch=epoch)
                
                # Save best model
                if val_info["val/loss"] < self.best_val:
                    self.best_val = val_info["val/loss"]
                    self.save_model(epoch=epoch, model_name="best.pt")
            
            # Regular checkpoint
            if (epoch + 1) % self.run_config.save_interval == 0:
                self.save_model(epoch=epoch, model_name=f"epoch_{epoch}.pt")
                
            # Log training progress
            if is_master():
                log_str = f"Epoch {epoch}: "
                log_str += f"train_loss={train_info['train/loss']:.4f}"
                if (epoch + 1) % self.run_config.eval_interval == 0:
                    log_str += f", val_loss={val_info['val/loss']:.4f}"
                    log_str += f", val_psnr={val_info['val/psnr']:.2f}"
                    log_str += f", val_ssim={val_info['val/ssim']:.4f}"
                    log_str += f", val_lpips={val_info['val/lpips']:.4f}"
                self.write_log(log_str)

    def write_metric(self, metric_dict: Dict[str, Any], metric_type: str):
        """Override write_metric to ensure proper wandb logging"""
        if wandb_available and wandb.run is not None:
            wandb.log(metric_dict)