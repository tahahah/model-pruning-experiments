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
                 **kwargs):
        super().__init__(**kwargs)
        # Model specific parameters
        self.reconstruction_weight = reconstruction_weight
        self.perceptual_weight = perceptual_weight
        
        # Training intervals
        self.save_interval = save_interval
        self.eval_interval = eval_interval

class DCAETrainer(Trainer):
    def __init__(self, path: str, model: DCAE, data_provider):
        super().__init__(path, model, data_provider)
        # Initialize metrics
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0)).cuda()
        self.psnr_stats = PSNRStats(PSNRStatsConfig())
        
        # Initialize best validation score
        self.best_val = float('inf')
        
    def _validate(self, model, data_loader, epoch) -> Dict[str, Any]:
        model.eval()
        val_loss = AverageMeter(is_distributed=False)
        val_recon_loss = AverageMeter(is_distributed=False)
        val_perceptual_loss = AverageMeter(is_distributed=False)
        val_psnr = AverageMeter(is_distributed=False)
        val_ssim = AverageMeter(is_distributed=False)
        val_lpips = AverageMeter(is_distributed=False)
        
        with torch.no_grad():
            with tqdm(total=len(data_loader), desc=f"Validation Epoch #{epoch}") as t:
                for batch_id, feed_dict in enumerate(data_loader):
                    images = feed_dict["data"].cuda()
                    
                    # Forward pass
                    with torch.cuda.amp.autocast(enabled=self.enable_amp):
                        # Get the actual model from DDP wrapper if needed
                        model_unwrapped = model.module if hasattr(model, 'module') else model
                        
                        encoded = model_unwrapped.encode(images)
                        reconstructed = model_unwrapped.decode(encoded)
                        
                        # Calculate losses
                        recon_loss = F.mse_loss(reconstructed, images)
                        perceptual_loss = self.lpips(reconstructed * 0.5 + 0.5, images * 0.5 + 0.5)
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
                    
                    # Save sample reconstructions
                    if batch_id == 0 and is_master():
                        sample_dir = os.path.join(self.path, "samples", f"epoch_{epoch}")
                        os.makedirs(sample_dir, exist_ok=True)
                        
                        for i in range(min(8, images.size(0))):
                            sample_path = os.path.join(sample_dir, f"sample_{i}.png")
                            sample_image = torch.cat([
                                images[i:i+1] * 0.5 + 0.5,
                                reconstructed[i:i+1] * 0.5 + 0.5
                            ], dim=-1)
                            torchvision.utils.save_image(sample_image, sample_path)
                            
                        if wandb_available and wandb.run is not None:
                            wandb.log({
                                "val/reconstructions": [wandb.Image(sample_path) for sample_path in os.listdir(sample_dir)]
                            }, step=epoch)
        
        metrics = {
            "val/loss": val_loss.avg,
            "val/recon_loss": val_recon_loss.avg,
            "val/perceptual_loss": val_perceptual_loss.avg,
            "val/psnr": val_psnr.avg,
            "val/ssim": val_ssim.avg,
            "val/lpips": val_lpips.avg
        }
        
        if wandb_available and wandb.run is not None:
            wandb.log(metrics, step=epoch)
            
        return metrics
        
    def run_step(self, feed_dict):
        images = feed_dict["data"]
        
        with torch.cuda.amp.autocast('cuda', enabled=self.enable_amp):  
            # Get the actual model from DDP wrapper if needed
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            # Forward pass through DCAE
            encoded = model.encode(images)
            reconstructed = model.decode(encoded)
            
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(reconstructed, images)
            
            # Perceptual loss (LPIPS)
            perceptual_loss = self.lpips(reconstructed * 0.5 + 0.5, images * 0.5 + 0.5)
            
            # Total loss
            total_loss = (self.run_config.reconstruction_weight * recon_loss + 
                         self.run_config.perceptual_weight * perceptual_loss)
            
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "perceptual_loss": perceptual_loss,
            "reconstructed": reconstructed
        }
        
    def _train_one_epoch(self, epoch):
        train_loss = AverageMeter(is_distributed=False)
        train_recon_loss = AverageMeter(is_distributed=False)
        train_perceptual_loss = AverageMeter(is_distributed=False)
        
        with tqdm(total=len(self.data_provider.train), desc=f"Training Epoch #{epoch}") as t:
            for feed_dict in self.data_provider.train:
                feed_dict = self.before_step(feed_dict)
                self.optimizer.zero_grad()
                
                output_dict = self.run_step(feed_dict)
                
                # Scale loss and backward
                self.scaler.scale(output_dict["loss"]).backward()
                
                # Update metrics
                train_loss.update(output_dict["loss"].item(), feed_dict["data"].size(0))
                train_recon_loss.update(output_dict["recon_loss"].item(), feed_dict["data"].size(0))
                train_perceptual_loss.update(output_dict["perceptual_loss"].item(), feed_dict["data"].size(0))
                
                # Optimizer step
                self.after_step()
                
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
        
        if wandb_available and wandb.run is not None:
            wandb.log(metrics, step=epoch)
            
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