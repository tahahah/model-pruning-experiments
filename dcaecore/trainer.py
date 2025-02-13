import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from efficientvit.apps.utils.dist import dist_barrier, get_dist_size, is_dist_initialized, is_master, sync_tensor
from efficientvit.apps.utils.ema import EMA
from efficientvit.apps.utils.lr import ConstantLRwithWarmup
from efficientvit.apps.utils.metric import AverageMeter
from efficientvit.models.efficientvit.dc_ae import DCAE, DCAEConfig
from pacman_dataset_copy import SimplePacmanDataset, PacmanDatasetProviderConfig

__all__ = ["OptimizerConfig", "LRSchedulerConfig", "DCAETrainerConfig", "DCAETrainer"]


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-4
    warmup_lr: float = 0.0
    weight_decay: float = 0.05
    no_wd_keys: tuple[str, ...] = ()
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class LRSchedulerConfig:
    name: Any = "cosine_annealing"
    warmup_steps: int = 1000


@dataclass
class DCAETrainerConfig:
    # Model config
    model: DCAEConfig = field(default_factory=DCAEConfig)
    
    # Training dataset config
    dataset: PacmanDatasetProviderConfig = field(default_factory=PacmanDatasetProviderConfig)
    
    # Training config
    resume: bool = True
    resume_path: Optional[str] = None
    resume_schedule: bool = True
    num_epochs: Optional[int] = None
    max_steps: Optional[int] = None
    clip_grad: Optional[float] = None
    save_checkpoint_steps: int = 1000
    
    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    
    # Logging
    log: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = "dcae"
    
    # EMA
    ema_decay: float = 0.9998
    ema_warmup_steps: int = 2000
    evaluate_ema: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class DCAETrainer:
    def __init__(self, cfg: DCAETrainerConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Set random seed
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            
        # Initialize model
        self.model = DCAE(cfg.model).to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optimizer.lr,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay
        )
        
        # Setup learning rate scheduler
        self.lr_scheduler = ConstantLRwithWarmup(
            optimizer=self.optimizer,
            warmup_steps=cfg.lr_scheduler.warmup_steps,
            warmup_init_lr=cfg.optimizer.warmup_lr,
            max_lr=cfg.optimizer.lr,
        )
        
        # Setup EMA
        self.ema = EMA(self.model, cfg.ema_decay, cfg.ema_warmup_steps)
        
        # Setup logging
        if cfg.log and is_master():
            self.setup_wandb()
            
        self.global_step = 0
        self.start_epoch = 0
        if cfg.resume:
            self.try_resume_from_checkpoint()
            
        # Initialize loss tracking
        self.train_loss = AverageMeter()
        
    def setup_wandb(self):
        wandb.init(
            entity=self.cfg.wandb_entity,
            project=self.cfg.wandb_project,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        
    def try_resume_from_checkpoint(self):
        if self.cfg.resume_path is not None and os.path.exists(self.cfg.resume_path):
            print(f"Resuming from checkpoint: {self.cfg.resume_path}")
            checkpoint = torch.load(self.cfg.resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            if self.cfg.resume_schedule:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
                self.global_step = checkpoint["global_step"]
                self.start_epoch = checkpoint["epoch"]
            
    def save_checkpoint(self, epoch):
        if not is_master():
            return
            
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
        }
        
        save_path = f"checkpoints/dcae_step_{self.global_step}.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        
    def train_step(self, batch):
        self.model.train()
        images = batch["image"].to(self.device)
        
        # Forward pass through DCAE
        reconstructed = self.model(images, self.global_step)
        loss = torch.nn.functional.mse_loss(reconstructed, images)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.cfg.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            
        self.optimizer.step()
        self.lr_scheduler.step()
        self.ema.step()
        
        return loss.item()
        
    def train(self):
        """Main training loop"""
        # Setup dataset
        train_dataset = SimplePacmanDataset(self.cfg.dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=None,  # Batch size is handled by the dataset
            num_workers=4,
            pin_memory=True,
        )
        
        epoch = self.start_epoch
        while True:
            self.train_loss.reset()
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
                loss = self.train_step(batch)
                self.train_loss.update(loss)
                
                if self.global_step % 100 == 0 and is_master():
                    if self.cfg.log:
                        wandb.log(
                            {
                                "train/loss": self.train_loss.avg,
                                "train/lr": self.lr_scheduler.get_last_lr()[0],
                            },
                            step=self.global_step,
                        )
                        
                if self.global_step % self.cfg.save_checkpoint_steps == 0:
                    self.save_checkpoint(epoch)
                    
                self.global_step += 1
                
                if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                    return
                    
            epoch += 1
            if self.cfg.num_epochs is not None and epoch >= self.cfg.num_epochs:
                return
