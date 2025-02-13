import os
import sys
import argparse
import yaml
import torch
import wandb
import random
import numpy as np
import logging
from datetime import datetime
from omegaconf import OmegaConf

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)

from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.models.efficientvit.dc_ae import DCAE
from trainer import DCAETrainer, DCAERunConfig
from pacman_dataset_copy import SimplePacmanDatasetProvider, PacmanDatasetProviderConfig

def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(output_dir: str) -> logging.Logger:
    """Setup logger that outputs to both console and file."""
    logger = logging.getLogger('dcae_training')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_wandb(config: dict):
    """Initialize Weights & Biases logging."""
    if config['trainer']['logging']['wandb']['enabled']:
        wandb.init(
            project=config['trainer']['logging']['wandb']['project'],
            entity=config['trainer']['logging']['wandb']['entity'],
            config=config,
            tags=config['trainer']['logging']['wandb']['tags']
        )

def create_run_config(config: dict) -> DCAERunConfig:
    """Create RunConfig from configuration dictionary."""
    run_config = DCAERunConfig(
        num_epochs=config['trainer']['num_epochs'],
        reconstruction_weight=config['trainer']['reconstruction_weight'],
        perceptual_weight=config['trainer']['perceptual_weight'],
        grad_clip=config['trainer']['grad_clip']
    )
    
    # Setup optimizer
    opt_cfg = config['trainer']['optimizer']
    run_config.optimizer_name = opt_cfg['name']
    run_config.optimizer_params = {
        'lr': opt_cfg['lr'],
        'weight_decay': opt_cfg['weight_decay'],
        'betas': opt_cfg['betas'],
        'eps': opt_cfg['eps']
    }
    
    # Setup scheduler
    sched_cfg = config['trainer']['lr_scheduler']
    run_config.scheduler_name = sched_cfg['name']
    run_config.scheduler_params = {
        'warmup_epochs': sched_cfg['warmup_epochs'],
        'warmup_lr': sched_cfg['warmup_lr'],
        'min_lr': sched_cfg['min_lr']
    }
    
    return run_config

def main():
    parser = argparse.ArgumentParser(description='Train DCAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained', type=str, default='mit-han-lab/dc-ae-f32c32-in-1.0', help='Pretrained model name or path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup system
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    set_random_seed(config['system']['seed'])
    logger.info(f"Set random seed to {config['system']['seed']}")
    
    # Initialize wandb
    setup_wandb(config)
    logger.info("Initialized Weights & Biases logging")
    
    # Create data provider
    data_cfg = PacmanDatasetProviderConfig(**config['data_provider'])
    data_provider = SimplePacmanDatasetProvider(data_cfg)
    logger.info(f"Created data provider with batch size {data_cfg.batch_size}")
    
    # Load pretrained model
    logger.info(f"Loading pretrained model from {args.pretrained}")
    model = DCAE_HF.from_pretrained(args.pretrained)
    
    # Create run config
    run_config = create_run_config(config)
    logger.info("Created training configuration")
    
    # Create trainer
    trainer = DCAETrainer(
        path=args.output_dir,
        model=model,
        data_provider=data_provider
    )
    
    # Setup trainer
    trainer.prep_for_training(
        run_config=run_config,
        ema_decay=config['trainer']['ema']['decay'] if config['trainer']['ema']['enabled'] else None,
        amp=config['trainer']['amp']
    )
    logger.info("Trainer setup complete")
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Cleanup
    if wandb.run is not None:
        wandb.finish()
    logger.info("Training completed")

if __name__ == "__main__":
    main()
