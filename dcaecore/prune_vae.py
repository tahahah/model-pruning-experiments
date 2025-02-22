import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
import logging
from datetime import datetime

# Try importing wandb
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(ROOT_DIR)
import torch_pruning as tp
from efficientvit.ae_model_zoo import DCAE_HF
from efficientvit.models.efficientvit.dc_ae import DCAE
from trainer import DCAERunConfig
from pruning_trainer import VAEPruningTrainer
from diffusers import AutoencoderTiny
from pacman_dataset_copy import SimplePacmanDatasetProvider, PacmanDatasetProviderConfig

class AutoencoderTinyWrapper(AutoencoderTiny):
    """Wrapper for AutoencoderTiny to make it compatible with our interface"""
        
    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x).clamp(0, 1)
        
    def forward(self, x):
        latent = self.encode(x)
        return self.decode(latent)

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
    
    return logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_wandb(config):
    """Initialize wandb if enabled in config."""
    if not wandb_available:
        return

    wandb_config = config.get('logging', {}).get('wandb', {})
    if wandb_config.get('enabled', False):
        wandb.init(
            project=wandb_config.get('project', 'dcae-finetuning'),
            entity=wandb_config.get('entity', None),
            config=config,
            tags=wandb_config.get('tags', ['dcae', 'autoencoder'])
        )

def create_run_config(config: dict) -> DCAERunConfig:
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

def setup_dist_env(gpu):
    """Setup distributed training environment."""
    # Set default env vars for distributed training if not set
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    # Initialize process group
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Set cudnn benchmark
    torch.backends.cudnn.benchmark = True
    
    return device

def setup_seed(seed, resume=False):
    if not resume:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dist_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_dist_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=str, help="Comma-separated list of GPU IDs to use")
    args = parser.parse_args()

    # Setup distributed environment
    device = setup_dist_env(args.gpu)
    setup_seed(args.seed, resume=False)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create data config with all required parameters
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
    data_provider = SimplePacmanDatasetProvider(data_cfg)
    
    # Setup system
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir)
    logger.info(f"Set random seed to {args.seed}")
    
    # Initialize wandb
    setup_wandb(config)
    logger.info("Initialized Weights & Biases logging")
    
    logger.info(f"Created data provider with batch size {data_cfg.batch_size}")
    
    # Load pretrained model
    logger.info(f"Loading pretrained model from {'madebyollin/taesd'}")
    model = AutoencoderTinyWrapper.from_pretrained("madebyollin/taesd")
    
    # Create run config
    run_config = create_run_config(config)
    logger.info("Created training configuration")
    
    # Get logging config
    logging_cfg = config.get('logging', {})
    
    # Create trainer with logging config
    trainer = VAEPruningTrainer(
        path=args.output_dir,
        model=model,
        data_provider=data_provider,
    )
    
    # Configure trainer logging
    trainer.write_train_log = True  # Enable training step logging
    trainer.write_val_log = True    # Enable validation step logging
    trainer.log_interval = logging_cfg.get('log_interval', 100)  # Set logging frequency
    
    # Setup trainer with safe defaults
    trainer.prep_for_training(
        run_config=run_config,
        ema_decay=None,  # EMA is handled by EfficientViT's trainer
        amp=None  # AMP is handled by EfficientViT's trainer
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
