import os
import time
import torch
import torch.backends.cudnn
from typing import Optional

def setup_device(gpu_id: Optional[int] = None) -> torch.device:
    """Setup device for training."""
    if gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = True
    return device

def get_dist_local_rank() -> int:
    """Get local rank for distributed training."""
    return 0

def get_dist_rank() -> int:
    """Get global rank for distributed training."""
    return 0

def get_dist_size() -> int:
    """Get world size for distributed training."""
    return 1

def is_master() -> bool:
    """Check if current process is the master process."""
    return True

def setup_seed(seed: int) -> None:
    """Setup random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
