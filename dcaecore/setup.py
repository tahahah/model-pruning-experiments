import os
import time
import torch
import torch.backends.cudnn
import torch.distributed as dist
from typing import Optional

def setup_dist_env(gpu: Optional[str] = None) -> None:
    """Setup distributed training environment."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    # Set default env vars for distributed training if not set
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://"
        )
    
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.set_device(get_dist_local_rank())

def get_dist_local_rank() -> int:
    """Get local rank for distributed training."""
    if not dist.is_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))

def get_dist_rank() -> int:
    """Get global rank for distributed training."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_dist_size() -> int:
    """Get world size for distributed training."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_master() -> bool:
    """Check if current process is the master process."""
    return get_dist_rank() == 0

def setup_seed(manual_seed: int, resume: bool) -> None:
    """Setup random seed for reproducibility."""
    if resume:
        manual_seed = int(time.time())
    manual_seed = get_dist_rank() + manual_seed
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
