from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import load_dataset
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from efficientvit.aecore.data_provider.base import BaseDataProvider, BaseDataProviderConfig

@dataclass
class PacmanDatasetProviderConfig(BaseDataProviderConfig):
    name: str = "pacman_dataset"
    dataset_name: str = "Tahahah/PacmanDataset_3"
    split: str = "train"
    image_size: int = 512
    verification_mode: str = "no_checks"
    streaming: bool = True
    batch_size: int = 32
    n_worker: int = 1
    max_samples: Optional[int] = None  # Added max_samples parameter
    num_replicas: int = 1
    rank: int = 0

class SimplePacmanDataset(Dataset):
    def __init__(self, cfg: PacmanDatasetProviderConfig, transform=None):
        self.cfg = cfg
        self.dataset = load_dataset(
            cfg.dataset_name, 
            split=cfg.split, 
            verification_mode=cfg.verification_mode,
            streaming=cfg.streaming
        )
        self.transform = transform if transform is not None else T.Compose([
            T.Resize(cfg.image_size),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Convert iterator to list if not streaming
        if not cfg.streaming:
            self.dataset = list(self.dataset)
    
    def __len__(self):
        if self.cfg.streaming:
            return int(1000)  # Effectively infinite for streaming dataset
        if self.cfg.max_samples is not None:
            return min(len(self.dataset), self.cfg.max_samples)
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.cfg.streaming:
            # For streaming, get next item from iterator
            item = next(iter(self.dataset))
        else:
            item = self.dataset[idx]
        
        try:
            image = item['frame_image'].convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error processing image {item['image']}: {e}")
            # Return a blank image in case of error
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}

class SimplePacmanDatasetProvider(BaseDataProvider):
    def __init__(self, data_config: PacmanDatasetProviderConfig):
        super().__init__(data_config)
        self.data_config = data_config
        
        # Create dataset
        dataset = SimplePacmanDataset(data_config)
        
        # Create sampler for distributed training
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=data_config.num_replicas,
            rank=data_config.rank,
            shuffle=True
        ) if data_config.num_replicas > 1 else None
        
        # Create dataloader
        self.train = torch.utils.data.DataLoader(
            dataset,
            batch_size=data_config.batch_size,
            num_workers=data_config.n_worker,
            sampler=sampler,
            drop_last=True
        )
        
        # For validation/test, we use the same dataset instance but with different settings
        if not data_config.streaming:  # Only create val/test if not streaming
            self.valid = self.train  # Use same loader for validation
            self.test = self.train   # Use same loader for testing
