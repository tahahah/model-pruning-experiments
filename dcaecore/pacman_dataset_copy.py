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
    max_samples: Optional[int] = None  
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
            return int(1000)  
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
    def __init__(self, cfg: PacmanDatasetProviderConfig):
        super().__init__(cfg)
        self.cfg = cfg
    
    def build_transform(self):
        return T.Compose([
            T.Resize((self.cfg.image_size, self.cfg.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def build_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        # Create train dataset
        train_dataset = SimplePacmanDataset(self.cfg)
        
        # Create validation dataset with PacmanDataset_2
        val_cfg = PacmanDatasetProviderConfig(
            **{k: v for k, v in self.cfg.__dict__.items()}  
        )
        val_cfg.dataset_name = "Tahahah/PacmanDataset_2"  
        val_dataset = SimplePacmanDataset(val_cfg)
        
        # Use same as validation for test
        test_dataset = val_dataset
        
        return train_dataset, val_dataset, test_dataset
    
    def build_dataloaders(self, train_dataset: Dataset, val_dataset: Optional[Dataset], test_dataset: Optional[Dataset]):
        # Create sampler for distributed training
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=self.cfg.num_replicas,
            rank=self.cfg.rank,
            shuffle=True
        ) if self.cfg.num_replicas > 1 else None
        
        # Create train dataloader
        self.train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_worker,
            sampler=sampler,
            drop_last=True
        )
        
        # Create validation dataloader
        if val_dataset is not None:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset,
                num_replicas=self.cfg.num_replicas,
                rank=self.cfg.rank,
                shuffle=False
            ) if self.cfg.num_replicas > 1 else None
            
            self.valid = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_worker,
                sampler=val_sampler,
                drop_last=False
            )
        
        # Create test dataloader (same as validation for now)
        if test_dataset is not None:
            self.test = self.valid
