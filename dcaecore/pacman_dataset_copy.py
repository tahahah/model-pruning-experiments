from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from efficientvit.aecore.data_provider.base import BaseDataProvider, BaseDataProviderConfig

@dataclass
class PacmanDatasetProviderConfig(BaseDataProviderConfig):
    name: str = "SimplePacmanDatasetProvider"
    train_dataset: str = "Tahahah/PacmanDataset_3"
    val_dataset: str = "Tahahah/PacmanDataset_2"
    train_split: str = "train"
    val_split: str = "train"
    image_size: int = 512
    verification_mode: str = "no_checks"
    streaming: bool = True
    batch_size: int = 16
    n_worker: int = 4
    num_replicas: int = 1
    rank: int = 0
    val_steps: int = 100

class SimplePacmanDataset(Dataset):
    def __init__(self, cfg: PacmanDatasetProviderConfig, dataset_name: str, split: str):
        self.cfg = cfg
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=cfg.streaming,
            verification_mode=cfg.verification_mode
        )
        self.transform = self.build_transform()
    
    def __len__(self):
        if self.cfg.streaming:
            return int(1000)  # Effectively infinite for streaming dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['frame_image'].convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error loading image at index {idx}: {str(e)}")
            # Return a random noise image as fallback
            return {"data": torch.randn(3, self.cfg.image_size, self.cfg.image_size)}
    
    def build_transform(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class SimplePacmanDatasetProvider(BaseDataProvider):
    def __init__(self, cfg: PacmanDatasetProviderConfig):
        self.cfg = cfg
        self.train = None
        self.valid = None
        self.build_datasets()
    
    def build_datasets(self):
        train_dataset = SimplePacmanDataset(self.cfg, self.cfg.train_dataset, self.cfg.train_split)
        val_dataset = SimplePacmanDataset(self.cfg, self.cfg.val_dataset, self.cfg.val_split)
        
        train_sampler = None
        val_sampler = None
        
        if self.cfg.num_replicas > 1:
            # Add distributed sampler if needed
            pass
        
        self.train = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_worker,
            sampler=train_sampler,
            drop_last=True,
            pin_memory=True
        )
        
        self.valid = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.n_worker,
            sampler=val_sampler,
            drop_last=False,
            pin_memory=True
        )
        return self.train, self.valid, self.valid
    
    def set_epoch(self, epoch):
        """Override set_epoch to handle epoch updates without requiring sampler.set_epoch"""
        # Since we're using a streaming dataset, we don't need to actually do anything here
        pass
