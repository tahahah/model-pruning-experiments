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
            return int(1e9)  # Effectively infinite for streaming dataset
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.cfg.streaming:
            # For streaming, get next item from iterator
            item = next(iter(self.dataset))
        else:
            item = self.dataset[idx]
        
        try:
            image = Image.open(item['image']).convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error processing image {item['image']}: {e}")
            # Return a blank image in case of error
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}

class SimplePacmanDatasetProvider(BaseDataProvider):
    def __init__(self, cfg: PacmanDatasetProviderConfig):
        super().__init__(cfg)
        self.cfg: PacmanDatasetProviderConfig
    
    def build_transform(self):
        return T.Compose([
            T.Resize(self.cfg.image_size),
            T.CenterCrop(self.cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    def build_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        transform = self.build_transform()
        # Create train dataset
        train_dataset = SimplePacmanDataset(
            self.cfg,
            transform=transform
        )
        # We don't have validation/test splits for now
        val_dataset = train_dataset
        test_dataset = train_dataset
        return train_dataset, val_dataset, test_dataset
