from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset
from torchvision import transforms
from efficientvit.aecore.data_provider.base import BaseDataProvider, BaseDataProviderConfig

@dataclass
class PacmanDatasetProviderConfig(BaseDataProviderConfig):
    name: str = "SimplePacmanDatasetProvider"
    train_dataset: str = "Tahahah/PacmanDataset_3"
    val_dataset: str = "Tahahah/PacmanDataset_2"
    train_split: str = "train[:200]"  # Only get first 200 samples if not streaming
    val_split: str = "train[:200]"    # Only get first 200 samples if not streaming
    image_size: int = 512
    verification_mode: str = "no_checks"
    streaming: bool = False  # Set to True for original streaming behavior
    batch_size: int = 16
    n_worker: int = 4
    num_replicas: int = 1
    rank: int = 0
    val_steps: int = 100

class StreamingPacmanDataset(IterableDataset):
    """Original streaming dataset implementation - unchanged"""
    def __init__(self, cfg: PacmanDatasetProviderConfig, dataset_name: str, split: str):
        super().__init__()
        self.cfg = cfg
        # Remove [:200] from split if present when streaming
        split = split.split("[")[0] if "[" in split else split
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=True,  # Always use streaming for IterableDataset
            verification_mode=cfg.verification_mode
        )
        self.transform = self.build_transform()
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self.dataset)
        return self

    def __next__(self):
        try:
            item = next(self._iterator)
            image = item['frame_image']
            if not isinstance(image, Image.Image):
                image = image.convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except StopIteration:
            self._iterator = iter(self.dataset)  # Reset iterator
            raise
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}

    def __len__(self): return 100

    def build_transform(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class CachedPacmanDataset(Dataset):
    """New cached dataset implementation for faster startup"""
    def __init__(self, cfg: PacmanDatasetProviderConfig, dataset_name: str, split: str):
        super().__init__()
        self.cfg = cfg
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            streaming=False,
            verification_mode=cfg.verification_mode
        )
        self.transform = self.build_transform()

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item['frame_image']
            if not isinstance(image, Image.Image):
                image = image.convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}

    def __len__(self):
        return len(self.dataset)

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
        self.test = None
        self.build_datasets()
    
    def build_datasets(self):
        if self.cfg.streaming:
            # Use original streaming implementation
            train_dataset = StreamingPacmanDataset(self.cfg, self.cfg.train_dataset, self.cfg.train_split)
            val_dataset = StreamingPacmanDataset(self.cfg, self.cfg.val_dataset, self.cfg.val_split)
            
            # Original dataloader config for streaming
            self.train = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_worker,
                pin_memory=True
            )
            
            self.valid = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_worker,
                pin_memory=True
            )
        else:
            # Use new cached implementation
            train_dataset = CachedPacmanDataset(self.cfg, self.cfg.train_dataset, self.cfg.train_split)
            val_dataset = CachedPacmanDataset(self.cfg, self.cfg.val_dataset, self.cfg.val_split)
            
            # Add shuffle for cached training data
            self.train = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_worker,
                pin_memory=True,
                shuffle=True
            )
            
            self.valid = DataLoader(
                val_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_worker,
                pin_memory=True
            )
        
        self.test = self.valid
        return self.train, self.valid, self.test
    
    def set_epoch(self, epoch):
        """No need to set epoch for either dataset type"""
        pass
