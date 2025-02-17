from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from datasets import load_dataset
from torchvision import transforms
from efficientvit.aecore.data_provider.base import BaseDataProvider, BaseDataProviderConfig
import os
import pickle
from pathlib import Path

@dataclass
class PacmanDatasetProviderConfig(BaseDataProviderConfig):
    name: str = "SimplePacmanDatasetProvider"
    train_dataset: str = "Tahahah/PacmanDataset_3"
    val_dataset: str = "Tahahah/PacmanDataset_2"
    train_split: str = "train[:200]"  # Only get first 200 samples if not streaming
    val_split: str = "train[:200]"    # Only get first 200 samples if not streaming
    image_size: int = 256
    verification_mode: str = "no_checks"
    streaming: bool = False  # Set to True for original streaming behavior
    cache_dir: str = ".cache/pacman_dataset"  # Directory to store cached samples
    batch_size: int = 4
    n_worker: int = 2
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
            streaming=True,
            verification_mode=cfg.verification_mode
        )
        
        # Take only first 200 samples if specified in split
        if "[:200]" in cfg.train_split:
            self.dataset = self.dataset.take(200)
            
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

    def __len__(self): 
        return 200 if "[:200]" in self.cfg.train_split else 100  # Return actual size for subset

    def build_transform(self):
        return transforms.Compose([
            transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class CachedPacmanDataset(Dataset):
    """Dataset that efficiently caches downloaded samples"""
    def __init__(self, cfg: PacmanDatasetProviderConfig, dataset_name: str, split: str):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.split = split.split("[")[0] if "[" in split else split
        self.num_samples = 200 if "[:200]" in split else None
        self.transform = self.build_transform()
        
        # Create cache directory if it doesn't exist
        self.cache_dir = Path(self.cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file is specific to dataset and split
        self.cache_file = self.cache_dir / f"{dataset_name.replace('/', '_')}_{self.split}_{self.num_samples}.pkl"
        
        # Load or create cache
        self.samples = self._load_or_create_cache()

    def _load_or_create_cache(self):
        """Load cached samples or create new cache if doesn't exist"""
        if self.cache_file.exists():
            print(f"Loading cached samples from {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        
        print(f"Cache not found, downloading {self.num_samples} samples...")
        # Use streaming to efficiently download only needed samples
        streamed_dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            verification_mode=self.cfg.verification_mode
        )
        
        if self.num_samples:
            streamed_dataset = streamed_dataset.take(self.num_samples)
        
        # Download and cache samples
        samples = []
        for item in streamed_dataset:
            samples.append(item['frame_image'])
            if self.num_samples and len(samples) >= self.num_samples:
                break
        
        print(f"Caching {len(samples)} samples to {self.cache_file}")
        with open(self.cache_file, 'wb') as f:
            pickle.dump(samples, f)
        
        return samples

    def __getitem__(self, idx):
        try:
            image = self.samples[idx]
            if not isinstance(image, Image.Image):
                image = image.convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}

    def __len__(self):
        return len(self.samples)

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
            # Use cached implementation
            train_dataset = CachedPacmanDataset(self.cfg, self.cfg.train_dataset, self.cfg.train_split)
            val_dataset = CachedPacmanDataset(self.cfg, self.cfg.val_dataset, self.cfg.val_split)
            
            self.train = DataLoader(
                train_dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.n_worker,
                pin_memory=True,
                shuffle=True  # Enable shuffle for training
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
