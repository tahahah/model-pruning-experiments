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
    image_size: int = 512
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
        self.split = split
        
        # Parse sample limit from split string if present (e.g., "train[:1000]" -> 1000)
        self.sample_limit = None
        if "[:" in split:
            try:
                self.sample_limit = int(split.split("[:")[-1].rstrip("]"))
            except ValueError:
                print(f"Warning: Could not parse sample limit from split: {split}")
        
        # Remove sample limit from split for dataset loading
        clean_split = split.split("[")[0] if "[" in split else split
        self.dataset = load_dataset(
            dataset_name,
            split=clean_split,
            streaming=True,
            verification_mode=cfg.verification_mode
        )
        
        # Only take limited samples if not streaming
        if not self.cfg.streaming and self.sample_limit is not None:
            self.dataset = self.dataset.take(self.sample_limit)
            
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
        if self.cfg.streaming:
            # For streaming datasets, use steps_per_epoch from run_config
            return self.cfg.val_steps * self.cfg.batch_size
        # For non-streaming, return parsed limit or default
        return self.sample_limit if self.sample_limit is not None else 100

    
    def build_transform(self):

        def make_square(image):
            # Calculate the necessary padding to make the image square
            width, height = image.size
            max_dim = max(width, height)
            padding = [
                (max_dim - width) // 2,  # Left padding
                (max_dim - height) // 2, # Top padding
                (max_dim - width + 1) // 2,  # Right padding
                (max_dim - height + 1) // 2  # Bottom padding
            ]
            return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

        def convert_to_rgb(img):
            return img.convert("RGB")

        def rotate_90_clockwise(img):
            return img.rotate(90, expand=True)

        return transforms.Compose([
                        transforms.Lambda(convert_to_rgb),
                        transforms.Lambda(make_square),  # Make the image square with padding
                        transforms.Resize(self.cfg.image_size),          # Resize to 512x512
                        transforms.functional.hflip,     # Horizontal mirror flip
                        transforms.Lambda(rotate_90_clockwise),  # Rotate 90 degrees clockwise
                        transforms.ToTensor()
                    ])

class CachedPacmanDataset(Dataset):
    """Dataset that efficiently caches downloaded samples"""
    def __init__(self, cfg: PacmanDatasetProviderConfig, dataset_name: str, split: str):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.split = split
        
        # Parse sample limit from split string if present (e.g., "train[:1000]" -> 1000)
        self.sample_limit = None
        if "[:" in split:
            try:
                self.sample_limit = int(split.split("[:")[-1].rstrip("]"))
            except ValueError:
                print(f"Warning: Could not parse sample limit from split: {split}")
        
        # Remove sample limit from split for dataset loading
        clean_split = split.split("[")[0] if "[" in split else split
        self.num_samples = self.sample_limit
        
        # Create cache directory if it doesn't exist
        self.cache_dir = Path(self.cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file is specific to dataset and split
        self.cache_file = self.cache_dir / f"{dataset_name.replace('/', '_')}_{clean_split}_{self.num_samples}.pkl"
        
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
        # Remove sample limit from split for dataset loading
        clean_split = self.split.split('[')[0] if '[' in self.split else self.split

        streamed_dataset = load_dataset(
            self.dataset_name,
            split=clean_split,
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

        def make_square(image):
            # Calculate the necessary padding to make the image square
            width, height = image.size
            max_dim = max(width, height)
            padding = [
                (max_dim - width) // 2,  # Left padding
                (max_dim - height) // 2, # Top padding
                (max_dim - width + 1) // 2,  # Right padding
                (max_dim - height + 1) // 2  # Bottom padding
            ]
            return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

        def convert_to_rgb(img):
            return img.convert("RGB")

        def rotate_90_clockwise(img):
            return img.rotate(90, expand=True)

        return transforms.Compose([
                        transforms.Lambda(convert_to_rgb),
                        transforms.Lambda(make_square),  # Make the image square with padding
                        transforms.Resize(self.cfg.image_size),          # Resize to 512x512
                        transforms.functional.hflip,     # Horizontal mirror flip
                        transforms.Lambda(rotate_90_clockwise),  # Rotate 90 degrees clockwise
                        transforms.ToTensor()
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
