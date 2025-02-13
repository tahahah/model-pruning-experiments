from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import load_dataset
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, IterableDataset
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

class SimplePacmanDataset(IterableDataset if PacmanDatasetProviderConfig.streaming else Dataset):
    def __init__(self, cfg: PacmanDatasetProviderConfig, transform=None, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.transform = transform if transform is not None else T.Compose([
            T.Resize(cfg.image_size),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Delay dataset loading until first use
        self.dataset = None
        
    def _init_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset(
                self.cfg.dataset_name, 
                split=self.cfg.split, 
                verification_mode=self.cfg.verification_mode,
                streaming=self.cfg.streaming
            )
            if not self.cfg.streaming:
                self.dataset = list(self.dataset)
    
    def __iter__(self):
        self._init_dataset()
        if self.cfg.streaming:
            return self
        else:
            raise NotImplementedError("Iteration not supported for non-streaming dataset")
    
    def __next__(self):
        if not self.cfg.streaming:
            raise NotImplementedError("Iteration not supported for non-streaming dataset")
        try:
            item = next(iter(self.dataset))
            image = Image.open(item['frame_image']).convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error processing image: {e}")
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}
    
    def __len__(self):
        if self.cfg.streaming:
            return int(1000)  # Limited for streaming dataset
        self._init_dataset()
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.cfg.streaming:
            raise NotImplementedError("Random access not supported for streaming dataset")
        
        self._init_dataset()
        try:
            item = self.dataset[idx]
            image = Image.open(item['frame_image']).convert('RGB')
            image = self.transform(image)
            return {"data": image}
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            return {"data": torch.zeros(3, self.cfg.image_size, self.cfg.image_size)}

class SimplePacmanDatasetProvider(BaseDataProvider):
    def __init__(self, cfg: PacmanDatasetProviderConfig):
        super().__init__(cfg)
        self.cfg: PacmanDatasetProviderConfig
        self._transform = None
    
    @property
    def transform(self):
        if self._transform is None:
            self._transform = T.Compose([
                T.Resize(self.cfg.image_size),
                T.CenterCrop(self.cfg.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        return self._transform
    
    def build_datasets(self) -> Tuple[Dataset, Optional[Dataset], Optional[Dataset]]:
        # Create separate dataset instances for train/val/test
        train_dataset = SimplePacmanDataset(
            self.cfg,
            transform=self.transform,
            is_train=True
        )
        
        # Create separate instances for val/test with same config
        val_dataset = SimplePacmanDataset(
            self.cfg,
            transform=self.transform,
            is_train=False
        )
        test_dataset = SimplePacmanDataset(
            self.cfg,
            transform=self.transform,
            is_train=False
        )
        
        return train_dataset, val_dataset, test_dataset
