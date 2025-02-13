from dataclasses import dataclass
from typing import Optional

import torch
import torchvision.transforms as T
from datasets import load_dataset
from torch.utils.data import IterableDataset
from PIL import Image

from efficientvit.diffusioncore.data_provider.base import BaseDataProvider


@dataclass
class PacmanDatasetProviderConfig:
    name: str = "pacman_dataset"
    dataset_name: str = "Tahahah/PacmanDataset_3"
    split: str = "train"
    image_size: int = 512
    verification_mode: str = "no_checks"
    streaming: bool = True
    batch_size: int = 32


class SimplePacmanDataset(IterableDataset):
    def __init__(self, cfg: PacmanDatasetProviderConfig):
        self.cfg = cfg
        self.dataset = load_dataset(
            cfg.dataset_name, 
            split=cfg.split, 
            verification_mode=cfg.verification_mode,
            streaming=cfg.streaming
        )
        
        # Transform pipeline for images
        self.transform = T.Compose([
            T.Resize(cfg.image_size),
            T.CenterCrop(cfg.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def process_image(self, image_path):
        """Process a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return a blank image in case of error
            return torch.zeros(3, self.cfg.image_size, self.cfg.image_size)

    def __iter__(self):
        """Iterator that yields batches of images."""
        iterator = iter(self.dataset)
        batch = []
        
        for item in iterator:
            image = self.process_image(item['image'])
            batch.append(image)
            
            if len(batch) == self.cfg.batch_size:
                # Stack the batch of images and yield
                images = torch.stack(batch)
                yield {"data": images}
                batch = []

        # Yield remaining images if any
        if batch:
            images = torch.stack(batch)
            yield {"data": images}


class SimplePacmanDatasetProvider(BaseDataProvider):
    def __init__(self, cfg: PacmanDatasetProviderConfig):
        super().__init__()
        self.train_dataset = SimplePacmanDataset(cfg)
        self.test_dataset = None

    def get_train_dataset(self):
        return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset
