from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd

from config import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, DEFAULT_BATCH_SIZE


class ImageDataset(Dataset):
    """
    Simple dataset that reads image paths from a metadata CSV and applies
    ImageNet-style preprocessing.
    """

    def __init__(
        self,
        metadata_csv: str,
        images_root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.metadata_path = Path(metadata_csv)
        self.images_root = Path(images_root)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_path}")

        if not self.images_root.exists():
            raise FileNotFoundError(f"Images root not found: {self.images_root}")

        self.df = pd.read_csv(self.metadata_path)

        if "filename" not in self.df.columns:
            raise ValueError("Expected 'filename' column in metadata CSV.")

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        row = self.df.iloc[idx]
        filename = row["filename"]

        img_path = self.images_root / filename
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Return the raw row as metadata dict alongside the tensor
        metadata = row.to_dict()
        metadata["index"] = idx

        return image, metadata


def create_dataloader(
    metadata_csv: str,
    images_root: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """
    Convenience helper to create a DataLoader for the impressionism subset.
    """
    dataset = ImageDataset(
        metadata_csv=metadata_csv,
        images_root=images_root,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader

