import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torchvision import models

from dataset_loader import create_dataloader


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = BASE_DIR / "dataset_impressionism"
IMAGES_DIR = DATASET_DIR / "images"
METADATA_CSV = DATASET_DIR / "metadata_subset.csv"

EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_NPY = EMBEDDINGS_DIR / "image_embeddings.npy"
EMBEDDINGS_META_CSV = EMBEDDINGS_DIR / "embedding_metadata.csv"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_feature_extractor(device: torch.device) -> nn.Module:
    """
    Load a pretrained ResNet50 and remove the final classification layer so we
    obtain a 2048-dim feature vector per image.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modules = list(model.children())[:-1]  # remove final FC
    backbone = nn.Sequential(*modules)
    backbone.eval()
    backbone.to(device)
    return backbone


@torch.no_grad()
def extract_all_embeddings(
    batch_size: int = 32,
) -> None:
    device = get_device()
    print(f"Using device: {device}")

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

    # Data loader
    loader = create_dataloader(
        metadata_csv=str(METADATA_CSV),
        images_root=str(IMAGES_DIR),
        batch_size=batch_size,
        shuffle=False,
    )

    df_meta = pd.read_csv(METADATA_CSV)
    num_images = len(df_meta)
    print(f"Found {num_images} entries in metadata.")

    # Model
    model = load_feature_extractor(device)

    # Allocate storage for all embeddings
    embedding_dim = 2048
    all_embeddings = np.zeros((num_images, embedding_dim), dtype=np.float32)

    # We will also re-build a metadata frame aligned with embedding indices
    meta_rows = []

    current_index = 0

    for images, batch_meta in tqdm(loader, desc="Extracting embeddings"):
        bs = images.size(0)
        images = images.to(device, non_blocking=True)

        features = model(images)
        features = features.view(bs, -1).cpu().numpy()

        all_embeddings[current_index : current_index + bs] = features

    # FIX: batch_meta is a dict of lists, not a list of dicts
    # Unpack it by iterating over the batch dimension
        for i in range(bs):
            row = {}
            for key, val in batch_meta.items():
                v = val[i]
                # Convert tensors to plain Python values
                if isinstance(v, torch.Tensor):
                    v = v.item()
                row[key] = v
            meta_rows.append(row)

        current_index += bs

    # Truncate in case of mismatch (safety)
    all_embeddings = all_embeddings[:current_index]

    # Save embeddings
    np.save(EMBEDDINGS_NPY, all_embeddings)
    print(f"Saved embeddings to: {EMBEDDINGS_NPY}")

    # Build and save aligned metadata
    meta_df = pd.DataFrame(meta_rows)
    meta_df["embedding_index"] = np.arange(len(meta_df))
    meta_df.to_csv(EMBEDDINGS_META_CSV, index=False)
    print(f"Saved embedding metadata to: {EMBEDDINGS_META_CSV}")


if __name__ == "__main__":
    extract_all_embeddings(batch_size=32)

