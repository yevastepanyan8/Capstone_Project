"""
extract_embeddings.py
---------------------
Extracts 2048-dim ResNet-50 embeddings for any genre dataset.

Usage
-----
# Default (Impressionism, backward-compatible)
python src/extract_embeddings.py

# New genre
python src/extract_embeddings.py \\
    --dataset_dir dataset_realism \\
    --output_dir  embeddings/realism
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from torchvision import models

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_loader import create_dataloader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_feature_extractor(device: torch.device) -> nn.Module:
    """ResNet-50 with final FC removed → 2048-dim feature vector."""
    model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval().to(device)
    return backbone


@torch.no_grad()
def extract_all_embeddings(dataset_dir: Path, output_dir: Path, batch_size: int = 32):
    images_dir   = dataset_dir / 'images'
    metadata_csv = dataset_dir / 'metadata_subset.csv'

    if not metadata_csv.exists():
        raise FileNotFoundError(f'Metadata CSV not found: {metadata_csv}')
    if not images_dir.exists():
        raise FileNotFoundError(f'Images directory not found: {images_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    out_npy  = output_dir / 'image_embeddings.npy'
    out_meta = output_dir / 'embedding_metadata.csv'

    device = get_device()
    print(f'Device       : {device}')
    print(f'Dataset dir  : {dataset_dir}')
    print(f'Output dir   : {output_dir}')

    # Pre-filter metadata to only rows whose image file actually exists on disk.
    # This gracefully handles cases where create_subset.py recorded a filename
    # that couldn't be copied (e.g. special characters in artist names).
    df_meta = pd.read_csv(metadata_csv)
    exists  = df_meta['filename'].apply(lambda f: (images_dir / f).exists())
    missing = df_meta[~exists]
    if len(missing) > 0:
        print(f'[warn] Skipping {len(missing)} missing image(s):')
        for fn in missing['filename']:
            print(f'       {fn}')
    df_meta = df_meta[exists].reset_index(drop=True)

    # Write filtered metadata back as a temp CSV for the dataloader
    filtered_csv = output_dir / '_filtered_metadata.csv'
    df_meta.to_csv(filtered_csv, index=False)

    loader     = create_dataloader(
        metadata_csv=str(filtered_csv),
        images_root=str(images_dir),
        batch_size=batch_size,
        shuffle=False,
    )
    num_images = len(df_meta)
    print(f'Images found : {num_images} (after filtering)')

    model          = load_feature_extractor(device)
    all_embeddings = np.zeros((num_images, 2048), dtype=np.float32)
    meta_rows      = []
    current_index  = 0

    for images, batch_meta in tqdm(loader, desc='Extracting embeddings'):
        bs     = images.size(0)
        images = images.to(device, non_blocking=True)
        feats  = model(images).view(bs, -1).cpu().numpy()
        all_embeddings[current_index: current_index + bs] = feats

        for i in range(bs):
            row = {}
            for key, val in batch_meta.items():
                v = val[i]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                row[key] = v
            meta_rows.append(row)

        current_index += bs

    all_embeddings = all_embeddings[:current_index]

    np.save(out_npy, all_embeddings)
    print(f'\nSaved embeddings → {out_npy}  {all_embeddings.shape}')

    meta_df = pd.DataFrame(meta_rows)
    meta_df['embedding_index'] = np.arange(len(meta_df))
    meta_df.to_csv(out_meta, index=False)
    print(f'Saved metadata   → {out_meta}')

    # Remove temp filtered CSV
    filtered_csv.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract ResNet-50 embeddings for a genre dataset.')
    parser.add_argument(
        '--dataset_dir', type=Path,
        default=BASE_DIR / 'dataset_impressionism',
        help='Path to genre dataset folder (must contain images/ and metadata_subset.csv)'
    )
    parser.add_argument(
        '--output_dir', type=Path,
        default=BASE_DIR / 'embeddings',
        help='Directory to save image_embeddings.npy and embedding_metadata.csv'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for embedding extraction'
    )
    args = parser.parse_args()

    extract_all_embeddings(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
