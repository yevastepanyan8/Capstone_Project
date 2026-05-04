"""
extract_embeddings_multilayer.py
--------------------------------
Extracts 3584-dim ResNet-50 multi-layer embeddings (concat of GAP'd
layer2 + layer3 + layer4) for any genre dataset.

Multi-layer features capture both texture/style (layer2-3) and semantic
content (layer4), which the default single-layer 2048-dim setup misses.

Output is a drop-in replacement for image_embeddings.npy in a separate
directory; the rest of the pipeline (PCA reduction, injection, analysis)
works unchanged once EMBEDDINGS_DIR is pointed at the new root.

Usage
-----
python src/extract_embeddings_multilayer.py \\
    --dataset_dir dataset_impressionism \\
    --output_dir  embeddings_multilayer/impressionism
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models
from tqdm import tqdm

from config import PROJECT_ROOT, DEFAULT_BATCH_SIZE
from utils import get_device
from dataset_loader import create_dataloader


# ResNet-50 channel counts: layer2=512, layer3=1024, layer4=2048 → total 3584
MULTILAYER_DIM = 512 + 1024 + 2048


class ResNet50MultiLayer(nn.Module):
    """ResNet-50 returning concatenated GAP features from layer2, layer3, layer4."""

    def __init__(self):
        super().__init__()
        r = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.stem   = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.layer4 = r.layer4
        self.gap    = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        x  = self.layer2(x); f2 = self.gap(x).flatten(1)   # (B, 512)
        x  = self.layer3(x); f3 = self.gap(x).flatten(1)   # (B, 1024)
        x  = self.layer4(x); f4 = self.gap(x).flatten(1)   # (B, 2048)
        return torch.cat([f2, f3, f4], dim=1)              # (B, 3584)


@torch.no_grad()
def extract_all_embeddings(
    dataset_dir: Path, output_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE
):
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
    print(f'Device         : {device}')
    print(f'Dataset dir    : {dataset_dir}')
    print(f'Output dir     : {output_dir}')
    print(f'Embedding dim  : {MULTILAYER_DIM}  (layer2 512 + layer3 1024 + layer4 2048)')

    # Pre-filter metadata to only rows whose image file actually exists on disk.
    df_meta = pd.read_csv(metadata_csv)
    exists  = df_meta['filename'].apply(lambda f: (images_dir / f).exists())
    missing = df_meta[~exists]
    if len(missing) > 0:
        print(f'[warn] Skipping {len(missing)} missing image(s):')
        for fn in missing['filename']:
            print(f'       {fn}')
    df_meta = df_meta[exists].reset_index(drop=True)

    filtered_csv = output_dir / '_filtered_metadata.csv'
    df_meta.to_csv(filtered_csv, index=False)

    loader = create_dataloader(
        metadata_csv=str(filtered_csv),
        images_root=str(images_dir),
        batch_size=batch_size,
        shuffle=False,
    )
    num_images = len(df_meta)
    print(f'Images found   : {num_images} (after filtering)')

    if num_images == 0:
        print('[error] No valid images found. Check your dataset directory.')
        filtered_csv.unlink(missing_ok=True)
        sys.exit(1)

    model          = ResNet50MultiLayer().eval().to(device)
    all_embeddings = np.zeros((num_images, MULTILAYER_DIM), dtype=np.float32)
    meta_rows      = []
    current_index  = 0

    for images, batch_meta in tqdm(loader, desc='Extracting multi-layer embeddings'):
        bs     = images.size(0)
        images = images.to(device, non_blocking=True)
        feats  = model(images).cpu().numpy()
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

    filtered_csv.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract ResNet-50 multi-layer (3584-dim) embeddings for a genre dataset.'
    )
    parser.add_argument(
        '--dataset_dir', type=Path,
        default=PROJECT_ROOT / 'dataset_impressionism',
        help='Path to genre dataset folder (must contain images/ and metadata_subset.csv)'
    )
    parser.add_argument(
        '--output_dir', type=Path,
        default=PROJECT_ROOT / 'embeddings_multilayer' / 'impressionism',
        help='Directory to save image_embeddings.npy and embedding_metadata.csv'
    )
    parser.add_argument(
        '--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
        help='Batch size for embedding extraction'
    )
    args = parser.parse_args()

    extract_all_embeddings(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
