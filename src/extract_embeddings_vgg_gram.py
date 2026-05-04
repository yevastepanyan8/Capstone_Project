"""
extract_embeddings_vgg_gram.py
------------------------------
Extracts 32,896-dim VGG-19 Gram matrix style embeddings for any genre dataset.

Approach (Gatys et al., 2016 — neural style transfer):
  1. Pass each image through VGG-19 up to conv3_1 (post-ReLU activation).
  2. Reshape feature map (B, 256, 56, 56) → (B, 256, 56*56).
  3. Compute Gram matrix G = F @ F.T / (C*H*W)  → (B, 256, 256).
  4. Flatten the upper triangle (including diagonal) → 32,896-dim style vector.

The Gram matrix encodes second-order channel statistics (which features co-occur
spatially), which is a classical mathematical formulation of "artistic style" —
texture, brushwork, color correlation patterns. Unlike ResNet's pooled features
(which capture global content) or DINOv2's CLS token (which captures semantics),
Gram-matrix embeddings are explicitly style-focused.

Output is a drop-in replacement for image_embeddings.npy in a separate directory;
the rest of the pipeline (PCA reduction, injection, analysis) works unchanged once
EMBEDDINGS_DIR is pointed at the new root.

Usage
-----
python src/extract_embeddings_vgg_gram.py \\
    --dataset_dir dataset_impressionism \\
    --output_dir  embeddings_vgg_gram/impressionism
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


# VGG-19 features Sequential indices:
#   0: conv1_1  | 5: conv2_1  | 10: conv3_1  | 19: conv4_1  | 28: conv5_1
# We use the post-ReLU activation of conv2_1 (index 6) — fine texture & brushwork style.
# conv3_1 (index 11, 32,896-dim) is richer but impractical on CPU for raw-space methods.
VGG_LAYER_INDEX = 6
VGG_LAYER_CHANNELS = 128
GRAM_DIM = VGG_LAYER_CHANNELS * (VGG_LAYER_CHANNELS + 1) // 2  # 8,256


class VGGGramExtractor(nn.Module):
    """VGG-19 truncated at conv3_1 → normalized Gram matrix → upper triangle."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Slice off everything after the post-ReLU of conv3_1
        self.features = nn.Sequential(
            *list(vgg.features.children())[: VGG_LAYER_INDEX + 1]
        )
        for p in self.features.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        feat = self.features(x)                      # (B, 256, 56, 56)
        B, C, H, W = feat.shape
        F_ = feat.view(B, C, H * W)                  # (B, 256, N)
        gram = torch.bmm(F_, F_.transpose(1, 2)) / (C * H * W)  # (B, 256, 256)
        idx = torch.triu_indices(C, C, device=gram.device)
        return gram[:, idx[0], idx[1]]               # (B, GRAM_DIM)


@torch.no_grad()
def extract_all_embeddings(
    dataset_dir: Path, output_dir: Path, batch_size: int = DEFAULT_BATCH_SIZE
):
    images_dir   = dataset_dir / "images"
    metadata_csv = dataset_dir / "metadata_subset.csv"

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_npy  = output_dir / "image_embeddings.npy"
    out_meta = output_dir / "embedding_metadata.csv"

    device = get_device()
    print(f"Device         : {device}")
    print(f"Dataset dir    : {dataset_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Embedding dim  : {GRAM_DIM}  (VGG-19 conv3_1 Gram upper triangle)")

    df_meta = pd.read_csv(metadata_csv)
    exists  = df_meta["filename"].apply(lambda f: (images_dir / f).exists())
    missing = df_meta[~exists]
    if len(missing) > 0:
        print(f"[warn] Skipping {len(missing)} missing image(s):")
        for fn in missing["filename"]:
            print(f"       {fn}")
    df_meta = df_meta[exists].reset_index(drop=True)

    filtered_csv = output_dir / "_filtered_metadata.csv"
    df_meta.to_csv(filtered_csv, index=False)

    loader = create_dataloader(
        metadata_csv=str(filtered_csv),
        images_root=str(images_dir),
        batch_size=batch_size,
        shuffle=False,
    )
    num_images = len(df_meta)
    print(f"Images found   : {num_images} (after filtering)")

    if num_images == 0:
        print("[error] No valid images found. Check your dataset directory.")
        filtered_csv.unlink(missing_ok=True)
        sys.exit(1)

    model          = VGGGramExtractor().eval().to(device)
    all_embeddings = np.zeros((num_images, GRAM_DIM), dtype=np.float32)
    meta_rows      = []
    current_index  = 0

    for images, batch_meta in tqdm(loader, desc="Extracting VGG Gram embeddings"):
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
    print(f"\nSaved embeddings → {out_npy}  {all_embeddings.shape}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df["embedding_index"] = np.arange(len(meta_df))
    meta_df.to_csv(out_meta, index=False)
    print(f"Saved metadata   → {out_meta}")

    filtered_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract VGG-19 Gram matrix style embeddings (32,896-dim) for a genre dataset."
    )
    parser.add_argument(
        "--dataset_dir", type=Path,
        default=PROJECT_ROOT / "dataset_impressionism",
        help="Path to genre dataset folder (must contain images/ and metadata_subset.csv)",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=PROJECT_ROOT / "embeddings_vgg_gram" / "impressionism",
        help="Directory to save image_embeddings.npy and embedding_metadata.csv",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size for embedding extraction",
    )
    args = parser.parse_args()

    extract_all_embeddings(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
