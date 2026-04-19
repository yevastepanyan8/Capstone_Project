"""
Error-Weighted CAM Heatmaps for Autoencoder Anomaly Detection.

Generates Grad-CAM–style heatmaps that show which spatial regions of a
painting contribute most to the autoencoder reconstruction error.

Approach:
  1. Pass each image through ResNet-50 up to the last conv layer (layer4)
     to get spatial feature maps of shape (2048, 7, 7).
  2. Global-average-pool the spatial maps → 2048-dim embedding vector.
  3. Feed the embedding through the trained autoencoder → reconstruction.
  4. Compute per-channel squared error between input and reconstruction.
  5. Weight each of the 2048 spatial feature maps by its channel error.
  6. Sum across channels → (7, 7) heatmap, upsample to image size.
  7. Overlay on the original painting as a jet-coloured heatmap.

Usage:
    python run_gradcam_heatmaps.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (
    GENRES, RANDOM_STATE, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    genre_dataset_dir, genre_results_dir,
)

warnings.filterwarnings("ignore")

DATASET_TYPE = "injected"
N_SAMPLES = 4          # number of normal + anomaly paintings to visualise per genre
TOP_K_ANOMALY = 4      # pick highest-error anomalies
TOP_K_NORMAL = 4       # pick lowest-error normals (well-reconstructed)

# ── Autoencoder (must match run_autoencoder_analysis.py) ──────────────────────
LATENT_DIM = 16
HIDDEN_DIMS = [128, 64, 32]


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── ResNet-50 feature extractor (spatial + pooled) ───────────────────────────
class ResNetSpatial(nn.Module):
    """Returns both spatial feature maps (2048, H, W) and pooled vector (2048,)."""
    def __init__(self, device):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Everything up to (and including) layer4
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.eval().to(device)

    @torch.no_grad()
    def forward(self, x):
        feat_maps = self.features(x)       # (B, 2048, 7, 7)
        pooled = self.pool(feat_maps)       # (B, 2048, 1, 1)
        pooled = pooled.flatten(1)          # (B, 2048)
        return feat_maps, pooled


# ── Image loading ─────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Inverse normalisation for display
_inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
    std=[1.0 / s for s in IMAGENET_STD],
)


def load_image_tensor(img_path, device):
    """Load a single image, return (preprocessed_tensor, original_PIL)."""
    img = Image.open(img_path).convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(device)
    return tensor, img


def resolve_image_path(filename, genre):
    """Find the actual image file, checking dataset dir first, then wikiart."""
    dataset_path = Path(f"dataset_{genre}/images") / filename
    if dataset_path.exists():
        return dataset_path
    wikiart_path = Path("wikiart/wikiart") / filename
    if wikiart_path.exists():
        return wikiart_path
    return None


# ── Heatmap generation ────────────────────────────────────────────────────────
def compute_error_cam(feat_maps, pooled_emb, ae_model, device):
    """
    Compute error-weighted CAM heatmap.

    Args:
        feat_maps: (1, 2048, H, W) spatial features from ResNet
        pooled_emb: (1, 2048) global-average-pooled embedding
        ae_model: trained autoencoder

    Returns:
        heatmap: (H, W) numpy array, normalised to [0, 1]
        channel_errors: (2048,) per-channel squared error
    """
    ae_model.eval()
    with torch.no_grad():
        recon = ae_model(pooled_emb)  # (1, 2048)
        channel_err = (pooled_emb - recon) ** 2  # (1, 2048)
        channel_err = channel_err.squeeze(0)     # (2048,)

    # Weight spatial maps by channel error: (2048, H, W) * (2048, 1, 1)
    weights = channel_err.unsqueeze(-1).unsqueeze(-1)  # (2048, 1, 1)
    spatial = feat_maps.squeeze(0)                      # (2048, H, W)
    cam = (spatial * weights).sum(dim=0)                # (H, W)

    # ReLU — only keep positive contributions
    cam = F.relu(cam)

    # Normalise to [0, 1]
    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, channel_err.cpu().numpy()


def overlay_heatmap(original_img, cam, alpha=0.5):
    """
    Overlay a CAM heatmap on the original image.

    Args:
        original_img: PIL Image
        cam: (H, W) numpy array in [0, 1]
        alpha: transparency of the heatmap overlay

    Returns:
        overlay: numpy array (H_orig, W_orig, 3) in [0, 1]
    """
    # Resize cam to original image size
    cam_pil = Image.fromarray(np.uint8(255 * cam)).resize(
        original_img.size, resample=Image.BILINEAR
    )
    cam_resized = np.float32(cam_pil) / 255.0

    # Apply colourmap (jet)
    cmap = plt.cm.jet
    heatmap = cmap(cam_resized)[..., :3]  # (H, W, 3) in [0, 1]

    # Original image normalised to [0, 1]
    orig_np = np.float32(original_img.resize(original_img.size)) / 255.0

    # Blend
    overlay = alpha * heatmap + (1 - alpha) * orig_np
    overlay = np.clip(overlay, 0, 1)
    return overlay, cam_resized


# ── Main per-genre pipeline ──────────────────────────────────────────────────
def generate_heatmaps_for_genre(genre, resnet, device):
    print(f"\n{'─' * 70}")
    print(f"  Genre: {genre.upper()}")
    print(f"{'─' * 70}")

    data_dir = genre_dataset_dir(genre, DATASET_TYPE)
    results_dir = genre_results_dir(genre, DATASET_TYPE)
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(data_dir / "metadata.csv")
    embeddings_raw = np.load(data_dir / "embeddings.npy")
    is_anomaly = metadata["is_anomaly"].values

    # Load trained autoencoder (raw space)
    model_path = results_dir / "autoencoder_raw_model.pt"
    if not model_path.exists():
        print(f"  ⚠  No trained model found at {model_path}. Run run_autoencoder_analysis.py first.")
        return

    ae = Autoencoder(2048, HIDDEN_DIMS, LATENT_DIM).to(device)
    ae.load_state_dict(torch.load(model_path, map_location=device))
    ae.eval()

    # Compute reconstruction errors for sample selection
    all_emb = torch.tensor(embeddings_raw, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon = ae(all_emb)
        per_sample_error = ((all_emb - recon) ** 2).mean(dim=1).cpu().numpy()

    # Select samples: top anomalies + well-reconstructed normals
    anomaly_idx = np.where(is_anomaly == 1)[0]
    normal_idx = np.where(is_anomaly == 0)[0]

    # Highest-error anomalies
    anomaly_ranked = anomaly_idx[np.argsort(per_sample_error[anomaly_idx])[::-1]]
    selected_anomalies = anomaly_ranked[:TOP_K_ANOMALY]

    # Lowest-error normals (best reconstructed) + highest-error normals
    normal_ranked_asc = normal_idx[np.argsort(per_sample_error[normal_idx])]
    selected_normals = normal_ranked_asc[:TOP_K_NORMAL]

    print(f"  Selected {len(selected_anomalies)} anomalies (highest error) + {len(selected_normals)} normals (lowest error)")

    # Generate heatmaps
    all_selected = list(selected_normals) + list(selected_anomalies)
    labels = ["normal"] * len(selected_normals) + ["anomaly"] * len(selected_anomalies)

    rows = []
    for idx, label in zip(all_selected, labels):
        row = metadata.iloc[idx]
        filename = row["filename"]
        img_path = resolve_image_path(filename, genre)
        if img_path is None:
            print(f"    ⚠  Image not found: {filename}")
            continue

        # Load image and extract spatial features
        img_tensor, orig_img = load_image_tensor(img_path, device)
        feat_maps, pooled = resnet(img_tensor)

        # Compute error-weighted CAM
        cam, channel_errors = compute_error_cam(feat_maps, pooled, ae, device)

        total_error = per_sample_error[idx]
        rows.append({
            "idx": idx,
            "label": label,
            "filename": filename,
            "artist": row.get("artist", ""),
            "description": row.get("description", ""),
            "total_error": total_error,
            "cam": cam,
            "orig_img": orig_img,
        })

    if not rows:
        print("  ⚠  No images could be loaded.")
        return

    # ── Plot: grid of heatmaps ────────────────────────────────────────────
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, r in enumerate(rows):
        orig = r["orig_img"]
        cam = r["cam"]
        overlay, cam_resized = overlay_heatmap(orig, cam, alpha=0.45)

        # Row 0: original image
        axes[0, col].imshow(orig)
        axes[0, col].set_title(
            f"{'ANOMALY' if r['label'] == 'anomaly' else 'Normal'}\n"
            f"{r['artist'][:25]}\nMSE={r['total_error']:.4f}",
            fontsize=8,
            color="red" if r["label"] == "anomaly" else "green",
            fontweight="bold",
        )
        axes[0, col].axis("off")

        # Row 1: heatmap overlay
        axes[1, col].imshow(overlay)
        axes[1, col].set_title("Error-Weighted CAM", fontsize=8)
        axes[1, col].axis("off")

    fig.suptitle(
        f"{genre.title()} — Reconstruction Error Heatmaps (Autoencoder, Raw 2048-dim)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out_path = results_dir / "ae_error_cam_heatmaps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path.name}")

    # ── Plot: individual high-res heatmaps for the top anomaly ────────────
    anomaly_rows = [r for r in rows if r["label"] == "anomaly"]
    if anomaly_rows:
        r = anomaly_rows[0]  # highest-error anomaly
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes2[0].imshow(r["orig_img"])
        axes2[0].set_title(f"Original\n{r['artist']}", fontsize=10)
        axes2[0].axis("off")

        # Heatmap alone
        cam_resized_full = np.array(
            Image.fromarray(np.uint8(255 * r["cam"])).resize(
                r["orig_img"].size, resample=Image.BILINEAR
            )
        ) / 255.0
        axes2[1].imshow(cam_resized_full, cmap="jet")
        axes2[1].set_title("Error-Weighted CAM", fontsize=10)
        axes2[1].axis("off")

        # Overlay
        overlay, _ = overlay_heatmap(r["orig_img"], r["cam"], alpha=0.45)
        axes2[2].imshow(overlay)
        axes2[2].set_title(f"Overlay (MSE={r['total_error']:.4f})", fontsize=10)
        axes2[2].axis("off")

        fig2.suptitle(
            f"{genre.title()} — Top Anomaly: {r['description'][:60]}",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout()
        out2 = results_dir / "ae_error_cam_top_anomaly.png"
        fig2.savefig(out2, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Saved → {out2.name}")

    # ── Plot: side-by-side normal vs anomaly comparison ───────────────────
    normal_rows = [r for r in rows if r["label"] == "normal"]
    if normal_rows and anomaly_rows:
        fig3, axes3 = plt.subplots(2, 2, figsize=(10, 10))

        for col, (r, title) in enumerate([
            (normal_rows[0], "Normal (Low Error)"),
            (anomaly_rows[0], "Anomaly (High Error)"),
        ]):
            axes3[0, col].imshow(r["orig_img"])
            axes3[0, col].set_title(
                f"{title}\n{r['artist'][:30]}\nMSE={r['total_error']:.4f}",
                fontsize=9,
                color="green" if col == 0 else "red",
                fontweight="bold",
            )
            axes3[0, col].axis("off")

            overlay, _ = overlay_heatmap(r["orig_img"], r["cam"], alpha=0.45)
            axes3[1, col].imshow(overlay)
            axes3[1, col].set_title("Error-Weighted CAM", fontsize=9)
            axes3[1, col].axis("off")

        fig3.suptitle(
            f"{genre.title()} — Normal vs Anomaly Error Attribution",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        out3 = results_dir / "ae_error_cam_comparison.png"
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"  Saved → {out3.name}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("  Error-Weighted CAM Heatmaps — Autoencoder Anomaly Attribution")
    print("=" * 70)

    # Load ResNet-50 once (shared across genres)
    print("  Loading ResNet-50 ...")
    resnet = ResNetSpatial(device)

    for genre in GENRES:
        generate_heatmaps_for_genre(genre, resnet, device)

    print(f"\n{'=' * 70}")
    print("  Done. Heatmaps saved to results/<genre>/injected/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
