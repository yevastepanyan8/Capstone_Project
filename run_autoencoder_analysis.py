"""
Autoencoder-based anomaly detection for all 3 genres.

Pipeline per genre:
  1. Load injected dataset (embeddings + metadata with is_anomaly labels).
  2. Train an autoencoder on NORMAL paintings only (is_anomaly == 0).
  3. Score ALL paintings by reconstruction error (MSE).
  4. Compute AUC-ROC using is_anomaly as ground truth.
  5. Save per-painting scores to results/<genre>/injected/.

Usage:
    python run_autoencoder_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import GENRES, RANDOM_STATE, genre_dataset_dir, genre_results_dir

warnings.filterwarnings("ignore")

DATASET_TYPE = "injected"

# ── Autoencoder hyper-parameters ──────────────────────────────────────────────
LATENT_DIM = 16
HIDDEN_DIMS = [128, 64, 32]       # encoder layers (decoder mirrors)
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15                     # early-stopping patience


# ── Reproducibility ───────────────────────────────────────────────────────────
def seed_everything(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Autoencoder model ─────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


# ── Training utilities ────────────────────────────────────────────────────────
def train_autoencoder(model, train_loader, val_loader, device, epochs=EPOCHS):
    """Train with MSE loss + early stopping on validation loss."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            x_hat = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_hat = model(batch)
                val_loss += criterion(x_hat, batch).item() * len(batch)
        val_loss /= len(val_loader.dataset)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    epoch {epoch:>4d}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    early stop at epoch {epoch} (best val_loss={best_val_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def compute_reconstruction_error(model, embeddings, device, batch_size=512):
    """Per-sample MSE reconstruction error."""
    model.eval()
    tensor = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat = model(batch)
            mse = ((batch - x_hat) ** 2).mean(dim=1)
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# ── Per-genre analysis ────────────────────────────────────────────────────────
def run_autoencoder_for_genre(genre, device):
    print(f"\n{'─' * 70}")
    print(f"  Genre: {genre.upper()}")
    print(f"{'─' * 70}")

    # Load data
    data_dir = genre_dataset_dir(genre, DATASET_TYPE)
    embeddings_pca = np.load(data_dir / "embeddings_pca50.npy")
    embeddings_raw = np.load(data_dir / "embeddings.npy")
    metadata = pd.read_csv(data_dir / "metadata.csv")

    is_anomaly = metadata["is_anomaly"].values
    n_normal = (is_anomaly == 0).sum()
    n_anomaly = (is_anomaly == 1).sum()
    print(f"  Data: {len(metadata)} paintings ({n_normal} normal, {n_anomaly} anomalies)")

    results_dir = genre_results_dir(genre, DATASET_TYPE)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Run on both embedding spaces ──────────────────────────────────────
    for emb_name, embeddings in [("pca50", embeddings_pca), ("raw", embeddings_raw)]:
        input_dim = embeddings.shape[1]
        print(f"\n  [{emb_name.upper()} space] dim={input_dim}")

        # Split normal data into train/val (80/20)
        normal_idx = np.where(is_anomaly == 0)[0]
        rng = np.random.RandomState(RANDOM_STATE)
        rng.shuffle(normal_idx)
        split = int(0.8 * len(normal_idx))
        train_idx, val_idx = normal_idx[:split], normal_idx[split:]

        train_data = torch.tensor(embeddings[train_idx], dtype=torch.float32)
        val_data = torch.tensor(embeddings[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(val_data), batch_size=BATCH_SIZE, shuffle=False
        )

        # Choose hidden dims based on embedding space
        if emb_name == "pca50":
            hidden = [32, 24]
            latent = 10
        else:
            hidden = HIDDEN_DIMS
            latent = LATENT_DIM

        seed_everything(RANDOM_STATE)
        model = Autoencoder(input_dim, hidden, latent).to(device)
        print(f"    Model: {input_dim} → {hidden} → {latent} → {hidden[::-1]} → {input_dim}")

        model = train_autoencoder(model, train_loader, val_loader, device)

        # Score ALL paintings
        recon_errors = compute_reconstruction_error(model, embeddings, device)

        # Normalize to [0, 1]
        e_min, e_max = recon_errors.min(), recon_errors.max()
        if e_max > e_min:
            norm_scores = (recon_errors - e_min) / (e_max - e_min)
        else:
            norm_scores = np.zeros_like(recon_errors)

        # AUC-ROC
        auc = roc_auc_score(is_anomaly, norm_scores)
        print(f"    AUC-ROC = {auc:.4f}  (score range: {norm_scores.min():.4f} – {norm_scores.max():.4f})")

        # Save results
        df = metadata[["filename", "artist"]].copy()
        df["ae_recon_error"] = recon_errors
        df["ae_anomaly_score"] = norm_scores
        out_path = results_dir / f"autoencoder_{emb_name}_scores.csv"
        df.to_csv(out_path, index=False)
        print(f"    Saved → {out_path.relative_to(data_dir.parent.parent.parent)}")

        # Save model
        model_path = results_dir / f"autoencoder_{emb_name}_model.pt"
        torch.save(model.state_dict(), model_path)

    return metadata, results_dir


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    seed_everything(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("  Autoencoder Anomaly Detection — All Genres")
    print("=" * 70)

    summary = {}

    for genre in GENRES:
        metadata, results_dir = run_autoencoder_for_genre(genre, device)
        is_anomaly = metadata["is_anomaly"].values

        genre_auc = {}
        for emb_name in ["pca50", "raw"]:
            csv_path = results_dir / f"autoencoder_{emb_name}_scores.csv"
            scores_df = pd.read_csv(csv_path)
            auc = roc_auc_score(is_anomaly, scores_df["ae_anomaly_score"])
            genre_auc[f"AE ({emb_name})"] = auc
        summary[genre] = genre_auc

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Autoencoder AUC-ROC Across All Genres")
    print(f"{'=' * 70}")
    print(f"  {'Method':20s} {'Impressionism':>15s} {'Realism':>15s} {'Romanticism':>15s} {'Mean':>10s}")
    print(f"  {'─' * 65}")
    for method in ["AE (pca50)", "AE (raw)"]:
        aucs = [summary[g].get(method, float("nan")) for g in GENRES]
        mean_auc = np.nanmean(aucs)
        vals = [f"{a:.4f}" for a in aucs]
        print(f"  {method:20s} {vals[0]:>15s} {vals[1]:>15s} {vals[2]:>15s} {mean_auc:>10.4f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
