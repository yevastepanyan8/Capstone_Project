"""
Per-Artist Anomaly Detection — Find the most unusual paintings within
each artist's body of work (no injection, purely unsupervised).

Pipeline per artist:
  1. Copy images from WikiArt into dataset_artists/<artist_name>/images/.
  2. Extract 2048-dim ResNet-50 embeddings.
  3. PCA reduction to 50 dims (fitted on the artist's own distribution).
  4. Run ALL anomaly detection methods on both raw and PCA embeddings.
  5. Rank paintings by combined anomaly score (ensemble).
  6. Generate per-method scores + visualisations.

Selected Artists:
  - Claude Monet   (1334 paintings, primarily Impressionism)
  - Rembrandt       (776 paintings, all Baroque)
  - Pablo Picasso   (763 paintings, highly multi-genre)

Usage:
    python run_artist_analysis.py
"""

import shutil
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (
    PROJECT_ROOT, WIKIART_DIR, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    EMBEDDING_DIM, N_PCA_COMPONENTS, RANDOM_STATE, KNN_K, CONTAMINATION,
    IF_N_ESTIMATORS, DEFAULT_BATCH_SIZE,
)

warnings.filterwarnings("ignore")

# ── Artists to analyse ────────────────────────────────────────────────────────
ARTISTS = ["claude monet", "rembrandt", "pablo picasso"]

# ── Directories ───────────────────────────────────────────────────────────────
ARTISTS_BASE   = PROJECT_ROOT / "dataset_artists"
ARTISTS_EMB    = PROJECT_ROOT / "embeddings" / "artists"
ARTISTS_RES    = PROJECT_ROOT / "results" / "artists"

# ── Autoencoder hyper-parameters ──────────────────────────────────────────────
AE_HIDDEN_RAW  = [128, 64, 32]
AE_HIDDEN_PCA  = [32, 24]
AE_LATENT_RAW  = 16
AE_LATENT_PCA  = 10
AE_EPOCHS      = 150
AE_BATCH       = 64
AE_LR          = 1e-3
AE_WD          = 1e-5
AE_PATIENCE    = 15

# ── Number of top outliers to highlight in visualisations ─────────────────────
TOP_K = 10

# ── Reproducibility ──────────────────────────────────────────────────────────
def seed_everything(seed=RANDOM_STATE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Dataset preparation (copy images, create metadata)
# ═════════════════════════════════════════════════════════════════════════════
def prepare_artist_dataset(artist_name: str) -> tuple[Path, pd.DataFrame]:
    """Copy artist images from WikiArt into a dedicated folder."""
    slug = artist_name.replace(" ", "_")
    artist_dir = ARTISTS_BASE / slug
    images_dir = artist_dir / "images"
    meta_csv   = artist_dir / "metadata.csv"

    if meta_csv.exists():
        print(f"  Dataset already exists at {artist_dir}")
        return artist_dir, pd.read_csv(meta_csv)

    images_dir.mkdir(parents=True, exist_ok=True)

    # Read global metadata
    global_meta = pd.read_csv(PROJECT_ROOT / "metadata" / "classes.csv")
    artist_rows = global_meta[global_meta["artist"] == artist_name].copy()

    if len(artist_rows) == 0:
        raise ValueError(f"Artist '{artist_name}' not found in metadata.")

    # Copy images
    copied = []
    for _, row in artist_rows.iterrows():
        src = WIKIART_DIR / row["filename"]
        if not src.exists():
            continue
        # Flatten into images/ using artist-style_title.jpg
        dst = images_dir / Path(row["filename"]).name
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append({
            "filename": Path(row["filename"]).name,
            "original_path": row["filename"],
            "artist": row["artist"],
            "genre": row["genre"],
            "description": row.get("description", ""),
        })

    df = pd.DataFrame(copied)
    df.to_csv(meta_csv, index=False)
    print(f"  Copied {len(df)} paintings → {artist_dir}")
    return artist_dir, df


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Embedding extraction
# ═════════════════════════════════════════════════════════════════════════════
_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def extract_embeddings(artist_dir: Path, metadata: pd.DataFrame, device):
    """Extract ResNet-50 embeddings for all images of an artist."""
    slug = artist_dir.name
    emb_dir = ARTISTS_EMB / slug
    emb_npy = emb_dir / "embeddings.npy"
    meta_csv = emb_dir / "metadata.csv"

    if emb_npy.exists() and meta_csv.exists():
        print(f"  Embeddings already exist at {emb_dir}")
        return np.load(emb_npy), pd.read_csv(meta_csv)

    emb_dir.mkdir(parents=True, exist_ok=True)
    images_dir = artist_dir / "images"

    # Load ResNet-50
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval().to(device)

    embeddings_list = []
    valid_rows = []

    for i, row in metadata.iterrows():
        img_path = images_dir / row["filename"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = _transform(img).unsqueeze(0).to(device)
            emb = backbone(tensor).squeeze().cpu().numpy()
            embeddings_list.append(emb)
            valid_rows.append(row)
        except Exception as e:
            print(f"    ⚠ Skipping {row['filename']}: {e}")

    embeddings = np.stack(embeddings_list)
    meta_out = pd.DataFrame(valid_rows).reset_index(drop=True)

    np.save(emb_npy, embeddings)
    meta_out.to_csv(meta_csv, index=False)
    print(f"  Extracted {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim) → {emb_dir}")
    return embeddings, meta_out


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 3 — PCA reduction
# ═════════════════════════════════════════════════════════════════════════════
def reduce_embeddings(embeddings: np.ndarray, emb_dir: Path):
    """PCA reduce and save."""
    pca_npy = emb_dir / "embeddings_pca50.npy"
    if pca_npy.exists():
        print(f"  PCA embeddings already exist")
        return np.load(pca_npy)

    n_components = min(N_PCA_COMPONENTS, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)
    np.save(pca_npy, reduced)
    var = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {embeddings.shape[1]} → {n_components} dims ({var:.1%} variance)")
    return reduced


# ═════════════════════════════════════════════════════════════════════════════
#  ANOMALY DETECTION METHODS
# ═════════════════════════════════════════════════════════════════════════════

# ── 1. Cosine Similarity ─────────────────────────────────────────────────────
def run_cosine_similarity(embeddings):
    """Score = 1 - cosine_similarity to centroid (higher = more anomalous)."""
    centroid = embeddings.mean(axis=0, keepdims=True)
    # cosine distance
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings, centroid).flatten()
    scores = 1.0 - sim
    return scores


# ── 2. Isolation Forest ──────────────────────────────────────────────────────
def run_isolation_forest(embeddings):
    """Score = negated decision function (higher = more anomalous)."""
    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    iso.fit(embeddings)
    scores = -iso.decision_function(embeddings)
    return scores


# ── 3. KS Test ───────────────────────────────────────────────────────────────
def run_ks_test(embeddings, k=KNN_K):
    """Per-painting KS-test: compare KNN neighbourhood vs global per dim."""
    from scipy.stats import ks_2samp
    n, d = embeddings.shape
    dists = pairwise_distances(embeddings, metric="euclidean")
    scores = np.zeros(n)
    for i in range(n):
        nn_idx = np.argsort(dists[i])[1:k+1]
        d_stats = []
        for dim in range(d):
            global_vals = embeddings[:, dim]
            local_vals = embeddings[nn_idx, dim]
            stat, _ = ks_2samp(local_vals, global_vals)
            d_stats.append(stat)
        scores[i] = np.mean(d_stats)
    return scores


# ── 4. Sliced Wasserstein Distance ───────────────────────────────────────────
def run_wasserstein(embeddings, k=KNN_K, n_proj=200):
    """Sliced Wasserstein distance between KNN neighbourhood and global."""
    from scipy.stats import wasserstein_distance
    n, d = embeddings.shape
    dists = pairwise_distances(embeddings, metric="euclidean")
    rng = np.random.RandomState(RANDOM_STATE)
    projections = rng.randn(n_proj, d)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    global_proj = embeddings @ projections.T  # (n, n_proj)
    scores = np.zeros(n)
    for i in range(n):
        nn_idx = np.argsort(dists[i])[1:k+1]
        local_proj = embeddings[nn_idx] @ projections.T
        swd = np.mean([
            wasserstein_distance(global_proj[:, j], local_proj[:, j])
            for j in range(n_proj)
        ])
        scores[i] = swd
    return scores


# ── 5. LOF ────────────────────────────────────────────────────────────────────
def run_lof(embeddings, k=KNN_K):
    """Local Outlier Factor (higher = more anomalous)."""
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False, contamination=CONTAMINATION)
    lof.fit_predict(embeddings)
    scores = -lof.negative_outlier_factor_
    return scores


# ── 6. GMM ────────────────────────────────────────────────────────────────────
def run_gmm(embeddings, n_components=10):
    """GMM negative log-likelihood (higher = more anomalous)."""
    nc = min(n_components, embeddings.shape[0] // 10)
    nc = max(nc, 2)
    gmm = GaussianMixture(n_components=nc, random_state=RANDOM_STATE, covariance_type="full")
    gmm.fit(embeddings)
    scores = -gmm.score_samples(embeddings)
    return scores


# ── 7. HDBSCAN ───────────────────────────────────────────────────────────────
def run_hdbscan(embeddings, min_cluster_size=50):
    """HDBSCAN outlier probabilities (higher = more anomalous)."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            prediction_data=True,
        )
        clusterer.fit(embeddings)
        scores = clusterer.outlier_scores_
        return scores
    except Exception as e:
        print(f"    ⚠ HDBSCAN failed: {e}")
        return None


# ── 8. Autoencoder ───────────────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def run_autoencoder(embeddings, device, hidden_dims, latent_dim):
    """Train autoencoder on ALL paintings, score by reconstruction error."""
    n = embeddings.shape[0]
    input_dim = embeddings.shape[1]

    # 80/20 train/val split
    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_data = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    val_data = torch.tensor(embeddings[val_idx], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_data), batch_size=AE_BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=AE_BATCH, shuffle=False)

    seed_everything()
    model = Autoencoder(input_dim, hidden_dims, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=AE_WD)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss += loss.item() * len(batch)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                v_loss += criterion(model(batch), batch).item() * len(batch)
        v_loss /= len(val_loader.dataset)

        if epoch % 50 == 0 or epoch == 1:
            print(f"      epoch {epoch:>4d}  train={t_loss:.6f}  val={v_loss:.6f}")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= AE_PATIENCE:
                print(f"      early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Score all paintings
    model.eval()
    all_tensor = torch.tensor(embeddings, dtype=torch.float32)
    loader = DataLoader(TensorDataset(all_tensor), batch_size=512, shuffle=False)
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            mse = ((batch - model(batch)) ** 2).mean(dim=1)
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# ═════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE SCORING
# ═════════════════════════════════════════════════════════════════════════════
def ensemble_scores(score_dict: dict) -> np.ndarray:
    """Combine all method scores into a single ensemble ranking.
    Normalise each to [0, 1] then take the mean."""
    scaler = MinMaxScaler()
    normalised = {}
    for name, scores in score_dict.items():
        if scores is None:
            continue
        s = scores.reshape(-1, 1)
        normalised[name] = scaler.fit_transform(s).flatten()
    stacked = np.stack(list(normalised.values()), axis=0)
    return stacked.mean(axis=0)


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(metadata, score_dict, ensemble, artist_name, results_dir, artist_dir):
    """Generate summary visualisations."""
    n_methods = sum(1 for v in score_dict.values() if v is not None)

    # ── 1. Ensemble score distribution + top outliers ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(ensemble, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
    top_idx = np.argsort(ensemble)[::-1][:TOP_K]
    for rank, idx in enumerate(top_idx):
        ax.axvline(ensemble[idx], color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Ensemble Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(f"{artist_name.title()} — Ensemble Anomaly Score Distribution (top-{TOP_K} marked)")
    plt.tight_layout()
    fig.savefig(results_dir / "ensemble_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 2. Per-method score heatmap for top outliers ──────────────────────
    scaler = MinMaxScaler()
    methods_present = {k: v for k, v in score_dict.items() if v is not None}
    norm_scores = {}
    for name, scores in methods_present.items():
        norm_scores[name] = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    top_idx = np.argsort(ensemble)[::-1][:TOP_K]
    heatmap_data = pd.DataFrame({
        name: scores[top_idx] for name, scores in norm_scores.items()
    })
    heatmap_data.index = [metadata.iloc[i]["filename"][:40] for i in top_idx]

    fig2, ax2 = plt.subplots(figsize=(14, max(5, TOP_K * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2,
                cbar_kws={"label": "Normalised Anomaly Score"})
    ax2.set_title(f"{artist_name.title()} — Top-{TOP_K} Outliers: Per-Method Score Heatmap")
    ax2.set_ylabel("Painting")
    plt.tight_layout()
    fig2.savefig(results_dir / "top_outliers_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── 3. Gallery of top outliers (actual images) ────────────────────────
    top_idx = np.argsort(ensemble)[::-1][:min(TOP_K, 8)]
    n_show = len(top_idx)
    cols = min(4, n_show)
    rows = (n_show + cols - 1) // cols
    fig3, axes3 = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes3 = np.array([[axes3]])
    elif rows == 1:
        axes3 = axes3.reshape(1, -1)
    elif cols == 1:
        axes3 = axes3.reshape(-1, 1)

    images_dir = artist_dir / "images"
    for pos, idx in enumerate(top_idx):
        r, c = divmod(pos, cols)
        ax = axes3[r, c]
        img_path = images_dir / metadata.iloc[idx]["filename"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        desc = str(metadata.iloc[idx].get("description", ""))[:35]
        genre = str(metadata.iloc[idx].get("genre", ""))[:30]
        ax.set_title(f"#{pos+1} score={ensemble[idx]:.3f}\n{desc}\n{genre}",
                      fontsize=7, color="red", fontweight="bold")
        ax.axis("off")

    # Hide unused axes
    for pos in range(n_show, rows * cols):
        r, c = divmod(pos, cols)
        axes3[r, c].axis("off")

    fig3.suptitle(f"{artist_name.title()} — Most Anomalous Paintings",
                  fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig3.savefig(results_dir / "top_outliers_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── 4. Gallery of most NORMAL paintings (lowest ensemble score) ───
    normal_idx = np.argsort(ensemble)[:min(TOP_K, 8)]   # ascending = most normal
    n_show_n = len(normal_idx)
    cols_n = min(4, n_show_n)
    rows_n = (n_show_n + cols_n - 1) // cols_n
    fig_n, axes_n = plt.subplots(rows_n, cols_n, figsize=(4 * cols_n, 4.5 * rows_n))
    if rows_n == 1 and cols_n == 1:
        axes_n = np.array([[axes_n]])
    elif rows_n == 1:
        axes_n = axes_n.reshape(1, -1)
    elif cols_n == 1:
        axes_n = axes_n.reshape(-1, 1)

    for pos, idx in enumerate(normal_idx):
        r, c = divmod(pos, cols_n)
        ax = axes_n[r, c]
        img_path = images_dir / metadata.iloc[idx]["filename"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        desc = str(metadata.iloc[idx].get("description", ""))[:35]
        genre = str(metadata.iloc[idx].get("genre", ""))[:30]
        ax.set_title(f"#{pos+1} score={ensemble[idx]:.3f}\n{desc}\n{genre}",
                      fontsize=7, color="green", fontweight="bold")
        ax.axis("off")

    for pos in range(n_show_n, rows_n * cols_n):
        r, c = divmod(pos, cols_n)
        axes_n[r, c].axis("off")

    fig_n.suptitle(f"{artist_name.title()} — Most Normal (Typical) Paintings",
                   fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig_n.savefig(results_dir / "most_normal_gallery.png", dpi=150, bbox_inches="tight")
    plt.close(fig_n)

    # ── 5. Method correlation matrix ──────────────────────────────────
    corr_df = pd.DataFrame(norm_scores)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df.corr(method="spearman"), annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax4, vmin=-1, vmax=1)
    ax4.set_title(f"{artist_name.title()} — Spearman Rank Correlation Between Methods")
    plt.tight_layout()
    fig4.savefig(results_dir / "method_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    print(f"  Saved 5 plots → {results_dir}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN PER-ARTIST PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def analyse_artist(artist_name: str, device):
    slug = artist_name.replace(" ", "_")
    print(f"\n{'═' * 70}")
    print(f"  ARTIST: {artist_name.upper()} ({slug})")
    print(f"{'═' * 70}")

    # 1. Prepare dataset
    print("\n  [Step 1] Preparing dataset ...")
    artist_dir, metadata = prepare_artist_dataset(artist_name)

    # 2. Extract embeddings
    print("  [Step 2] Extracting embeddings ...")
    embeddings_raw, metadata = extract_embeddings(artist_dir, metadata, device)

    # 3. PCA reduction
    emb_dir = ARTISTS_EMB / slug
    print("  [Step 3] PCA reduction ...")
    embeddings_pca = reduce_embeddings(embeddings_raw, emb_dir)

    # 4. Run all methods
    results_dir = ARTISTS_RES / slug
    results_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}

    for space_name, emb in [("raw", embeddings_raw), ("pca50", embeddings_pca)]:
        print(f"\n  ── {space_name.upper()} space ({emb.shape[1]}-dim) ──")

        # Cosine Similarity
        print(f"    [Cosine Similarity] ...", end=" ")
        s = run_cosine_similarity(emb)
        all_scores[f"Cosine ({space_name})"] = s
        print("done")

        # Isolation Forest
        print(f"    [Isolation Forest] ...", end=" ")
        s = run_isolation_forest(emb)
        all_scores[f"IF ({space_name})"] = s
        print("done")

        # LOF
        print(f"    [LOF] ...", end=" ")
        s = run_lof(emb, k=KNN_K)
        all_scores[f"LOF ({space_name})"] = s
        print("done")

        # GMM
        print(f"    [GMM] ...", end=" ")
        s = run_gmm(emb)
        all_scores[f"GMM ({space_name})"] = s
        print("done")

        # HDBSCAN (PCA only for speed)
        if space_name == "pca50":
            print(f"    [HDBSCAN] ...", end=" ")
            s = run_hdbscan(emb)
            if s is not None:
                all_scores[f"HDBSCAN ({space_name})"] = s
                print("done")

        # KS Test (PCA only — too slow on 2048-dim)
        if space_name == "pca50":
            print(f"    [KS Test] ...", end=" ")
            s = run_ks_test(emb, k=KNN_K)
            all_scores[f"KS Test ({space_name})"] = s
            print("done")

        # Wasserstein (PCA only — too slow on 2048-dim)
        if space_name == "pca50":
            print(f"    [Wasserstein] ...", end=" ")
            s = run_wasserstein(emb, k=KNN_K)
            all_scores[f"Wasserstein ({space_name})"] = s
            print("done")

        # Autoencoder
        if space_name == "raw":
            hidden, latent = AE_HIDDEN_RAW, AE_LATENT_RAW
        else:
            hidden, latent = AE_HIDDEN_PCA, AE_LATENT_PCA
        print(f"    [Autoencoder] training ...")
        s = run_autoencoder(emb, device, hidden, latent)
        all_scores[f"AE ({space_name})"] = s
        print(f"    [Autoencoder] done")

    # 5. Ensemble
    print("\n  [Step 5] Computing ensemble scores ...")
    ens = ensemble_scores(all_scores)

    # 6. Save all scores
    scores_df = metadata[["filename", "artist", "genre", "description"]].copy()
    for name, scores in all_scores.items():
        if scores is not None:
            scores_df[name] = scores
    scores_df["Ensemble"] = ens
    scores_df = scores_df.sort_values("Ensemble", ascending=False)
    scores_df.to_csv(results_dir / "all_anomaly_scores.csv", index=False)
    print(f"  Saved scores → all_anomaly_scores.csv ({len(scores_df)} paintings)")

    # 7. Print top outliers & most normal
    print(f"\n  {'─' * 60}")
    print(f"  TOP-{TOP_K} MOST ANOMALOUS PAINTINGS:")
    print(f"  {'─' * 60}")
    for rank, (_, row) in enumerate(scores_df.head(TOP_K).iterrows()):
        desc = str(row.get("description", ""))[:50]
        genre = str(row.get("genre", ""))[:25]
        print(f"    #{rank+1:>2d}  score={row['Ensemble']:.4f}  {desc:<50s}  {genre}")

    print(f"\n  {'─' * 60}")
    print(f"  TOP-{TOP_K} MOST NORMAL (TYPICAL) PAINTINGS:")
    print(f"  {'─' * 60}")
    for rank, (_, row) in enumerate(scores_df.tail(TOP_K).iloc[::-1].iterrows()):
        desc = str(row.get("description", ""))[:50]
        genre = str(row.get("genre", ""))[:25]
        print(f"    #{rank+1:>2d}  score={row['Ensemble']:.4f}  {desc:<50s}  {genre}")

    # 8. Visualisations
    print(f"\n  [Step 8] Generating visualisations ...")
    # Need to restore original index order for plotting
    scores_df_orig_order = metadata[["filename", "artist", "genre", "description"]].copy()
    for name, scores in all_scores.items():
        if scores is not None:
            scores_df_orig_order[name] = scores
    scores_df_orig_order["Ensemble"] = ens

    plot_results(metadata, all_scores, ens, artist_name, results_dir, artist_dir)

    return scores_df


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("=" * 70)
    print("  Per-Artist Anomaly Detection — Unsupervised Outlier Discovery")
    print("  Artists: " + ", ".join(a.title() for a in ARTISTS))
    print("=" * 70)

    all_results = {}
    for artist in ARTISTS:
        result_df = analyse_artist(artist, device)
        all_results[artist] = result_df

    # Cross-artist summary
    print(f"\n\n{'=' * 70}")
    print("  CROSS-ARTIST SUMMARY")
    print(f"{'=' * 70}")
    for artist, df in all_results.items():
        top = df.iloc[0]
        print(f"\n  {artist.title()}:")
        print(f"    Most anomalous: {top.get('description', '')[:60]}")
        print(f"    Genre: {top.get('genre', '')}")
        print(f"    Ensemble score: {top['Ensemble']:.4f}")

    print(f"\n{'=' * 70}")
    print("  Done. Results saved to results/artists/<artist_name>/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
