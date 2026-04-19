"""
Run all 4 anomaly detection analyses for all 3 genres, then compute AUC-ROC.

This script replicates the logic from the analysis notebooks but runs
everything programmatically, avoiding the need to edit notebook config
cells 12 times.

Usage:
    python run_all_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.multitest import multipletests

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (
    CONTAMINATION,
    GENRES,
    IF_N_ARTIST_DIMS,
    IF_N_ESTIMATORS,
    KNN_K,
    MIN_ARTIST_IMAGES,
    N_PCA_COMPONENTS,
    RANDOM_STATE,
    SWD_N_PROJECTIONS,
    genre_dataset_dir,
    genre_results_dir,
)

warnings.filterwarnings("ignore")
DATASET_TYPE = "injected"


def load_genre_data(genre):
    """Load embeddings and metadata for a genre."""
    data_dir = genre_dataset_dir(genre, DATASET_TYPE)
    embeddings_raw = np.load(data_dir / "embeddings.npy")
    embeddings_pca = np.load(data_dir / "embeddings_pca50.npy")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    return embeddings_raw, embeddings_pca, metadata


def ensure_results_dir(genre):
    """Create results directory if needed."""
    results_dir = genre_results_dir(genre, DATASET_TYPE)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ── Cosine Similarity Analysis ────────────────────────────────────────────────

def run_cosine_similarity(genre, embeddings_raw, metadata, results_dir):
    """Cosine similarity: 1 - cos_sim(painting, global_centroid)."""
    print(f"  [Cosine Similarity] ", end="", flush=True)

    # L2 normalize
    norms = np.linalg.norm(embeddings_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_norm = embeddings_raw / norms

    # Global centroid
    centroid = embeddings_norm.mean(axis=0)
    centroid /= np.linalg.norm(centroid)

    # Cosine similarity per painting
    cos_sims = embeddings_norm @ centroid
    anomaly_scores = 1 - cos_sims

    df = metadata[["filename", "artist"]].copy()
    df["global_cosine_sim"] = cos_sims
    df["global_anomaly_score"] = anomaly_scores
    df.to_csv(results_dir / "phase1_global_cosine_scores.csv", index=False)

    print(f"done (score range: {anomaly_scores.min():.4f} - {anomaly_scores.max():.4f})")
    return df


# ── Wasserstein Distance Analysis ────────────────────────────────────────────

def sliced_wasserstein_distance(X, Y, n_projections=SWD_N_PROJECTIONS, random_state=RANDOM_STATE):
    """Compute sliced Wasserstein distance between two point clouds."""
    rng = np.random.RandomState(random_state)
    d = X.shape[1]
    projections = rng.randn(n_projections, d)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    distances = []
    for proj in projections:
        x_proj = X @ proj
        y_proj = Y @ proj
        distances.append(wasserstein_distance(x_proj, y_proj))
    return np.mean(distances)


def run_wasserstein(genre, embeddings_pca, metadata, results_dir):
    """Wasserstein: KNN neighbourhood vs global distribution."""
    print(f"  [Wasserstein Distance] ", end="", flush=True)

    n = len(embeddings_pca)
    nn = NearestNeighbors(n_neighbors=min(KNN_K + 1, n), metric="euclidean")
    nn.fit(embeddings_pca)
    _, indices = nn.kneighbors(embeddings_pca)

    swd_scores = np.zeros(n)
    for i in range(n):
        knn_idx = indices[i, 1:]  # exclude self
        knn_embeddings = embeddings_pca[knn_idx]
        swd_scores[i] = sliced_wasserstein_distance(
            knn_embeddings, embeddings_pca, random_state=RANDOM_STATE
        )
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{n}...", end="", flush=True)

    # Normalize to [0, 1]
    swd_min, swd_max = swd_scores.min(), swd_scores.max()
    if swd_max > swd_min:
        swd_norm = (swd_scores - swd_min) / (swd_max - swd_min)
    else:
        swd_norm = np.zeros(n)

    # INVERT: high SWD means painting is in dense cluster (normal)
    # Low SWD means painting is isolated (anomaly)
    swd_norm = 1 - swd_norm

    df = metadata[["filename", "artist"]].copy()
    df["painting_swd"] = swd_scores
    df["painting_swd_norm"] = swd_norm
    df.to_csv(results_dir / "wasserstein_phase2_painting_scores.csv", index=False)

    print(f"done (score range: {swd_norm.min():.4f} - {swd_norm.max():.4f})")
    return df


# ── KS Test Analysis ─────────────────────────────────────────────────────────

def run_ks_test(genre, embeddings_pca, metadata, results_dir):
    """KS test: KNN neighbourhood vs global distribution per PCA dimension."""
    print(f"  [KS Test] ", end="", flush=True)

    n, d = embeddings_pca.shape
    nn = NearestNeighbors(n_neighbors=min(KNN_K + 1, n), metric="euclidean")
    nn.fit(embeddings_pca)
    _, indices = nn.kneighbors(embeddings_pca)

    ks_mean_d = np.zeros(n)
    ks_n_sig = np.zeros(n, dtype=int)

    for i in range(n):
        knn_idx = indices[i, 1:]
        knn_emb = embeddings_pca[knn_idx]

        d_stats = np.zeros(d)
        p_vals = np.zeros(d)
        for dim in range(d):
            stat, pval = ks_2samp(knn_emb[:, dim], embeddings_pca[:, dim])
            d_stats[dim] = stat
            p_vals[dim] = pval

        ks_mean_d[i] = d_stats.mean()

        # BH FDR correction
        reject, _, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
        ks_n_sig[i] = reject.sum()

        if (i + 1) % 500 == 0:
            print(f"{i+1}/{n}...", end="", flush=True)

    df = metadata[["filename", "artist"]].copy()
    df["ks_mean_d"] = ks_mean_d
    df["ks_n_sig"] = ks_n_sig
    df.to_csv(results_dir / "ks_phase2_painting_scores.csv", index=False)

    print(f"done (score range: {ks_mean_d.min():.4f} - {ks_mean_d.max():.4f})")
    return df


# ── Isolation Forest Analysis ─────────────────────────────────────────────────

def run_isolation_forest(genre, embeddings_pca, metadata, results_dir):
    """Isolation Forest on PCA embeddings."""
    print(f"  [Isolation Forest] ", end="", flush=True)

    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )
    iso.fit(embeddings_pca)

    # Negate so higher = more anomalous
    raw_scores = -iso.decision_function(embeddings_pca)

    # Normalize to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros(len(raw_scores))

    labels = iso.predict(embeddings_pca)
    flags = labels == -1

    df = metadata[["filename", "artist"]].copy()
    df["if_global_score"] = raw_scores
    df["if_global_norm"] = norm_scores
    df["if_global_flag"] = flags
    df.to_csv(results_dir / "if_phase1_global_scores.csv", index=False)

    print(f"done (score range: {norm_scores.min():.4f} - {norm_scores.max():.4f})")
    return df


# ── AUC-ROC Evaluation ───────────────────────────────────────────────────────

METHODS = {
    "Cosine Similarity":    "global_anomaly_score",
    "Wasserstein Distance": "painting_swd_norm",
    "KS Test":              "ks_mean_d",
    "Isolation Forest":     "if_global_norm",
}


def compute_auc(genre, results_dir, metadata):
    """Compute AUC-ROC for all methods."""
    y_true = metadata["is_anomaly"].values

    csv_map = {
        "Cosine Similarity":    "phase1_global_cosine_scores.csv",
        "Wasserstein Distance": "wasserstein_phase2_painting_scores.csv",
        "KS Test":              "ks_phase2_painting_scores.csv",
        "Isolation Forest":     "if_phase1_global_scores.csv",
    }

    results = {}
    for method_name, score_col in METHODS.items():
        csv_path = results_dir / csv_map[method_name]
        if not csv_path.exists():
            print(f"    {method_name}: CSV not found")
            continue
        scores_df = pd.read_csv(csv_path)
        # Merge on filename to ensure alignment
        merged = metadata[["filename", "is_anomaly"]].merge(
            scores_df[["filename", score_col]], on="filename", how="inner"
        )
        auc = roc_auc_score(merged["is_anomaly"], merged[score_col])
        results[method_name] = auc

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Running All Anomaly Detection Analyses")
    print("=" * 70)

    all_auc = {}

    for genre in GENRES:
        print(f"\n{'─' * 70}")
        print(f"  Genre: {genre.upper()}")
        print(f"{'─' * 70}")

        embeddings_raw, embeddings_pca, metadata = load_genre_data(genre)
        results_dir = ensure_results_dir(genre)

        n_normal = (metadata["is_anomaly"] == 0).sum()
        n_anomaly = (metadata["is_anomaly"] == 1).sum()
        print(f"  Data: {len(metadata)} paintings ({n_normal} normal, {n_anomaly} anomalies)")
        print(f"  Embeddings: raw {embeddings_raw.shape}, PCA {embeddings_pca.shape}")
        print()

        # Run all 4 methods
        run_cosine_similarity(genre, embeddings_raw, metadata, results_dir)
        run_wasserstein(genre, embeddings_pca, metadata, results_dir)
        run_ks_test(genre, embeddings_pca, metadata, results_dir)
        run_isolation_forest(genre, embeddings_pca, metadata, results_dir)

        # Compute AUC
        print(f"\n  AUC-ROC Results:")
        genre_auc = compute_auc(genre, results_dir, metadata)
        all_auc[genre] = genre_auc
        for method, auc in genre_auc.items():
            status = "OK" if auc > 0.5 else "BELOW RANDOM"
            print(f"    {method:25s} AUC = {auc:.4f}  [{status}]")

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY: AUC-ROC Across All Genres")
    print(f"{'=' * 70}")
    print(f"  {'Method':25s} {'Impressionism':>15s} {'Realism':>15s} {'Romanticism':>15s} {'Mean':>10s}")
    print(f"  {'─' * 70}")
    for method in METHODS:
        aucs = [all_auc.get(g, {}).get(method, float("nan")) for g in GENRES]
        mean_auc = np.nanmean(aucs)
        vals = [f"{a:.4f}" for a in aucs]
        print(f"  {method:25s} {vals[0]:>15s} {vals[1]:>15s} {vals[2]:>15s} {mean_auc:>10.4f}")
    print(f"{'=' * 70}")

    # Check all > 0.5
    all_ok = all(
        auc > 0.5
        for genre_aucs in all_auc.values()
        for auc in genre_aucs.values()
    )
    if all_ok:
        print("\n  All AUC scores are above 0.5 - fixes confirmed working!")
    else:
        print("\n  WARNING: Some AUC scores are still below 0.5!")

    return all_auc


if __name__ == "__main__":
    main()
