"""
Clustering-based anomaly detection: LOF, HDBSCAN, GMM for all 3 genres.

Each method scores every painting; AUC-ROC is computed against injected
ground-truth labels. Methods run on both raw 2048-dim and PCA 50-dim spaces.

Usage:
    python run_clustering_analysis.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture

try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import GENRES, KNN_K, RANDOM_STATE, genre_dataset_dir, genre_results_dir

warnings.filterwarnings("ignore")

DATASET_TYPE = "injected"


def load_genre_data(genre):
    """Load embeddings and metadata for a genre."""
    data_dir = genre_dataset_dir(genre, DATASET_TYPE)
    embeddings_raw = np.load(data_dir / "embeddings.npy")
    embeddings_pca = np.load(data_dir / "embeddings_pca50.npy")
    metadata = pd.read_csv(data_dir / "metadata.csv")
    return embeddings_raw, embeddings_pca, metadata


# ── Local Outlier Factor ──────────────────────────────────────────────────────

def run_lof(embeddings, metadata, results_dir, space_name, n_neighbors=KNN_K):
    """LOF: higher negative_outlier_factor means more anomalous."""
    print(f"    [LOF {space_name}] n_neighbors={n_neighbors} ... ", end="", flush=True)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination="auto",
        novelty=False,
        metric="cosine" if space_name == "raw" else "euclidean",
    )
    labels = lof.fit_predict(embeddings)

    # negative_outlier_factor_: more negative = more anomalous
    # Negate so higher = more anomalous
    raw_scores = -lof.negative_outlier_factor_

    # Normalize to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(raw_scores)

    df = metadata[["filename", "artist"]].copy()
    df["lof_raw_score"] = raw_scores
    df["lof_anomaly_score"] = norm_scores
    df["lof_flag"] = (labels == -1).astype(int)

    out_path = results_dir / f"lof_{space_name}_scores.csv"
    df.to_csv(out_path, index=False)

    auc = roc_auc_score(metadata["is_anomaly"], norm_scores)
    print(f"AUC = {auc:.4f}")
    return auc


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def run_hdbscan(embeddings, metadata, results_dir, space_name, min_cluster_size=50):
    """HDBSCAN: outlier scores from probabilities. Points with low probability
    or labelled -1 (noise) are anomalous."""
    print(f"    [HDBSCAN {space_name}] min_cluster_size={min_cluster_size} ... ", end="", flush=True)

    if not HAS_HDBSCAN:
        print("SKIPPED (hdbscan not installed)")
        return None

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric="euclidean",
        core_dist_n_jobs=-1,
    )
    clusterer.fit(embeddings)

    # outlier_scores_: higher = more anomalous (already [0, 1])
    raw_scores = clusterer.outlier_scores_

    # Normalize to [0, 1] in case not already bounded
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(raw_scores)

    df = metadata[["filename", "artist"]].copy()
    df["hdbscan_raw_score"] = raw_scores
    df["hdbscan_anomaly_score"] = norm_scores
    df["hdbscan_label"] = clusterer.labels_
    df["hdbscan_noise"] = (clusterer.labels_ == -1).astype(int)

    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise = (clusterer.labels_ == -1).sum()

    out_path = results_dir / f"hdbscan_{space_name}_scores.csv"
    df.to_csv(out_path, index=False)

    auc = roc_auc_score(metadata["is_anomaly"], norm_scores)
    print(f"AUC = {auc:.4f}  (clusters={n_clusters}, noise={n_noise})")
    return auc


# ── Gaussian Mixture Model ────────────────────────────────────────────────────

def run_gmm(embeddings, metadata, results_dir, space_name, n_components=10):
    """GMM: use negative log-likelihood as anomaly score."""
    print(f"    [GMM {space_name}] n_components={n_components} ... ", end="", flush=True)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full" if space_name == "pca50" else "diag",
        random_state=RANDOM_STATE,
        max_iter=200,
        n_init=3,
    )
    gmm.fit(embeddings)

    # score_samples returns log-likelihood; negate so higher = more anomalous
    log_likelihoods = gmm.score_samples(embeddings)
    raw_scores = -log_likelihoods

    # Normalize to [0, 1]
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max > s_min:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)
    else:
        norm_scores = np.zeros_like(raw_scores)

    df = metadata[["filename", "artist"]].copy()
    df["gmm_neg_loglik"] = raw_scores
    df["gmm_anomaly_score"] = norm_scores

    out_path = results_dir / f"gmm_{space_name}_scores.csv"
    df.to_csv(out_path, index=False)

    auc = roc_auc_score(metadata["is_anomaly"], norm_scores)
    print(f"AUC = {auc:.4f}")
    return auc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Clustering-Based Anomaly Detection — LOF, HDBSCAN, GMM")
    print("=" * 70)

    if not HAS_HDBSCAN:
        print("  ⚠  hdbscan package not found — installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan", "-q"])
        import importlib
        globals()["hdbscan"] = importlib.import_module("hdbscan")
        globals()["HAS_HDBSCAN"] = True
        print("  ✓  hdbscan installed successfully")

    all_results = {}

    for genre in GENRES:
        print(f"\n{'─' * 70}")
        print(f"  Genre: {genre.upper()}")
        print(f"{'─' * 70}")

        embeddings_raw, embeddings_pca, metadata = load_genre_data(genre)
        results_dir = genre_results_dir(genre, DATASET_TYPE)
        results_dir.mkdir(parents=True, exist_ok=True)

        n_normal = (metadata["is_anomaly"] == 0).sum()
        n_anomaly = (metadata["is_anomaly"] == 1).sum()
        print(f"  Data: {len(metadata)} paintings ({n_normal} normal, {n_anomaly} anomalies)")
        print(f"  Embeddings: raw {embeddings_raw.shape}, PCA {embeddings_pca.shape}\n")

        genre_results = {}

        # ── LOF ───────────────────────────────────────────────────────────
        for space_name, emb in [("raw", embeddings_raw), ("pca50", embeddings_pca)]:
            auc = run_lof(emb, metadata, results_dir, space_name)
            genre_results[f"LOF ({space_name})"] = auc

        # ── HDBSCAN (PCA only — infeasible on raw 2048-dim) ─────────────
        auc = run_hdbscan(embeddings_pca, metadata, results_dir, "pca50")
        if auc is not None:
            genre_results["HDBSCAN (pca50)"] = auc

        # ── GMM ──────────────────────────────────────────────────────────
        for space_name, emb in [("raw", embeddings_raw), ("pca50", embeddings_pca)]:
            auc = run_gmm(emb, metadata, results_dir, space_name)
            genre_results[f"GMM ({space_name})"] = auc

        all_results[genre] = genre_results

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY: Clustering AUC-ROC Across All Genres")
    print(f"{'=' * 70}")

    methods = sorted(
        set(m for g in all_results.values() for m in g),
        key=lambda m: -np.mean([all_results[g].get(m, 0) for g in GENRES]),
    )

    print(f"  {'Method':25s} {'Impressionism':>15s} {'Realism':>15s} {'Romanticism':>15s} {'Mean':>10s}")
    print(f"  {'─' * 70}")
    for method in methods:
        aucs = [all_results[g].get(method, float("nan")) for g in GENRES]
        mean_auc = np.nanmean(aucs)
        vals = [f"{a:.4f}" if not np.isnan(a) else "   N/A" for a in aucs]
        status = "✓" if mean_auc > 0.5 else "✗"
        print(f"  {method:25s} {vals[0]:>15s} {vals[1]:>15s} {vals[2]:>15s} {mean_auc:>10.4f}  {status}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
