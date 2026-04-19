"""
reduce_embeddings.py
--------------------
Fits PCA on L2-normalised embeddings and saves:
  <output_dir>/embeddings_pca50.npy  — reduced embeddings (n, 50)
  <output_dir>/pca_model.pkl         — fitted PCA object
  outputs/figures/pca_variance_<genre>.png — scree + cumulative variance plot

Run once per genre before any analysis notebook:
    # Impressionism (default / backward-compatible)
    python src/reduce_embeddings.py

    # New genre
    python src/reduce_embeddings.py \\
        --embeddings_path embeddings/realism/image_embeddings.npy \\
        --output_dir      embeddings/realism \\
        --genre           realism
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from config import (
    PROJECT_ROOT, EMBEDDINGS_DIR, FIGURES_DIR, N_PCA_COMPONENTS, RANDOM_STATE,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def reduce(embeddings_path: Path, output_dir: Path, genre: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & normalise ──────────────────────────────────────────────────────
    print(f'Loading embeddings from {embeddings_path}...')
    embeddings_raw  = np.load(embeddings_path)
    embeddings_norm = normalize(embeddings_raw, norm='l2')
    print(f'  Shape  : {embeddings_raw.shape}')
    print(f'  Dtype  : {embeddings_raw.dtype}')

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    print(f'\nFitting PCA ({N_PCA_COMPONENTS} components)...')
    pca            = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    embeddings_pca = pca.fit_transform(embeddings_norm)
    cumvar         = np.cumsum(pca.explained_variance_ratio_)

    print(f'  Variance explained : {cumvar[-1]:.2%}')
    print(f'  Dims for 80%       : {(cumvar >= 0.80).argmax() + 1}')
    print(f'  Dims for 90%       : {(cumvar >= 0.90).argmax() + 1}')

    # ── Save reduced embeddings ───────────────────────────────────────────────
    out_npy = output_dir / 'embeddings_pca50.npy'
    np.save(out_npy, embeddings_pca)
    print(f'\nSaved reduced embeddings → {out_npy}')

    # ── Save fitted PCA model ─────────────────────────────────────────────────
    out_pkl = output_dir / 'pca_model.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump(pca, f)
    print(f'Saved PCA model         → {out_pkl}')

    # ── Variance plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(range(1, N_PCA_COMPONENTS + 1), pca.explained_variance_ratio_,
                 marker='o', markersize=3, color='steelblue')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot')

    axes[1].plot(range(1, N_PCA_COMPONENTS + 1), cumvar,
                 marker='o', markersize=3, color='steelblue')
    axes[1].axhline(0.80, color='crimson', linestyle='--', linewidth=1, label='80%')
    axes[1].axhline(0.90, color='orange',  linestyle='--', linewidth=1, label='90%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Variance Explained')
    axes[1].legend()

    plt.suptitle(
        f'PCA on ResNet-50 {genre.title()} Embeddings (n={len(embeddings_raw)})',
        fontsize=12, y=1.02
    )
    plt.tight_layout()

    out_fig = FIGURES_DIR / f'pca_variance_{genre}.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved variance plot     → {out_fig}')
    print(f'\nDone. Notebooks can load embeddings_pca50.npy from {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PCA reduction for any genre embeddings.')
    parser.add_argument(
        '--embeddings_path', type=Path,
        default=EMBEDDINGS_DIR / 'image_embeddings.npy',
        help='Path to image_embeddings.npy'
    )
    parser.add_argument(
        '--output_dir', type=Path,
        default=EMBEDDINGS_DIR,
        help='Directory to save embeddings_pca50.npy and pca_model.pkl'
    )
    parser.add_argument(
        '--genre', type=str, default='impressionism',
        help='Genre label used for the variance plot filename'
    )
    args = parser.parse_args()

    reduce(
        embeddings_path=args.embeddings_path,
        output_dir=args.output_dir,
        genre=args.genre,
    )
