"""
reduce_embeddings.py
--------------------
Fits PCA on the L2-normalised image embeddings and saves:
  - embeddings/embeddings_pca50.npy   : reduced embeddings (n, 50)
  - embeddings/pca_model.pkl          : fitted PCA object (for transforming new data)
  - outputs/figures/pca_variance.png  : scree + cumulative variance plot

Run once before any analysis notebook that needs PCA-reduced embeddings:
    python src/reduce_embeddings.py
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR  = BASE_DIR / 'embeddings'
OUTPUTS_DIR     = BASE_DIR / 'outputs' / 'figures'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS = 50

# ── Load & normalise ─────────────────────────────────────────────────────────
print('Loading embeddings...')
embeddings_raw = np.load(EMBEDDINGS_DIR / 'image_embeddings.npy')
embeddings_norm = normalize(embeddings_raw, norm='l2')
print(f'  Raw shape      : {embeddings_raw.shape}')
print(f'  Dtype          : {embeddings_raw.dtype}')

# ── Fit PCA ──────────────────────────────────────────────────────────────────
print(f'\nFitting PCA ({N_COMPONENTS} components)...')
pca = PCA(n_components=N_COMPONENTS, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_norm)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f'  Variance explained : {cumvar[-1]:.2%}')
print(f'  Dims for 80%       : {(cumvar >= 0.80).argmax() + 1}')
print(f'  Dims for 90%       : {(cumvar >= 0.90).argmax() + 1}')

# ── Save reduced embeddings ───────────────────────────────────────────────────
out_npy = EMBEDDINGS_DIR / 'embeddings_pca50.npy'
np.save(out_npy, embeddings_pca)
print(f'\nSaved reduced embeddings → {out_npy}')

# ── Save fitted PCA model ─────────────────────────────────────────────────────
out_pkl = EMBEDDINGS_DIR / 'pca_model.pkl'
with open(out_pkl, 'wb') as f:
    pickle.dump(pca, f)
print(f'Saved PCA model         → {out_pkl}')

# ── Variance plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(range(1, N_COMPONENTS + 1), pca.explained_variance_ratio_,
             marker='o', markersize=3, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot')

axes[1].plot(range(1, N_COMPONENTS + 1), cumvar,
             marker='o', markersize=3, color='steelblue')
axes[1].axhline(0.80, color='crimson', linestyle='--', linewidth=1, label='80%')
axes[1].axhline(0.90, color='orange',  linestyle='--', linewidth=1, label='90%')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()

plt.suptitle(f'PCA on ResNet-50 Impressionism Embeddings (n={len(embeddings_raw)})',
             fontsize=12, y=1.02)
plt.tight_layout()

out_fig = OUTPUTS_DIR / 'pca_variance.png'
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved variance plot     → {out_fig}')

print('\nDone. All analysis notebooks can now load embeddings_pca50.npy.')
