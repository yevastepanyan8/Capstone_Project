"""
Shared configuration for the artwork anomaly detection pipeline.

All magic numbers, paths, and constants live here so every script
and notebook imports from a single source of truth.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
METADATA_DIR = PROJECT_ROOT / "metadata"
WIKIART_DIR  = PROJECT_ROOT / "wikiart" / "wikiart"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR  = PROJECT_ROOT / "results"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
FIGURES_DIR  = OUTPUTS_DIR / "figures"

# ── ImageNet preprocessing ────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = (224, 224)
EMBEDDING_DIM = 2048

# ── PCA ───────────────────────────────────────────────────────────────────────
N_PCA_COMPONENTS = 50

# ── Anomaly detection (shared across methods) ────────────────────────────────
CONTAMINATION     = 0.05
MIN_ARTIST_IMAGES = 20
KNN_K             = 20
RANDOM_STATE      = 42

# ── Isolation Forest ──────────────────────────────────────────────────────────
IF_N_ESTIMATORS  = 200
IF_N_ARTIST_DIMS = 10

# ── Sliced Wasserstein Distance ───────────────────────────────────────────────
SWD_N_PROJECTIONS = 200

# ── Supported genres ──────────────────────────────────────────────────────────
GENRES = ["impressionism", "realism", "romanticism"]

# ── Default anomaly injection ─────────────────────────────────────────────────
DEFAULT_N_ANOMALIES    = 75
DEFAULT_ANOMALY_GENRES = ["Cubism", "Expressionism", "Abstract_Expressionism"]

# ── Batch processing ─────────────────────────────────────────────────────────
DEFAULT_BATCH_SIZE = 32

# ── Supported image extensions ────────────────────────────────────────────────
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ── Path helpers ──────────────────────────────────────────────────────────────
def genre_embeddings_dir(genre: str) -> Path:
    """Return the embeddings directory for a genre, e.g. embeddings/impressionism/."""
    return EMBEDDINGS_DIR / genre


def genre_results_dir(genre: str, dataset_type: str) -> Path:
    """Return the results directory, e.g. results/impressionism/injected/."""
    return RESULTS_DIR / genre / dataset_type


def genre_dataset_dir(genre: str, dataset_type: str) -> Path:
    """Return the embeddings sub-directory, e.g. embeddings/impressionism/injected/."""
    return EMBEDDINGS_DIR / genre / dataset_type
