"""
run_pipeline.py
================

One-command reproduction of every numerical result reported in the
capstone paper.

What this script does, in order:

  1. For each of the five host genres (Impressionism, Realism,
     Romanticism, Baroque, Northern Renaissance) and each of the four
     embedding types (ResNet-50 standard, ResNet-50 multi-layer,
     DINOv2 ViT-B/14, VGG-19 Gram matrix):
       a. Extract the raw embedding for every painting in the genre
          subset (1500 paintings per genre).
       b. Fit a PCA model on those embeddings and write a 50-component
          projection plus the fitted PCA object.
       c. Build the standard injection dataset (Cubism, Expressionism,
          Abstract Expressionism).
       d. Build the hard injection dataset (Fauvism, Post-Impressionism,
          Pointillism).

  2. For each (embedding type, injection regime) combination, run the
     clustering-based detectors (LOF, HDBSCAN, GMM) and the autoencoder
     detector. Each writes per-painting anomaly scores plus AUC-ROC.

Usage
-----
    python run_pipeline.py

Prerequisites
-------------
  - WikiArt downloaded to wikiart/wikiart/ with one folder per genre.
  - Genre subsets created under dataset_<genre>/ via src/create_subset.py.
  - Python 3.10+ with dependencies from requirements.txt.

Re-runs
-------
Each stage checks whether its primary output already exists on disk.
If it does, the stage is skipped. Delete the relevant output directory
to force a stage to re-run.

Notes
-----
A full cold run on a single GPU takes several hours. The script prints
a clear progress banner per stage so it is easy to monitor.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

GENRES = [
    "impressionism",
    "realism",
    "romanticism",
    "baroque",
    "northern_renaissance",
]

EMBEDDING_VARIANTS = [
    {
        "name": "resnet50_standard",
        "extract_script": "src/extract_embeddings.py",
        "emb_dir": "embeddings",
        "results_dir": "results",
    },
    {
        "name": "resnet50_multilayer",
        "extract_script": "src/extract_embeddings_multilayer.py",
        "emb_dir": "embeddings_multilayer",
        "results_dir": "results_multilayer",
    },
    {
        "name": "dinov2_vitb14",
        "extract_script": "src/extract_embeddings_dinov2.py",
        "emb_dir": "embeddings_dinov2",
        "results_dir": "results_dinov2",
    },
    {
        "name": "vgg19_gram",
        "extract_script": "src/extract_embeddings_vgg_gram.py",
        "emb_dir": "embeddings_vgg_gram",
        "results_dir": "results_vgg_gram",
    },
]

INJECTION_REGIMES = [
    {
        "name": "injected",
        "anomaly_genres": ["Cubism", "Expressionism", "Abstract_Expressionism"],
        "label": "standard",
    },
    {
        "name": "injected_hard",
        "anomaly_genres": ["Fauvism", "Post_Impressionism", "Pointillism"],
        "label": "hard",
    },
]


def banner(text, level=1):
    """Print a labeled banner. Level controls the bar character."""
    bar = "=" * 70 if level == 1 else "-" * 70
    print(f"\n{bar}\n  {text}\n{bar}")


def run(cmd, description, env_extra=None):
    """Run a subprocess with the project root as CWD. Exit on failure."""
    print(f"\n[run] {description}")
    print(f"      {' '.join(str(c) for c in cmd)}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if result.returncode != 0:
        print(f"[fail] {description}")
        sys.exit(1)


def file_exists(path):
    return (PROJECT_ROOT / path).is_file()


def dir_has_file(directory, filename):
    return (PROJECT_ROOT / directory / filename).is_file()


# =====================================================================
# Stage 1: extract embeddings, fit PCA, and build injection datasets
# =====================================================================

def stage_extract_and_inject():
    banner("Stage 1: extract embeddings, fit PCA, build injection datasets")

    for variant in EMBEDDING_VARIANTS:
        emb_root = variant["emb_dir"]
        extract_script = variant["extract_script"]

        for genre in GENRES:
            dataset_dir = f"dataset_{genre}"
            genre_emb_dir = f"{emb_root}/{genre}"

            banner(f"{variant['name']} | {genre}", level=2)

            # 1a. Extract raw embeddings
            if dir_has_file(genre_emb_dir, "image_embeddings.npy"):
                print(f"[skip] embeddings already exist at {genre_emb_dir}")
            else:
                run(
                    [extract_script,
                     "--dataset_dir", dataset_dir,
                     "--output_dir", genre_emb_dir],
                    f"extract {variant['name']} embeddings for {genre}",
                )

            # 1b. Fit PCA to 50 components
            if dir_has_file(genre_emb_dir, "pca_model.pkl") and \
               dir_has_file(genre_emb_dir, "embeddings_pca50.npy"):
                print(f"[skip] PCA already fitted at {genre_emb_dir}")
            else:
                run(
                    ["src/reduce_embeddings.py",
                     "--embeddings_path", f"{genre_emb_dir}/image_embeddings.npy",
                     "--output_dir", genre_emb_dir,
                     "--genre", genre],
                    f"fit PCA for {variant['name']} | {genre}",
                )

            # 1c, 1d. Build both injection regimes
            for regime in INJECTION_REGIMES:
                regime_dir = f"{genre_emb_dir}/{regime['name']}"
                if dir_has_file(regime_dir, "embeddings.npy") and \
                   dir_has_file(regime_dir, "metadata.csv"):
                    print(f"[skip] {regime['name']} dataset already built at {regime_dir}")
                    continue

                run(
                    ["src/create_injection_dataset.py",
                     "--genre_dir", genre_emb_dir,
                     "--genre", genre,
                     "--anomaly_genres", *regime["anomaly_genres"],
                     "--injection_name", regime["name"]],
                    f"build {regime['label']} injection for {variant['name']} | {genre}",
                )


# =====================================================================
# Stage 2: run all detectors for every (embedding, regime) pair
# =====================================================================

def stage_run_detectors():
    banner("Stage 2: run anomaly detectors for every (embedding, regime) pair")

    for variant in EMBEDDING_VARIANTS:
        for regime in INJECTION_REGIMES:
            banner(
                f"{variant['name']} | regime={regime['name']}",
                level=2,
            )

            env_extra = {
                "EMBEDDINGS_DIR": variant["emb_dir"],
                "RESULTS_DIR": variant["results_dir"],
                "DATASET_TYPE": regime["name"],
            }

            run(
                ["run_clustering_analysis.py"],
                f"clustering detectors (LOF, HDBSCAN, GMM) | {variant['name']} | {regime['label']}",
                env_extra=env_extra,
            )

            run(
                ["run_autoencoder_analysis.py"],
                f"autoencoder detector | {variant['name']} | {regime['label']}",
                env_extra=env_extra,
            )


# =====================================================================
# Entry point
# =====================================================================

def main():
    banner("Capstone reproduction pipeline")
    print(f"  Project root      : {PROJECT_ROOT}")
    print(f"  Genres            : {len(GENRES)}")
    print(f"  Embedding variants: {len(EMBEDDING_VARIANTS)}")
    print(f"  Injection regimes : {len(INJECTION_REGIMES)}")

    stage_extract_and_inject()
    stage_run_detectors()

    banner("Pipeline finished")
    print("  All anomaly scores and AUC-ROC values are now written to the")
    print("  results_* directories. Open notebooks/final_results_summary.ipynb")
    print("  to regenerate the paper tables and figures.")


if __name__ == "__main__":
    main()
