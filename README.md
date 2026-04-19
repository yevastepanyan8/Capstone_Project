<div align="center">

# 🎨 Anomaly Detection in Artwork Using Deep Learning Embeddings

**Capstone Project — American University of Armenia**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f7931e?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Detecting stylistically anomalous paintings within art genres using CNN embeddings and multiple anomaly detection techniques*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#-key-results)
- [Methods](#-methods)
- [Pipeline Architecture](#-pipeline-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Detailed Results](#-detailed-results)

---

## Overview

This project investigates whether **stylistically out-of-distribution artwork** can be automatically detected using deep learning embeddings. We extract feature representations from paintings using a pretrained **ResNet-50** CNN, then apply **eight distinct anomaly detection methods** — ranging from classical statistical tests to deep autoencoders and clustering-based approaches — to identify paintings that deviate from their genre's learned distribution.

The evaluation framework uses **controlled anomaly injection**: 75 cross-genre paintings (Cubism, Expressionism, Abstract Expressionism) are injected into each genre dataset (~5% contamination), providing ground-truth labels for rigorous AUC-ROC benchmarking across three art genres: **Impressionism**, **Realism**, and **Romanticism**.

---

## 🏆 Key Results

### AUC-ROC Comparison Across All Methods and Genres

| Method | Impressionism | Realism | Romanticism | **Mean** |
|:---|:---:|:---:|:---:|:---:|
| **Autoencoder (raw 2048-dim)** | **0.9128** | **0.8833** | **0.9048** | **0.9003** |
| LOF (raw 2048-dim) | 0.8025 | 0.8038 | 0.7931 | 0.7998 |
| Cosine Similarity | 0.8334 | 0.7492 | 0.7988 | 0.7938 |
| KS Test | 0.8008 | 0.6790 | 0.7683 | 0.7494 |
| GMM (raw 2048-dim) | 0.7435 | 0.7283 | 0.7271 | 0.7330 |
| Autoencoder (PCA 50-dim) | 0.5534 | 0.5896 | 0.5781 | 0.5737 |
| Sliced Wasserstein Distance | 0.4461 | 0.6242 | 0.5520 | 0.5408 |
| LOF (PCA 50-dim) | 0.4556 | 0.4353 | 0.4101 | 0.4337 |
| GMM (PCA 50-dim) | 0.2995 | 0.4361 | 0.4983 | 0.4113 |
| Isolation Forest | 0.3960 | 0.2679 | 0.3355 | 0.3331 |
| HDBSCAN (PCA 50-dim) | 0.2523 | 0.3136 | 0.2870 | 0.2843 |

> **Key Findings:**
> - The **Autoencoder on raw 2048-dim embeddings** is the top-performing method with a mean AUC of **0.9003**, significantly outperforming all other methods.
> - **LOF (raw)** is the strongest unsupervised baseline (mean AUC = 0.7998), nearly matching Cosine Similarity, confirming that local density estimation in high-dimensional space is effective.
> - **Cosine Similarity** is the strongest non-ML method (mean AUC = 0.7938), confirming that centroid-based distance captures genre coherence.
> - **Raw 2048-dim space consistently outperforms PCA 50-dim** — LOF drops from 0.80 to 0.43, GMM from 0.73 to 0.41, and Autoencoder from 0.90 to 0.57 after PCA reduction.
> - **HDBSCAN and Isolation Forest underperform** (AUC < 0.35), suggesting that injected anomalies are not spatially isolated in the PCA-reduced embedding space.

---

## 🔬 Methods

### 1. Feature Extraction — ResNet-50

| Component | Detail |
|:---|:---|
| **Model** | ResNet-50 (ImageNet1K_V2 weights) |
| **Output** | 2048-dimensional feature vector per image |
| **Preprocessing** | Resize to 224×224, ImageNet normalization |
| **Reduction** | PCA: 2048 → 50 dims (~90% variance explained) |

### 2. Anomaly Detection Techniques

| # | Method | Space | Approach | Score Interpretation |
|:---:|:---|:---:|:---|:---|
| 1 | **Cosine Similarity** | Raw 2048-dim | Distance from genre centroid | `1 - cos_sim` (higher = anomalous) |
| 2 | **Sliced Wasserstein Distance** | PCA 50-dim | KNN neighbourhood vs. global distribution | Inverted normalized SWD |
| 3 | **Kolmogorov-Smirnov Test** | PCA 50-dim | Per-dimension distribution test + BH-FDR correction | Mean D-statistic |
| 4 | **Isolation Forest** | PCA 50-dim | Tree-based isolation scoring | Negated decision function |
| 5 | **Autoencoder** | Raw 2048-dim | Reconstruction error (MSE) | Normalized MSE |
| 6 | **LOF** | Raw / PCA | Local density deviation from k-nearest neighbours | Negated LOF score |
| 7 | **HDBSCAN** | PCA 50-dim | Density-based clustering outlier scores | Outlier probability |
| 8 | **GMM** | Raw / PCA | Gaussian mixture negative log-likelihood | Normalized neg-loglik |

### 3. Evaluation Strategy

- **Ground Truth**: 75 cross-genre paintings injected as known anomalies (~5% contamination)
- **Anomaly Sources**: Cubism, Expressionism, Abstract Expressionism
- **Metric**: AUC-ROC (area under receiver operating characteristic curve)
- **PCA Integrity**: PCA model fitted on clean data only — anomalies projected with pre-learned transform (no data leakage)

---

## 🏗 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WikiArt Dataset ──► Genre Subset (1500 imgs) ──► ResNet-50        │
│                       create_subset.py              extract_embeddings.py
│                                                         │          │
│                                                    2048-dim embeddings
│                                                         │          │
│                                    ┌────────────────────┤          │
│                                    │                    │          │
│                               PCA (50-dim)         Raw (2048-dim)  │
│                              reduce_embeddings.py       │          │
│                                    │                    │          │
├────────────────────────────────────┼────────────────────┤──────────┤
│                     ANOMALY INJECTION                              │
├────────────────────────────────────┼────────────────────┤──────────┤
│                                    │                    │          │
│    create_injection_dataset.py ──► Clean (1500) + Injected (1575)  │
│    (75 cross-genre anomalies)      with is_anomaly ground truth    │
│                                    │                    │          │
├────────────────────────────────────┼────────────────────┤──────────┤
│                     ANOMALY DETECTION                              │
├────────────────────────────────────┼────────────────────┤──────────┤
│                                    │                    │          │
│  ┌──────────────┐  ┌────────────┐  │  ┌──────────────┐  │          │
│  │   Wasserstein │  │  KS Test   │  │  │  Isolation   │  │          │
│  │   Distance    │  │ + BH-FDR   │  │  │   Forest     │  │          │
│  └──────┬───────┘  └─────┬──────┘  │  └──────┬───────┘  │          │
│         │                │         │         │          │          │
│         │     PCA 50-dim space     │         │     Raw 2048-dim    │
│         │                │         │         │          │          │
│         │                │         │  ┌──────┴───────┐  │          │
│         │                │         │  │   Cosine     │  │          │
│         │                │         │  │  Similarity  │  │          │
│         │                │         │  └──────┬───────┘  │          │
│         │                │         │         │          │          │
│         │                │         │  ┌──────┴───────┐  │          │
│         │                │         │  │ Autoencoder  │  │          │
│         │                │         │  │  (MSE loss)  │  │          │
│         │                │         │  └──────┬───────┘  │          │
│         │                │         │         │          │          │
├─────────┴────────────────┴─────────┴─────────┴──────────┴──────────┤
│                        EVALUATION                                  │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    AUC-ROC  ◄── Per-painting anomaly scores vs. is_anomaly labels  │
│    Ensemble ◄── Score fusion, bootstrap CIs, significance tests    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Capstone_Project/
│
├── src/                                # Core pipeline modules
│   ├── config.py                       # Shared constants, paths, hyperparameters
│   ├── utils.py                        # Device selection, model loading utilities
│   ├── create_subset.py                # Filter WikiArt by genre, sample images
│   ├── dataset_loader.py               # PyTorch Dataset / DataLoader
│   ├── extract_embeddings.py           # ResNet-50 → 2048-dim embeddings
│   ├── reduce_embeddings.py            # PCA reduction (2048 → 50 dims)
│   └── create_injection_dataset.py     # Clean + anomaly-injected datasets
│
├── notebooks/                          # Interactive analysis notebooks
│   ├── cosine_similarity_analysis.ipynb
│   ├── wasserstein_analysis.ipynb
│   ├── ks_test_analysis.ipynb
│   ├── isolation_forest_analysis.ipynb
│   ├── embedding_analysis.ipynb        # PCA / UMAP visualisations
│   ├── auc_roc_evaluation.ipynb        # Cross-method AUC benchmark
│   ├── ensemble_analysis.ipynb         # Score fusion & significance tests
│   └── sensitivity_analysis.ipynb      # Hyperparameter robustness
│
├── metadata/                           # Dataset metadata
│   ├── classes.csv                     # WikiArt painting metadata
│   └── wclasses.csv                    # Numeric class encodings
│
├── run_pipeline.py                     # End-to-end data preparation pipeline
├── run_all_analysis.py                 # Run all 4 statistical methods + AUC
├── run_autoencoder_analysis.py         # Autoencoder anomaly detection + AUC
├── run_clustering_analysis.py          # LOF, HDBSCAN, GMM clustering analysis + AUC
├── Data Preprocessing.ipynb            # Exploratory data analysis
├── requirements.txt                    # Python dependencies
└── README.md
```

**Generated directories** (gitignored — reproduced by running the pipeline):
```
├── embeddings/<genre>/                 # Raw + PCA embeddings per genre
│   ├── clean/                          #   Original genre embeddings (1500)
│   └── injected/                       #   Genre + anomaly embeddings (1575)
├── results/<genre>/injected/           # Per-method anomaly score CSVs + models
└── outputs/figures/                    # Visualisation outputs
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- CUDA-capable GPU recommended (CPU works but is slower for embedding extraction and autoencoder training)

### Installation

```bash
git clone git@github.com:yevastepanyan8/Capstone_Project.git
cd Capstone_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Data

Download the [WikiArt dataset](https://www.kaggle.com/datasets/steubk/wikiart) and place it under `wikiart/wikiart/` so that genre folders (`Impressionism/`, `Realism/`, etc.) are direct children.

---

## 💻 Usage

### Option A: Full Automated Pipeline

```bash
# Step 1 — Prepare data: subsets → embeddings → PCA → injection
python run_pipeline.py

# Step 2 — Run all statistical anomaly detection methods
python run_all_analysis.py

# Step 3 — Run autoencoder anomaly detection
python run_autoencoder_analysis.py

# Step 4 — Run clustering-based anomaly detection (LOF, HDBSCAN, GMM)
python run_clustering_analysis.py
```

### Option B: Step-by-Step

<details>
<summary>Click to expand manual steps</summary>

**1. Create genre subsets**
```bash
python src/create_subset.py \
    --metadata metadata/classes.csv \
    --images wikiart/wikiart \
    --output dataset_impressionism \
    --genre impressionism \
    --size 1500
```

**2. Extract embeddings**
```bash
python src/extract_embeddings.py \
    --dataset_dir dataset_impressionism \
    --output_dir embeddings/impressionism
```

**3. PCA reduction**
```bash
python src/reduce_embeddings.py \
    --embeddings_path embeddings/impressionism/image_embeddings.npy \
    --output_dir embeddings/impressionism \
    --genre impressionism
```

**4. Create injection datasets**
```bash
python src/create_injection_dataset.py \
    --genre_dir embeddings/impressionism \
    --genre impressionism
```

**5. Run analysis notebooks**

Set `GENRE` and `DATASET_TYPE` in the first cell of each notebook:
```python
GENRE        = 'impressionism'    # or 'realism', 'romanticism'
DATASET_TYPE = 'injected'         # 'clean' for exploration, 'injected' for evaluation
```

</details>

### Option C: Interactive Notebooks

Open notebooks in `notebooks/` in the recommended order:

1. `cosine_similarity_analysis.ipynb`
2. `wasserstein_analysis.ipynb`
3. `ks_test_analysis.ipynb`
4. `isolation_forest_analysis.ipynb`
5. `auc_roc_evaluation.ipynb` — requires methods 1–4 complete
6. `ensemble_analysis.ipynb` — score fusion, bootstrap CIs, significance tests
7. `sensitivity_analysis.ipynb` — hyperparameter robustness

---

## ⚙ Configuration

All shared hyperparameters are centralised in [`src/config.py`](src/config.py):

| Parameter | Default | Description |
|:---|:---:|:---|
| `N_PCA_COMPONENTS` | 50 | PCA dimensions for reduced embeddings |
| `CONTAMINATION` | 0.05 | Top 5% flagged as anomalies |
| `MIN_ARTIST_IMAGES` | 20 | Minimum paintings for per-artist analysis |
| `KNN_K` | 20 | Neighbours for per-painting scoring |
| `IF_N_ESTIMATORS` | 200 | Isolation Forest trees |
| `SWD_N_PROJECTIONS` | 200 | Random projections for Sliced Wasserstein |
| `DEFAULT_N_ANOMALIES` | 75 | Injected anomalies (~5% of 1500) |
| `RANDOM_STATE` | 42 | Global seed for reproducibility |

### Autoencoder Hyperparameters

| Parameter | Raw (2048-dim) | PCA (50-dim) |
|:---|:---:|:---:|
| Hidden layers | [128, 64, 32] | [32, 24] |
| Latent dimension | 16 | 10 |
| Epochs (max) | 150 | 150 |
| Batch size | 64 | 64 |
| Learning rate | 1e-3 | 1e-3 |
| Early stopping patience | 15 | 15 |

---

## 📊 Detailed Results

### Per-Genre Breakdown

<details>
<summary><strong>Impressionism</strong> (1500 normal + 75 anomalies)</summary>

| Method | AUC-ROC | Status |
|:---|:---:|:---:|
| Autoencoder (raw) | **0.9128** | ✅ |
| Cosine Similarity | 0.8334 | ✅ |
| KS Test | 0.8008 | ✅ |
| LOF (raw) | 0.8025 | ✅ |
| GMM (raw) | 0.7435 | ✅ |
| Autoencoder (PCA) | 0.5534 | ✅ |
| LOF (PCA) | 0.4556 | ⚠️ |
| Wasserstein Distance | 0.4461 | ⚠️ |
| Isolation Forest | 0.3960 | ⚠️ |
| GMM (PCA) | 0.2995 | ⚠️ |
| HDBSCAN (PCA) | 0.2523 | ⚠️ |

</details>

<details>
<summary><strong>Realism</strong> (1500 normal + 75 anomalies)</summary>

| Method | AUC-ROC | Status |
|:---|:---:|:---:|
| Autoencoder (raw) | **0.8833** | ✅ |
| LOF (raw) | 0.8038 | ✅ |
| Cosine Similarity | 0.7492 | ✅ |
| GMM (raw) | 0.7283 | ✅ |
| KS Test | 0.6790 | ✅ |
| Wasserstein Distance | 0.6242 | ✅ |
| Autoencoder (PCA) | 0.5896 | ✅ |
| GMM (PCA) | 0.4361 | ⚠️ |
| LOF (PCA) | 0.4353 | ⚠️ |
| HDBSCAN (PCA) | 0.3136 | ⚠️ |
| Isolation Forest | 0.2679 | ⚠️ |

</details>

<details>
<summary><strong>Romanticism</strong> (1498 normal + 75 anomalies)</summary>

| Method | AUC-ROC | Status |
|:---|:---:|:---:|
| Autoencoder (raw) | **0.9048** | ✅ |
| Cosine Similarity | 0.7988 | ✅ |
| LOF (raw) | 0.7931 | ✅ |
| KS Test | 0.7683 | ✅ |
| GMM (raw) | 0.7271 | ✅ |
| Autoencoder (PCA) | 0.5781 | ✅ |
| Wasserstein Distance | 0.5520 | ✅ |
| GMM (PCA) | 0.4983 | ⚠️ |
| LOF (PCA) | 0.4101 | ⚠️ |
| Isolation Forest | 0.3355 | ⚠️ |
| HDBSCAN (PCA) | 0.2870 | ⚠️ |

</details>

### Method Rankings (by Mean AUC-ROC)

```
 Rank  Method                       Mean AUC    Embedding Space
 ───── ──────────────────────────── ────────── ─────────────────
  1.   Autoencoder (raw)              0.9003    Raw 2048-dim
  2.   LOF (raw)                      0.7998    Raw 2048-dim
  3.   Cosine Similarity              0.7938    Raw 2048-dim
  4.   KS Test                        0.7494    PCA 50-dim
  5.   GMM (raw)                      0.7330    Raw 2048-dim
  6.   Autoencoder (PCA)              0.5737    PCA 50-dim
  7.   Wasserstein Distance           0.5408    PCA 50-dim
  8.   LOF (PCA)                      0.4337    PCA 50-dim
  9.   GMM (PCA)                      0.4113    PCA 50-dim
 10.   Isolation Forest               0.3331    PCA 50-dim
 11.   HDBSCAN (PCA)                  0.2843    PCA 50-dim
```

### Observations

1. **Raw embedding space is superior** — All top-4 methods (Autoencoder, LOF, Cosine Similarity, GMM) operate on the full 2048-dim ResNet-50 embeddings. PCA reduction consistently degrades performance across all methods.
2. **Reconstruction-based detection excels** — The autoencoder's ability to learn the normal manifold and flag high-reconstruction-error samples is the most effective approach (AUC = 0.9003).
3. **Local density methods are strong** — LOF (raw) achieves mean AUC = 0.80, confirming that anomalous paintings occupy locally sparse regions in the original embedding space.
4. **Consistent performance** — The autoencoder achieves AUC > 0.88 across all three genres, demonstrating robustness to genre-specific characteristics.
5. **PCA degrades all methods uniformly** — LOF: 0.80 → 0.43, GMM: 0.73 → 0.41, Autoencoder: 0.90 → 0.57. The 50-dim reduction discards discriminative features needed for anomaly detection.
6. **HDBSCAN and Isolation Forest fail** — Both density-isolation methods score below random (AUC < 0.35), indicating that anomalies in art embedding space are not globally isolated but rather interspersed with normal samples.