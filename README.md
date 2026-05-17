# Automated Stylistic Anomaly Detection in Artwork Collections Using Deep Learning

Capstone Project, American University of Armenia.
Author: Yeva Stepanyan. Supervisor: Gurgen Hovakimyan.

## 1. Project Objective

This project asks whether a system can identify paintings that
stylistically do not belong in a genre collection, without ever being
trained on labeled anomalies. The framing is unsupervised: the system
learns what normal looks like for a given genre from the embedding
distribution of that genre, and assigns a continuous anomaly score to
every painting. The intended use is as a quantitative decision-support
tool for curators, attribution researchers, and art historians working
with large digital archives. The project is not an authentication
system and does not claim to detect forgeries.

The full study covers:

- Four deep visual embeddings (ResNet-50 standard, ResNet-50
  multi-layer, DINOv2 ViT-B/14, VGG-19 Gram matrix).
- Eleven anomaly detection algorithms covering distance, density,
  isolation, reconstruction, distributional, and clustering paradigms.
- Five canonical genres (Impressionism, Realism, Romanticism, Baroque,
  Northern Renaissance), 1,500 paintings each.
- Two injection regimes for controlled evaluation: standard injection
  (Cubism, Expressionism, Abstract Expressionism) and hard injection
  (Fauvism, Post-Impressionism, Pointillism).

The headline finding is that multi-layer ResNet-50 paired with a deep
autoencoder on raw embeddings reaches a mean AUC-ROC of 0.95 on
standard injection and 0.92 on hard injection. The most interesting
finding is that the ranking of optimal detectors flips between
semantic-content embeddings and style-texture embeddings, which means
stylistic anomaly is not a single problem but a family of geometrically
distinct ones.

## 2. Required Software and Libraries

- Python 3.10 or newer.
- A CUDA-capable GPU is recommended for embedding extraction and
  autoencoder training. The pipeline also runs on CPU but is much
  slower.
- All Python dependencies are listed in `requirements.txt`. The main
  libraries used are PyTorch, torchvision, scikit-learn, hdbscan,
  statsmodels, matplotlib, seaborn, pandas, and jupyter.

Install everything with:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows the activation command is `venv\Scripts\activate`.

## 3. Data Sources and Preprocessing

The raw painting images come from the WikiArt dataset, mirrored on
Kaggle at `https://www.kaggle.com/datasets/steubk/wikiart`. Download it
and unpack into the project so that the genre folders sit at
`wikiart/wikiart/<Genre>/`. Each genre folder contains the painting
JPEGs.

Genre subsets used by the pipeline are produced by
`src/create_subset.py`, which samples 1,500 paintings from each of the
five host genres using a fixed random seed of 42. The script writes
each subset to a directory named `dataset_<genre>/` with an `images/`
folder and a `metadata_subset.csv` file.

The injection step in `src/create_injection_dataset.py` then samples
75 paintings from the chosen anomaly genres and appends their
embeddings to the host-genre embeddings, producing a 5 percent
contamination rate with known binary labels for evaluation.

Variable descriptions for the most relevant CSV columns:

- `filename`: image filename, used as the join key across stages.
- `artist`: artist name parsed from the filename.
- `genre`: genre label as stored in the WikiArt metadata.
- `is_anomaly`: 0 for host-genre paintings and 1 for injected
  anomalies. Used as the ground truth for AUC-ROC.
- `anomaly_genre`: the genre an injected painting was sampled from.

## 4. Steps Required to Reproduce the Results

The project is designed to be reproduced with a single command from
the project root:

```
python run_pipeline.py
```

This script runs the full study in two stages.

Stage 1 builds the data for every combination of embedding type and
genre. For each of the five host genres and each of the four embedding
types it extracts raw embeddings, fits a PCA model and writes a 50
component projection, builds the standard injection dataset (Cubism,
Expressionism, Abstract Expressionism), and builds the hard injection
dataset (Fauvism, Post-Impressionism, Pointillism).

Stage 2 runs the anomaly detectors for every combination of embedding
type and injection regime. For each pair it executes
`run_clustering_analysis.py` (LOF, HDBSCAN, GMM) and
`run_autoencoder_analysis.py` (deep autoencoder with reconstruction
error). Each detector writes per-painting anomaly scores and per-genre
AUC-ROC values into the appropriate `results_*` directory.

The pipeline is idempotent. Each stage checks for its primary output
file and skips itself if the file already exists, so re-running the
script does not redo finished work. To force a stage to re-run, delete
its output directory.

A full cold run on a single GPU takes several hours. CPU-only runs
take substantially longer because of embedding extraction and
autoencoder training time.

## 5. How to Run the Code

The recommended entry point is `run_pipeline.py` as described above.
If a stage needs to be run by itself, the individual scripts are:

- `src/create_subset.py` builds the 1,500-painting subset for one
  genre from the full WikiArt dataset.
- `src/extract_embeddings.py` extracts standard ResNet-50 embeddings
  (2,048 dimensions) for one genre subset.
- `src/extract_embeddings_multilayer.py` extracts multi-layer
  ResNet-50 embeddings (3,584 dimensions) by concatenating pooled
  activations from layers 2, 3, and 4.
- `src/extract_embeddings_dinov2.py` extracts DINOv2 ViT-B/14 CLS-token
  embeddings (768 dimensions).
- `src/extract_embeddings_vgg_gram.py` extracts VGG-19 Gram-matrix
  style embeddings (8,256 dimensions) from `conv2_1` activations.
- `src/reduce_embeddings.py` fits a PCA model to 50 components on the
  clean host-genre embeddings and writes both the projection and the
  fitted PCA object.
- `src/create_injection_dataset.py` builds the clean and injected
  versions of a genre dataset. Pass `--anomaly_genres` and
  `--injection_name` to choose between the standard and hard regimes.
- `run_clustering_analysis.py` runs LOF, HDBSCAN, and GMM on the raw
  and PCA-50 spaces and writes per-painting scores plus AUC-ROC.
  Reads `EMBEDDINGS_DIR`, `RESULTS_DIR`, and `DATASET_TYPE` from
  environment variables.
- `run_autoencoder_analysis.py` trains a deep autoencoder on the
  clean host-genre paintings, scores every painting by reconstruction
  error, and writes per-dimension error attribution plots.
- `run_all_analysis.py` runs the statistical detectors (cosine
  similarity, sliced Wasserstein, Kolmogorov-Smirnov, Isolation
  Forest) for the standard regime.
- `run_gradcam_heatmaps.py` produces error-weighted spatial heatmaps
  for paintings flagged by the autoencoder.

Hyperparameters and paths live in `src/config.py`. Two environment
variables control which embedding and which injection regime the
analysis scripts read:

- `EMBEDDINGS_DIR` selects the embedding root, for example
  `embeddings_vgg_gram`. Defaults to `embeddings_vgg_gram`.
- `RESULTS_DIR` selects the results root, for example
  `results_vgg_gram`. Defaults to `results_vgg_gram`.
- `DATASET_TYPE` selects the injection regime. Use `injected` for
  the standard regime and `injected_hard` for the hard regime.
  Defaults to `injected_hard`.

## 6. How the Paper Figures and Tables Were Generated

All figures and tables in the paper are regenerated programmatically.
No values are hand-edited.

- `figures/pipeline.png` is generated by `pipeline_diagram.py` at the
  project root, which builds the end-to-end pipeline diagram using
  matplotlib.
- `figures/heatmap.png` (embedding-by-method AUC matrix under both
  injection regimes), `figures/rank.png` (per-embedding method
  rankings), and `figures/cam.png` (autoencoder reconstruction-error
  attribution) are generated from
  `notebooks/final_results_summary.ipynb`. The notebook reads the
  per-method AUC CSVs from every `results_*` directory and produces
  the figures used in the Results and Analysis sections.
- The two summary tables in the paper (mean AUC by embedding and mean
  AUC by detection method) are computed in the same notebook from the
  full set of 440 AUC values across the four embeddings, eleven
  methods, five genres, and two regimes.

To regenerate the figures after a fresh pipeline run, open
`notebooks/final_results_summary.ipynb` and execute all cells. Saved
figures are written under `outputs/figures/`.

## 7. Directory Layout

```
Capstone_Project/
  src/
    config.py
    utils.py
    create_subset.py
    dataset_loader.py
    extract_embeddings.py
    extract_embeddings_multilayer.py
    extract_embeddings_dinov2.py
    extract_embeddings_vgg_gram.py
    reduce_embeddings.py
    create_injection_dataset.py
  notebooks/
    cosine_similarity_analysis.ipynb
    wasserstein_analysis.ipynb
    ks_test_analysis.ipynb
    isolation_forest_analysis.ipynb
    embedding_analysis.ipynb
    auc_roc_evaluation.ipynb
    ensemble_analysis.ipynb
    sensitivity_analysis.ipynb
    final_results_summary.ipynb
    genre_anomaly_galleries.ipynb
  metadata/
    classes.csv
    wclasses.csv
  run_pipeline.py
  run_all_analysis.py
  run_autoencoder_analysis.py
  run_clustering_analysis.py
  run_artist_analysis.py
  run_gradcam_heatmaps.py
  pipeline_diagram.py
  paper.tex
  requirements.txt
  README.md
```

Directories produced by the pipeline (not stored in git):

```
  embeddings/<genre>/                      ResNet-50 standard
  embeddings_multilayer/<genre>/           ResNet-50 multi-layer
  embeddings_dinov2/<genre>/               DINOv2 ViT-B/14
  embeddings_vgg_gram/<genre>/             VGG-19 Gram matrix
  results/<genre>/<regime>/                Detection outputs per embedding
  results_multilayer/<genre>/<regime>/
  results_dinov2/<genre>/<regime>/
  results_vgg_gram/<genre>/<regime>/
  outputs/figures/                         Notebook-generated figures
```

## 8. Reproducibility Notes

- All random seeds are fixed at 42 across embedding extraction, PCA
  fitting, injection sampling, GMM initialization, Isolation Forest,
  Wasserstein projections, and autoencoder training.
- Hyperparameters are held fixed across all five genres. No per-genre
  tuning was performed, so reported AUC values reflect a single
  hyperparameter setting rather than the best-tuned setting per
  genre.
- The PCA model is fitted on clean host-genre embeddings only.
  Injected anomalies are projected with the pre-fitted PCA so they
  never influence the principal directions.
- A fresh clone followed by `pip install -r requirements.txt` and
  `python run_pipeline.py` reproduces every numerical value reported
  in the paper.

## 9. Acknowledgments

The author thanks Prof. Emma Chookaszian for the qualitative
art-historical validation of the model outputs, and Prof. Narine
Sarvazyan for guidance on the structure and presentation of this
work.
