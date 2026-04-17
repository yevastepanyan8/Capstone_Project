"""
create_injection_dataset.py
---------------------------
Creates two versioned embedding datasets for anomaly detection evaluation:

  embeddings/<genre>/clean/
      embeddings.npy        — original genre embeddings  (N, 2048)
      embeddings_pca50.npy  — PCA-reduced                (N, 50)
      metadata.csv          — metadata with is_anomaly=0

  embeddings/<genre>/injected/
      embeddings.npy        — genre + anomaly embeddings      (N+M, 2048)
      embeddings_pca50.npy  — PCA-reduced using SAME model    (N+M, 50)
      metadata.csv          — combined metadata with is_anomaly labels

PCA is applied using the model fitted on CLEAN data only (pca_model.pkl),
so anomalies never influence the projection — this is the correct evaluation setup.

Usage
-----
# Impressionism (default)
python src/create_injection_dataset.py

# Realism
python src/create_injection_dataset.py --genre_dir embeddings/realism --genre realism

# Romanticism
python src/create_injection_dataset.py --genre_dir embeddings/romanticism --genre romanticism

# Custom anomaly genres and count
python src/create_injection_dataset.py \\
    --genre_dir embeddings/realism --genre realism \\
    --n_anomalies 100 \\
    --anomaly_genres Cubism Expressionism Baroque
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
WIKIART_DIR = BASE_DIR / 'wikiart'

# ImageNet preprocessing — must match what was used for the original embeddings
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Dataset for anomaly images ────────────────────────────────────────────────
class AnomalyImageDataset(Dataset):
    """Loads anomaly images directly from a list of paths."""

    def __init__(self, image_paths: list[Path]):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        return TRANSFORM(image), str(path)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():     return torch.device('cuda')
    if torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def load_resnet50(device: torch.device) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = nn.Sequential(*list(model.children())[:-1])
    backbone.eval().to(device)
    return backbone


def extract_embeddings(image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
    """Extract 2048-dim ResNet-50 embeddings for a list of image paths."""
    device  = get_device()
    model   = load_resnet50(device)
    dataset = AnomalyImageDataset(image_paths)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_emb = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc='  Extracting anomaly embeddings'):
            images = images.to(device, non_blocking=True)
            feats  = model(images).view(images.size(0), -1).cpu().numpy()
            all_emb.append(feats)

    return np.concatenate(all_emb, axis=0).astype(np.float32)


def artist_from_filename(path: Path) -> str:
    """Extract artist name from wikiart filename: 'pablo-picasso_title.jpg' → 'pablo picasso'."""
    stem = path.stem  # 'pablo-picasso_guernica-1937'
    artist_slug = stem.split('_')[0]
    return artist_slug.replace('-', ' ')


def sample_anomaly_images(
    anomaly_genres: list[str],
    n_total: int,
    random_state: int = 42,
) -> tuple[list[Path], list[str]]:
    """
    Sample n_total images evenly across the specified genre folders.
    Returns (image_paths, genre_labels).
    """
    rng = np.random.RandomState(random_state)

    # Map genre name to folder (handle spacing/casing variations)
    genre_folder_map = {f.name.lower().replace('_', ' '): f
                        for f in WIKIART_DIR.iterdir() if f.is_dir()}

    sampled_paths  = []
    sampled_genres = []
    n_per_genre    = max(1, n_total // len(anomaly_genres))

    for genre in anomaly_genres:
        genre_key = genre.lower().replace('_', ' ')
        if genre_key not in genre_folder_map:
            print(f'  [warn] Genre folder not found: {genre} — skipping')
            continue

        folder = genre_folder_map[genre_key]
        all_images = sorted(folder.glob('*.jpg')) + sorted(folder.glob('*.png'))

        if len(all_images) == 0:
            print(f'  [warn] No images found in {folder} — skipping')
            continue

        n_sample = min(n_per_genre, len(all_images))
        chosen   = rng.choice(len(all_images), size=n_sample, replace=False)
        chosen_paths = [all_images[i] for i in chosen]

        sampled_paths.extend(chosen_paths)
        sampled_genres.extend([genre] * n_sample)
        print(f'  Sampled {n_sample} images from {folder.name}')

    return sampled_paths, sampled_genres


# ── Main ──────────────────────────────────────────────────────────────────────
def create_datasets(
    genre_dir: Path,
    genre: str,
    n_anomalies: int,
    anomaly_genres: list[str],
    random_state: int = 42,
):
    existing_embeddings = genre_dir / 'image_embeddings.npy'
    existing_metadata   = genre_dir / 'embedding_metadata.csv'
    pca_model_path      = genre_dir / 'pca_model.pkl'
    clean_dir           = genre_dir / 'clean'
    injected_dir        = genre_dir / 'injected'

    # ── Validate prerequisites ────────────────────────────────────────────────
    for p, name in [
        (existing_embeddings, 'image_embeddings.npy'),
        (existing_metadata,   'embedding_metadata.csv'),
        (pca_model_path,      'pca_model.pkl'),
    ]:
        if not p.exists():
            print(f'[error] Missing prerequisite: {p}')
            print('  Run src/extract_embeddings.py and src/reduce_embeddings.py first.')
            sys.exit(1)

    # ── Load clean assets ─────────────────────────────────────────────────────
    print(f'Loading existing {genre} embeddings...')
    normal_emb  = np.load(existing_embeddings)
    normal_meta = pd.read_csv(existing_metadata).reset_index(drop=True)

    print(f'  Normal embeddings : {normal_emb.shape}')
    print(f'  Normal metadata   : {normal_meta.shape}')

    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    print(f'  PCA model loaded  : {pca.n_components_} components')

    # ── Save clean dataset ────────────────────────────────────────────────────
    print('\nSaving CLEAN dataset...')
    clean_dir.mkdir(parents=True, exist_ok=True)

    clean_meta = normal_meta.copy()
    clean_meta['is_anomaly']    = 0
    clean_meta['anomaly_type']  = 'normal'
    clean_meta['anomaly_genre'] = genre

    # PCA on clean (L2-normalise first, same as reduce_embeddings.py)
    normal_emb_norm = normalize(normal_emb, norm='l2')
    normal_pca      = pca.transform(normal_emb_norm)

    np.save(clean_dir / 'embeddings.npy',        normal_emb)
    np.save(clean_dir / 'embeddings_pca50.npy',  normal_pca)
    clean_meta.to_csv(clean_dir / 'metadata.csv', index=False)

    print(f'  Saved to {clean_dir}')
    print(f'  embeddings.npy       : {normal_emb.shape}')
    print(f'  embeddings_pca50.npy : {normal_pca.shape}')

    # ── Sample and embed anomaly images ──────────────────────────────────────
    print(f'\nSampling {n_anomalies} anomaly images from: {anomaly_genres}')
    anomaly_paths, anomaly_genre_labels = sample_anomaly_images(
        anomaly_genres, n_anomalies, random_state
    )

    if len(anomaly_paths) == 0:
        print('[error] No anomaly images found. Check --anomaly_genres values.')
        sys.exit(1)

    print(f'  Total anomaly images sampled: {len(anomaly_paths)}')
    print('\nExtracting anomaly embeddings...')
    anomaly_emb = extract_embeddings(anomaly_paths)
    print(f'  Anomaly embeddings shape: {anomaly_emb.shape}')

    # ── Build anomaly metadata ────────────────────────────────────────────────
    anomaly_rows = []
    for path, genre_label in zip(anomaly_paths, anomaly_genre_labels):
        # Filename relative to wikiart root: e.g. "Cubism/pablo-picasso_guernica.jpg"
        rel_filename = f'{path.parent.name}/{path.name}'
        anomaly_rows.append({
            'filename'     : rel_filename,
            'artist'       : artist_from_filename(path),
            'genre'        : f"['{genre_label}']",
            'description'  : path.stem,
            'phash'        : '',
            'width'        : -1,
            'height'       : -1,
            'genre_count'  : 1,
            'subset'       : 'injected',
            'is_anomaly'   : 1,
            'anomaly_type' : 'cross_genre',
            'anomaly_genre': genre_label,
        })

    anomaly_meta = pd.DataFrame(anomaly_rows)

    # ── Save injected dataset ─────────────────────────────────────────────────
    print('\nSaving INJECTED dataset...')
    injected_dir.mkdir(parents=True, exist_ok=True)

    injected_emb     = np.vstack([normal_emb, anomaly_emb])
    anomaly_emb_norm = normalize(anomaly_emb, norm='l2')
    anomaly_pca      = pca.transform(anomaly_emb_norm)
    injected_pca     = np.vstack([normal_pca, anomaly_pca])

    # Align columns before concat
    all_cols = set(clean_meta.columns) | set(anomaly_meta.columns)
    for col in all_cols - set(clean_meta.columns):
        clean_meta[col] = None
    for col in all_cols - set(anomaly_meta.columns):
        anomaly_meta[col] = None

    injected_meta = pd.concat(
        [clean_meta, anomaly_meta], ignore_index=True
    )[sorted(all_cols)]

    np.save(injected_dir / 'embeddings.npy',        injected_emb)
    np.save(injected_dir / 'embeddings_pca50.npy',  injected_pca)
    injected_meta.to_csv(injected_dir / 'metadata.csv', index=False)

    print(f'  Saved to {injected_dir}')
    print(f'  embeddings.npy       : {injected_emb.shape}')
    print(f'  embeddings_pca50.npy : {injected_pca.shape}')
    print(f'  Normal paintings     : {len(normal_meta)}')
    print(f'  Injected anomalies   : {len(anomaly_meta)}')
    print(f'  Contamination rate   : {len(anomaly_meta)/len(injected_meta):.1%}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n── Summary ──────────────────────────────────────────────────────')
    print(f'  Clean    → {clean_dir}')
    print(f'  Injected → {injected_dir}')
    print()
    print('  Anomaly breakdown:')
    for genre_label in anomaly_genres:
        n = (anomaly_meta['anomaly_genre'] == genre_label).sum()
        print(f'    {genre_label:<35} {n} images')
    print()
    print(f'  Set GENRE="{genre}" and DATASET_TYPE="injected" in notebook config cells.')
    print('  Use the is_anomaly column as ground truth for AUC-ROC evaluation.')


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create clean and anomaly-injected embedding datasets.'
    )
    parser.add_argument(
        '--genre_dir', type=Path,
        default=BASE_DIR / 'embeddings' / 'impressionism',
        help='Path to genre embeddings folder (contains image_embeddings.npy, pca_model.pkl)'
    )
    parser.add_argument(
        '--genre', type=str, default='impressionism',
        help='Genre name — used as label in metadata and notebook config'
    )
    parser.add_argument(
        '--n_anomalies', type=int, default=75,
        help='Total number of anomaly images to inject (default: 75 ≈ 5%% of 1500)'
    )
    parser.add_argument(
        '--anomaly_genres', nargs='+',
        default=['Cubism', 'Expressionism', 'Abstract_Expressionism'],
        help='Wikiart genre folder names to sample anomalies from'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random seed for reproducibility'
    )
    args = parser.parse_args()

    print('── Injection Dataset Creator ─────────────────────────────────────')
    print(f'  Genre dir       : {args.genre_dir}')
    print(f'  Genre           : {args.genre}')
    print(f'  Anomaly genres  : {args.anomaly_genres}')
    print(f'  N anomalies     : {args.n_anomalies}')
    print(f'  Random state    : {args.random_state}')
    print()

    create_datasets(
        genre_dir      = args.genre_dir,
        genre          = args.genre,
        n_anomalies    = args.n_anomalies,
        anomaly_genres = args.anomaly_genres,
        random_state   = args.random_state,
    )
