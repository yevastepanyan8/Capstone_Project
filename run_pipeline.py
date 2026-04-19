"""
Full pipeline: extract embeddings → PCA → injection datasets → all genres.
Run from project root:  python run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
GENRES = ["impressionism", "realism", "romanticism"]


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(PROJECT),
    )
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)
    print(f"  ✓ {desc}")


for genre in GENRES:
    dataset_dir = f"dataset_{genre}"
    emb_dir = f"embeddings/{genre}"

    # Step 1: Extract embeddings
    run(
        ["src/extract_embeddings.py",
         "--dataset_dir", dataset_dir,
         "--output_dir", emb_dir],
        f"Extract embeddings — {genre}",
    )

    # Step 2: PCA reduction
    run(
        ["src/reduce_embeddings.py",
         "--embeddings_path", f"{emb_dir}/image_embeddings.npy",
         "--output_dir", emb_dir,
         "--genre", genre],
        f"PCA reduction — {genre}",
    )

    # Step 3: Create injection datasets
    run(
        ["src/create_injection_dataset.py",
         "--genre_dir", emb_dir,
         "--genre", genre],
        f"Create injection datasets — {genre}",
    )

print(f"\n{'='*60}")
print("  ALL DONE — embeddings and injection datasets ready")
print(f"{'='*60}")
