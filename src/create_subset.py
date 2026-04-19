import os
import shutil
import argparse
import ast
import logging

import pandas as pd

from config import IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


def _parse_genre_cell(value):
    """
    Parse the 'genre' column from classes.csv.

    Examples of raw values:
      "['Impressionism']"
      "['Abstract Expressionism', 'Post Impressionism']"
    This returns a list of lower‑cased genre strings, e.g. ["impressionism"].
    """
    if pd.isna(value):
        return []

    # Already a list
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value]

    text = str(value).strip()

    # Try to interpret as a Python literal list
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip().lower() for v in parsed]
        return [str(parsed).strip().lower()]
    except (ValueError, SyntaxError):
        # Fallback: split a string like "['Impressionism']" manually
        cleaned = text.strip("[]")
        parts = [p.strip(" '\"") for p in cleaned.split(",")]
        return [p.lower() for p in parts if p]


def create_subset(
    metadata_path,
    images_dir,
    output_dir,
    genre=None,
    sample_size=2000,
    random_state=42,
):
    # Load metadata from classes.csv
    df = pd.read_csv(metadata_path)

    print(f"Total rows in metadata: {len(df)}")

    # Parse genre column into a list-of-strings column for robust filtering
    if "genre" not in df.columns:
        raise ValueError("Expected a 'genre' column in the metadata CSV.")

    df["genre_parsed"] = df["genre"].apply(_parse_genre_cell)

    # Filter by genre (e.g. --genre impressionism)
    if genre is not None:
        genre_norm = genre.strip().lower()
        df = df[df["genre_parsed"].apply(lambda genres: genre_norm in genres)]
        print(f"After genre filter ({genre}): {len(df)}")

    if len(df) == 0:
        raise ValueError("No rows left after filtering. Check your --genre value.")

    # Sample subset
    subset_size = min(sample_size, len(df))
    df_sample = df.sample(n=subset_size, random_state=random_state).copy()

    # Drop helper column before saving
    if "genre_parsed" in df_sample.columns:
        df_sample = df_sample.drop(columns=["genre_parsed"])

    print(f"Subset size: {len(df_sample)}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    images_out = os.path.join(output_dir, "images")
    os.makedirs(images_out, exist_ok=True)

    copied = 0
    missing = 0
    missing_images = []

    for _, row in df_sample.iterrows():
        filename = row["filename"]
        src_path = os.path.join(images_dir, filename)

        if os.path.exists(src_path):
            # Ensure subdirectories for classes (e.g. Impressionism/...) are created
            dst_path = os.path.join(images_out, filename)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            # Try alternative extensions before giving up
            found = False
            base, ext = os.path.splitext(src_path)
            for alt_ext in IMAGE_EXTENSIONS:
                alt_path = base + alt_ext
                if alt_path != src_path and os.path.exists(alt_path):
                    dst_path = os.path.join(images_out, os.path.splitext(filename)[0] + alt_ext)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(alt_path, dst_path)
                    copied += 1
                    found = True
                    break
            if not found:
                missing += 1
                missing_images.append(str(src_path))

    # Save subset metadata
    metadata_out = os.path.join(output_dir, "metadata_subset.csv")
    df_sample.to_csv(metadata_out, index=False)

    print("\nSubset creation complete")
    print(f"Images copied: {copied}")
    print(f"Images missing: {missing}")
    if missing_images:
        logger.warning(f"Missing images ({missing}): {missing_images[:10]}{'...' if missing > 10 else ''}")
    print(f"Metadata saved to: {metadata_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata", required=True, help="Path to metadata CSV")
    parser.add_argument("--images", required=True, help="Path to full image folder")
    parser.add_argument("--output", required=True, help="Output subset directory")

    parser.add_argument(
        "--genre",
        default=None,
        help="Filter by genre (e.g. 'impressionism') based on the 'genre' column in metadata",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Subset size (maximum number of images)",
    )

    args = parser.parse_args()

    create_subset(
        metadata_path=args.metadata,
        images_dir=args.images,
        output_dir=args.output,
        genre=args.genre,
        sample_size=args.size,
    )