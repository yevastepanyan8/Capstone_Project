import os
import cv2
import numpy as np
import logging
from skimage import filters, color


# for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

# 1. going through data
def get_image_paths(data_dir):
    genre_paths = {}
    for genre in os.listdir(data_dir):
        genre_dir = os.path.join(data_dir, genre)
        if os.path.isdir(genre_dir):
            genre_paths[genre] = [
                os.path.join(genre_dir, f) for f in os.listdir(genre_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            logging.info(f"Found {len(genre_paths[genre])} images in genre '{genre}'")
    return genre_paths

# 2. loading data and changing color plane
def load_and_convert(image_path, color_space="RGB"):
    if not os.path.exists(image_path):
        logging.warning(f"File not found: {image_path}")
        return None
    img = cv2.imread(image_path)
    if img is None:
        logging.warning(f"Failed to read image: {image_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if color_space == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif color_space == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


# 3. adds more variety
def augment_image(image):
    applied = []
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        applied.append("horizontal flip")
    if np.random.rand() > 0.2:
        image = cv2.flip(image, 0)
        applied.append("vertical flip")
    angle = np.random.uniform(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h))
    applied.append(f"rotation {angle:.2f}°")
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.uniform(-30, 30)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    applied.append(f"contrast {alpha:.2f}, brightness {beta:.2f}")
    if np.random.rand() > 0.5:
        ksize = np.random.choice([3, 5])
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        applied.append(f"Gaussian blur {ksize}x{ksize}")
    logging.info(f"Applied augmentations: {', '.join(applied)}")
    return image

# 4. normalizing
def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    logging.debug(f"Resized to {size} and normalized")
    return image

# 5. texture filtering
def apply_texture_filters(image):
    gray = color.rgb2gray(image)
    edges = filters.sobel(gray)
    logging.debug("Applied Sobel edge filter")
    return edges

# 6. saving
def save_image(image, save_path):
    if image.max() <= 1.0:
        image = (image * 255).astype("uint8")
    cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    logging.info(f"Saved processed image to {save_path}")

# 7. running
def run_preprocessing(data_dir, output_dir):
    genre_paths = get_image_paths(data_dir)
    total_processed, total_skipped = 0, 0

    for genre, paths in genre_paths.items():
        save_genre_dir = os.path.join(output_dir, genre)
        os.makedirs(save_genre_dir, exist_ok=True)

        for path in paths:
            logging.info(f"Processing {path}")
            img = load_and_convert(path, color_space="RGB")

            if img is None:
                logging.warning(f"Skipping unreadable file: {path}")
                total_skipped += 1
                continue  # skip this file safely

            img = augment_image(img)
            img = preprocess_image(img)
            save_name = os.path.join(save_genre_dir, os.path.basename(path))
            save_image(img, save_name)
            total_processed += 1

    logging.info(f"Summary: processed {total_processed} images, skipped {total_skipped} unreadable files")


if __name__ == "__main__":
    data_dir = "../data"          
    output_dir = "../processed_data"
    run_preprocessing(data_dir, output_dir)
