#!/usr/bin/env python3
"""
process_tmd_samples.py

Load each tmd_sample image from the dataset directory, convert to Lab color space,
cluster pixels to identify the substrate (background) color, compute a per-pixel
distance from the substrate, normalize to a probability-like map, threshold to
obtain a binary mask, and save both the probability map and binary mask.

Outputs:
  - tmd_sample_N_probability.png  (grayscale probability map, 0-255)
  - tmd_sample_N_mask.png         (binary mask, 0 or 255)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from pathlib import Path


def load_image(path):
    """Load an image from disk and return it in RGB (HxWx3, uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_flake_probability_map(rgb_img, n_clusters=3, method='kmeans', random_state=42):
    """
    Given an RGB image (HxWx3, uint8), compute a flake-likelihood probability map.

    Steps:
      1. Convert RGB -> Lab.
      2. Reshape pixels to (N, 3) for clustering.
      3. Cluster pixels using KMeans or GMM on Lab values.
      4. Identify the dominant cluster (largest pixel count) as substrate.
      5. Compute Euclidean distance from each pixel to the substrate center in Lab space.
      6. Normalize distances to [0, 1] as a probability-like map.

    Returns:
      prob_map (HxW float32 in [0,1]): flake likelihood per pixel.
      labels (HxW int32): cluster label for each pixel.
      centers (np.ndarray): cluster centers in Lab space (n_clusters x 3).
    """
    # Convert to Lab
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    lab_float = lab.astype(np.float32)

    h, w = lab.shape[:2]
    pixels = lab_float.reshape(-1, 3)  # (N, 3)

    # Cluster
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        labels = model.fit_predict(pixels)
        centers = model.cluster_centers_
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(pixels)
        centers = model.means_
    else:
        raise ValueError(f"Unknown method: {method}")

    # Identify dominant cluster (largest pixel count)
    unique, counts = np.unique(labels, return_counts=True)
    dominant_label = unique[np.argmax(counts)]
    substrate_center = centers[dominant_label]  # (3,) in Lab

    # Compute Euclidean distance from each pixel to substrate center
    distances = np.linalg.norm(pixels - substrate_center, axis=1)  # (N,)

    # Normalize to [0, 1]
    d_min, d_max = distances.min(), distances.max()
    if d_max > d_min:
        prob = (distances - d_min) / (d_max - d_min)
    else:
        prob = np.zeros_like(distances)

    prob_map = prob.reshape(h, w).astype(np.float32)
    labels_map = labels.reshape(h, w).astype(np.int32)

    return prob_map, labels_map, centers


def threshold_probability_map(prob_map, threshold=0.5):
    """
    Apply a global threshold to the probability map to obtain a binary mask.
    Returns a uint8 mask (0 or 255).
    """
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    return mask


def process_image(image_path, output_dir, n_clusters=3, method='kmeans',
                  threshold=0.5, random_state=42):
    """
    Process a single image: compute probability map and binary mask, save both.
    """
    print(f"Processing: {image_path.name}")

    # Load
    rgb = load_image(image_path)

    # Compute probability map
    prob_map, labels_map, centers = compute_flake_probability_map(
        rgb, n_clusters=n_clusters, method=method, random_state=random_state
    )

    # Threshold to binary mask
    mask = threshold_probability_map(prob_map, threshold=threshold)

    # Save outputs
    stem = image_path.stem  # e.g. "tmd_sample_0"
    prob_path = output_dir / f"{stem}_probability.png"
    mask_path = output_dir / f"{stem}_mask.png"

    # Probability map: scale to 0-255 for saving as grayscale PNG
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(str(prob_path), prob_uint8)
    cv2.imwrite(str(mask_path), mask)

    # Also save a color overlay visualization
    overlay = rgb.copy()
    overlay[mask > 0] = (0, 255, 0)  # green overlay on detected flakes
    overlay_path = output_dir / f"{stem}_overlay.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    print(f"  -> {prob_path.name}, {mask_path.name}, {overlay_path.name}")
    # Determine dominant cluster center for display
    unique_labels, counts = np.unique(labels_map, return_counts=True)
    dominant_label = unique_labels[np.argmax(counts)]
    print(f"  -> Dominant cluster (substrate) Lab center: {centers[dominant_label]}")
    print(f"  -> Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")

    return prob_map, mask


def main():
    parser = argparse.ArgumentParser(
        description="Process tmd_sample images to detect flakes via color distance from substrate."
    )
    parser.add_argument(
        "--input_dir", type=str, default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing tmd_sample_*.jpg images (default: ./dataset)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to save output probability maps and masks (default: ./results)"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=3,
        help="Number of clusters for pixel grouping (default: 3)"
    )
    parser.add_argument(
        "--method", type=str, choices=['kmeans', 'gmm'], default='kmeans',
        help="Clustering method: kmeans or gmm (default: kmeans)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Global threshold for probability-to-mask conversion (default: 0.5)"
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all tmd_sample images
    image_paths = sorted(input_dir.glob("tmd_sample_*.jpg"))
    if not image_paths:
        print(f"No tmd_sample_*.jpg images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Using method={args.method}, n_clusters={args.n_clusters}, threshold={args.threshold}")
    print()

    for img_path in image_paths:
        process_image(
            img_path, output_dir,
            n_clusters=args.n_clusters,
            method=args.method,
            threshold=args.threshold,
            random_state=args.random_state,
        )
        print()

    print("All images processed successfully.")


if __name__ == "__main__":
    main()