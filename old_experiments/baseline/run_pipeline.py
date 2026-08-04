#!/usr/bin/env python3
"""
run_pipeline.py

Main runner script that orchestrates the load, predict, and post-processing pipeline.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

from load import load_image, find_sample_images
from predict import compute_flake_probability_map
from post_processing import threshold_probability_map, save_results


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
    prob_path, mask_path, overlay_path = save_results(rgb, prob_map, mask, output_dir, stem)

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
    image_paths = find_sample_images(input_dir)
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