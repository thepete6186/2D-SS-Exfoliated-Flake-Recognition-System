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
from post_processing import threshold_probability_map, filter_noise_with_blur, filter_noise_with_edges, save_results


def process_image(image_path, output_dir, n_clusters=3, method='kmeans',
                  threshold=0.5, random_state=42, blur_ksize=5, min_flake_area=100, low_threshold=0.02, use_edge_filter=False, edge_weight=0.2, edge_method='canny', use_morphology=True, morph_kernel=3):
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

    # Filter noise with blur and connected components
    if use_edge_filter:
        prob_map = filter_noise_with_edges(prob_map, rgb, blur_ksize=blur_ksize, min_area=min_flake_area, low_threshold=low_threshold, edge_weight=edge_weight, edge_method=edge_method, use_morphology=use_morphology, morph_kernel=morph_kernel)
    else:
        prob_map = filter_noise_with_blur(prob_map, blur_ksize=blur_ksize, min_area=min_flake_area, low_threshold=low_threshold, use_morphology=use_morphology, morph_kernel=morph_kernel)

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
    parser.add_argument(
        "--blur_ksize", type=int, default=5,
        help="Kernel size for Gaussian blur (default: 5)"
    )
    parser.add_argument(
        "--min_flake_area", type=int, default=100,
        help="Minimum area for a component to be kept (default: 100 pixels)"
    )
    parser.add_argument(
        "--low_threshold", type=float, default=0.02,
        help="Threshold to capture low-probability areas (default: 0.02)"
    )
    parser.add_argument(
        "--use_edge_filter", action="store_true",
        help="Use edge-aware filtering to preserve flake edges"
    )
    parser.add_argument(
        "--edge_weight", type=float, default=0.2,
        help="Weight for edge map in combination (default: 0.2)"
    )
    parser.add_argument(
        "--edge_method", type=str, choices=['canny', 'sobel', 'log'], default='canny',
        help="Edge detection method: canny, sobel, or log (default: canny)"
    )
    parser.add_argument(
        "--use_morphology", action="store_true", default=True,
        help="Use morphological cleanup on binary mask (default: True)"
    )
    parser.add_argument(
        "--morph_kernel", type=int, default=3,
        help="Kernel size for morphological operations (default: 3)"
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
            blur_ksize=args.blur_ksize,
            min_flake_area=args.min_flake_area,
            low_threshold=args.low_threshold,
            use_edge_filter=args.use_edge_filter,
            edge_weight=args.edge_weight,
            edge_method=args.edge_method,
            use_morphology=args.use_morphology,
            morph_kernel=args.morph_kernel,
        )
        print()

    print("All images processed successfully.")


if __name__ == "__main__":
    main()