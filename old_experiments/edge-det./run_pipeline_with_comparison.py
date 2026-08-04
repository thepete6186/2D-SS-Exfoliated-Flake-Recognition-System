#!/usr/bin/env python3
"""
run_pipeline_with_comparison.py

Main runner script that saves probability maps before and after post-processing
with varying hyperparameters.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

from load import load_image, find_sample_images
from predict import compute_flake_probability_map
from post_processing import threshold_probability_map, filter_noise_with_blur, filter_noise_with_edges, save_results, save_thresholded_probability


def process_image_with_comparison(image_path, output_dir, n_clusters=3, method='kmeans',
                                  threshold=0.5, random_state=42, blur_ksize=5, min_flake_area=100, 
                                  low_threshold=0.02, use_edge_filter=False, edge_weight=0.2, 
                                  edge_method='canny', use_morphology=True, morph_kernel=3):
    """
    Process a single image: compute probability map and save before/after post-processing.
    """
    print(f"Processing: {image_path.name}")

    # Load
    rgb = load_image(image_path)

    # Compute probability map
    prob_map, labels_map, centers = compute_flake_probability_map(
        rgb, n_clusters=n_clusters, method=method, random_state=random_state
    )

    # Save raw probability map (before post-processing)
    stem = image_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_prob_path = output_dir / f"{stem}_probability_raw.png"
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(str(raw_prob_path), prob_uint8)
    print(f"  -> Saved raw probability: {raw_prob_path.name}")

    # Apply post-processing
    if use_edge_filter:
        prob_map_processed = filter_noise_with_edges(
            prob_map, rgb, blur_ksize=blur_ksize, min_area=min_flake_area, 
            low_threshold=low_threshold, edge_weight=edge_weight, edge_method=edge_method, 
            use_morphology=use_morphology, morph_kernel=morph_kernel
        )
    else:
        prob_map_processed = filter_noise_with_blur(
            prob_map, blur_ksize=blur_ksize, min_area=min_flake_area, 
            low_threshold=low_threshold, use_morphology=use_morphology, morph_kernel=morph_kernel
        )

    # Save processed probability map
    processed_prob_path = output_dir / f"{stem}_probability_processed.png"
    prob_uint8_processed = (prob_map_processed * 255).astype(np.uint8)
    cv2.imwrite(str(processed_prob_path), prob_uint8_processed)
    print(f"  -> Saved processed probability: {processed_prob_path.name}")

    # Threshold to binary mask
    mask = threshold_probability_map(prob_map_processed, threshold=threshold)

    # Save mask and overlay
    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    
    overlay = rgb.copy()
    overlay[mask > 0] = (0, 255, 0)  # green overlay
    overlay_path = output_dir / f"{stem}_overlay.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    
    print(f"  -> Saved: {mask_path.name}, {overlay_path.name}")
    
    # Determine dominant cluster center for display
    unique_labels, counts = np.unique(labels_map, return_counts=True)
    dominant_label = unique_labels[np.argmax(counts)]
    print(f"  -> Dominant cluster (substrate) Lab center: {centers[dominant_label]}")
    print(f"  -> Raw probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    print(f"  -> Processed probability range: [{prob_map_processed.min():.4f}, {prob_map_processed.max():.4f}]")

    return prob_map, prob_map_processed, mask


def main():
    parser = argparse.ArgumentParser(
        description="Process tmd_sample images with probability map comparison."
    )
    parser.add_argument(
        "--input_dir", type=str, default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Directory containing tmd_sample_*.jpg images (default: ./dataset)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to save output (default: ./results)"
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
        "--min_flake_area", type=int, default=1000,
        help="Minimum area for a component to be kept (default: 1000 pixels)"
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
    print(f"min_flake_area={args.min_flake_area}, blur_ksize={args.blur_ksize}")
    print()

    for img_path in image_paths:
        process_image_with_comparison(
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