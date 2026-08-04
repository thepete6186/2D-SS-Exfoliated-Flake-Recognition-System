#!/usr/bin/env python3
"""
test_edge_aware_denoise.py

Test runner for edge-aware denoising on 5 sample images.
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from load import load_image, find_sample_images
from predict import compute_flake_probability_map
from edge_aware_denoise import (
    clean_probability_with_blurred_edges,
    save_debug_images,
    save_component_stats_csv,
    get_default_config,
    EdgeAwareDenoiseConfig
)


def process_image(
    image_path: Path,
    output_dir: Path,
    config: EdgeAwareDenoiseConfig,
    n_clusters: int = 3,
    method: str = 'kmeans',
    random_state: int = 42
):
    """
    Process a single image with edge-aware denoising.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
        config: Edge-aware denoising configuration
        n_clusters: Number of clusters for probability map
        method: Clustering method
        random_state: Random seed
        
    Returns:
        Tuple of (cleaned_prob_map, final_mask, component_stats)
    """
    print(f"Processing: {image_path.name}")
    
    # Load image
    rgb = load_image(image_path)
    
    # Compute probability map
    prob_map, labels_map, centers = compute_flake_probability_map(
        rgb, n_clusters=n_clusters, method=method, random_state=random_state
    )
    
    # Get substrate color (dominant cluster center in Lab)
    unique_labels, counts = np.unique(labels_map, return_counts=True)
    dominant_label = unique_labels[np.argmax(counts)]
    substrate_color = centers[dominant_label]
    
    print(f"  -> Substrate Lab center: {substrate_color}")
    print(f"  -> Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")
    
    # Apply edge-aware denoising
    cleaned_prob_map, final_mask, debug_images, component_stats = clean_probability_with_blurred_edges(
        rgb, prob_map, config, substrate_color=substrate_color
    )
    
    # Save debug images
    stem = image_path.stem
    save_debug_images(debug_images, output_dir, stem)
    
    # Save component stats CSV
    csv_path = save_component_stats_csv(component_stats, output_dir, stem)
    
    # Print summary
    num_components = len(component_stats)
    num_kept = sum(1 for s in component_stats if s["score"] >= config.score_threshold)
    print(f"  -> Found {num_components} components, kept {num_kept} with score >= {config.score_threshold}")
    print(f"  -> Final mask pixels: {np.sum(final_mask > 0)}")
    
    return cleaned_prob_map, final_mask, component_stats


def main():
    parser = argparse.ArgumentParser(
        description="Test edge-aware denoising on TMD sample images."
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
        help="Number of clusters for probability map (default: 3)"
    )
    parser.add_argument(
        "--method", type=str, choices=['kmeans', 'gmm'], default='kmeans',
        help="Clustering method (default: kmeans)"
    )
    parser.add_argument(
        "--num_images", type=int, default=5,
        help="Number of images to process (default: 5)"
    )
    parser.add_argument(
        "--combine_mode", type=str, choices=['boost-only', 'weighted-max'], default='weighted-max',
        help="Combine mode (default: weighted-max)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.25,
        help="Edge weight for weighted-max mode (default: 0.25)"
    )
    parser.add_argument(
        "--edge_boost", type=float, default=0.1,
        help="Boost value for boost-only mode (default: 0.1)"
    )
    parser.add_argument(
        "--candidate_threshold", type=float, default=0.02,
        help="Candidate threshold (default: 0.02)"
    )
    parser.add_argument(
        "--final_threshold", type=float, default=0.4,
        help="Final threshold (default: 0.4)"
    )
    parser.add_argument(
        "--min_speck_area", type=int, default=30,
        help="Minimum speck area (default: 30)"
    )
    parser.add_argument(
        "--min_final_area", type=int, default=100,
        help="Minimum final area (default: 100)"
    )
    parser.add_argument(
        "--score_threshold", type=float, default=0.3,
        help="Score threshold for keeping components (default: 0.3)"
    )
    args = parser.parse_args()
    
    # Create config from arguments
    config = EdgeAwareDenoiseConfig(
        combine_mode=args.combine_mode,
        alpha=args.alpha,
        edge_boost=args.edge_boost,
        candidate_threshold=args.candidate_threshold,
        final_threshold=args.final_threshold,
        min_speck_area=args.min_speck_area,
        min_final_area=args.min_final_area,
        score_threshold=args.score_threshold
    )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all tmd_sample images
    image_paths = find_sample_images(input_dir)
    if not image_paths:
        print(f"No tmd_sample_*.jpg images found in {input_dir}")
        sys.exit(1)
    
    # Select first N images
    image_paths = image_paths[:args.num_images]
    
    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Using combine_mode={config.combine_mode}, alpha={config.alpha}")
    print(f"candidate_threshold={config.candidate_threshold}, final_threshold={config.final_threshold}")
    print()
    
    for img_path in image_paths:
        process_image(
            img_path, output_dir, config,
            n_clusters=args.n_clusters,
            method=args.method
        )
        print()
    
    print("All images processed successfully.")


if __name__ == "__main__":
    main()