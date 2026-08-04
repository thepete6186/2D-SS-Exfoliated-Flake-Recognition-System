#!/usr/bin/env python3
"""
apply_prob_threshold.py

Apply additional thresholding to probability maps to remove low-confidence pixels.
Saves thresholded probability maps showing only the retained high-probability regions.
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path


def load_probability_map(path):
    """Load a probability map from disk and return it as float32 in [0,1]."""
    prob = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if prob is None:
        raise FileNotFoundError(f"Could not load probability map: {path}")
    return prob.astype(np.float32) / 255.0


def apply_probability_threshold(prob_map, threshold=0.1):
    """
    Apply a threshold to the probability map, setting all pixels below threshold to 0.
    This removes low-confidence pixels from the probability map.
    
    Args:
        prob_map: Probability map (HxW float32 in [0,1])
        threshold: Threshold value - pixels below this are set to 0 (default: 0.1)
    
    Returns:
        Thresholded probability map (HxW float32 in [0,1])
    """
    thresholded = prob_map.copy()
    thresholded[prob_map < threshold] = 0
    return thresholded


def find_probability_maps(input_dir):
    """Find all probability map images in the given directory."""
    input_dir = Path(input_dir)
    prob_paths = sorted(input_dir.glob("tmd_sample_*_probability.png"))
    return prob_paths


def process_probability_map(prob_path, output_dir, thresholds=[0.05, 0.1, 0.2]):
    """
    Process a single probability map with multiple thresholds.
    
    Args:
        prob_path: Path to the probability map
        output_dir: Directory to save thresholded outputs
        thresholds: List of threshold values to apply
    """
    print(f"Processing: {prob_path.name}")
    
    # Load probability map
    prob_map = load_probability_map(prob_path)
    
    # Get stem for naming
    stem = prob_path.stem.replace("_probability", "")
    
    # Apply each threshold and save
    for thresh in thresholds:
        thresholded = apply_probability_threshold(prob_map, threshold=thresh)
        
        # Save thresholded probability map
        prob_uint8 = (thresholded * 255).astype(np.uint8)
        output_path = output_dir / f"{stem}_probability_thresh{thresh:.2f}.png"
        cv2.imwrite(str(output_path), prob_uint8)
        
        # Count pixels removed
        original_pixels = np.sum(prob_map > 0)
        remaining_pixels = np.sum(thresholded > 0)
        removed_pixels = original_pixels - remaining_pixels
        
        print(f"  -> {output_path.name}: {remaining_pixels} pixels retained, {removed_pixels} pixels removed (threshold={thresh})")
    
    return prob_map


def main():
    parser = argparse.ArgumentParser(
        description="Apply additional thresholding to probability maps to remove low-confidence pixels."
    )
    parser.add_argument(
        "--input_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory containing probability maps (default: ./results)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory to save thresholded probability maps (default: ./results)"
    )
    parser.add_argument(
        "--thresholds", type=float, nargs='+', default=[0.05, 0.1, 0.2],
        help="Threshold values to apply (default: 0.05 0.1 0.2)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all probability map images
    prob_paths = find_probability_maps(input_dir)
    if not prob_paths:
        print(f"No tmd_sample_*_probability.png images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(prob_paths)} probability maps in {input_dir}")
    print(f"Using thresholds: {args.thresholds}")
    print()

    for prob_path in prob_paths:
        process_probability_map(prob_path, output_dir, thresholds=args.thresholds)
        print()

    print("All probability maps processed successfully.")


if __name__ == "__main__":
    main()