#!/usr/bin/env python3
"""
post_processing.py

Post-processing functions for thresholding and saving results.
"""

import numpy as np
import cv2
from pathlib import Path


def threshold_probability_map(prob_map, threshold=0.5):
    """
    Apply a global threshold to the probability map to obtain a binary mask.
    Returns a uint8 mask (0 or 255).
    """
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    return mask


def save_results(rgb_img, prob_map, mask, output_dir, stem):
    """
    Save probability map, binary mask, and overlay visualization.
    
    Args:
        rgb_img: Original RGB image (HxWx3, uint8)
        prob_map: Probability map (HxW float32 in [0,1])
        mask: Binary mask (HxW uint8, 0 or 255)
        output_dir: Directory to save outputs
        stem: Base filename (e.g., "tmd_sample_0")
    
    Returns:
        Tuple of (prob_path, mask_path, overlay_path) as Path objects
    """
    output_dir = Path(output_dir)
    
    # Probability map: scale to 0-255 for saving as grayscale PNG
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    prob_path = output_dir / f"{stem}_probability.png"
    cv2.imwrite(str(prob_path), prob_uint8)
    
    # Binary mask
    mask_path = output_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask)
    
    # Color overlay visualization
    overlay = rgb_img.copy()
    overlay[mask > 0] = (0, 255, 0)  # green overlay on detected flakes
    overlay_path = output_dir / f"{stem}_overlay.png"
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)
    
    return prob_path, mask_path, overlay_path