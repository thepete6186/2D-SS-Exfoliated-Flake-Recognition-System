#!/usr/bin/env python3
"""
post_processing.py

Post-processing functions for thresholding and saving results.
"""

import numpy as np
import cv2
from pathlib import Path


def morphological_cleanup(mask, kernel_size=3, iterations=1):
    """
    Apply morphological operations to clean up binary mask.
    Uses opening to remove small noise and closing to fill gaps.
    
    Args:
        mask: Binary mask (HxW uint8, 0 or 255)
        kernel_size: Size of the morphological kernel (default: 3)
        iterations: Number of iterations for each operation (default: 1)
    
    Returns:
        Cleaned binary mask (HxW uint8, 0 or 255)
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Opening: remove small noise
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    # Closing: fill small gaps
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return cleaned


def adaptive_threshold(prob_map, block_size=51, offset=10):
    """
    Apply adaptive threshold to handle varying background conditions.
    
    Args:
        prob_map: Probability map (HxW float32 in [0,1])
        block_size: Size of local neighborhood for threshold calculation (default: 51)
        offset: Subtracted from local mean (default: 10)
    
    Returns:
        Binary mask (HxW uint8, 0 or 255)
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    # Use adaptive threshold for local contrast handling
    binary = cv2.adaptiveThreshold(
        prob_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block_size, offset
    )
    return binary


def sobel_edge_detection(rgb_img):
    """
    Compute edge map using Sobel operator (alternative to Canny).
    
    Args:
        rgb_img: Original RGB image (HxWx3, uint8)
    
    Returns:
        Edge map (HxW float32 in [0,1])
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    # Sobel in x and y directions
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Combine and normalize
    edges = np.sqrt(sobelx**2 + sobely**2)
    edges = edges / edges.max() if edges.max() > 0 else edges
    return edges.astype(np.float32)


def laplacian_of_gaussian(rgb_img, sigma=1.0):
    """
    Compute edge map using Laplacian of Gaussian for multi-scale edge detection.
    
    Args:
        rgb_img: Original RGB image (HxWx3, uint8)
        sigma: Standard deviation for Gaussian kernel (default: 1.0)
    
    Returns:
        Edge map (HxW float32 in [0,1])
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    # Apply Laplacian
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    # Take absolute value and normalize
    edges = np.abs(laplacian)
    edges = edges / edges.max() if edges.max() > 0 else edges
    return edges.astype(np.float32)


def threshold_probability_map(prob_map, threshold=0.5):
    """
    Apply a global threshold to the probability map to obtain a binary mask.
    Returns a uint8 mask (0 or 255).
    """
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    return mask


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


def filter_noise_with_blur(prob_map, blur_ksize=5, min_area=100, low_threshold=0.02, use_morphology=True, morph_kernel=3):
    """
    Filter noise from probability map by:
    1. Apply Gaussian blur to smooth the probability map
    2. Threshold to get binary mask (using low_threshold)
    3. Remove small connected components
    4. Apply morphological cleanup (optional)
    5. Return updated probability map with noise removed
    
    Args:
        prob_map: Probability map (HxW float32 in [0,1])
        blur_ksize: Kernel size for Gaussian blur (default: 5)
        min_area: Minimum area for a component to be kept (default: 100)
        low_threshold: Threshold to capture low-probability areas (default: 0.02)
        use_morphology: Whether to apply morphological cleanup (default: True)
        morph_kernel: Kernel size for morphological operations (default: 3)
    
    Returns:
        Cleaned probability map (HxW float32 in [0,1])
    """
    # Apply Gaussian blur to smooth the probability map
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    if blur_ksize > 1:
        prob_blurred = cv2.GaussianBlur(prob_uint8, (blur_ksize, blur_ksize), 0)
    else:
        prob_blurred = prob_uint8
    
    # Threshold to get binary mask (use very low threshold to include low-probability thin flakes)
    # This removes 0-probability pixels (substrate)
    _, binary = cv2.threshold(prob_blurred, int(low_threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create cleaned probability map (use BLURRED values, not original)
    cleaned = np.zeros_like(prob_map)
    
    # Keep only components with area >= min_area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = prob_blurred[labels == i] / 255.0
    
    # Apply morphological cleanup to the binary mask
    if use_morphology:
        binary = morphological_cleanup(binary, kernel_size=morph_kernel, iterations=1)
    
    return cleaned


def filter_noise_with_edges(prob_map, rgb_img, blur_ksize=5, min_area=100, low_threshold=0.02, edge_weight=0.2, edge_method='canny', use_morphology=True, morph_kernel=3):
    """
    Filter noise from probability map using edge-aware denoising:
    1. Apply Gaussian blur to smooth the probability map
    2. Compute edge map from RGB image (Canny, Sobel, or LoG)
    3. Combine probability map with edge map
    4. Threshold to get binary mask
    5. Remove small connected components
    6. Apply morphological cleanup (optional)
    7. Return updated probability map with noise removed
    
    Args:
        prob_map: Probability map (HxW float32 in [0,1])
        rgb_img: Original RGB image (HxWx3, uint8)
        blur_ksize: Kernel size for Gaussian blur (default: 5)
        min_area: Minimum area for a component to be kept (default: 100)
        low_threshold: Threshold to capture low-probability areas (default: 0.02)
        edge_weight: Weight for edge map in combination (default: 0.2)
        edge_method: Edge detection method - 'canny', 'sobel', or 'log' (default: 'canny')
        use_morphology: Whether to apply morphological cleanup (default: True)
        morph_kernel: Kernel size for morphological operations (default: 3)
    
    Returns:
        Cleaned probability map (HxW float32 in [0,1])
    """
    # Apply Gaussian blur to smooth the probability map
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    if blur_ksize > 1:
        prob_blurred = cv2.GaussianBlur(prob_uint8, (blur_ksize, blur_ksize), 0)
    else:
        prob_blurred = prob_uint8
    
    # Compute edge map from RGB image using selected method
    if edge_method == 'canny':
        gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edge_map = edges.astype(np.float32) / 255.0
    elif edge_method == 'sobel':
        edge_map = sobel_edge_detection(rgb_img)
    elif edge_method == 'log':
        edge_map = laplacian_of_gaussian(rgb_img, sigma=1.0)
    else:
        raise ValueError(f"Unknown edge_method: {edge_method}")
    
    # Combine probability map with edge map
    # Lower edge_weight so edges help but do not dominate
    combined = (prob_blurred / 255.0) * (1 - edge_weight) + edge_map * edge_weight
    
    # Threshold to get binary mask (use very low threshold to include low-probability thin flakes)
    _, binary = cv2.threshold((combined * 255).astype(np.uint8), int(low_threshold * 255), 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Create cleaned probability map
    cleaned = np.zeros_like(prob_map)
    
    # Keep only components with area >= min_area
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = prob_blurred[labels == i] / 255.0
    
    # Apply morphological cleanup to the binary mask
    if use_morphology:
        binary = morphological_cleanup(binary, kernel_size=morph_kernel, iterations=1)
    
    return cleaned


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


def save_thresholded_probability(prob_map, output_dir, stem, threshold=0.1):
    """
    Save a thresholded probability map where low-probability pixels are removed.
    
    Args:
        prob_map: Probability map (HxW float32 in [0,1])
        output_dir: Directory to save outputs
        stem: Base filename (e.g., "tmd_sample_0")
        threshold: Threshold value - pixels below this are set to 0 (default: 0.1)
    
    Returns:
        Path to the saved thresholded probability map
    """
    output_dir = Path(output_dir)
    
    # Apply threshold to remove low-probability pixels
    thresholded = apply_probability_threshold(prob_map, threshold)
    
    # Save thresholded probability map
    prob_uint8 = (thresholded * 255).astype(np.uint8)
    prob_path = output_dir / f"{stem}_probability_thresh{threshold:.2f}.png"
    cv2.imwrite(str(prob_path), prob_uint8)
    
    return prob_path