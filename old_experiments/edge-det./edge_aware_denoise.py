#!/usr/bin/env python3
"""
edge_aware_denoise.py

Edge-aware denoising for flake probability maps using blurred RGB edges.
Designed to preserve thin, low-contrast flakes while removing background noise.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import csv


@dataclass
class EdgeAwareDenoiseConfig:
    """Configuration for edge-aware denoising."""
    
    # Image preprocessing
    blur_kernel_image: int = 5  # Gaussian blur kernel for grayscale
    
    # Probability map smoothing
    blur_kernel_prob: int = 5  # Gaussian blur kernel for probability map
    
    # Edge detection
    canny_low: int = 30  # Canny low threshold
    canny_high: int = 100  # Canny high threshold
    
    # Combine modes
    combine_mode: str = "weighted-max"  # "boost-only" or "weighted-max"
    alpha: float = 0.25  # Edge weight for weighted-max mode
    edge_boost: float = 0.1  # Boost value for boost-only mode
    
    # Thresholding
    candidate_threshold: float = 0.02  # Low threshold for candidate collection
    final_threshold: float = 0.4  # Final probability threshold
    
    # Morphology and small object removal
    min_speck_area: int = 30  # Remove components below this area
    min_final_area: int = 100  # Final small object removal
    morph_kernel: int = 3  # Morphological kernel size (3x3 disk)
    
    # Component scoring weights
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "area": 1.0,
        "contrast": 1.0,
        "solidity": 1.0,
    })
    
    # Optional: keep top K components by score (None = keep all above threshold)
    keep_top_k: Optional[int] = None
    
    # Score threshold for keeping components
    score_threshold: float = 0.3


def preprocess_image(rgb_img: np.ndarray, config: EdgeAwareDenoiseConfig) -> np.ndarray:
    """
    Convert RGB to grayscale and apply Gaussian blur.
    
    Args:
        rgb_img: RGB image (HxWx3, uint8)
        config: Configuration object
        
    Returns:
        Blurred grayscale image (HxW, float32 in [0, 1])
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    if config.blur_kernel_image > 1:
        blurred = cv2.GaussianBlur(gray, (config.blur_kernel_image, config.blur_kernel_image), 0)
    else:
        blurred = gray
    return blurred.astype(np.float32) / 255.0


def compute_edge_map(blurred_gray: np.ndarray, config: EdgeAwareDenoiseConfig) -> np.ndarray:
    """
    Compute edge map from blurred grayscale using Canny.
    
    Args:
        blurred_gray: Blurred grayscale image (HxW, float32 in [0, 1])
        config: Configuration object
        
    Returns:
        Edge map (HxW, float32 in [0, 1])
    """
    gray_uint8 = (blurred_gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray_uint8, config.canny_low, config.canny_high)
    return edges.astype(np.float32) / 255.0


def smooth_probability_map(prob_map: np.ndarray, config: EdgeAwareDenoiseConfig) -> np.ndarray:
    """
    Apply Gaussian blur to smooth the probability map.
    
    Args:
        prob_map: Probability map (HxW, float32 in [0, 1])
        config: Configuration object
        
    Returns:
        Blurred probability map (HxW, float32 in [0, 1])
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    if config.blur_kernel_prob > 1:
        prob_blurred = cv2.GaussianBlur(prob_uint8, (config.blur_kernel_prob, config.blur_kernel_prob), 0)
    else:
        prob_blurred = prob_uint8
    return prob_blurred.astype(np.float32) / 255.0


def combine_signals(prob_blurred: np.ndarray, edge_map: np.ndarray, config: EdgeAwareDenoiseConfig) -> np.ndarray:
    """
    Combine probability map with edge map using specified mode.
    
    Args:
        prob_blurred: Blurred probability map (HxW, float32 in [0, 1])
        edge_map: Edge map (HxW, float32 in [0, 1])
        config: Configuration object
        
    Returns:
        Combined map (HxW, float32 in [0, 1])
    """
    if config.combine_mode == "boost-only":
        combined = prob_blurred.copy()
        # Boost probability at edge locations
        edge_mask = edge_map > 0
        combined[edge_mask] = np.maximum(
            combined[edge_mask],
            np.minimum(combined[edge_mask] + config.edge_boost, 1.0)
        )
    elif config.combine_mode == "weighted-max":
        # Use maximum of probability and weighted edge
        combined = np.maximum(prob_blurred, config.alpha * edge_map)
    else:
        raise ValueError(f"Unknown combine_mode: {config.combine_mode}")
    
    return combined


def get_component_features(
    component_mask: np.ndarray,
    prob_blurred: np.ndarray,
    rgb_img: np.ndarray,
    substrate_color: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute features for a connected component.
    
    Args:
        component_mask: Binary mask of single component (HxW, uint8)
        prob_blurred: Blurred probability map (HxW, float32)
        rgb_img: Original RGB image (HxWx3, uint8)
        substrate_color: Optional substrate color in Lab space
        
    Returns:
        Dictionary of features: area, mean_prob, mean_contrast, solidity, perimeter_area_ratio
    """
    features = {}
    
    # Area
    features["area"] = float(np.sum(component_mask > 0))
    
    # Mean probability
    features["mean_prob"] = float(np.mean(prob_blurred[component_mask > 0]))
    
    # Mean contrast to substrate (if provided)
    if substrate_color is not None:
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab).astype(np.float32)
        # Compute distance to substrate for pixels in component
        distances = np.linalg.norm(
            lab[component_mask > 0] - substrate_color.reshape(1, 3),
            axis=1
        )
        features["mean_contrast"] = float(np.mean(distances))
    else:
        features["mean_contrast"] = 0.0
    
    # Solidity: ratio of area to convex hull area
    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        features["solidity"] = features["area"] / hull_area if hull_area > 0 else 0.0
    else:
        features["solidity"] = 0.0
    
    # Perimeter/Area ratio
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        features["perimeter_area_ratio"] = perimeter / features["area"] if features["area"] > 0 else 0.0
    else:
        features["perimeter_area_ratio"] = 0.0
    
    return features


def score_component(features: Dict[str, float], config: EdgeAwareDenoiseConfig) -> float:
    """
    Compute score for a component based on features.
    
    score = area_norm * (1 - normalized_mean_contrast) * solidity_weighted
    
    Args:
        features: Dictionary of component features
        config: Configuration object
        
    Returns:
        Component score (float)
    """
    # Normalize area (assuming max area of 100000 for 1024x1024 images)
    max_area = 100000.0
    area_norm = min(features["area"] / max_area, 1.0)
    
    # Normalize contrast (assuming max contrast of 100 in Lab space)
    max_contrast = 100.0
    contrast_norm = min(features["mean_contrast"] / max_contrast, 1.0)
    
    # Compute score
    score = (
        area_norm ** config.scoring_weights.get("area", 1.0) *
        (1.0 - contrast_norm) ** config.scoring_weights.get("contrast", 1.0) *
        features["solidity"] ** config.scoring_weights.get("solidity", 1.0)
    )
    
    return float(score)


def clean_probability_with_blurred_edges(
    rgb_img: np.ndarray,
    prob_map: np.ndarray,
    config: EdgeAwareDenoiseConfig,
    substrate_color: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[Dict]]:
    """
    Main edge-aware denoising function.
    
    Args:
        rgb_img: Original RGB image (HxWx3, uint8)
        prob_map: Model probability map (HxW, float32 in [0, 1])
        config: Configuration object
        substrate_color: Optional substrate color in Lab space
        
    Returns:
        cleaned_prob_map: Cleaned probability map (HxW, float32)
        final_mask: Final binary mask (HxW, uint8, 0 or 255)
        debug_images: Dictionary of intermediate images
        component_stats: List of component statistics dictionaries
    """
    debug_images = {}
    
    # Step 1: Preprocess image
    blurred_gray = preprocess_image(rgb_img, config)
    debug_images["blurred_gray"] = (blurred_gray * 255).astype(np.uint8)
    
    # Step 2: Compute edge map
    edge_map = compute_edge_map(blurred_gray, config)
    debug_images["edge_map"] = (edge_map * 255).astype(np.uint8)
    
    # Step 3: Smooth probability map
    prob_blurred = smooth_probability_map(prob_map, config)
    debug_images["prob_blurred"] = (prob_blurred * 255).astype(np.uint8)
    
    # Step 4: Combine signals
    combined = combine_signals(prob_blurred, edge_map, config)
    debug_images["combined"] = (combined * 255).astype(np.uint8)
    
    # Step 5: Candidate mask (low threshold)
    binary_candidate = (combined >= config.candidate_threshold).astype(np.uint8) * 255
    debug_images["binary_candidate"] = binary_candidate
    
    # Step 6: Morphological opening and closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.morph_kernel, config.morph_kernel))
    binary_candidate = cv2.morphologyEx(binary_candidate, cv2.MORPH_OPEN, kernel)
    binary_candidate = cv2.morphologyEx(binary_candidate, cv2.MORPH_CLOSE, kernel)
    
    # Step 7: Connected components with stats
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_candidate, connectivity=8)
    
    # Step 8: Remove small components
    candidate_components = np.zeros_like(binary_candidate)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= config.min_speck_area:
            candidate_components[labels == i] = 255
    
    debug_images["candidate_components"] = candidate_components
    
    # Step 9: Component scoring and selection
    component_stats = []
    final_candidate_mask = np.zeros_like(binary_candidate)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < config.min_speck_area:
            continue
        
        component_mask = (labels == i).astype(np.uint8) * 255
        features = get_component_features(component_mask, prob_blurred, rgb_img, substrate_color)
        score = score_component(features, config)
        
        features["label"] = i
        features["score"] = score
        component_stats.append(features)
    
    # Sort by score descending
    component_stats.sort(key=lambda x: x["score"], reverse=True)
    
    # Keep components based on score threshold or top-k
    kept_labels = set()
    for stat in component_stats:
        if config.keep_top_k is not None:
            if len(kept_labels) < config.keep_top_k:
                kept_labels.add(stat["label"])
        else:
            if stat["score"] >= config.score_threshold:
                kept_labels.add(stat["label"])
    
    for label in kept_labels:
        final_candidate_mask[labels == label] = 255
    
    debug_images["final_candidate_mask"] = final_candidate_mask
    
    # Step 10: Create cleaned probability map
    cleaned_prob_map = prob_blurred * (final_candidate_mask > 0).astype(np.float32)
    
    # Step 11: Final threshold
    final_mask = (cleaned_prob_map >= config.final_threshold).astype(np.uint8) * 255
    
    # Step 12: Morphological hole filling and small object removal
    # Hole filling
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small objects
    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    for i in range(1, num_labels_final):
        if stats_final[i, cv2.CC_STAT_AREA] < config.min_final_area:
            final_mask[labels_final == i] = 0
    
    debug_images["final_mask"] = final_mask
    
    # Create overlay
    overlay = rgb_img.copy()
    overlay[final_mask > 0] = (0, 255, 0)  # green overlay
    debug_images["overlay"] = overlay
    
    return cleaned_prob_map, final_mask, debug_images, component_stats


def save_debug_images(debug_images: Dict[str, np.ndarray], output_dir: Path, stem: str) -> None:
    """
    Save all debug images to disk.
    
    Args:
        debug_images: Dictionary of images
        output_dir: Output directory
        stem: Base filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, img in debug_images.items():
        if name == "edge_map":
            # Save edge map as heatmap (jet colormap)
            edge_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            edge_color = cv2.cvtColor(edge_color, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(output_dir / f"{stem}_{name}.png"), cv2.cvtColor(edge_color, cv2.COLOR_RGB2BGR))
        elif name == "overlay":
            # Save overlay in BGR
            cv2.imwrite(str(output_dir / f"{stem}_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            # Save grayscale images
            cv2.imwrite(str(output_dir / f"{stem}_{name}.png"), img)


def save_component_stats_csv(component_stats: List[Dict], output_dir: Path, stem: str) -> Path:
    """
    Save component statistics to CSV.
    
    Args:
        component_stats: List of component statistics
        output_dir: Output directory
        stem: Base filename
        
    Returns:
        Path to saved CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{stem}_component_stats.csv"
    
    if not component_stats:
        # Write empty file with headers
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["label", "area", "mean_prob", "mean_contrast", "solidity", "perimeter_area_ratio", "score"])
        return csv_path
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "area", "mean_prob", "mean_contrast", "solidity", "perimeter_area_ratio", "score"])
        for stat in component_stats:
            writer.writerow([
                stat.get("label", 0),
                stat.get("area", 0),
                stat.get("mean_prob", 0),
                stat.get("mean_contrast", 0),
                stat.get("solidity", 0),
                stat.get("perimeter_area_ratio", 0),
                stat.get("score", 0)
            ])
    
    return csv_path


def get_default_config() -> EdgeAwareDenoiseConfig:
    """Get default configuration for 1024x1024 images."""
    return EdgeAwareDenoiseConfig(
        blur_kernel_image=5,
        blur_kernel_prob=5,
        canny_low=30,
        canny_high=100,
        combine_mode="weighted-max",
        alpha=0.25,
        edge_boost=0.1,
        candidate_threshold=0.02,
        min_speck_area=30,
        min_final_area=100,
        morph_kernel=3,
        scoring_weights={"area": 1.0, "contrast": 1.0, "solidity": 1.0},
        keep_top_k=None,
        score_threshold=0.3,
        final_threshold=0.4
    )