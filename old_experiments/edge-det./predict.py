#!/usr/bin/env python3
"""
predict.py

Predict flake probability maps using clustering on Lab color space.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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
    import cv2
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