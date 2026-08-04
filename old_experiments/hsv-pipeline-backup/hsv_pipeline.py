#!/usr/bin/env python3
"""
hsv_pipeline.py

Channel-Segregated Substrate Peak Normalization (CSSPN) pipeline for
unsupervised 2D material flake probability estimation.

This module implements a fully statistical, pixel-level approach that:
  1. Converts RGB to floating-point HSV.
  2. Locates the global substrate baseline via 1D histogram modes.
  3. Estimates per-channel substrate noise floors (standard deviations).
  4. Computes channel-segregated, standard-deviation-normalized distance maps.
  5. Applies saturation-gated hue suppression to stabilize hue at low saturation.
  6. Combines channels into a continuous probability map via exponential CDF.

No spatial denoising, edge detection, superpixels, or morphological
post-processing is used. The pipeline is entirely pixel-level.
"""

import numpy as np
import cv2
from typing import Dict, Tuple


class HSVPipeline:
    """
    Unsupervised flake probability engine based on Channel-Segregated
    Substrate Peak Normalization.

    The pipeline determines the global substrate background baseline
    (histogram modes) across the entire image and computes a multi-channel
    normalized probability heatmap.

    Parameters
    ----------
    w_S : float, default=1.0
        Weight for the Saturation channel distance.
    w_V : float, default=0.0
        Weight for the Value channel distance.
    w_H : float, default=0.0
        Weight for the (gated) Hue channel distance.
    substrate_tolerance : float, default=15.0
        Absolute channel tolerance used to build the substrate mask for
        noise-floor estimation (|S - S_sub| < tol AND |V - V_sub| < tol).
    epsilon : float, default=1e-5
        Small constant added to each channel standard deviation to prevent
        division-by-zero.
    """

    def __init__(
        self,
        w_S: float = 1.0,
        w_V: float = 0.0,
        w_H: float = 0.0,
        substrate_tolerance: float = 15.0,
        epsilon: float = 1e-5,
    ) -> None:
        # Validate weights sum to 1.0
        total = w_S + w_V + w_H
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(
                f"Channel weights must sum to 1.0, got {total:.6f}"
            )
        self.w_S = float(w_S)
        self.w_V = float(w_V)
        self.w_H = float(w_H)
        self.substrate_tolerance = float(substrate_tolerance)
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, rgb_image: np.ndarray) -> Dict[str, object]:
        """
        Run the full pipeline on a single RGB image.

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB uint8 image of shape (H, W, 3).

        Returns
        -------
        dict
            A dictionary containing:
              - 'probability_map'        : float32 (H, W) in [0.0, 1.0]
              - 'D_S'                    : float32 (H, W) saturation distance
              - 'D_V'                    : float32 (H, W) value distance
              - 'D_H_gated'              : float32 (H, W) gated hue distance
              - 'substrate_peak'         : (H_sub, S_sub, V_sub) tuple
              - 'substrate_std'          : (sigma_H, sigma_S, sigma_V) tuple
              - 'probability_heatmap_rgb': uint8 (H, W, 3) RGB heatmap
        """
        # Step 1: Convert to floating-point HSV
        hsv = self._convert_to_hsv(rgb_image)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Step 2: Global histogram mode detection (substrate baseline)
        substrate_peak = self._find_substrate_peak(hsv)
        H_sub, S_sub, V_sub = substrate_peak

        # Step 3: Substrate noise floor estimation
        substrate_std = self._estimate_substrate_std(hsv, substrate_peak)
        sigma_H, sigma_S, sigma_V = substrate_std

        # Step 4: Segregated channel-distance calculations
        D_S = self._compute_saturation_distance(S, S_sub, sigma_S)
        D_V = self._compute_value_distance(V, V_sub, sigma_V)
        D_H = self._compute_hue_distance(H, H_sub, sigma_H)

        # Step 5: Saturation-gated hue suppression
        D_H_gated = self._apply_saturation_gate(D_H, S, S_sub, sigma_S)

        # Step 6: Multi-channel weighted probability combination
        probability_map = self._combine_probability(D_S, D_V, D_H_gated)

        # Build heatmap visualization
        heatmap = self._build_heatmap(probability_map)

        return {
            "probability_map": probability_map,
            "D_S": D_S,
            "D_V": D_V,
            "D_H_gated": D_H_gated,
            "substrate_peak": substrate_peak,
            "substrate_std": substrate_std,
            "probability_heatmap_rgb": heatmap,
        }

    # ------------------------------------------------------------------
    # Step 1: Image Input & Basic Validation
    # ------------------------------------------------------------------
    def _convert_to_hsv(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert an RGB uint8 image to floating-point HSV.

        Hue range: [0, 180) degrees (OpenCV standard).
        Saturation range: [0, 255].
        Value range: [0, 255].

        Returns a float32 array of shape (H, W, 3).
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image of shape (H, W, 3), got {rgb_image.shape}"
            )
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: Global Histogram Mode Detection
    # ------------------------------------------------------------------
    def _compute_histograms(self, hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute 1D histograms for each HSV channel.

        Returns
        -------
        (h_hist, s_hist, v_hist) : tuple of 1D float arrays
            - h_hist: 180 bins (bin width = 1.0)
            - s_hist: 256 bins (bin width = 1.0)
            - v_hist: 256 bins (bin width = 1.0)
        """
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
        s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
        v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

        return h_hist, s_hist, v_hist

    def _find_substrate_peak(self, hsv: np.ndarray) -> Tuple[float, float, float]:
        """
        Locate the global substrate baseline (H_sub, S_sub, V_sub) by
        finding the bin index with the maximum pixel count in each channel.

        Returns
        -------
        (H_sub, S_sub, V_sub) : tuple of floats
        """
        h_hist, s_hist, v_hist = self._compute_histograms(hsv)

        H_sub = float(np.argmax(h_hist))
        S_sub = float(np.argmax(s_hist))
        V_sub = float(np.argmax(v_hist))

        return (H_sub, S_sub, V_sub)

    # ------------------------------------------------------------------
    # Step 3: Substrate Noise Floor Estimation
    # ------------------------------------------------------------------
    def _estimate_substrate_std(
        self, hsv: np.ndarray, substrate_peak: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """
        Estimate per-channel standard deviations using only pixels that
        fall within a tight range around the substrate peaks.

        The substrate mask is defined as:
            |S - S_sub| < tolerance AND |V - V_sub| < tolerance

        A small epsilon is added to each sigma to prevent division by zero.

        Returns
        -------
        (sigma_H, sigma_S, sigma_V) : tuple of floats
        """
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        H_sub, S_sub, V_sub = substrate_peak

        # Build substrate mask
        substrate_mask = (
            (np.abs(S - S_sub) < self.substrate_tolerance)
            & (np.abs(V - V_sub) < self.substrate_tolerance)
        )

        # Compute standard deviations on substrate pixels only
        if substrate_mask.any():
            sigma_H = float(np.std(H[substrate_mask]))
            sigma_S = float(np.std(S[substrate_mask]))
            sigma_V = float(np.std(V[substrate_mask]))
        else:
            # Fallback: use full-image std if no substrate pixels found
            sigma_H = float(np.std(H))
            sigma_S = float(np.std(S))
            sigma_V = float(np.std(V))

        # Add epsilon to prevent division-by-zero
        sigma_H += self.epsilon
        sigma_S += self.epsilon
        sigma_V += self.epsilon

        return (sigma_H, sigma_S, sigma_V)

    # ------------------------------------------------------------------
    # Step 4: Segregated Channel-Distance Calculations
    # ------------------------------------------------------------------
    def _compute_saturation_distance(
        self,
        S: np.ndarray,
        S_sub: float,
        sigma_S: float,
    ) -> np.ndarray:
        """
        Saturation channel distance map.

        Linear absolute difference from the substrate saturation peak,
        normalized by the substrate saturation standard deviation:

            D_S = |S - S_sub| / sigma_S

        This yields high values for pixels whose saturation deviates
        significantly from the substrate baseline.
        """
        D_S = np.abs(S - S_sub) / sigma_S
        return D_S.astype(np.float32)

    def _compute_value_distance(
        self, V: np.ndarray, V_sub: float, sigma_V: float
    ) -> np.ndarray:
        """
        Value channel distance: linear absolute difference normalized by
        the substrate value standard deviation.

        D_V = |V - V_sub| / sigma_V
        """
        delta_V = np.abs(V - V_sub)
        return (delta_V / sigma_V).astype(np.float32)

    def _compute_hue_distance(
        self, H: np.ndarray, H_sub: float, sigma_H: float
    ) -> np.ndarray:
        """
        Hue channel distance with circular math.

        Hue is angular in [0, 180). The shortest circular distance is:
            delta_H = |((H - H_sub + 90) % 180) - 90|

        Normalized distance:
            D_H = delta_H / sigma_H
        """
        delta_H = np.abs(((H - H_sub + 90.0) % 180.0) - 90.0)
        return (delta_H / sigma_H).astype(np.float32)

    # ------------------------------------------------------------------
    # Step 5: Saturation-Gated Hue Suppression
    # ------------------------------------------------------------------
    def _apply_saturation_gate(
        self,
        D_H: np.ndarray,
        S: np.ndarray,
        S_sub: float,
        sigma_S: float,
    ) -> np.ndarray:
        """
        Zero out the hue distance where saturation is too low for hue to be
        statistically meaningful.

        Saturation gate: S_gate = S_sub + (2.0 * sigma_S)
        hue_valid_mask = S > S_gate

        D_H_gated = D_H * hue_valid_mask
        """
        S_gate = S_sub + (2.0 * sigma_S)
        hue_valid_mask = S > S_gate
        D_H_gated = D_H * hue_valid_mask
        return D_H_gated.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 6: Multi-Channel Weighted Probability Combination
    # ------------------------------------------------------------------
    def _combine_probability(
        self,
        D_S: np.ndarray,
        D_V: np.ndarray,
        D_H_gated: np.ndarray,
    ) -> np.ndarray:
        """
        Combine the segregated channel distance maps into a unified
        continuous probability map.

        Weighted composite distance:
            D_composite = (w_S * D_S) + (w_V * D_V) + (w_H * D_H_gated)

        Exponential CDF scaling:
            P_map = 1.0 - exp(-0.5 * D_composite^2)

        Pure substrate pixels (D_composite ~ 0) evaluate to P ~ 0.0.
        Strong flake outliers (high distance) approach P ~ 1.0.
        """
        D_composite = (
            self.w_S * D_S
            + self.w_V * D_V
            + self.w_H * D_H_gated
        )

        P_map = 1.0 - np.exp(-0.5 * D_composite**2)
        return P_map.astype(np.float32)

    # ------------------------------------------------------------------
    # Visualization Helper
    # ------------------------------------------------------------------
    def _build_heatmap(self, probability_map: np.ndarray) -> np.ndarray:
        """
        Convert a probability map to an RGB heatmap using OpenCV's
        COLORMAP_INFERNO.

        Returns a uint8 (H, W, 3) RGB image.
        """
        prob_uint8 = (np.clip(probability_map, 0.0, 1.0) * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)