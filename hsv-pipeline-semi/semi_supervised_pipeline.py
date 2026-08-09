#!/usr/bin/env python3
"""
semi_supervised_pipeline.py

Semi-supervised HSV flake detection pipeline.

Instead of purely unsupervised thresholding, this pipeline:
  1. Automatically detects the substrate background (histogram mode + noise floor).
  2. Accepts user-provided "seed" clicks on flake regions.
  3. Extracts a flake HSV signature (mean + per-channel std) from patches around clicks.
  4. Computes directional deviation maps (ΔH, ΔS, ΔV) from the substrate peak.
  5. Scores every pixel by how closely its deviation-from-substrate matches the
     flake's deviation-from-substrate using Gaussian likelihoods.

This is "semi-supervised" because the substrate detection is automatic, but the
flake profile comes from minimal user interaction (a few clicks).
"""

import numpy as np
import cv2
from typing import Dict, Tuple, List, Optional


class SemiSupervisedPipeline:
    """
    Semi-supervised flake probability engine.

    Parameters
    ----------
    substrate_tolerance : float
        Tolerance for building the substrate mask for noise-floor estimation.
    patch_radius : int
        Half-size of the square patch extracted around each user click (patch
        size = 2*patch_radius + 1).
    w_H : float
        Weight for the Hue channel likelihood (heavier = stricter hue match).
    w_S : float
        Weight for the Saturation channel likelihood.
    w_V : float
        Weight for the Value channel likelihood.
    epsilon : float
        Small constant to prevent division-by-zero in std normalization.
    """

    def __init__(
        self,
        substrate_tolerance: float = 15.0,
        patch_radius: int = 7,
        w_H: float = 1.5,
        w_S: float = 1.0,
        w_V: float = 1.0,
        epsilon: float = 1e-5,
    ) -> None:
        self.substrate_tolerance = float(substrate_tolerance)
        self.patch_radius = int(patch_radius)
        self.w_H = float(w_H)
        self.w_S = float(w_S)
        self.w_V = float(w_V)
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(
        self,
        rgb_image: np.ndarray,
        seed_points: List[Tuple[int, int]],
        substrate_peak: Optional[Tuple[float, float, float]] = None,
        substrate_std: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, object]:
        """
        Run the full semi-supervised pipeline.

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB uint8 image of shape (H, W, 3).
        seed_points : list of (x, y) tuples
            Pixel coordinates where the user clicked on flake regions.
        substrate_peak : (H, S, V) tuple, optional
            Operator-supplied substrate reference (e.g. from click-to-calibrate
            in the annotator). When given, the automatic whole-frame histogram
            detection is skipped and deviations are measured from this peak.
        substrate_std : (sigma_H, sigma_S, sigma_V) tuple, optional
            Operator-supplied substrate noise floor. When omitted while
            substrate_peak is given, the noise floor is auto-estimated from
            this image around the supplied peak.

        Returns
        -------
        dict with keys:
            - 'probability_map'   : float32 (H, W) in [0, 1]
            - 'delta_H'           : signed hue deviation from substrate
            - 'delta_S'           : signed saturation deviation from substrate
            - 'delta_V'           : signed value deviation from substrate
            - 'substrate_peak'    : (H_sub, S_sub, V_sub)
            - 'substrate_std'     : (sigma_H, sigma_S, sigma_V)
            - 'substrate_source'  : 'auto' or 'override'
            - 'flake_signature'   : dict with mean, std, and reference deviations
            - 'heatmap_rgb'       : uint8 (H, W, 3) Inferno heatmap
            - 'seed_patches'      : list of patch bounding boxes for visualization
        """
        # Step 1: Convert to HSV
        hsv = self._convert_to_hsv(rgb_image)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Step 2: Substrate reference — operator-supplied or automatic
        substrate_source = "auto" if substrate_peak is None else "override"
        if substrate_peak is None:
            substrate_peak = self._find_substrate_peak(hsv)
        else:
            substrate_peak = tuple(float(c) for c in substrate_peak)
        H_sub, S_sub, V_sub = substrate_peak

        # Step 3: Substrate noise floor
        if substrate_std is None:
            substrate_std = self._estimate_substrate_std(hsv, substrate_peak)
        else:
            substrate_std = tuple(float(c) for c in substrate_std)
        sigma_H, sigma_S, sigma_V = substrate_std

        # Step 4: Extract flake signature from seed clicks
        flake_sig = self._extract_flake_signature(
            H, S, V, seed_points, substrate_peak
        )

        # Step 5: Compute signed deviation maps
        delta_H = self._signed_hue_deviation(H, H_sub)
        delta_S = (S - S_sub).astype(np.float32)
        delta_V = (V - V_sub).astype(np.float32)

        # Step 6: Compute probability map via Gaussian likelihood matching
        prob_map = self._compute_probability(
            delta_H, delta_S, delta_V, flake_sig
        )

        # Step 7: Build heatmap
        heatmap = self._build_heatmap(prob_map)

        # Collect seed patch boxes for visualization
        seed_patches = self._get_seed_patch_boxes(seed_points, H.shape)

        return {
            "probability_map": prob_map,
            "delta_H": delta_H,
            "delta_S": delta_S,
            "delta_V": delta_V,
            "substrate_peak": substrate_peak,
            "substrate_std": substrate_std,
            "substrate_source": substrate_source,
            "flake_signature": flake_sig,
            "heatmap_rgb": heatmap,
            "seed_patches": seed_patches,
        }

    # ------------------------------------------------------------------
    # Step 1: HSV conversion
    # ------------------------------------------------------------------
    def _convert_to_hsv(self, rgb_image: np.ndarray) -> np.ndarray:
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got {rgb_image.shape}")
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        return hsv.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: Substrate peak detection (histogram mode)
    # ------------------------------------------------------------------
    def _find_substrate_peak(self, hsv: np.ndarray) -> Tuple[float, float, float]:
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
        s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
        v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()
        H_sub = float(np.argmax(h_hist))
        S_sub = float(np.argmax(s_hist))
        V_sub = float(np.argmax(v_hist))
        return (H_sub, S_sub, V_sub)

    # ------------------------------------------------------------------
    # Step 3: Substrate noise floor estimation
    # ------------------------------------------------------------------
    def _estimate_substrate_std(
        self, hsv: np.ndarray, substrate_peak: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        H_sub, S_sub, V_sub = substrate_peak

        mask = (
            (np.abs(S - S_sub) < self.substrate_tolerance)
            & (np.abs(V - V_sub) < self.substrate_tolerance)
        )

        if mask.any():
            sigma_H = float(np.std(H[mask]))
            sigma_S = float(np.std(S[mask]))
            sigma_V = float(np.std(V[mask]))
        else:
            sigma_H = float(np.std(H))
            sigma_S = float(np.std(S))
            sigma_V = float(np.std(V))

        sigma_H += self.epsilon
        sigma_S += self.epsilon
        sigma_V += self.epsilon
        return (sigma_H, sigma_S, sigma_V)

    # ------------------------------------------------------------------
    # Step 4: Flake signature extraction from seed clicks
    # ------------------------------------------------------------------
    def _extract_flake_signature(
        self,
        H: np.ndarray,
        S: np.ndarray,
        V: np.ndarray,
        seed_points: List[Tuple[int, int]],
        substrate_peak: Tuple[float, float, float],
    ) -> Dict:
        """
        Extract the flake's HSV signature from patches around user clicks.

        Returns a dict with:
            - 'mean_H', 'mean_S', 'mean_V'  : mean HSV of flake patches
            - 'std_H', 'std_S', 'std_V'     : per-channel std within patches
            - 'delta_H_ref', 'delta_S_ref', 'delta_V_ref' : deviation from substrate
            - 'n_pixels'                    : total pixels sampled
        """
        H_sub, S_sub, V_sub = substrate_peak
        r = self.patch_radius

        h_vals, s_vals, v_vals = [], [], []

        for (x, y) in seed_points:
            x, y = int(x), int(y)
            x0, x1 = max(0, x - r), min(H.shape[1], x + r + 1)
            y0, y1 = max(0, y - r), min(H.shape[0], y + r + 1)

            h_vals.append(H[y0:y1, x0:x1].ravel())
            s_vals.append(S[y0:y1, x0:x1].ravel())
            v_vals.append(V[y0:y1, x0:x1].ravel())

        h_all = np.concatenate(h_vals) if h_vals else np.array([0.0])
        s_all = np.concatenate(s_vals) if s_vals else np.array([0.0])
        v_all = np.concatenate(v_vals) if v_vals else np.array([0.0])

        # Hue is circular (OpenCV 0..179 wraps: 0 and 179 are both red) —
        # linear mean/std of a red flake straddling the wrap would give a
        # spurious cyan mean (~90) with a huge std (~87)
        mean_H, std_H = self._circular_mean_std(h_all)
        mean_S = float(np.mean(s_all))
        mean_V = float(np.mean(v_all))

        std_H = std_H + self.epsilon
        std_S = float(np.std(s_all)) + self.epsilon
        std_V = float(np.std(v_all)) + self.epsilon

        # Signed deviation from substrate (circular for hue)
        delta_H_ref = self._circular_delta(mean_H, H_sub)
        delta_S_ref = mean_S - S_sub
        delta_V_ref = mean_V - V_sub

        return {
            "mean_H": mean_H, "mean_S": mean_S, "mean_V": mean_V,
            "std_H": std_H, "std_S": std_S, "std_V": std_V,
            "delta_H_ref": float(delta_H_ref),
            "delta_S_ref": float(delta_S_ref),
            "delta_V_ref": float(delta_V_ref),
            "n_pixels": int(len(h_all)),
        }

    # ------------------------------------------------------------------
    # Step 5: Signed deviation maps
    # ------------------------------------------------------------------
    def _signed_hue_deviation(self, H: np.ndarray, H_sub: float) -> np.ndarray:
        """
        Signed circular hue deviation in [-90, +90].
        Positive = hue shifted clockwise from substrate, negative = counter-clockwise.
        """
        raw = (H - H_sub + 90.0) % 180.0 - 90.0
        return raw.astype(np.float32)

    @staticmethod
    def _circular_delta(h1: float, h2: float) -> float:
        """Signed circular difference h1 - h2 in [-90, +90]."""
        return float(((h1 - h2 + 90.0) % 180.0) - 90.0)

    @staticmethod
    def _circular_mean_std(h_vals: np.ndarray) -> Tuple[float, float]:
        """Circular mean and std of OpenCV hue values (0..179).

        Hue is mapped onto the full circle (x2), averaged as unit
        vectors, and mapped back; std comes from the resultant length
        (sqrt(-2 ln R)), which matches the linear std for tight
        clusters away from the wrap.
        """
        ang = np.asarray(h_vals, dtype=np.float64) * (np.pi / 90.0)
        c = float(np.mean(np.cos(ang)))
        s = float(np.mean(np.sin(ang)))
        mean = (np.degrees(np.arctan2(s, c)) / 2.0) % 180.0
        r = min(1.0, float(np.hypot(c, s)))
        std = np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(r, 1e-12))))) / 2.0
        return float(mean), float(std)

    # ------------------------------------------------------------------
    # Step 6: Probability computation via Gaussian likelihood matching
    # ------------------------------------------------------------------
    def _compute_probability(
        self,
        delta_H: np.ndarray,
        delta_S: np.ndarray,
        delta_V: np.ndarray,
        flake_sig: Dict,
    ) -> np.ndarray:
        """
        Score each pixel by how closely its deviation-from-substrate matches
        the flake's deviation-from-substrate.

        Per-channel Gaussian likelihood:
            P_c = exp(-0.5 * ((delta_c - delta_c_ref) / sigma_c_flake)^2)

        Combined:
            P = (P_S^w_S) * (P_V^w_V) * (P_H^w_H)

        Hue is weighted heavier (w_H > w_S, w_V by default) to enforce
        stricter hue matching.
        """
        dH_ref = flake_sig["delta_H_ref"]
        dS_ref = flake_sig["delta_S_ref"]
        dV_ref = flake_sig["delta_V_ref"]
        sH = flake_sig["std_H"]
        sS = flake_sig["std_S"]
        sV = flake_sig["std_V"]

        # Gaussian likelihoods
        P_H = np.exp(-0.5 * ((delta_H - dH_ref) / sH) ** 2)
        P_S = np.exp(-0.5 * ((delta_S - dS_ref) / sS) ** 2)
        P_V = np.exp(-0.5 * ((delta_V - dV_ref) / sV) ** 2)

        # Weighted combination (exponentiation by weight)
        P = (P_S ** self.w_S) * (P_V ** self.w_V) * (P_H ** self.w_H)
        return P.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 7: Heatmap
    # ------------------------------------------------------------------
    def _build_heatmap(self, prob_map: np.ndarray) -> np.ndarray:
        p8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(p8, cv2.COLORMAP_INFERNO)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # ------------------------------------------------------------------
    # Calibrate-then-apply: process with a pre-existing flake signature
    # ------------------------------------------------------------------
    def process_with_signature(
        self,
        rgb_image: np.ndarray,
        flake_sig: Dict,
        substrate_peak: Optional[Tuple[float, float, float]] = None,
        substrate_std: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, object]:
        """
        Apply a pre-existing flake signature to a new image.

        This is used in calibrate mode: the flake signature (deviation
        vector + per-channel std) is extracted from one calibration image,
        then applied to all other images in the same sample. Each image
        gets its own automatic substrate detection unless an operator
        substrate_peak / substrate_std override is supplied (see process()).

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB uint8 image of shape (H, W, 3).
        flake_sig : dict
            Flake signature dict (from a previous process() call's
            'flake_signature' field). Must contain:
            delta_H_ref, delta_S_ref, delta_V_ref, std_H, std_S, std_V.
        substrate_peak, substrate_std : tuples, optional
            Operator substrate override, same semantics as process().

        Returns
        -------
        dict (same keys as process(), except no 'seed_patches')
        """
        # Step 1: Convert to HSV
        hsv = self._convert_to_hsv(rgb_image)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        # Step 2: Substrate reference — operator-supplied or per-image automatic
        substrate_source = "auto" if substrate_peak is None else "override"
        if substrate_peak is None:
            substrate_peak = self._find_substrate_peak(hsv)
        else:
            substrate_peak = tuple(float(c) for c in substrate_peak)
        H_sub, S_sub, V_sub = substrate_peak

        # Step 3: Substrate noise floor
        if substrate_std is None:
            substrate_std = self._estimate_substrate_std(hsv, substrate_peak)
        else:
            substrate_std = tuple(float(c) for c in substrate_std)

        # Step 4: Compute signed deviation maps (relative to THIS image's substrate)
        delta_H = self._signed_hue_deviation(H, H_sub)
        delta_S = (S - S_sub).astype(np.float32)
        delta_V = (V - V_sub).astype(np.float32)

        # Step 5: Compute probability using the provided flake signature
        prob_map = self._compute_probability(delta_H, delta_S, delta_V, flake_sig)

        # Step 6: Build heatmap
        heatmap = self._build_heatmap(prob_map)

        return {
            "probability_map": prob_map,
            "delta_H": delta_H,
            "delta_S": delta_S,
            "delta_V": delta_V,
            "substrate_peak": substrate_peak,
            "substrate_std": substrate_std,
            "substrate_source": substrate_source,
            "flake_signature": flake_sig,
            "heatmap_rgb": heatmap,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_seed_patch_boxes(
        self, seed_points: List[Tuple[int, int]], shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Return (x0, y0, x1, y1) bounding boxes for each seed patch."""
        r = self.patch_radius
        boxes = []
        for (x, y) in seed_points:
            x, y = int(x), int(y)
            x0 = max(0, x - r)
            y0 = max(0, y - r)
            x1 = min(shape[1], x + r + 1)
            y1 = min(shape[0], y + r + 1)
            boxes.append((x0, y0, x1, y1))
        return boxes


# ----------------------------------------------------------------------
# Click-to-calibrate substrate sampling (module-level helpers)
#
# Used by the camera annotator to turn operator clicks on bare substrate
# into a (peak, std) reference that process() / process_with_signature()
# accept as substrate_peak / substrate_std overrides. Kept here so the
# circular-hue statistics live next to the pipeline that consumes them.
# ----------------------------------------------------------------------

def _rgb_to_hsv_f32(rgb_image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to float32 OpenCV HSV (H 0-179, S/V 0-255)."""
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got {rgb_image.shape}")
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)


def _clipped_patch_pixels(
    hsv: np.ndarray, x: int, y: int, radius: int
) -> np.ndarray:
    """Flattened (N, 3) pixels of a border-clipped square patch."""
    h, w = hsv.shape[:2]
    x, y = int(x), int(y)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return hsv[y0:y1, x0:x1].reshape(-1, 3)


def extract_hsv_patch(
    rgb_image: np.ndarray, x: int, y: int, radius: int = 7
) -> np.ndarray:
    """
    Extract HSV pixels from a square patch around (x, y).

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image of shape (H, W, 3).
    x, y : int
        Patch center in image coordinates.
    radius : int
        Half-size of the patch (patch size = 2*radius + 1, clipped at
        image borders).

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, 3) with OpenCV HSV pixels.
    """
    return _clipped_patch_pixels(_rgb_to_hsv_f32(rgb_image), x, y, radius)


def substrate_stats_from_pixels(
    hsv_pixels: np.ndarray, epsilon: float = 1e-5
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Substrate (peak, std) from sampled HSV pixels.

    Hue uses circular statistics (OpenCV hue wraps at 180: 0 and 179 are
    both red); saturation and value use linear mean/std.

    Parameters
    ----------
    hsv_pixels : np.ndarray
        Array of shape (N, 3) with OpenCV HSV pixels.
    epsilon : float
        Added to each std component to prevent zero noise floors.

    Returns
    -------
    ((H, S, V), (sigma_H, sigma_S, sigma_V))
    """
    px = np.asarray(hsv_pixels, dtype=np.float32).reshape(-1, 3)
    if px.shape[0] == 0:
        raise ValueError("No pixels supplied for substrate statistics")

    mean_H, std_H = SemiSupervisedPipeline._circular_mean_std(px[:, 0])
    peak = (
        float(mean_H),
        float(np.mean(px[:, 1])),
        float(np.mean(px[:, 2])),
    )
    std = (
        float(std_H) + epsilon,
        float(np.std(px[:, 1])) + epsilon,
        float(np.std(px[:, 2])) + epsilon,
    )
    return peak, std


def substrate_from_samples(
    rgb_image: np.ndarray,
    points: List[Tuple[int, int]],
    patch_radius: int = 7,
    epsilon: float = 1e-5,
) -> Dict[str, object]:
    """
    Build a substrate calibration from operator clicks on bare substrate.

    Parameters
    ----------
    rgb_image : np.ndarray
        RGB image of shape (H, W, 3).
    points : list of (x, y) tuples
        Click positions on bare substrate.
    patch_radius : int
        Half-size of the patch sampled around each click.
    epsilon : float
        Std floor, see substrate_stats_from_pixels.

    Returns
    -------
    dict
        {'peak': [H, S, V], 'std': [sH, sS, sV],
         'n_pixels': int, 'n_samples': int}
    """
    hsv = _rgb_to_hsv_f32(rgb_image)
    arrays = [
        _clipped_patch_pixels(hsv, x, y, patch_radius) for (x, y) in points
    ]
    arrays = [a for a in arrays if a.shape[0] > 0]
    if not arrays:
        raise ValueError("No in-bounds substrate samples")

    all_px = np.concatenate(arrays, axis=0)
    peak, std = substrate_stats_from_pixels(all_px, epsilon=epsilon)
    return {
        "peak": [float(c) for c in peak],
        "std": [float(c) for c in std],
        "n_pixels": int(all_px.shape[0]),
        "n_samples": len(arrays),
    }