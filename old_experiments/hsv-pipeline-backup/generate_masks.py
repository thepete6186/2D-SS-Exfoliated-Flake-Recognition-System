#!/usr/bin/env python3
"""
generate_masks.py

Generate binary flake masks from the HSV pipeline's raw probability maps
using connected component analysis + contour detection.

Approach:
  1. Load the source image
  2. Compute raw probability map using OLD (main peak) S logic
  3. Apply a hard gate (either 2σ or 1.5σ) to suppress H/V deviations
  4. Threshold the probability map to create a binary mask
  5. Use connected components to identify individual flakes
  6. Save:
     - Binary mask PNG
     - Mask overlay on original image
     - Statistical summary (flake count, areas, centroids)

Outputs per gate variant:
  <output_dir>/mask_<variant>.png
  <output_dir>/overlay_<variant>.png
  <output_dir>/stats_<variant>.txt

Usage:
    python generate_masks.py --image ../dataset/sample2/tmd_sample_13.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rgb_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_base_diagnostics(rgb: np.ndarray):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

    H_sub = float(np.argmax(h_hist))
    S_sub = float(np.argmax(s_hist))
    V_sub = float(np.argmax(v_hist))

    substrate_tolerance = 15.0
    substrate_mask = (
        (np.abs(S - S_sub) < substrate_tolerance)
        & (np.abs(V - V_sub) < substrate_tolerance)
    )
    if substrate_mask.any():
        sigma_H = float(np.std(H[substrate_mask]))
        sigma_S = float(np.std(S[substrate_mask]))
        sigma_V = float(np.std(V[substrate_mask]))
    else:
        sigma_H = float(np.std(H))
        sigma_S = float(np.std(S))
        sigma_V = float(np.std(V))

    sigma_H += 1e-5
    sigma_S += 1e-5
    sigma_V += 1e-5

    s_peaks, _ = find_peaks(s_hist)
    threshold = S_sub + 2.0 * sigma_S
    candidates = s_peaks[s_peaks > threshold]
    S_thin = float(candidates[0]) if len(candidates) > 0 else None

    return {
        "hsv": hsv,
        "H": H, "S": S, "V": V,
        "H_sub": H_sub, "S_sub": S_sub, "V_sub": V_sub,
        "sigma_H": sigma_H, "sigma_S": sigma_S, "sigma_V": sigma_V,
        "S_thin": S_thin,
    }


def compute_probability_old_with_gate(d: dict, v_thresh: float = 2.0,
                                      h_thresh: float = 2.0) -> np.ndarray:
    """
    OLD: linear D_S from S_sub, then hard gate on V and H.
    Returns raw probability map (not thresholded).
    """
    S = d["S"]
    D_S = np.abs(S - d["S_sub"]) / d["sigma_S"]

    V = d["V"]
    D_V = np.abs(V - d["V_sub"]) / d["sigma_V"]

    H = d["H"]
    delta_H = np.abs(((H - d["H_sub"] + 90.0) % 180.0) - 90.0)
    D_H = delta_H / d["sigma_H"]
    S_gate = d["S_sub"] + (2.0 * d["sigma_S"])
    D_H_gated = D_H * (S > S_gate)

    # Composite: S only
    D_composite = 1.0 * D_S
    P = 1.0 - np.exp(-0.5 * D_composite**2)

    # Hard gate
    mask = (D_V < v_thresh) & (D_H_gated < h_thresh)
    P = (P * mask).astype(np.float32)
    return P


def generate_mask(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold probability map to binary mask (0/255)."""
    return (prob_map >= threshold).astype(np.uint8) * 255


def analyze_components(mask: np.ndarray, min_area: int = 100):
    """
    Use connected components to find discrete flakes.
    Returns list of (label, area, centroid_x, centroid_y, bbox).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    components = []
    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[label]
        components.append({
            "label": label,
            "area": area,
            "bbox": (x, y, w, h),
            "center": (cx, cy),
        })
    return components


def draw_overlay(rgb: np.ndarray, mask: np.ndarray, components: list,
                 title: str = "") -> np.ndarray:
    """Draw mask as semi-transparent red overlay + contours + labels."""
    overlay = rgb.copy()
    # Red overlay on mask regions
    mask_bool = mask > 0
    overlay[mask_bool] = (
        0.5 * overlay[mask_bool].astype(np.float32)
        + 0.5 * np.array([255, 0, 0], dtype=np.float32)
    ).astype(np.uint8)

    # Draw contours for each component
    for comp in components:
        x, y, w, h = comp["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cx, cy = comp["center"]
        cv2.circle(overlay, (int(cx), int(cy)), 8, (255, 255, 0), -1)

    return overlay


def write_stats(components: list, path: Path, threshold: float, v_thresh: float,
                h_thresh: float) -> None:
    """Write component statistics to a text file."""
    with open(path, "w") as f:
        f.write(f"Flake Detection Statistics\n")
        f.write(f"==========================\n\n")
        f.write(f"Probability threshold: {threshold}\n")
        f.write(f"V hard gate: {v_thresh}σ\n")
        f.write(f"H hard gate: {h_thresh}σ\n")
        f.write(f"Minimum area: {100}px\n")
        f.write(f"\nDetected components: {len(components)}\n\n")
        f.write(f"{'#':>3} {'Area(px)':>10} {'Width':>6} {'Height':>6} {'CenterX':>10} {'CenterY':>10}\n")
        f.write("-" * 55 + "\n")
        for i, comp in enumerate(components, 1):
            x, y, w, h = comp["bbox"]
            cx, cy = comp["center"]
            f.write(f"{i:3d} {comp['area']:10d} {w:6d} {h:6d} {cx:10.1f} {cy:10.1f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "dataset" / "sample2" / "tmd_sample_13.jpg"
        ),
        help="Path to the RGB image",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "masks13"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.3,0.5,0.7",
        help="Comma-separated probability thresholds to try (default: 0.3,0.5,0.7)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image_path = Path(args.image)
    print(f"Loading image: {image_path}")
    rgb = load_rgb_image(image_path)
    print(f"  Shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Compute diagnostics
    d = compute_base_diagnostics(rgb)
    print(f"  Substrate peak (H, S, V): ({d['H_sub']:.0f}, {d['S_sub']:.0f}, {d['V_sub']:.0f})")
    print(f"  Substrate std  (H, S, V): ({d['sigma_H']:.3f}, {d['sigma_S']:.3f}, {d['sigma_V']:.3f})")
    print(f"  Thin-flake peak (S_thin): {d['S_thin']}")

    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(",")]

    # Generate masks for both hard gate variants (2σ and 1.5σ)
    gate_variants = [
        ("gate2sig", 2.0, 2.0),
        ("gate1p5sig", 1.5, 1.5),
    ]

    for gate_name, v_thresh, h_thresh in gate_variants:
        print(f"\n=== Hard Gate: D_V<{v_thresh}σ, D_H<{h_thresh}σ ===")
        P = compute_probability_old_with_gate(d, v_thresh, h_thresh)
        print(f"  Probability range: [{P.min():.4f}, {P.max():.4f}]")

        for threshold in thresholds:
            mask = generate_mask(P, threshold=threshold)
            mask_area = int(np.sum(mask > 0))
            total_pixels = mask.size
            coverage = 100.0 * mask_area / total_pixels
            print(f"  Threshold {threshold}: mask area={mask_area}px ({coverage:.2f}% coverage)")

            # Connected components
            components = analyze_components(mask, min_area=100)
            print(f"    Detected {len(components)} flakes (min_area=100px)")

            # Save binary mask
            mask_path = output_dir / f"mask_{gate_name}_th{threshold}.png"
            cv2.imwrite(str(mask_path), mask)
            print(f"    Saved: {mask_path}")

            # Save overlay
            overlay = draw_overlay(rgb, mask, components)
            overlay_path = output_dir / f"overlay_{gate_name}_th{threshold}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"    Saved: {overlay_path}")

            # Save stats
            stats_path = output_dir / f"stats_{gate_name}_th{threshold}.txt"
            write_stats(components, stats_path, threshold, v_thresh, h_thresh)
            print(f"    Saved: {stats_path}")

    # Also save a combined figure: row 0 = 2σ gate, row 1 = 1.5σ gate
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Flake Masks for {image_path.name}", fontsize=14)

    # 2σ gate at 0.3, 0.5, 0.7
    for i, th in enumerate([0.3, 0.5, 0.7]):
        P = compute_probability_old_with_gate(d, 2.0, 2.0)
        mask = generate_mask(P, threshold=th)
        components = analyze_components(mask, min_area=100)
        overlay = draw_overlay(rgb, mask, components)
        ax = axes[0, i]
        ax.imshow(overlay)
        ax.set_title(f"2σ Gate, P>={th} ({len(components)} flakes)")
        ax.axis("off")

    # 1.5σ gate
    P15 = compute_probability_old_with_gate(d, 1.5, 1.5)
    for i, th in enumerate([0.3, 0.5, 0.7]):
        mask = generate_mask(P15, threshold=th)
        components = analyze_components(mask, min_area=100)
        overlay = draw_overlay(rgb, mask, components)
        ax = axes[1, i]
        ax.imshow(overlay)
        ax.set_title(f"1.5σ Gate, P>={th} ({len(components)} flakes)")
        ax.axis("off")

    plt.tight_layout()
    combined_path = output_dir / "combined_masks.png"
    plt.savefig(str(combined_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved combined figure: {combined_path}")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()