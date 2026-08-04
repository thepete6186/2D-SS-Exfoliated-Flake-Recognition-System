#!/usr/bin/env python3
"""
run_diagnostics_old_ds.py

Generates a diagnostic figure (diagnostics2.png) that shows the D_S map
using the PREVIOUS logic — a Gaussian kernel centered on the second
saturation peak (S_thin), i.e., proximity to the thin-flake shoulder.

This is for comparison against the current D_S map which uses linear
distance from the main substrate peak (S_sub).

Usage:
    python run_diagnostics_old_ds.py [--image PATH]
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
    """Load an image from disk and return it in RGB (HxWx3, uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate D_S map using OLD (2nd peak) logic."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "dataset" / "sample2" / "tmd_sample_3.jpg"
        ),
        help="Path to the RGB image to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "diagnostics2.png"),
        help="Path to save the diagnostic figure",
    )
    args = parser.parse_args()

    # Load image
    image_path = Path(args.image)
    print(f"Loading image: {image_path}")
    rgb = load_rgb_image(image_path)
    print(f"  Shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Convert to HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Compute histograms
    h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

    # Find substrate peaks (global max)
    H_sub = float(np.argmax(h_hist))
    S_sub = float(np.argmax(s_hist))
    V_sub = float(np.argmax(v_hist))

    # Build substrate mask and estimate noise floor
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

    # Find S_thin (2nd peak) — previous logic
    s_peaks, _ = find_peaks(s_hist)
    threshold = S_sub + 2.0 * sigma_S
    candidates = s_peaks[s_peaks > threshold]
    S_thin = float(candidates[0]) if len(candidates) > 0 else None

    print(f"  Substrate peak (H, S, V): ({H_sub:.0f}, {S_sub:.0f}, {V_sub:.0f})")
    print(f"  Substrate std  (H, S, V): ({sigma_H:.3f}, {sigma_S:.3f}, {sigma_V:.3f})")
    print(f"  Thin-flake peak (S_thin): {S_thin}")

    # OLD D_S: Gaussian centered on S_thin (2nd peak)
    if S_thin is not None:
        D_S_old = np.exp(-0.5 * ((S - S_thin) / sigma_S) ** 2)
        dS_title = f"Old D_S: Gaussian @ S_thin={S_thin:.0f}"
    else:
        D_S_old = np.abs(S - S_sub) / sigma_S
        dS_title = "Old D_S: (no 2nd peak) linear |S-S_sub|/σ"

    # NEW/current D_S: linear distance from substrate peak
    D_S_new = np.abs(S - S_sub) / sigma_S

    # Build 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"D_S Map Comparison — {image_path.name}\n"
        f"S_sub={S_sub:.0f}, S_thin={S_thin if S_thin is not None else 'None'}, "
        f"σ_S={sigma_S:.2f}",
        fontsize=14,
    )

    # 1. Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original RGB Image")
    axes[0, 0].axis("off")

    # 2. Old D_S (Gaussian on 2nd peak)
    im_old = axes[0, 1].imshow(D_S_old, cmap="viridis")
    axes[0, 1].set_title(dS_title)
    axes[0, 1].axis("off")
    plt.colorbar(im_old, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. New D_S (linear from main peak)
    im_new = axes[1, 0].imshow(D_S_new, cmap="viridis")
    axes[1, 0].set_title(f"New D_S: linear |S-S_sub|/σ (S_sub={S_sub:.0f})")
    axes[1, 0].axis("off")
    plt.colorbar(im_new, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4. S histogram with peaks marked
    ax_hist = axes[1, 1]
    s_hist_n = s_hist / s_hist.max() if s_hist.max() > 0 else s_hist
    ax_hist.plot(s_hist_n, color="green", linewidth=0.8, label="Saturation")
    ax_hist.axvline(x=S_sub, color="green", linestyle="--", linewidth=1.5,
                    label=f"S_sub={S_sub:.0f}")
    if S_thin is not None:
        ax_hist.axvline(x=S_thin, color="yellow", linestyle="--", linewidth=2.0,
                        label=f"S_thin={S_thin:.0f}")
    ax_hist.axvline(x=S_sub + 2.0 * sigma_S, color="red", linestyle=":",
                    linewidth=1.0, label=f"S_sub+2σ={S_sub + 2.0 * sigma_S:.1f}")
    ax_hist.set_title("S Histogram with Peak Markers")
    ax_hist.set_xlabel("Bin")
    ax_hist.set_ylabel("Normalized Count")
    ax_hist.legend(fontsize=8, loc="upper right")
    ax_hist.set_xlim(0, 255)

    plt.tight_layout()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic figure: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()