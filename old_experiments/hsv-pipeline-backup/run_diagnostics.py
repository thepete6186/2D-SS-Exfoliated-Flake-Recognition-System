#!/usr/bin/env python3
"""
run_diagnostics.py

Diagnostic visualizer for the HSVPipeline.

Loads an RGB image, executes the HSVPipeline, and plots a 2x3 Matplotlib
figure showing:
  1. Original RGB Image
  2. Raw S-Channel Distance Map (D_S)
  3. Raw V-Channel Distance Map (D_V)
  4. Gated H-Channel Distance Map (D_H_gated)
  5. Channel Histograms with overlay markers showing detected peak locations
  6. Final Combined Probability Heatmap (P_map)

Usage:
    python run_diagnostics.py [--image PATH] [--output PATH]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsv_pipeline import HSVPipeline


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image from disk and return it in RGB (HxWx3, uint8)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_histograms_with_peaks(
    hsv: np.ndarray,
    substrate_peak: tuple,
    ax: plt.Axes,
) -> None:
    """
    Plot the H, S, V histograms with vertical lines marking the detected
    substrate peaks.
    """
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H_sub, S_sub, V_sub = substrate_peak

    # Compute histograms
    h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

    # Normalize for display
    h_hist_n = h_hist / h_hist.max() if h_hist.max() > 0 else h_hist
    s_hist_n = s_hist / s_hist.max() if s_hist.max() > 0 else s_hist
    v_hist_n = v_hist / v_hist.max() if v_hist.max() > 0 else v_hist

    # Plot each channel
    ax.plot(h_hist_n, color="red", linewidth=0.8, label="Hue")
    ax.plot(s_hist_n, color="green", linewidth=0.8, label="Saturation")
    ax.plot(v_hist_n, color="blue", linewidth=0.8, label="Value")

    # Mark substrate peaks
    ax.axvline(x=H_sub, color="red", linestyle="--", linewidth=1.5,
               label=f"H_sub={H_sub:.0f}")
    ax.axvline(x=S_sub, color="green", linestyle="--", linewidth=1.5,
               label=f"S_sub={S_sub:.0f}")
    ax.axvline(x=V_sub, color="blue", linestyle="--", linewidth=1.5,
               label=f"V_sub={V_sub:.0f}")

    ax.set_title("Channel Histograms with Substrate Peaks")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized Count")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 255)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HSVPipeline diagnostics on a single image."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "dataset" / "sample2" / "tmd_sample_3.jpg"
        ),
        help="Path to the RGB image to process (default: dataset/sample2/tmd_sample_3.jpg)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "diagnostics.png"),
        help="Path to save the diagnostic figure (default: hsv-pipeline/results/diagnostics.png)",
    )
    parser.add_argument(
        "--w_s", type=float, default=1.0, help="Saturation channel weight (default: 1.0)"
    )
    parser.add_argument(
        "--w_v", type=float, default=0.0, help="Value channel weight (default: 0.0)"
    )
    parser.add_argument(
        "--w_h", type=float, default=0.0, help="Hue channel weight (default: 0.0)"
    )
    args = parser.parse_args()

    # Load image
    image_path = Path(args.image)
    print(f"Loading image: {image_path}")
    rgb = load_rgb_image(image_path)
    print(f"  Shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Run pipeline
    print("Running HSVPipeline...")
    pipeline = HSVPipeline(w_S=args.w_s, w_V=args.w_v, w_H=args.w_h)
    result = pipeline.process(rgb)

    prob_map = result["probability_map"]
    D_S = result["D_S"]
    D_V = result["D_V"]
    D_H_gated = result["D_H_gated"]
    substrate_peak = result["substrate_peak"]
    substrate_std = result["substrate_std"]
    heatmap = result["probability_heatmap_rgb"]

    print(f"  Substrate peak (H, S, V): {substrate_peak}")
    print(f"  Substrate std  (H, S, V): {substrate_std}")
    print(f"  Probability range: [{prob_map.min():.4f}, {prob_map.max():.4f}]")

    # Convert HSV for histogram plotting
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"HSVPipeline Diagnostics — {image_path.name}\n"
        f"Substrate peak: H={substrate_peak[0]:.0f}, S={substrate_peak[1]:.0f}, "
        f"V={substrate_peak[2]:.0f} | "
        f"σ: H={substrate_std[0]:.2f}, S={substrate_std[1]:.2f}, "
        f"V={substrate_std[2]:.2f}",
        fontsize=12,
    )

    # 1. Original RGB Image
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original RGB Image")
    axes[0, 0].axis("off")

    # 2. Raw S-Channel Distance Map
    im_s = axes[0, 1].imshow(D_S, cmap="viridis")
    axes[0, 1].set_title("2. S-Channel Distance (D_S)")
    axes[0, 1].axis("off")
    plt.colorbar(im_s, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Raw V-Channel Distance Map
    im_v = axes[0, 2].imshow(D_V, cmap="viridis")
    axes[0, 2].set_title("3. V-Channel Distance (D_V)")
    axes[0, 2].axis("off")
    plt.colorbar(im_v, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Gated H-Channel Distance Map
    im_h = axes[1, 0].imshow(D_H_gated, cmap="viridis")
    axes[1, 0].set_title("4. Gated H-Channel Distance (D_H_gated)")
    axes[1, 0].axis("off")
    plt.colorbar(im_h, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Channel Histograms with peaks
    plot_histograms_with_peaks(hsv, substrate_peak, axes[1, 1])

    # 6. Final Combined Probability Heatmap
    axes[1, 2].imshow(heatmap)
    axes[1, 2].set_title("6. Final Probability Heatmap (P_map)")
    axes[1, 2].axis("off")

    plt.tight_layout()

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved diagnostic figure: {output_path}")

    # Also save the probability heatmap separately
    heatmap_path = output_path.parent / f"{image_path.stem}_heatmap.png"
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print(f"Saved heatmap: {heatmap_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()