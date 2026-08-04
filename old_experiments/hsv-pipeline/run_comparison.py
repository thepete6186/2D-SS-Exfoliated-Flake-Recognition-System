#!/usr/bin/env python3
"""
run_comparison.py

Generate diagnostics and heatmaps for BOTH S-channel approaches:

  OLD = 1st main peak (S_sub): D_S = |S - S_sub| / sigma_S
  NEW = 2nd peak (S_thin):    D_S = exp(-0.5 * ((S - S_thin) / sigma_S)^2)

Both use heavily downweighted V and H channels (w_S=1.0, w_V=0.0, w_H=0.0),
so the probability is driven purely by the S channel.

Outputs:
  results/diagnostics_old.png  - full 2x3 diagnostic using old (main peak) logic
  results/diagnostics_new.png  - full 2x3 diagnostic using new (2nd peak) logic
  results/heatmap_old.png      - heatmap using old (main peak) logic
  results/heatmap_new.png      - heatmap using new (2nd peak) logic

Usage:
    python run_comparison.py [--image PATH]
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


def compute_base_diagnostics(rgb: np.ndarray):
    """
    Compute shared HSV statistics needed by both approaches.
    Returns a dict with S_sub, S_thin, sigma_S, H_sub, V_sub, etc.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Histograms
    h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

    # Substrate peaks (global max)
    H_sub = float(np.argmax(h_hist))
    S_sub = float(np.argmax(s_hist))
    V_sub = float(np.argmax(v_hist))

    # Substrate noise floor
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

    # Find S_thin (2nd peak)
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
        "h_hist": h_hist, "s_hist": s_hist, "v_hist": v_hist,
    }


def compute_probability_old(d: dict) -> np.ndarray:
    """
    OLD: D_S from main peak: |S - S_sub| / sigma_S
    Then composite probability with heavy V/H downweighting.
    """
    S = d["S"]
    S_sub = d["S_sub"]
    sigma_S = d["sigma_S"]

    D_S = np.abs(S - S_sub) / sigma_S
    D_composite = 1.0 * D_S
    P = 1.0 - np.exp(-0.5 * D_composite**2)
    return P.astype(np.float32), D_S


def compute_probability_new(d: dict) -> np.ndarray:
    """
    NEW: D_S from 2nd peak: Gaussian centered on S_thin
    Then composite probability with heavy V/H downweighting.
    """
    S = d["S"]
    S_thin = d["S_thin"]
    sigma_S = d["sigma_S"]

    if S_thin is not None:
        D_S = np.exp(-0.5 * ((S - S_thin) / sigma_S) ** 2)
    else:
        D_S = np.abs(S - d["S_sub"]) / sigma_S
    D_composite = 1.0 * D_S
    P = 1.0 - np.exp(-0.5 * D_composite**2)
    return P.astype(np.float32), D_S


def build_heatmap_rgb(prob_map: np.ndarray) -> np.ndarray:
    """Convert probability map to inferno RGB heatmap."""
    prob_uint8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def plot_histograms(ax, d: dict, title: str) -> None:
    """Plot H, S, V histograms with peaks marked."""
    h_hist_n = d["h_hist"] / d["h_hist"].max() if d["h_hist"].max() > 0 else d["h_hist"]
    s_hist_n = d["s_hist"] / d["s_hist"].max() if d["s_hist"].max() > 0 else d["s_hist"]
    v_hist_n = d["v_hist"] / d["v_hist"].max() if d["v_hist"].max() > 0 else d["v_hist"]

    ax.plot(h_hist_n, color="red", linewidth=0.8, label="Hue")
    ax.plot(s_hist_n, color="green", linewidth=0.8, label="Saturation")
    ax.plot(v_hist_n, color="blue", linewidth=0.8, label="Value")

    ax.axvline(x=d["H_sub"], color="red", linestyle="--", linewidth=1.5,
               label=f"H_sub={d['H_sub']:.0f}")
    ax.axvline(x=d["S_sub"], color="green", linestyle="--", linewidth=1.5,
               label=f"S_sub={d['S_sub']:.0f}")
    ax.axvline(x=d["V_sub"], color="blue", linestyle="--", linewidth=1.5,
               label=f"V_sub={d['V_sub']:.0f}")
    if d["S_thin"] is not None:
        ax.axvline(x=d["S_thin"], color="yellow", linestyle="--", linewidth=2.0,
                   label=f"S_thin={d['S_thin']:.0f}")

    ax.set_title(title)
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized Count")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 255)


def build_diagnostic_figure(
    rgb: np.ndarray,
    d: dict,
    prob_map: np.ndarray,
    D_S: np.ndarray,
    heatmap_rgb: np.ndarray,
    title_prefix: str,
) -> plt.Figure:
    """Build a 2x3 diagnostic figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{title_prefix} — Substrate H={d['H_sub']:.0f}, S={d['S_sub']:.0f}, V={d['V_sub']:.0f}\n"
        f"σ: H={d['sigma_H']:.2f}, S={d['sigma_S']:.2f}, V={d['sigma_V']:.2f} | "
        f"S_thin={d['S_thin'] if d['S_thin'] is not None else 'N/A'}"
        f" | weights: w_S=1.0, w_V=0.0, w_H=0.0",
        fontsize=12,
    )

    # 1. Original RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original RGB Image")
    axes[0, 0].axis("off")

    # 2. S-Channel Distance Map
    im_s = axes[0, 1].imshow(D_S, cmap="viridis")
    axes[0, 1].set_title("2. S-Channel Distance (D_S)")
    axes[0, 1].axis("off")
    plt.colorbar(im_s, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. V-Channel Distance Map
    D_V = np.abs(d["V"] - d["V_sub"]) / d["sigma_V"]
    im_v = axes[0, 2].imshow(D_V, cmap="viridis")
    axes[0, 2].set_title("3. V-Channel Distance (D_V)")
    axes[0, 2].axis("off")
    plt.colorbar(im_v, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Gated H-Channel Distance Map
    delta_H = np.abs(((d["H"] - d["H_sub"] + 90.0) % 180.0) - 90.0)
    D_H = delta_H / d["sigma_H"]
    S_gate = d["S_sub"] + (2.0 * d["sigma_S"])
    D_H_gated = D_H * (d["S"] > S_gate)
    im_h = axes[1, 0].imshow(D_H_gated, cmap="viridis")
    axes[1, 0].set_title("4. Gated H-Channel Distance (D_H_gated)")
    axes[1, 0].axis("off")
    plt.colorbar(im_h, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Channel Histograms
    plot_histograms(axes[1, 1], d, "5. Channel Histograms with Peaks")

    # 6. Final Probability Heatmap
    axes[1, 2].imshow(heatmap_rgb)
    axes[1, 2].set_title("6. Probability Heatmap (P_map)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diagnostics and heatmaps for OLD (main peak) and NEW (2nd peak) S logic."
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
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip saving heatmap images (diagnostics only)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image_path = Path(args.image)
    print(f"Loading image: {image_path}")
    rgb = load_rgb_image(image_path)
    print(f"  Shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Compute shared diagnostics
    d = compute_base_diagnostics(rgb)
    print(f"  Substrate peak (H, S, V): ({d['H_sub']:.0f}, {d['S_sub']:.0f}, {d['V_sub']:.0f})")
    print(f"  Substrate std  (H, S, V): ({d['sigma_H']:.3f}, {d['sigma_S']:.3f}, {d['sigma_V']:.3f})")
    print(f"  Thin-flake peak (S_thin): {d['S_thin']}")

    # === OLD: 1st main peak ===
    print("\nComputing OLD (1st main peak) probability...")
    P_old, D_S_old = compute_probability_old(d)
    heatmap_old = build_heatmap_rgb(P_old)
    print(f"  OLD Probability range: [{P_old.min():.4f}, {P_old.max():.4f}]")
    print(f"  OLD D_S range: [{D_S_old.min():.4f}, {D_S_old.max():.4f}]")

    fig_old = build_diagnostic_figure(
        rgb, d, P_old, D_S_old, heatmap_old, "OLD: 1st Main Peak (S_sub)"
    )
    old_diag_path = output_dir / "diagnostics_old.png"
    fig_old.savefig(str(old_diag_path), dpi=150, bbox_inches="tight")
    print(f"  Saved: {old_diag_path}")
    plt.close(fig_old)

    if not args.no_heatmap:
        old_heat_path = output_dir / "heatmap_old.png"
        cv2.imwrite(str(old_heat_path), cv2.cvtColor(heatmap_old, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {old_heat_path}")

    # === NEW: 2nd peak ===
    if d["S_thin"] is not None:
        print("\nComputing NEW (2nd peak) probability...")
        P_new, D_S_new = compute_probability_new(d)
        heatmap_new = build_heatmap_rgb(P_new)
        print(f"  NEW Probability range: [{P_new.min():.4f}, {P_new.max():.4f}]")
        print(f"  NEW D_S range: [{D_S_new.min():.4f}, {D_S_new.max():.4f}]")

        fig_new = build_diagnostic_figure(
            rgb, d, P_new, D_S_new, heatmap_new, "NEW: 2nd Peak (S_thin)"
        )
        new_diag_path = output_dir / "diagnostics_new.png"
        fig_new.savefig(str(new_diag_path), dpi=150, bbox_inches="tight")
        print(f"  Saved: {new_diag_path}")
        plt.close(fig_new)

        if not args.no_heatmap:
            new_heat_path = output_dir / "heatmap_new.png"
            cv2.imwrite(str(new_heat_path), cv2.cvtColor(heatmap_new, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {new_heat_path}")
    else:
        print("\nNo S_thin detected — skipping NEW approach.")


if __name__ == "__main__":
    main()