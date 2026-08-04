#!/usr/bin/env python3
"""
run_aggressive_variants.py

Run the HSVPipeline on sample3 images with various H/V penalization schemes.

The idea: use the S channel as the primary flake detector (OLD: linear from
main peak), but aggressively cancel pixels that show ANY significant H or V
deviation from substrate. This creates a "pure saturation" flake detector.

Outputs:
  results/diagnostics_old.png   - diagnostic using OLD (main peak) S logic
  results/diagnostics_new.png   - diagnostic using NEW (2nd peak) S logic
  results/heatmap_<variant>.png - multiple heatmap variants for each image

Usage:
    python run_aggressive_variants.py
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
        "h_hist": h_hist, "s_hist": s_hist, "v_hist": v_hist,
    }


def build_heatmap_rgb(prob_map: np.ndarray) -> np.ndarray:
    prob_uint8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def plot_histograms(ax, d: dict, title: str) -> None:
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
    D_S: np.ndarray,
    D_V: np.ndarray,
    D_H_gated: np.ndarray,
    heatmap_rgb: np.ndarray,
    title_prefix: str,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{title_prefix} — Substrate H={d['H_sub']:.0f}, S={d['S_sub']:.0f}, V={d['V_sub']:.0f}\n"
        f"σ: H={d['sigma_H']:.2f}, S={d['sigma_S']:.2f}, V={d['sigma_V']:.2f} | "
        f"S_thin={d['S_thin'] if d['S_thin'] is not None else 'N/A'}",
        fontsize=12,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original RGB Image")
    axes[0, 0].axis("off")

    im_s = axes[0, 1].imshow(D_S, cmap="viridis")
    axes[0, 1].set_title("2. S-Channel Distance (D_S)")
    axes[0, 1].axis("off")
    plt.colorbar(im_s, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im_v = axes[0, 2].imshow(D_V, cmap="viridis")
    axes[0, 2].set_title("3. V-Channel Distance (D_V)")
    axes[0, 2].axis("off")
    plt.colorbar(im_v, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_h = axes[1, 0].imshow(D_H_gated, cmap="viridis")
    axes[1, 0].set_title("4. Gated H-Channel Distance (D_H_gated)")
    axes[1, 0].axis("off")
    plt.colorbar(im_h, ax=axes[1, 0], fraction=0.046, pad=0.04)

    plot_histograms(axes[1, 1], d, "5. Channel Histograms with Peaks")

    axes[1, 2].imshow(heatmap_rgb)
    axes[1, 2].set_title("6. Probability Heatmap (P_map)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


def compute_probability_old(d: dict, w_S=1.0, w_V=0.0, w_H=0.0):
    """OLD: linear D_S from S_sub; combined with optional H/V weights."""
    S = d["S"]
    D_S = np.abs(S - d["S_sub"]) / d["sigma_S"]

    V = d["V"]
    D_V = np.abs(V - d["V_sub"]) / d["sigma_V"]

    H = d["H"]
    delta_H = np.abs(((H - d["H_sub"] + 90.0) % 180.0) - 90.0)
    D_H = delta_H / d["sigma_H"]
    S_gate = d["S_sub"] + (2.0 * d["sigma_S"])
    D_H_gated = D_H * (S > S_gate)

    D_composite = w_S * D_S + w_V * D_V + w_H * D_H_gated
    P = 1.0 - np.exp(-0.5 * D_composite**2)
    return P.astype(np.float32), D_S, D_V, D_H_gated


def compute_probability_new(d: dict, w_S=1.0, w_V=0.0, w_H=0.0):
    """NEW: Gaussian D_S from S_thin; combined with optional H/V weights."""
    S = d["S"]
    if d["S_thin"] is not None:
        D_S = np.exp(-0.5 * ((S - d["S_thin"]) / d["sigma_S"]) ** 2)
    else:
        D_S = np.abs(S - d["S_sub"]) / d["sigma_S"]

    V = d["V"]
    D_V = np.abs(V - d["V_sub"]) / d["sigma_V"]

    H = d["H"]
    delta_H = np.abs(((H - d["H_sub"] + 90.0) % 180.0) - 90.0)
    D_H = delta_H / d["sigma_H"]
    S_gate = d["S_sub"] + (2.0 * d["sigma_S"])
    D_H_gated = D_H * (S > S_gate)

    D_composite = w_S * D_S + w_V * D_V + w_H * D_H_gated
    P = 1.0 - np.exp(-0.5 * D_composite**2)
    return P.astype(np.float32), D_S, D_V, D_H_gated


def apply_hard_gate(P: np.ndarray, D_V: np.ndarray, D_H_gated: np.ndarray,
                    v_thresh: float = 2.0, h_thresh: float = 2.0) -> np.ndarray:
    """Hard gate: zero out pixels where D_V or D_H exceed thresholds."""
    mask = (D_V < v_thresh) & (D_H_gated < h_thresh)
    return (P * mask).astype(np.float32)


def apply_soft_gate(P: np.ndarray, D_V: np.ndarray, D_H_gated: np.ndarray,
                    v_sigma: float = 1.0, h_sigma: float = 1.0) -> np.ndarray:
    """Soft gate: multiply by exp(-0.5*(D_V/v_sigma)^2) * exp(-0.5*(D_H/h_sigma)^2)."""
    gate_V = np.exp(-0.5 * (D_V / v_sigma) ** 2)
    gate_H = np.exp(-0.5 * (D_H_gated / h_sigma) ** 2)
    return (P * gate_V * gate_H).astype(np.float32)


def apply_aggressive_v_gate(P: np.ndarray, D_V: np.ndarray, D_H_gated: np.ndarray,
                            v_thresh: float = 1.0, h_thresh: float = 2.0) -> np.ndarray:
    """
    Aggressive V hard gate: zero out pixels where D_V exceeds a very tight
    threshold (e.g., 0.5σ or 1σ), while keeping H at a more lenient 2σ.
    This heavily penalizes ANY V deviation from substrate.
    """
    mask = (D_V < v_thresh) & (D_H_gated < h_thresh)
    return (P * mask).astype(np.float32)


def apply_aggressive_v_soft_gate(P: np.ndarray, D_V: np.ndarray, D_H_gated: np.ndarray,
                                 v_sigma: float = 0.5, h_sigma: float = 2.0) -> np.ndarray:
    """
    Aggressive V soft gate: multiply by exp(-0.5*(D_V/v_sigma)^2) with a very
    tight v_sigma (e.g., 0.3 or 0.5), while keeping H at a more lenient 2σ.
    This creates a sharp Gaussian that kills pixels with even minor V shifts.
    """
    gate_V = np.exp(-0.5 * (D_V / v_sigma) ** 2)
    gate_H = np.exp(-0.5 * (D_H_gated / h_sigma) ** 2)
    return (P * gate_V * gate_H).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent / "dataset" / "sample3" / "ws2-251104161326950.jpg"
        ),
        help="Path to the RGB image to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(args.image)
    print(f"Loading image: {image_path}")
    rgb = load_rgb_image(image_path)
    print(f"  Shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Compute diagnostics
    d = compute_base_diagnostics(rgb)
    print(f"  Substrate peak (H, S, V): ({d['H_sub']:.0f}, {d['S_sub']:.0f}, {d['V_sub']:.0f})")
    print(f"  Substrate std  (H, S, V): ({d['sigma_H']:.3f}, {d['sigma_S']:.3f}, {d['sigma_V']:.3f})")
    print(f"  Thin-flake peak (S_thin): {d['S_thin']}")

    # === OLD: 1st main peak, no H/V penalty (baseline) ===
    P_old_base, D_S, D_V, D_H_gated = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    heatmap_old_base = build_heatmap_rgb(P_old_base)
    print(f"\nOLD baseline (S-only): P range [{P_old_base.min():.4f}, {P_old_base.max():.4f}]")

    # Save diagnostic (OLD, S-only) in results root
    fig_old = build_diagnostic_figure(
        rgb, d, D_S, D_V, D_H_gated, heatmap_old_base, "OLD: 1st Main Peak (S_only)"
    )
    fig_old.savefig(str(output_dir / "diagnostics_old.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {output_dir / 'diagnostics_old.png'}")
    plt.close(fig_old)

    # === NEW: 2nd peak, no H/V penalty (baseline) ===
    if d["S_thin"] is not None:
        P_new_base, D_S_new, D_V_new, D_H_gated_new = compute_probability_new(d, w_S=1.0, w_V=0.0, w_H=0.0)
        heatmap_new_base = build_heatmap_rgb(P_new_base)
        print(f"NEW baseline (S-only): P range [{P_new_base.min():.4f}, {P_new_base.max():.4f}]")

        fig_new = build_diagnostic_figure(
            rgb, d, D_S_new, D_V_new, D_H_gated_new, heatmap_new_base, "NEW: 2nd Peak (S_only)"
        )
        fig_new.savefig(str(output_dir / "diagnostics_new.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {output_dir / 'diagnostics_new.png'}")
        plt.close(fig_new)

    # === VARIANTS: aggressive H/V penalization on OLD D_S ===
    variants = {}

    # Variant 1: moderate weights w_S=0.7, w_V=0.2, w_H=0.1
    P_v1, _, _, _ = compute_probability_old(d, w_S=0.7, w_V=0.2, w_H=0.1)
    variants["heatmap_old_w070_v020_h010.png"] = (P_v1, "w_S=0.7, w_V=0.2, w_H=0.1")

    # Variant 2: aggressive weights w_S=0.6, w_V=0.25, w_H=0.15
    P_v2, _, _, _ = compute_probability_old(d, w_S=0.6, w_V=0.25, w_H=0.15)
    variants["heatmap_old_w060_v025_h015.png"] = (P_v2, "w_S=0.6, w_V=0.25, w_H=0.15")

    # Variant 3: very aggressive weights w_S=0.5, w_V=0.3, w_H=0.2
    P_v3, _, _, _ = compute_probability_old(d, w_S=0.5, w_V=0.3, w_H=0.2)
    variants["heatmap_old_w050_v030_h020.png"] = (P_v3, "w_S=0.5, w_V=0.3, w_H=0.2")

    # Variant 4: hard gate with threshold 2σ on V and H
    P_v4, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v4 = apply_hard_gate(P_v4, D_V, D_H_gated, v_thresh=2.0, h_thresh=2.0)
    variants["heatmap_old_hardgate_2sig.png"] = (P_v4, "hard gate D_V<2σ, D_H<2σ")

    # Variant 5: hard gate with threshold 1.5σ
    P_v5, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v5 = apply_hard_gate(P_v5, D_V, D_H_gated, v_thresh=1.5, h_thresh=1.5)
    variants["heatmap_old_hardgate_1p5sig.png"] = (P_v5, "hard gate D_V<1.5σ, D_H<1.5σ")

    # Variant 6: hard gate with threshold 1σ (very strict)
    P_v6, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v6 = apply_hard_gate(P_v6, D_V, D_H_gated, v_thresh=1.0, h_thresh=1.0)
    variants["heatmap_old_hardgate_1sig.png"] = (P_v6, "hard gate D_V<1σ, D_H<1σ")

    # Variant 7: soft gaussian gate with σ=2
    P_v7, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v7 = apply_soft_gate(P_v7, D_V, D_H_gated, v_sigma=2.0, h_sigma=2.0)
    variants["heatmap_old_softgate_2sig.png"] = (P_v7, "soft gate exp(-0.5(D/2σ)²)")

    # Variant 8: soft gaussian gate with σ=1
    P_v8, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v8 = apply_soft_gate(P_v8, D_V, D_H_gated, v_sigma=1.0, h_sigma=1.0)
    variants["heatmap_old_softgate_1sig.png"] = (P_v8, "soft gate exp(-0.5(D/1σ)²)")

    # Variant 9: soft gaussian gate with σ=0.5 (extremely aggressive)
    P_v9, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v9 = apply_soft_gate(P_v9, D_V, D_H_gated, v_sigma=0.5, h_sigma=0.5)
    variants["heatmap_old_softgate_0p5sig.png"] = (P_v9, "soft gate exp(-0.5(D/0.5σ)²)")

    # Variant 10: S-only baseline for reference
    variants["heatmap_old_sonly.png"] = (P_old_base, "S only (baseline, w_S=1)")

    # === EXTRA-AGGRESSIVE V PENALIZATION VARIANTS ===
    # These specifically target the V channel with much tighter thresholds
    # than the standard 2σ gate, while keeping H at 2σ.

    # Variant 11: hard gate D_V < 1σ, D_H < 2σ
    P_v11, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v11 = apply_aggressive_v_gate(P_v11, D_V, D_H_gated, v_thresh=1.0, h_thresh=2.0)
    variants["heatmap_old_hardgate_v1sig_h2sig.png"] = (P_v11, "hard gate D_V<1σ, D_H<2σ")

    # Variant 12: hard gate D_V < 0.5σ, D_H < 2σ (very aggressive)
    P_v12, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v12 = apply_aggressive_v_gate(P_v12, D_V, D_H_gated, v_thresh=0.5, h_thresh=2.0)
    variants["heatmap_old_hardgate_v0p5sig_h2sig.png"] = (P_v12, "hard gate D_V<0.5σ, D_H<2σ")

    # Variant 13: hard gate D_V < 0.25σ, D_H < 2σ (extremely aggressive)
    P_v13, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v13 = apply_aggressive_v_gate(P_v13, D_V, D_H_gated, v_thresh=0.25, h_thresh=2.0)
    variants["heatmap_old_hardgate_v0p25sig_h2sig.png"] = (P_v13, "hard gate D_V<0.25σ, D_H<2σ")

    # Variant 14: soft gate V σ=1, H σ=2
    P_v14, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v14 = apply_aggressive_v_soft_gate(P_v14, D_V, D_H_gated, v_sigma=1.0, h_sigma=2.0)
    variants["heatmap_old_softgate_v1sig_h2sig.png"] = (P_v14, "soft gate V σ=1, H σ=2")

    # Variant 15: soft gate V σ=0.5, H σ=2 (very aggressive)
    P_v15, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v15 = apply_aggressive_v_soft_gate(P_v15, D_V, D_H_gated, v_sigma=0.5, h_sigma=2.0)
    variants["heatmap_old_softgate_v0p5sig_h2sig.png"] = (P_v15, "soft gate V σ=0.5, H σ=2")

    # Variant 16: soft gate V σ=0.3, H σ=2 (extremely aggressive)
    P_v16, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v16 = apply_aggressive_v_soft_gate(P_v16, D_V, D_H_gated, v_sigma=0.3, h_sigma=2.0)
    variants["heatmap_old_softgate_v0p3sig_h2sig.png"] = (P_v16, "soft gate V σ=0.3, H σ=2")

    # Variant 17: hard gate D_V < 1σ, D_H < 1σ (both strict)
    P_v17, _, _, _ = compute_probability_old(d, w_S=1.0, w_V=0.0, w_H=0.0)
    P_v17 = apply_aggressive_v_gate(P_v17, D_V, D_H_gated, v_thresh=1.0, h_thresh=1.0)
    variants["heatmap_old_hardgate_v1sig_h1sig.png"] = (P_v17, "hard gate D_V<1σ, D_H<1σ")

    # Save all variant heatmaps
    for filename, (P_v, desc) in variants.items():
        hm = build_heatmap_rgb(P_v)
        out_path = output_dir / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(hm, cv2.COLOR_RGB2BGR))
        print(f"  {desc}: P range [{P_v.min():.4f}, {P_v.max():.4f}] → {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()