#!/usr/bin/env python3
"""
run_sample2_minimal.py

Process ALL sample2 images, generating only 3 outputs per sample into
results/sample2heat/:
  1. diagnostics_old.png            - full 2x3 diagnostic using OLD (main peak) logic
  2. heatmap_old_sonly.png          - S-only probability heatmap (baseline)
  3. heatmap_old_hardgate_2sig.png  - S probability with 2σ hard gate on V and H
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

    return {
        "H": H, "S": S, "V": V,
        "H_sub": H_sub, "S_sub": S_sub, "V_sub": V_sub,
        "sigma_H": sigma_H, "sigma_S": sigma_S, "sigma_V": sigma_V,
        "h_hist": h_hist, "s_hist": s_hist, "v_hist": v_hist,
    }


def compute_probability_old(d: dict, gate: bool = False) -> np.ndarray:
    """OLD: linear D_S from S_sub; option 2σ hard gate on V/H."""
    S = d["S"]
    D_S = np.abs(S - d["S_sub"]) / d["sigma_S"]

    V = d["V"]
    D_V = np.abs(V - d["V_sub"]) / d["sigma_V"]

    H = d["H"]
    delta_H = np.abs(((H - d["H_sub"] + 90.0) % 180.0) - 90.0)
    D_H = delta_H / d["sigma_H"]
    S_gate = d["S_sub"] + (2.0 * d["sigma_S"])
    D_H_gated = D_H * (S > S_gate)

    D_composite = 1.0 * D_S
    P = 1.0 - np.exp(-0.5 * D_composite**2)

    if gate:
        mask = (D_V < 2.0) & (D_H_gated < 2.0)
        P = (P * mask).astype(np.float32)

    return P.astype(np.float32), D_S, D_V, D_H_gated


def build_heatmap_rgb(prob_map: np.ndarray) -> np.ndarray:
    prob_uint8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def plot_histograms(ax, d: dict) -> None:
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

    ax.set_title("Channel Histograms with Substrate Peaks")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized Count")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 255)


def build_diagnostic_figure(rgb, d, D_S, D_V, D_H_gated, heatmap_rgb, title) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{title} — Substrate H={d['H_sub']:.0f}, S={d['S_sub']:.0f}, V={d['V_sub']:.0f}\n"
        f"σ: H={d['sigma_H']:.2f}, S={d['sigma_S']:.2f}, V={d['sigma_V']:.2f}",
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

    plot_histograms(axes[1, 1], d)

    axes[1, 2].imshow(heatmap_rgb)
    axes[1, 2].set_title("6. Probability Heatmap (P_map)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "dataset" / "sample2"),
        help="Directory containing sample2 images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results" / "sample2heat"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted(samples_dir.glob("tmd_sample_*.jpg"))
    print(f"Found {len(images)} images in {samples_dir}")

    for img_path in images:
        sample_name = img_path.stem
        out_dir = output_root / sample_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing: {img_path.name}")
        print(f"Output: {out_dir}")
        print(f"{'='*60}")

        rgb = load_rgb_image(img_path)
        d = compute_base_diagnostics(rgb)
        print(f"  Substrate peak (H, S, V): ({d['H_sub']:.0f}, {d['S_sub']:.0f}, {d['V_sub']:.0f})")
        print(f"  Substrate std  (H, S, V): ({d['sigma_H']:.3f}, {d['sigma_S']:.3f}, {d['sigma_V']:.3f})")

        # S-only probability
        P_sonly, D_S, D_V, D_H_gated = compute_probability_old(d, gate=False)
        hm_sonly = build_heatmap_rgb(P_sonly)
        print(f"  S-only P range: [{P_sonly.min():.4f}, {P_sonly.max():.4f}]")

        # Save S-only heatmap
        cv2.imwrite(str(out_dir / "heatmap_old_sonly.png"),
                    cv2.cvtColor(hm_sonly, cv2.COLOR_RGB2BGR))

        # 2σ hard gate heatmap
        P_gate, _, _, _ = compute_probability_old(d, gate=True)
        hm_gate = build_heatmap_rgb(P_gate)
        print(f"  2σ gate P range: [{P_gate.min():.4f}, {P_gate.max():.4f}]")

        # Save 2σ gate heatmap
        cv2.imwrite(str(out_dir / "heatmap_old_hardgate_2sig.png"),
                    cv2.cvtColor(hm_gate, cv2.COLOR_RGB2BGR))

        # Save diagnostic figure (using S-only heatmap as the reference)
        fig = build_diagnostic_figure(rgb, d, D_S, D_V, D_H_gated, hm_sonly,
                                      f"OLD: 1st Main Peak (S_only)")
        fig.savefig(str(out_dir / "diagnostics_old.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved 3 files to {out_dir}")

    print("\n\nAll samples processed.")


if __name__ == "__main__":
    main()