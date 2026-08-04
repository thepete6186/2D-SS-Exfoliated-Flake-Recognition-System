#!/usr/bin/env python3
"""
run_semi_supervised.py

Interactive semi-supervised flake detection.

Usage (interactive — click on flakes in the image window):
    python run_semi_supervised.py --image ../dataset/sample1/tmd_sample_0.jpg

Usage (CLI — provide clicks directly):
    python run_semi_supervised.py --image ../dataset/sample3/ws2-251104161326950.jpg \
        --clicks 150,200 300,400

Usage (batch — run on all images in a sample folder with pre-defined clicks):
    python run_semi_supervised.py --sample sample1 --batch

Workflow:
  1. The image is displayed in a matplotlib window.
  2. Click on one or more flake regions (the more clicks, the better the signature).
  3. Close the window (or press Enter) to proceed.
  4. The pipeline extracts the flake HSV signature and generates:
     - A diagnostic figure (deviation maps + histograms + heatmap)
     - A standalone probability heatmap PNG
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from semi_supervised_pipeline import SemiSupervisedPipeline


# ===========================================================================
# Image loading
# ===========================================================================

def load_rgb_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===========================================================================
# Interactive click collection
# ===========================================================================

def get_clicks_interactive(rgb: np.ndarray, title: str = "") -> list:
    """
    Display the image and collect click coordinates from the user.

    Returns a list of (x, y) tuples.
    """
    # Try to use an interactive backend
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        try:
            plt.switch_backend("Qt5Agg")
        except Exception:
            pass  # Fall back to whatever is available

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(rgb)
    ax.set_title(f"{title}\nClick on flake regions, then close window or press Enter")
    ax.axis("off")

    clicks = []

    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            clicks.append((x, y))
            ax.plot(x, y, "r+", markersize=15, markeredgewidth=2)
            ax.annotate(f"#{len(clicks)}", (x, y), textcoords="offset points",
                        xytext=(10, 10), color="red", fontsize=10, fontweight="bold")
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "enter":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    plt.show()
    return clicks


def parse_clicks_arg(clicks_str: str) -> list:
    """Parse a string like '150,200 300,400' into [(150, 200), (300, 400)]."""
    points = []
    for pair in clicks_str.split():
        parts = pair.split(",")
        if len(parts) == 2:
            points.append((int(parts[0]), int(parts[1])))
    return points


# ===========================================================================
# Diagnostic figure
# ===========================================================================

def build_diagnostic_figure(
    rgb: np.ndarray,
    result: dict,
    seed_points: list,
    image_name: str,
) -> plt.Figure:
    """
    Build a 2x3 diagnostic figure:
      1. Original with seed clicks marked
      2. ΔH (signed hue deviation)
      3. ΔS (signed saturation deviation)
      4. ΔV (signed value deviation)
      5. Channel histograms with substrate + flake signature markers
      6. Final probability heatmap
    """
    dH = result["delta_H"]
    dS = result["delta_S"]
    dV = result["delta_V"]
    sub_peak = result["substrate_peak"]
    sub_std = result["substrate_std"]
    flake_sig = result["flake_signature"]
    heatmap = result["heatmap_rgb"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Semi-Supervised Diagnostics — {image_name}\n"
        f"Substrate: H={sub_peak[0]:.0f}, S={sub_peak[1]:.0f}, V={sub_peak[2]:.0f} | "
        f"σ: H={sub_std[0]:.2f}, S={sub_std[1]:.2f}, V={sub_std[2]:.2f}\n"
        f"Flake sig: ΔH={flake_sig['delta_H_ref']:+.1f}, "
        f"ΔS={flake_sig['delta_S_ref']:+.1f}, "
        f"ΔV={flake_sig['delta_V_ref']:+.1f} | "
        f"σ_flake: H={flake_sig['std_H']:.2f}, S={flake_sig['std_S']:.2f}, "
        f"V={flake_sig['std_V']:.2f} | seeds={len(seed_points)}",
        fontsize=11,
    )

    # 1. Original with seed clicks
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("1. Original + Seed Clicks")
    axes[0, 0].axis("off")
    for i, (x, y) in enumerate(seed_points):
        axes[0, 0].plot(x, y, "r+", markersize=15, markeredgewidth=2)
        axes[0, 0].annotate(f"#{i+1}", (x, y), textcoords="offset points",
                            xytext=(10, 10), color="red", fontsize=9,
                            fontweight="bold")

    # 2. ΔH
    im_h = axes[0, 1].imshow(dH, cmap="RdBu_r", vmin=-30, vmax=30)
    axes[0, 1].set_title("2. ΔH (signed hue deviation)")
    axes[0, 1].axis("off")
    plt.colorbar(im_h, ax=axes[0, 1], fraction=0.046, pad=0.04)
    # Mark reference deviation
    axes[0, 1].plot([x for x, y in seed_points], [y for x, y in seed_points],
                    "g+", markersize=10, markeredgewidth=1.5)

    # 3. ΔS
    im_s = axes[0, 2].imshow(dS, cmap="RdBu_r", vmin=-80, vmax=80)
    axes[0, 2].set_title("3. ΔS (signed saturation deviation)")
    axes[0, 2].axis("off")
    plt.colorbar(im_s, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].plot([x for x, y in seed_points], [y for x, y in seed_points],
                    "g+", markersize=10, markeredgewidth=1.5)

    # 4. ΔV
    im_v = axes[1, 0].imshow(dV, cmap="RdBu_r", vmin=-80, vmax=80)
    axes[1, 0].set_title("4. ΔV (signed value deviation)")
    axes[1, 0].axis("off")
    plt.colorbar(im_v, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].plot([x for x, y in seed_points], [y for x, y in seed_points],
                    "g+", markersize=10, markeredgewidth=1.5)

    # 5. Histograms with substrate + flake markers
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

    ax = axes[1, 1]
    ax.plot(h_hist / h_hist.max() if h_hist.max() > 0 else h_hist,
            color="red", linewidth=0.8, label="Hue")
    ax.plot(s_hist / s_hist.max() if s_hist.max() > 0 else s_hist,
            color="green", linewidth=0.8, label="Saturation")
    ax.plot(v_hist / v_hist.max() if v_hist.max() > 0 else v_hist,
            color="blue", linewidth=0.8, label="Value")

    # Substrate peaks
    H_sub, S_sub, V_sub = sub_peak
    ax.axvline(x=H_sub, color="red", linestyle="--", linewidth=1.5,
               label=f"H_sub={H_sub:.0f}")
    ax.axvline(x=S_sub, color="green", linestyle="--", linewidth=1.5,
               label=f"S_sub={S_sub:.0f}")
    ax.axvline(x=V_sub, color="blue", linestyle="--", linewidth=1.5,
               label=f"V_sub={V_sub:.0f}")

    # Flake signature means
    ax.axvline(x=flake_sig["mean_H"], color="red", linestyle="-", linewidth=2.0,
               alpha=0.7, label=f"H_flake={flake_sig['mean_H']:.0f}")
    ax.axvline(x=flake_sig["mean_S"], color="green", linestyle="-", linewidth=2.0,
               alpha=0.7, label=f"S_flake={flake_sig['mean_S']:.0f}")
    ax.axvline(x=flake_sig["mean_V"], color="blue", linestyle="-", linewidth=2.0,
               alpha=0.7, label=f"V_flake={flake_sig['mean_V']:.0f}")

    ax.set_title("5. Histograms (substrate dashed, flake solid)")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized Count")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlim(0, 255)

    # 6. Final heatmap
    axes[1, 2].imshow(heatmap)
    axes[1, 2].set_title("6. Probability Heatmap")
    axes[1, 2].axis("off")
    for i, (x, y) in enumerate(seed_points):
        axes[1, 2].plot(x, y, "w+", markersize=12, markeredgewidth=1.5)

    plt.tight_layout()
    return fig


# ===========================================================================
# Batch mode: pre-defined clicks per sample
# ===========================================================================

# Pre-defined seed clicks for testing (x, y coordinates in the image)
# These are rough estimates — adjust after seeing the images
# Image sizes: sample1/tmd_sample_0=2736x1824, sample1/tmd_sample_1,2=5472x3648
#              sample3=2736x1824, sample4 mixed sizes
BATCH_CLICKS = {
    "sample1": {
        "tmd_sample_0": [(1200, 800), (1600, 1000)],
        "tmd_sample_1": [(2400, 1600), (3200, 2000)],
        "tmd_sample_2": [(2200, 1400), (3000, 1800)],
    },
    "sample3": {
        "ws2-251104161326950": [(1200, 800), (1600, 1000)],
        "ws2-251104162248915": [(1200, 800), (1600, 1000)],
    },
    "sample4": {
        "ws2-251104161126416": [(2400, 1600), (3200, 2000)],
        "ws2-251104162101190": [(1200, 800), (1600, 1000)],
        "ws2-251104163730007": [(1200, 800), (1600, 1000)],
        "ws2-251104163957775": [(1200, 800), (1600, 1000)],
    },
}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Semi-supervised HSV flake detection."
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image to process.",
    )
    parser.add_argument(
        "--sample", type=str, default=None,
        help="Sample folder name (e.g. sample1, sample3, sample4).",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: use pre-defined clicks for all images in the sample.",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Calibrate mode: click on first image in sample, apply signature to all.",
    )
    parser.add_argument(
        "--calibration-image", type=str, default=None,
        help="Specific image filename to use for calibration (e.g., tmd_sample_3.jpg).",
    )
    parser.add_argument(
        "--clicks", type=str, default=None,
        help="Pre-defined clicks as 'x1,y1 x2,y2 ...' (non-interactive mode).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--patch-radius", type=int, default=7,
        help="Patch radius around each click (default: 7).",
    )
    parser.add_argument(
        "--w-h", type=float, default=1.5,
        help="Hue channel weight (default: 1.5, heavier than S/V).",
    )
    parser.add_argument(
        "--w-s", type=float, default=1.0,
        help="Saturation channel weight (default: 1.0).",
    )
    parser.add_argument(
        "--w-v", type=float, default=1.0,
        help="Value channel weight (default: 1.0).",
    )
    args = parser.parse_args()

    pipeline = SemiSupervisedPipeline(
        patch_radius=args.patch_radius,
        w_H=args.w_h,
        w_S=args.w_s,
        w_V=args.w_v,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Single image mode ---
    if args.image and not args.batch:
        image_path = Path(args.image)
        rgb = load_rgb_image(image_path)

        if args.clicks:
            clicks = parse_clicks_arg(args.clicks)
        else:
            clicks = get_clicks_interactive(rgb, image_path.name)

        if not clicks:
            print("No clicks provided. Exiting.")
            return

        print(f"\nProcessing: {image_path.name}")
        print(f"  Clicks: {clicks}")
        result = pipeline.process(rgb, clicks)

        sub = result["substrate_peak"]
        sig = result["flake_signature"]
        prob = result["probability_map"]
        print(f"  Substrate: H={sub[0]:.0f}, S={sub[1]:.0f}, V={sub[2]:.0f}")
        print(f"  Flake sig: ΔH={sig['delta_H_ref']:+.1f}, "
              f"ΔS={sig['delta_S_ref']:+.1f}, ΔV={sig['delta_V_ref']:+.1f}")
        print(f"  Probability range: [{prob.min():.4f}, {prob.max():.4f}]")

        # Save outputs
        stem = image_path.stem
        out = output_dir / stem
        out.mkdir(parents=True, exist_ok=True)

        # Heatmap
        hm = result["heatmap_rgb"]
        cv2.imwrite(str(out / "heatmap_semi.png"),
                    cv2.cvtColor(hm, cv2.COLOR_RGB2BGR))

        # Diagnostic figure
        matplotlib.use("Agg")
        fig = build_diagnostic_figure(rgb, result, clicks, image_path.name)
        fig.savefig(str(out / "diagnostics_semi.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save clicks + signature as JSON
        meta = {
            "image": str(image_path),
            "clicks": clicks,
            "substrate_peak": list(sub),
            "substrate_std": list(result["substrate_std"]),
            "flake_signature": {k: v for k, v in sig.items()},
            "probability_range": [float(prob.min()), float(prob.max())],
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved: {out}/")
        print(f"    - heatmap_semi.png")
        print(f"    - diagnostics_semi.png")
        print(f"    - metadata.json")

    # --- Calibrate mode ---
    elif args.sample and args.calibrate:
        sample_name = args.sample
        dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
        sample_dir = dataset_dir / sample_name

        if not sample_dir.is_dir():
            print(f"Sample directory not found: {sample_dir}")
            return

        images = sorted(sample_dir.glob("*.jpg"))
        if not images:
            print(f"No images in {sample_dir}")
            return

        print(f"\n{'='*60}")
        print(f"Calibrate mode: {sample_name} ({len(images)} images)")
        print(f"{'='*60}")

        # Step 1: Choose calibration image
        if args.calibration_image:
            # Use specified image
            cal_img_path = None
            for img in images:
                if img.name == args.calibration_image:
                    cal_img_path = img
                    break
            if cal_img_path is None:
                print(f"ERROR: Calibration image '{args.calibration_image}' not found in {sample_dir}")
                print(f"Available images: {[img.name for img in images]}")
                return
            print(f"\n  Using specified calibration image: {cal_img_path.name}")
        else:
            # Use first image
            cal_img_path = images[0]
            print(f"\n  Calibration image (first): {cal_img_path.name}")
        cal_rgb = load_rgb_image(cal_img_path)
        clicks = get_clicks_interactive(cal_rgb, f"CALIBRATION — {sample_name}/{cal_img_path.name}")

        if not clicks:
            print("No clicks provided. Exiting.")
            return

        # Step 2: Run full pipeline on calibration image to get flake signature
        print(f"\n  Clicks: {clicks}")
        cal_result = pipeline.process(cal_rgb, clicks)
        flake_sig = cal_result["flake_signature"]

        cal_sub = cal_result["substrate_peak"]
        print(f"  Calibration substrate: H={cal_sub[0]:.0f}, S={cal_sub[1]:.0f}, V={cal_sub[2]:.0f}")
        print(f"  Flake signature: ΔH={flake_sig['delta_H_ref']:+.1f}, "
              f"ΔS={flake_sig['delta_S_ref']:+.1f}, ΔV={flake_sig['delta_V_ref']:+.1f}")
        print(f"  Flake σ: H={flake_sig['std_H']:.2f}, S={flake_sig['std_S']:.2f}, "
              f"V={flake_sig['std_V']:.2f}")

        # Save calibration image results
        cal_out = output_dir / sample_name / "calibration"
        cal_out.mkdir(parents=True, exist_ok=True)
        cal_hm = cal_result["heatmap_rgb"]
        cv2.imwrite(str(cal_out / "heatmap_semi.png"),
                    cv2.cvtColor(cal_hm, cv2.COLOR_RGB2BGR))

        matplotlib.use("Agg")
        cal_fig = build_diagnostic_figure(cal_rgb, cal_result, clicks,
                                          f"CALIBRATION {sample_name}/{cal_img_path.name}")
        cal_fig.savefig(str(cal_out / "diagnostics_semi.png"), dpi=150,
                        bbox_inches="tight")
        plt.close(cal_fig)

        cal_meta = {
            "image": str(cal_img_path),
            "clicks": clicks,
            "substrate_peak": list(cal_sub),
            "substrate_std": list(cal_result["substrate_std"]),
            "flake_signature": {k: v for k, v in flake_sig.items()},
            "probability_range": [float(cal_result["probability_map"].min()),
                                  float(cal_result["probability_map"].max())],
        }
        with open(cal_out / "metadata.json", "w") as f:
            json.dump(cal_meta, f, indent=2)
        print(f"  Saved calibration: {cal_out}/")

        # Step 3: Apply signature to all images in the sample
        print(f"\n  Applying signature to {len(images)} images...")

        for img_path in images:
            stem = img_path.stem
            print(f"\n    {img_path.name}")
            rgb = load_rgb_image(img_path)

            # Use process_with_signature for all images (including calibration image
            # for consistency, but we already saved it above with click markers)
            if img_path == cal_img_path:
                # Already processed — just note it
                print(f"      (calibration image — already saved)")
                continue

            result = pipeline.process_with_signature(rgb, flake_sig)

            sub = result["substrate_peak"]
            prob = result["probability_map"]
            print(f"      Substrate: H={sub[0]:.0f}, S={sub[1]:.0f}, V={sub[2]:.0f}")
            print(f"      Prob range: [{prob.min():.4f}, {prob.max():.4f}]")

            out = output_dir / sample_name / stem
            out.mkdir(parents=True, exist_ok=True)

            hm = result["heatmap_rgb"]
            cv2.imwrite(str(out / "heatmap_semi.png"),
                        cv2.cvtColor(hm, cv2.COLOR_RGB2BGR))

            # For non-calibration images, no seed clicks to show
            fig = build_diagnostic_figure(rgb, result, [], img_path.name)
            fig.savefig(str(out / "diagnostics_semi.png"), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

            meta = {
                "image": str(img_path),
                "calibration_image": str(cal_img_path),
                "calibration_clicks": clicks,
                "substrate_peak": list(sub),
                "substrate_std": list(result["substrate_std"]),
                "flake_signature": {k: v for k, v in flake_sig.items()},
                "probability_range": [float(prob.min()), float(prob.max())],
            }
            with open(out / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"      Saved: {out}/")

        print(f"\n{'='*60}\nCalibrate done.\n{'='*60}")

    # --- Batch mode ---
    elif args.sample and args.batch:
        sample_name = args.sample
        dataset_dir = Path(__file__).resolve().parent.parent / "dataset"
        sample_dir = dataset_dir / sample_name
        clicks_map = BATCH_CLICKS.get(sample_name, {})

        if not sample_dir.is_dir():
            print(f"Sample directory not found: {sample_dir}")
            return

        images = sorted(sample_dir.glob("*.jpg"))
        if not images:
            print(f"No images in {sample_dir}")
            return

        print(f"\n{'='*60}")
        print(f"Batch mode: {sample_name} ({len(images)} images)")
        print(f"{'='*60}")

        matplotlib.use("Agg")

        for img_path in images:
            stem = img_path.stem
            clicks = clicks_map.get(stem, [])

            if not clicks:
                print(f"\n  SKIP {img_path.name} — no pre-defined clicks")
                continue

            print(f"\n  {img_path.name}")
            print(f"    Clicks: {clicks}")
            rgb = load_rgb_image(img_path)
            result = pipeline.process(rgb, clicks)

            sub = result["substrate_peak"]
            sig = result["flake_signature"]
            prob = result["probability_map"]
            print(f"    Substrate: H={sub[0]:.0f}, S={sub[1]:.0f}, V={sub[2]:.0f}")
            print(f"    Flake sig: ΔH={sig['delta_H_ref']:+.1f}, "
                  f"ΔS={sig['delta_S_ref']:+.1f}, ΔV={sig['delta_V_ref']:+.1f}")
            print(f"    Prob range: [{prob.min():.4f}, {prob.max():.4f}]")

            out = output_dir / sample_name / stem
            out.mkdir(parents=True, exist_ok=True)

            hm = result["heatmap_rgb"]
            cv2.imwrite(str(out / "heatmap_semi.png"),
                        cv2.cvtColor(hm, cv2.COLOR_RGB2BGR))

            fig = build_diagnostic_figure(rgb, result, clicks, img_path.name)
            fig.savefig(str(out / "diagnostics_semi.png"), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

            meta = {
                "image": str(img_path),
                "clicks": clicks,
                "substrate_peak": list(sub),
                "substrate_std": list(result["substrate_std"]),
                "flake_signature": {k: v for k, v in sig.items()},
                "probability_range": [float(prob.min()), float(prob.max())],
            }
            with open(out / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            print(f"    Saved: {out}/")

        print(f"\n{'='*60}\nBatch done.\n{'='*60}")

    else:
        parser.print_help()
        print("\nExamples:")
        print("  Interactive:  python run_semi_supervised.py --image ../dataset/sample1/tmd_sample_0.jpg")
        print("  CLI clicks:   python run_semi_supervised.py --image ../dataset/sample3/ws2-251104161326950.jpg --clicks 200,300 400,200")
        print("  Batch:        python run_semi_supervised.py --sample sample1 --batch")


if __name__ == "__main__":
    main()