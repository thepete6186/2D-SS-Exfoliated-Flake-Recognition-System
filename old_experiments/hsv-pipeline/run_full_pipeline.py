#!/usr/bin/env python3
"""Full pipeline for sample1, sample3, sample4: diagnostics + heatmap + masks."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# Diagnostics + Heatmap
# ===========================================================================

def load_rgb_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_base_diagnostics(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
    s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
    v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()
    H_sub = float(np.argmax(h_hist))
    S_sub = float(np.argmax(s_hist))
    V_sub = float(np.argmax(v_hist))
    sm = (np.abs(S - S_sub) < 15.0) & (np.abs(V - V_sub) < 15.0)
    if sm.any():
        sigH, sigS, sigV = (np.std(H[sm]), np.std(S[sm]), np.std(V[sm]))
    else:
        sigH, sigS, sigV = (np.std(H), np.std(S), np.std(V))
    sigH += 1e-5; sigS += 1e-5; sigV += 1e-5
    peaks, _ = find_peaks(s_hist)
    cand = peaks[peaks > S_sub + 2.0 * sigS]
    S_thin = float(cand[0]) if len(cand) > 0 else None
    return {"hsv": hsv, "H": H, "S": S, "V": V,
            "H_sub": H_sub, "S_sub": S_sub, "V_sub": V_sub,
            "sigma_H": sigH, "sigma_S": sigS, "sigma_V": sigV,
            "S_thin": S_thin,
            "h_hist": h_hist, "s_hist": s_hist, "v_hist": v_hist}


def compute_probability(d):
    S, V, H = d["S"], d["V"], d["H"]
    D_S = np.abs(S - d["S_sub"]) / d["sigma_S"]
    D_V = np.abs(V - d["V_sub"]) / d["sigma_V"]
    dh = np.abs(((H - d["H_sub"] + 90.0) % 180.0) - 90.0)
    D_H = dh / d["sigma_H"]
    S_gate = d["S_sub"] + (2.0 * d["sigma_S"])
    D_H_gated = D_H * (S > S_gate)

    # Base probability from S channel
    P = 1.0 - np.exp(-0.5 * D_S**2)

    # --- Shrink V and H gate boundaries by 5 pixels (morphological erosion) ---
    # Build binary gate masks at the 2-sigma threshold and erode them by 5px
    # so that S cannot "pour out" at the edges where H and V are more restricted.
    gate_V = (D_V < 2.0).astype(np.uint8)
    gate_H = (D_H_gated < 2.0).astype(np.uint8)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    gate_V = cv2.erode(gate_V, erode_kernel, iterations=1)
    gate_H = cv2.erode(gate_H, erode_kernel, iterations=1)

    # --- Heavy exponential penalties for V and H (H penalized 2x heavier) ---
    # penalty = 1.0 near substrate, -> 0.0 as channel distance grows.
    # k_V = 5.0 (heavy), k_H = 10.0 (2x heavier than V)
    k_V = 5.0
    k_H = 10.0
    penalty_V = np.exp(-0.5 * (D_V / 2.0)**2 * k_V)
    penalty_H = np.exp(-0.5 * (D_H_gated / 2.0)**2 * k_H)

    # Apply eroded hard gates + heavy soft penalties
    P = (P * gate_V * gate_H * penalty_V * penalty_H).astype(np.float32)
    return P, D_S, D_V, D_H_gated


def build_heatmap_rgb(P):
    p8 = (np.clip(P, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(p8, cv2.COLORMAP_INFERNO),
                        cv2.COLOR_BGR2RGB)


def plot_histograms(ax, d):
    for hist, color, label, sub in [
        (d["h_hist"] / d["h_hist"].max(), "red", "Hue", d["H_sub"]),
        (d["s_hist"] / d["s_hist"].max(), "green", "Saturation", d["S_sub"]),
        (d["v_hist"] / d["v_hist"].max(), "blue", "Value", d["V_sub"]),
    ]:
        ax.plot(hist, color=color, linewidth=0.8, label=label)
        ax.axvline(x=sub, color=color, linestyle="--", linewidth=1.5,
                   label=f"{label}_sub={sub:.0f}")
    if d["S_thin"] is not None:
        ax.axvline(x=d["S_thin"], color="yellow", linestyle="--",
                   linewidth=2.0, label=f"S_thin={d['S_thin']:.0f}")
    ax.set_title("Channel Histograms")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Normalized Count")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 255)


def build_diag_fig(rgb, d, D_S, D_V, D_H, hm, title):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"{title} — Sub H={d['H_sub']:.0f}, S={d['S_sub']:.0f}, V={d['V_sub']:.0f}\n"
        f"σ: H={d['sigma_H']:.2f}, S={d['sigma_S']:.2f}, V={d['sigma_V']:.2f} | "
        f"S_thin={d['S_thin'] if d['S_thin'] is not None else 'N/A'}",
        fontsize=12)
    ax[0, 0].imshow(rgb); ax[0, 0].set_title("Original"); ax[0, 0].axis("off")
    for i, (data, name) in enumerate([(D_S, "D_S"), (D_V, "D_V"),
                                      (D_H, "D_H_gated")], 1):
        im = ax[0 if i < 3 else 1, i % 3].imshow(data, cmap="viridis")
        ax[0 if i < 3 else 1, i % 3].set_title(name)
        ax[0 if i < 3 else 1, i % 3].axis("off")
        plt.colorbar(im, ax=ax[0 if i < 3 else 1, i % 3], fraction=0.046, pad=0.04)
    plot_histograms(ax[1, 1], d)
    ax[1, 2].imshow(hm); ax[1, 2].set_title("P_map heatmap"); ax[1, 2].axis("off")
    plt.tight_layout()
    return fig

# ===========================================================================
# Mask extraction
# ===========================================================================

def extract_mask(hm, v_th, s_min, h_lo, h_hi, open_r, close_r, min_area):
    hsv = cv2.cvtColor(hm, cv2.COLOR_RGB2HSV).astype(np.float32)
    m = ((hsv[..., 2] / 255.0 >= v_th) & (hsv[..., 1] / 255.0 >= s_min) &
         (hsv[..., 0] >= h_lo) & (hsv[..., 0] <= h_hi)).astype(np.uint8) * 255
    if open_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * open_r + 1, 2 * open_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_r > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * close_r + 1, 2 * close_r + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_area > 0:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        f = np.zeros_like(m)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area:
                f[lab == i] = 255
        m = f
    return m


def analyze(mask, min_area):
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        comps.append({"area": a,
                      "bbox": (int(st[i, 0]), int(st[i, 1]),
                               int(st[i, 2]), int(st[i, 3])),
                      "center": (float(cent[i][0]), float(cent[i][1]))})
    total = sum(c["area"] for c in comps)
    areas = [c["area"] for c in comps]
    return comps, total, areas


def save_overlay(hm, mask, comps, path):
    ov = hm.copy()
    mb = mask > 0
    ov[mb] = (0.5 * ov[mb].astype(np.float32) +
              0.5 * np.array([255, 0, 0], dtype=np.float32)).astype(np.uint8)
    for c in comps:
        x, y, w, h = c["bbox"]
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cx, cy = c["center"]
        cv2.circle(ov, (int(cx), int(cy)), 8, (255, 255, 0), -1)
    cv2.imwrite(str(path), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))


def write_stats(comps, total, areas, path, tech, img_area):
    with open(path, "w") as f:
        f.write("Flake Detection Statistics\n==========================\n\n")
        f.write(f"Technique: {tech['name']}\n")
        f.write(f"V>={tech['v_th']}, S>={tech['s_min']}, "
                f"H={tech['h_lo']}-{tech['h_hi']}\n")
        f.write(f"Min area: {tech['min_area']}px, "
                f"open={tech['open_r']}, close={tech['close_r']}\n\n")
        f.write(f"Total flake area: {total} px\n")
        f.write(f"Image area: {img_area} px\n")
        f.write(f"Coverage: {100.0 * total / img_area:.2f}%\n")
        f.write(f"Detected: {len(comps)} flakes\n\n")
        if areas:
            f.write(f"Areas: min={min(areas)}, max={max(areas)}, "
                    f"mean={np.mean(areas):.1f}, median={np.median(areas):.1f}\n\n")
        f.write(f"{'#':>3} {'Area':>10} {'W':>5} {'H':>5} "
                f"{'CenterX':>10} {'CenterY':>10}\n")
        f.write("-" * 50 + "\n")
        for i, c in enumerate(comps, 1):
            x, y, w, h = c["bbox"]
            cx, cy = c["center"]
            f.write(f"{i:3d} {c['area']:10d} {w:5d} {h:5d} "
                    f"{cx:10.1f} {cy:10.1f}\n")

# ===========================================================================
# Main
# ===========================================================================

TECHNIQUES = [
    {"name": "v055", "v_th": 0.55, "s_min": 0.15,
     "h_lo": 15.0, "h_hi": 80.0, "min_area": 50, "open_r": 2, "close_r": 4},
    {"name": "v070", "v_th": 0.70, "s_min": 0.20,
     "h_lo": 20.0, "h_hi": 70.0, "min_area": 50, "open_r": 2, "close_r": 4},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "dataset"))
    ap.add_argument("--output-root", type=str,
                    default=str(Path(__file__).resolve().parent /
                                "results" / "full_pipeline"))
    ap.add_argument("--samples", type=str, default="sample1,sample3,sample4")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="Only generate diagnostic maps + heatmaps, skip mask extraction")
    args = ap.parse_args()

    sd = Path(args.samples_dir)
    out_root = Path(args.output_root)
    samples = [s.strip() for s in args.samples.split(",")]

    for sname in samples:
        sdir = sd / sname
        imgs = sorted(sdir.glob("*.jpg")) if sdir.is_dir() else []
        if not imgs:
            print(f"SKIP: {sdir}")
            continue
        print(f"\n{'='*60}\nSample: {sname} ({len(imgs)} images)\n{'='*60}")

        for ip in imgs:
            iname = ip.stem
            od = out_root / sname / iname
            od.mkdir(parents=True, exist_ok=True)

            print(f"\n  {ip.name}")
            rgb = load_rgb_image(ip)
            h, w = rgb.shape[:2]

            d = compute_base_diagnostics(rgb)
            print(f"    Sub: H={d['H_sub']:.0f} S={d['S_sub']:.0f} "
                  f"V={d['V_sub']:.0f} | σ: H={d['sigma_H']:.2f} "
                  f"S={d['sigma_S']:.2f} V={d['sigma_V']:.2f}")

            P, D_S, D_V, D_H = compute_probability(d)
            hm = build_heatmap_rgb(P)
            cv2.imwrite(str(od / "heatmap_old_hardgate_2sig.png"),
                        cv2.cvtColor(hm, cv2.COLOR_RGB2BGR))

            fig = build_diag_fig(rgb, d, D_S, D_V, D_H, hm,
                                 f"{sname}/{iname}")
            fig.savefig(str(od / "diagnostics_old.png"), dpi=150,
                        bbox_inches="tight")
            plt.close(fig)

            if args.diagnostics_only:
                print("    (skipping mask extraction --diagnostics-only)")
            else:
                for tech in TECHNIQUES:
                    td = od / "masks" / tech["name"]
                    td.mkdir(parents=True, exist_ok=True)
                    mask = extract_mask(hm, tech["v_th"], tech["s_min"],
                                        tech["h_lo"], tech["h_hi"],
                                        tech["open_r"], tech["close_r"],
                                        tech["min_area"])
                    comps, total, areas = analyze(mask, tech["min_area"])
                    cv2.imwrite(str(td / "flake_mask.png"), mask)
                    save_overlay(hm, mask, comps, td / "flake_overlay.png")
                    write_stats(comps, total, areas, td / "flake_stats.txt",
                                tech, h * w)
                    print(f"    [{tech['name']}] {len(comps)} flakes, "
                          f"total={total}px ({100.0 * total / (h * w):.2f}%)")

    print("\n\nDone.")


if __name__ == "__main__":
    main()