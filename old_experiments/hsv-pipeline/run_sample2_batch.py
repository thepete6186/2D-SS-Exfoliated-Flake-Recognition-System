#!/usr/bin/env python3
"""
run_sample2_batch.py

Run the aggressive variants pipeline on ALL sample2 images (tmd_sample_3 through
tmd_sample_20), saving diagnostics and heatmaps to results/sample2heat/.

For each image, generates:
  <output>/sample2heat/tmd_sample_XX/diagnostics_old.png
  <output>/sample2heat/tmd_sample_XX/diagnostics_new.png
  <output>/sample2heat/tmd_sample_XX/heatmap_old_*.png (all variants)
"""

import subprocess
import sys
from pathlib import Path

# All sample2 images
sample2_dir = Path(__file__).resolve().parent.parent / "dataset" / "sample2"
output_root = Path(__file__).resolve().parent / "results" / "sample2heat"

# Get all tmd_sample_*.jpg files
images = sorted(sample2_dir.glob("tmd_sample_*.jpg"))
print(f"Found {len(images)} images in sample2")

for img_path in images:
    sample_name = img_path.stem  # e.g., tmd_sample_3
    out_dir = output_root / sample_name
    print(f"\n{'='*60}")
    print(f"Processing: {img_path.name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_aggressive_variants.py"),
        "--image", str(img_path),
        "--output-dir", str(out_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)

print("\n\nAll sample2 images processed.")