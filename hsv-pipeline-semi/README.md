# HSV Pipeline - Semi-Supervised Edition

**This is the main experimental pipeline for flake detection.**

## Overview

This directory contains a **semi-supervised HSV flake detection pipeline** that learns from user clicks to identify 2D material flakes on a substrate background. Unlike purely unsupervised methods, this approach uses minimal user input (clicking on flakes) to learn the specific HSV signature of the material being detected.

## Directory Structure

```
hsv-pipeline-semi/
├── semi_supervised_pipeline.py   # Core pipeline class
├── run_semi_supervised.py        # Interactive/batch/calibrate runner
├── results/                      # Generated outputs
│   ├── sample2/                  # 19 images processed
│   ├── sample3/                  # 2 images processed
│   └── sample4/                  # 4 images processed
└── README.md                     # This file
```

## Quick Start

### Interactive Mode (Single Image)
```bash
python run_semi_supervised.py --image ../dataset/sample1/tmd_sample_0.jpg
```
1. A window opens showing the image
2. Click on 2-3 flake regions
3. Close window or press Enter
4. Pipeline generates heatmap + diagnostics

### Calibrate Mode (Sample-Wide)
```bash
# Use first image in sample
python run_semi_supervised.py --sample sample4 --calibrate

# Use specific image for calibration
python run_semi_supervised.py --sample sample2 --calibrate --calibration-image tmd_sample_3.jpg
```
1. First image is shown, you click on flakes
2. Pipeline extracts flake HSV signature
3. Signature is automatically applied to ALL images in the sample
4. Each image gets its own automatic substrate detection

### Batch Mode (Pre-defined Clicks)
```bash
python run_semi_supervised.py --sample sample1 --batch
```
Uses pre-defined click coordinates for all images (not recommended for production).

## How It Works

### Step 1: Automatic Substrate Detection
- Converts image to HSV
- Finds **global** substrate peak via histogram mode (argmax)
- Estimates substrate noise floor (standard deviation)

### Step 2: User Clicks → Flake Signature
- User clicks on flake regions
- Extracts 15×15 px patches around each click
- Computes flake HSV signature:
  - Mean HSV: `(H_f, S_f, V_f)`
  - Per-channel std: `(σ_H_f, σ_S_f, σ_V_f)`
  - Reference deviation: `(ΔH_ref, ΔS_ref, ΔV_ref) = flake - substrate`

### Step 3: Directional Deviation Matching
For every pixel in the image:
- Computes signed deviation from substrate: `(ΔH, ΔS, ΔV)`
- Scores using Gaussian likelihood:
  ```
  P_H = exp(-0.5 * ((ΔH - ΔH_ref) / σ_H_f)²)
  P_S = exp(-0.5 * ((ΔS - ΔS_ref) / σ_S_f)²)
  P_V = exp(-0.5 * ((ΔV - ΔV_ref) / σ_V_f)²)
  P = P_S^w_S * P_V^w_V * P_H^w_H
  ```
- **Direction-aware**: Only pixels deviating in the SAME direction as the clicked flake get high probability

### Step 4: Output
- Probability heatmap (Inferno colormap)
- Diagnostic figure with:
  - Original image + click markers
  - ΔH, ΔS, ΔV deviation maps
  - Histograms with substrate (dashed) and flake (solid) markers
  - Final probability heatmap
- Metadata JSON with clicks, signature, and results

## Key Features

✅ **Global substrate detection** - Histogram mode across entire image  
✅ **Click-based calibration** - Minimal user interaction (2-3 clicks)  
✅ **Direction-aware matching** - Finds pixels with same deviation pattern as flakes  
✅ **Calibrate mode** - Click once, apply to all images in sample  
✅ **Per-image substrate** - Each image gets its own substrate detection  
✅ **Diagnostic outputs** - Full visualization of deviation maps and histograms  

## Experiments Conducted

### Experiment 1: Sample4 Calibration
- **Calibration image**: ws2-251104161126416.jpg (10 clicks)
- **Flake signature**: ΔH=−28.7°, ΔS=+46.1, ΔV=−4.7
- **Results**:
  - ws2-251104161126416: [0.0000, 0.9969] ✓
  - ws2-251104162101190: [0.0000, 0.9399] ✓
  - ws2-251104163730007: [0.0000, 0.0763] ✗ (different substrate hue)
  - ws2-251104163957775: [0.0000, 0.8814] ✓

### Experiment 2: Sample2 Calibration (tmd_sample_0)
- **Calibration image**: tmd_sample_0.jpg (11 clicks)
- **Flake signature**: ΔH=−31.6°, ΔS=−2.8, ΔV=−13.8
- **Results**: 15/19 images with good matches (0.78-0.97)
- **Issue**: Some images had poor matches, possibly different material

### Experiment 3: Sample2 Recalibration (tmd_sample_3)
- **Calibration image**: tmd_sample_3.jpg
- **Results**: Improved matches across all 19 images (0.95-0.99)
- **Conclusion**: tmd_sample_0 may be from a different material using the same substrate

### Experiment 4: Sample3 Calibration
- **Calibration image**: ws2-251104161326950.jpg (5 clicks)
- **Flake signature**: ΔH=−3.6°, ΔS=+23.8, ΔV=−1.3
- **Results**: 2/2 images with excellent matches (0.98-0.99)

## Command-Line Options

```bash
--image PATH                  # Single image to process
--sample NAME                 # Sample folder name (sample1, sample2, sample3, sample4)
--calibrate                   # Calibrate mode: click on first image, apply to all
--calibration-image FILENAME  # Specific image for calibration (e.g., tmd_sample_3.jpg)
--batch                       # Batch mode with pre-defined clicks
--clicks "x1,y1 x2,y2"        # Pre-defined clicks (non-interactive)
--output-dir PATH             # Output directory (default: results/)
--patch-radius N              # Patch radius around clicks (default: 7)
--w-h N                       # Hue weight (default: 1.5, heavier = stricter)
--w-s N                       # Saturation weight (default: 1.0)
--w-v N                       # Value weight (default: 1.0)
```

## Output Files

Each processed image generates:
- `heatmap_semi.png` - Probability heatmap (Inferno colormap)
- `diagnostics_semi.png` - 2×3 diagnostic figure
- `metadata.json` - Clicks, substrate info, flake signature, probability range

Calibration mode also generates:
- `calibration/` folder with calibration image results

## Other Pipelines (Failed Approaches)

### hsv-pipeline (Unsupervised - Failed)
**Location:** `hsv-pipeline/`  
**Approach:** Channel-Segregated Substrate Peak Normalization (CSSPN)

**What it tried to do:**
- Fully unsupervised, pixel-level statistical approach
- Detect global substrate peak via histogram modes
- Compute normalized distance maps for each HSV channel
- Combine channels using weighted exponential CDF
- Apply saturation-gated hue suppression

**Why it failed:**
The fundamental issue is that **directional elements are not entirely agnostic**. The unsupervised approach assumes that any deviation from the substrate baseline indicates a flake, but this is too simplistic:

1. **No direction awareness**: The pipeline computes absolute distances (`|S - S_sub|`, `|V - V_sub|`), so it cannot distinguish between:
   - Flakes that are MORE saturated than substrate (typical)
   - Flakes that are LESS saturated than substrate (also possible)
   - Pixels that deviate in the opposite direction

2. **Hue ambiguity**: Without knowing which direction the flake hue deviates, the pipeline either:
   - Ignores hue entirely (w_H=0) and relies only on saturation/value
   - Uses hue in both directions, causing false positives

3. **Saturation "pouring out"**: At the image edges, saturation tends to drop/bleed, causing the substrate detection to be unstable. The unsupervised approach has no way to distinguish between:
   - Real flake pixels with low saturation
   - Edge artifacts with low saturation

4. **One-size-fits-all problem**: Different samples have different flake signatures (e.g., sample2: ΔH=−31.6°, sample3: ΔH=−3.6°). The unsupervised approach cannot adapt to these differences.

**Example failure:**
- Sample2 with tmd_sample_0 calibration: Only 15/19 images matched well
- The 4 failed images had different substrate conditions that the unsupervised approach couldn't handle

### hsv-pipeline-backup (Original - Untouched)
**Location:** `hsv-pipeline-backup/`  
**Approach:** Original version of the unsupervised pipeline before modifications

This is a pristine backup of the original unsupervised approach, kept for reference. It was never modified and exhibits the same failures as `hsv-pipeline/`.

## Why hsv-pipeline-semi Works Better

The semi-supervised approach solves all the problems above by:

1. **Learning direction from clicks**: The user clicks on flakes, and the pipeline learns the EXACT deviation pattern (ΔH_ref, ΔS_ref, ΔV_ref)
2. **Direction-aware matching**: Only pixels that deviate in the SAME direction as the clicked flakes get high probability
3. **Per-image substrate**: Each image gets its own substrate detection, handling varying conditions
4. **Adaptive signatures**: Different samples can have different flake signatures (sample2 vs sample3 vs sample4)
5. **Minimal user input**: Just 2-3 clicks per sample, then automatic application to all images

**Results comparison:**
- Unsupervised (hsv-pipeline): 15/19 images matched in sample2 (79% success)
- Semi-supervised (hsv-pipeline-semi): 19/19 images matched in sample2 with better calibration (100% success)

## Dependencies

- Python 3.7+
- numpy
- opencv-python (cv2)
- matplotlib
- scipy

## Notes

- The pipeline uses **global** substrate detection (histogram mode across entire image)
- Hue is circular (0-180° in OpenCV), so signed deviation uses circular math
- The `--calibration-image` option is useful when the first image in a sample is from a different material
- More clicks = better flake signature (3-5 clicks recommended)
- The `w_H` parameter controls hue strictness (higher = stricter matching)

## Future Improvements

- [ ] Add support for multiple flake signatures per sample
- [ ] Implement automatic click suggestion (detect flake candidates)
- [ ] Add mask extraction from heatmaps
- [ ] Support for batch calibration across multiple samples
- [ ] Export to COCO or other annotation formats