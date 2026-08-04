# Edge-Aware Denoising for Flake Probability Maps

This module implements edge-aware denoising to preserve thin, low-contrast flakes while removing background noise.

## Overview

The algorithm processes an RGB image and a model probability map to produce a cleaned probability map and final binary mask. It uses blurred RGB edges to boost or combine with probability values, ensuring that thin flakes with low probability are not lost during post-processing.

## Algorithm Steps

1. **Preprocess Image**: Convert RGB to grayscale and apply Gaussian blur
2. **Compute Edges**: Extract edges from blurred grayscale using Canny
3. **Smooth Probability**: Apply Gaussian blur to probability map
4. **Combine Signals**: Merge probability and edge information using one of two modes
5. **Candidate Mask**: Threshold combined map with low threshold to collect plausible regions
6. **Morphology**: Apply opening then closing with 3×3 disk kernel
7. **Component Analysis**: Extract connected components, compute features, and score them
8. **Final Selection**: Keep components based on score threshold
9. **Final Mask**: Apply final threshold and remove small objects

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `blur_kernel_image` | int | 5 | Gaussian blur kernel size for grayscale image |
| `blur_kernel_prob` | int | 5 | Gaussian blur kernel size for probability map |
| `canny_low` | int | 30 | Canny low threshold |
| `canny_high` | int | 100 | Canny high threshold |
| `combine_mode` | str | "weighted-max" | "boost-only" or "weighted-max" |
| `alpha` | float | 0.25 | Edge weight for weighted-max mode |
| `edge_boost` | float | 0.1 | Boost value for boost-only mode |
| `candidate_threshold` | float | 0.02 | Low threshold for candidate collection |
| `final_threshold` | float | 0.4 | Final probability threshold |
| `min_speck_area` | int | 30 | Remove components below this area (px) |
| `min_final_area` | int | 100 | Final small object removal (px) |
| `morph_kernel` | int | 3 | Morphological kernel size |
| `scoring_weights` | dict | {"area": 1.0, "contrast": 1.0, "solidity": 1.0} | Weights for component scoring |
| `keep_top_k` | int or None | None | Keep top K components (None = use score threshold) |
| `score_threshold` | float | 0.3 | Score threshold for keeping components |

## Combine Modes

### weighted-max (default)
```
combined = np.maximum(prob_blurred, alpha * edge_map)
```
Uses the maximum of the probability and weighted edge values. This ensures edges help identify flake boundaries without dominating the probability signal.

### boost-only
```
combined = prob_blurred
combined[edge_map > 0] = max(combined[edge_map > 0], prob_blurred[edge_map > 0] + edge_boost)
```
Boosts probability values at edge locations by a fixed amount, ensuring edge pixels are included in the candidate mask.

## Component Scoring

Each component is scored as:
```
score = area_norm * (1 - normalized_mean_contrast) * solidity
```

Where:
- `area_norm`: Normalized area (max 100,000 px for 1024×1024 images)
- `normalized_mean_contrast`: Mean Lab distance to substrate, normalized (max 100)
- `solidity`: Ratio of component area to convex hull area

Components with score >= `score_threshold` are kept.

## Default Config for 1024×1024 Images

```python
config = EdgeAwareDenoiseConfig(
    blur_kernel_image=5,
    blur_kernel_prob=5,
    canny_low=30,
    canny_high=100,
    combine_mode="weighted-max",
    alpha=0.25,
    edge_boost=0.1,
    candidate_threshold=0.02,
    min_speck_area=30,
    min_final_area=100,
    morph_kernel=3,
    scoring_weights={"area": 1.0, "contrast": 1.0, "solidity": 1.0},
    keep_top_k=None,
    score_threshold=0.3,
    final_threshold=0.4
)
```

## Usage

```python
from edge_aware_denoise import clean_probability_with_blurred_edges, get_default_config

# Load your RGB image and probability map
rgb_img = load_image("image.jpg")  # HxWx3, uint8, RGB
prob_map = ...  # HxW, float32 in [0, 1]

# Get default config
config = get_default_config()

# Process
cleaned_prob, final_mask, debug_images, stats = clean_probability_with_blurred_edges(
    rgb_img, prob_map, config
)
```

## Debug Outputs

The function returns a dictionary of intermediate images:
- `blurred_gray`: Blurred grayscale image
- `edge_map`: Edge map (save as heatmap with jet colormap)
- `prob_blurred`: Smoothed probability map
- `combined`: Combined probability/edge map
- `binary_candidate`: Low threshold binary mask
- `candidate_components`: After small speck removal
- `final_candidate_mask`: After component scoring
- `final_mask`: Final binary mask
- `overlay`: Mask overlaid on RGB image (green)

## Tuning Guidance

### If thin flakes are missing:
- Lower `candidate_threshold` (0.01–0.02)
- Lower `final_threshold` (0.25–0.35)
- Reduce `alpha` or `edge_boost` so edges don't suppress probability

### If background clumps persist:
- Increase `morph_kernel` size
- Raise `min_final_area`
- Tighten scoring (require higher `score_threshold` or adjust `scoring_weights`)

### If many false positives remain:
- Increase `alpha` slightly
- Raise `final_threshold`
- Increase `min_speck_area`

## Command Line

```bash
cd segment_3
python test_edge_aware_denoise.py --num_images 5
```

## Requirements

- OpenCV (cv2)
- NumPy
- scikit-learn (for probability map computation)