"""
The flake HSV signature must treat hue as circular. OpenCV hue 0 and 179
are both "red"; a red flake sampled across the wrap must not produce a
spurious cyan mean (~90) with a huge std (~87).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "hsv-pipeline-semi"))

from semi_supervised_pipeline import SemiSupervisedPipeline


def test_flake_signature_circular_hue():
    pipe = SemiSupervisedPipeline(patch_radius=3)
    h = 21
    # Patch pixels straddle the hue wrap: half ~178, half ~1 (all "red")
    H = np.zeros((h, h), np.float32)
    H[:, 0::2] = 178.0
    H[:, 1::2] = 1.0
    S = np.full((h, h), 120.0, np.float32)
    V = np.full((h, h), 200.0, np.float32)

    sig = pipe._extract_flake_signature(
        H, S, V, seed_points=[(10, 10)], substrate_peak=(90.0, 50.0, 100.0)
    )

    # Circular mean of {178, 1} is ~179.5 (equivalently -0.5): near the wrap
    dist_from_red = abs(pipe._circular_delta(sig["mean_H"], 0.0))
    assert dist_from_red < 3.0, (
        f"mean_H={sig['mean_H']} is not near the red wrap point "
        f"(circular distance {dist_from_red})"
    )
    # Spread of perceptually-identical reds must be small, not ~87
    assert sig["std_H"] < 5.0, f"std_H={sig['std_H']} blown up by wraparound"


def test_flake_signature_matches_linear_stats_away_from_wrap():
    """Away from the wrap, circular stats must agree with linear ones."""
    pipe = SemiSupervisedPipeline(patch_radius=3)
    h = 21
    rng = np.random.default_rng(0)
    H = np.clip(rng.normal(60.0, 2.0, (h, h)), 0, 179).astype(np.float32)
    S = np.full((h, h), 120.0, np.float32)
    V = np.full((h, h), 200.0, np.float32)

    sig = pipe._extract_flake_signature(
        H, S, V, seed_points=[(10, 10)], substrate_peak=(90.0, 50.0, 100.0)
    )
    patch = H[7:14, 7:14]
    assert abs(sig["mean_H"] - float(np.mean(patch))) < 1.0
    assert abs(sig["std_H"] - float(np.std(patch))) < 1.0
