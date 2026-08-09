"""Tests for click-to-calibrate substrate support in the HSV pipeline.

Covers the module-level sampling helpers (extract_hsv_patch,
substrate_stats_from_pixels, substrate_from_samples) and the
substrate_peak / substrate_std overrides on process() and
process_with_signature().
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "hsv-pipeline-semi"))

from semi_supervised_pipeline import (  # noqa: E402
    SemiSupervisedPipeline,
    extract_hsv_patch,
    substrate_from_samples,
    substrate_stats_from_pixels,
)


RGB_MAJORITY = (0, 128, 0)      # dominates the histogram
RGB_SUBSTRATE = (50, 50, 150)   # true substrate region, minority by area
RGB_FLAKE = (200, 180, 60)      # seed-click region


def hsv_of(rgb: tuple) -> tuple:
    """OpenCV HSV (H 0-179, S/V 0-255) of a single RGB color."""
    pixel = np.array([[rgb]], dtype=np.uint8)
    h, s, v = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    return (float(h), float(s), float(v))


def make_test_image() -> np.ndarray:
    """60x60 RGB image whose histogram mode disagrees with the substrate.

    Majority color fills the frame; the true substrate occupies a 20x20
    corner block; the flake occupies a 10x10 block for seed clicks.
    """
    rgb = np.empty((60, 60, 3), dtype=np.uint8)
    rgb[:, :] = RGB_MAJORITY
    rgb[0:20, 0:20] = RGB_SUBSTRATE
    rgb[40:50, 40:50] = RGB_FLAKE
    return rgb


FLAKE_SEEDS = [(45, 45)]


class TestProcessOverrides:
    def test_process_honors_supplied_substrate_peak(self):
        rgb = make_test_image()
        supplied = hsv_of(RGB_SUBSTRATE)
        pipe = SemiSupervisedPipeline(patch_radius=3)

        result = pipe.process(rgb, FLAKE_SEEDS, substrate_peak=supplied)

        assert result["substrate_peak"] == pytest.approx(supplied)
        assert result["substrate_source"] == "override"
        # Delta maps must be measured from the supplied peak, not the
        # histogram mode: a majority-color pixel deviates by (majority - supplied).
        maj = hsv_of(RGB_MAJORITY)
        assert result["delta_S"][30, 30] == pytest.approx(maj[1] - supplied[1])
        assert result["delta_V"][30, 30] == pytest.approx(maj[2] - supplied[2])

    def test_process_auto_derives_std_when_only_peak_supplied(self):
        rgb = make_test_image()
        supplied = hsv_of(RGB_SUBSTRATE)
        pipe = SemiSupervisedPipeline(patch_radius=3)

        result = pipe.process(rgb, FLAKE_SEEDS, substrate_peak=supplied)

        hsv = pipe._convert_to_hsv(rgb)
        expected_std = pipe._estimate_substrate_std(hsv, supplied)
        assert result["substrate_std"] == pytest.approx(expected_std)
        assert all(s > 0 for s in result["substrate_std"])

    def test_process_honors_supplied_substrate_std(self):
        rgb = make_test_image()
        supplied_peak = hsv_of(RGB_SUBSTRATE)
        supplied_std = (2.5, 3.5, 4.5)
        pipe = SemiSupervisedPipeline(patch_radius=3)

        result = pipe.process(
            rgb, FLAKE_SEEDS,
            substrate_peak=supplied_peak, substrate_std=supplied_std,
        )

        assert result["substrate_std"] == pytest.approx(supplied_std)

    def test_process_with_signature_override(self):
        rgb = make_test_image()
        supplied = hsv_of(RGB_SUBSTRATE)
        pipe = SemiSupervisedPipeline(patch_radius=3)
        sig = pipe.process(rgb, FLAKE_SEEDS)["flake_signature"]

        result = pipe.process_with_signature(rgb, sig, substrate_peak=supplied)

        assert result["substrate_peak"] == pytest.approx(supplied)
        assert result["substrate_source"] == "override"

    def test_process_positional_two_arg_backward_compatible(self):
        rgb = make_test_image()
        pipe = SemiSupervisedPipeline(patch_radius=3)

        result = pipe.process(rgb, FLAKE_SEEDS)

        hsv = pipe._convert_to_hsv(rgb)
        assert result["substrate_peak"] == pytest.approx(
            pipe._find_substrate_peak(hsv)
        )
        assert result["substrate_source"] == "auto"


class TestSubstrateStats:
    def test_substrate_stats_circular_wrap(self):
        # Red substrate straddling the OpenCV hue wrap (0/179): a linear
        # mean would land near cyan (~90) with a huge std.
        n = 200
        hsv_pixels = np.zeros((n, 3), dtype=np.float32)
        hsv_pixels[0::2, 0] = 178.0
        hsv_pixels[1::2, 0] = 1.0
        hsv_pixels[:, 1] = 120.0
        hsv_pixels[:, 2] = 200.0

        peak, std = substrate_stats_from_pixels(hsv_pixels)

        assert abs(SemiSupervisedPipeline._circular_delta(peak[0], 0.0)) < 3.0
        assert std[0] < 5.0
        assert peak[1] == pytest.approx(120.0)
        assert peak[2] == pytest.approx(200.0)

    def test_substrate_stats_linear_away_from_wrap(self):
        rng = np.random.default_rng(0)
        hsv_pixels = np.stack(
            [
                np.clip(rng.normal(60.0, 2.0, 500), 0, 179),
                np.clip(rng.normal(80.0, 5.0, 500), 0, 255),
                np.clip(rng.normal(150.0, 8.0, 500), 0, 255),
            ],
            axis=1,
        ).astype(np.float32)

        peak, std = substrate_stats_from_pixels(hsv_pixels)

        assert peak[0] == pytest.approx(np.mean(hsv_pixels[:, 0]), abs=1.0)
        assert std[0] == pytest.approx(np.std(hsv_pixels[:, 0]), abs=1.0)
        assert peak[1] == pytest.approx(np.mean(hsv_pixels[:, 1]), abs=1e-3)
        assert std[1] == pytest.approx(np.std(hsv_pixels[:, 1]), abs=1e-3)
        assert peak[2] == pytest.approx(np.mean(hsv_pixels[:, 2]), abs=1e-3)
        assert std[2] == pytest.approx(np.std(hsv_pixels[:, 2]), abs=1e-3)


class TestPatchSampling:
    def test_extract_hsv_patch_interior(self):
        rgb = make_test_image()
        patch = extract_hsv_patch(rgb, 30, 30, radius=7)

        assert patch.shape == (15 * 15, 3)
        assert patch.dtype == np.float32
        assert tuple(patch[0]) == pytest.approx(hsv_of(RGB_MAJORITY))

    def test_extract_hsv_patch_clips_at_border(self):
        rgb = make_test_image()
        patch = extract_hsv_patch(rgb, 0, 0, radius=7)

        # Only the 8x8 in-bounds quadrant survives; no wraparound sampling.
        assert patch.shape == (8 * 8, 3)

    def test_substrate_from_samples_counts_and_stats(self):
        rgb = make_test_image()
        # Two clicks inside the true-substrate block (patches stay inside it).
        result = substrate_from_samples(rgb, [(10, 10), (9, 9)], patch_radius=2)

        assert result["n_samples"] == 2
        assert result["n_pixels"] == 2 * 25
        assert result["peak"] == pytest.approx(hsv_of(RGB_SUBSTRATE))
        # Uniform color: std collapses to ~epsilon.
        assert all(s < 0.01 for s in result["std"])
