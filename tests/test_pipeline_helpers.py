#!/usr/bin/env python3
"""
Tests for the full-pipeline helpers:
- pixel <-> stage-pulse conversion (edge_offset_to_stage_steps)
- AutoAlignController rotation_sign forwarding
- ColorEdgeEstimator channel-order handling
"""

import cv2
import pytest

from auto_align import AutoAlignController
from chip_edge_detector import (
    DEFAULT_PIXELS_PER_NM,
    STAGE_STEP_SIZE_UM,
    edge_offset_to_stage_steps,
)
from color_edge_estimator import ColorEdgeEstimator
from stage.simulated import SimulatedStage


def _px_per_nm_at_5x():
    return DEFAULT_PIXELS_PER_NM * 5.0  # default calibration for 5x zoom


def test_edge_offset_to_stage_steps_known_value():
    """A known 100 px offset at 5x must become ~12 X pulses at 0.625 um/step."""
    ppn = _px_per_nm_at_5x()
    sx, sy = edge_offset_to_stage_steps(100.0, -80.0, ppn, STAGE_STEP_SIZE_UM)

    nm_per_px = 1.0 / ppn
    pulses_per_px = (nm_per_px / 1000.0) / STAGE_STEP_SIZE_UM
    assert sx == int(round(100.0 * pulses_per_px))
    assert sy == int(round(-80.0 * pulses_per_px))
    # sanity: a reasonable sub-0.2 pulses/px means 100 px -> tens of pulses
    assert 0.05 < pulses_per_px < 0.5
    assert sx > 0
    assert sy < 0


def test_edge_offset_center_returns_zero():
    """A feature already at the image centre maps to a zero move."""
    sx, sy = edge_offset_to_stage_steps(0.0, 0.0, _px_per_nm_at_5x())
    assert (sx, sy) == (0, 0)


def test_edge_offset_direction_reversal():
    """direction=-1 must flip both axis signs."""
    ppn = _px_per_nm_at_5x()
    sx_p, sy_p = edge_offset_to_stage_steps(50.0, 50.0, ppn, direction=1.0)
    sx_m, sy_m = edge_offset_to_stage_steps(50.0, 50.0, ppn, direction=-1.0)
    assert (sx_p, sy_p) == (-sx_m, -sy_m)


def test_edge_offset_validates_inputs():
    with pytest.raises(ValueError):
        edge_offset_to_stage_steps(1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        edge_offset_to_stage_steps(1.0, 1.0, 0.01, step_size_um=0.0)
    with pytest.raises(ValueError):
        edge_offset_to_stage_steps(1.0, 1.0, 0.01, direction=0.5)


def test_auto_align_forwards_rotation_sign():
    stage = SimulatedStage(axes=("x", "y", "r"))
    stage.connect()

    class _CamStub:
        def capture(self):
            return None

    ctrl_down = AutoAlignController(_CamStub(), stage, rotation_sign=-1.0)
    assert ctrl_down.rotation.rotation_sign == -1.0

    ctrl_up = AutoAlignController(_CamStub(), stage, rotation_sign=1.0)
    assert ctrl_up.rotation.rotation_sign == 1.0


def test_color_edge_estimator_channel_order_bgr():
    """channel_order='bgr' must feed a BGR frame through unchanged."""
    img_bgr = cv2.imread("Capture.PNG")
    assert img_bgr is not None

    est_bgr = ColorEdgeEstimator(channel_order="bgr")
    r_bgr = est_bgr.detect(img_bgr)
    assert r_bgr is not None and r_bgr.angle_deg is not None


def test_color_edge_estimator_rgb_and_bgr_agree():
    """Detecting on an RGB frame (channel_order=rgb) == detecting on the same
    scene in BGR (channel_order=bgr)."""
    img_bgr = cv2.imread("Capture.PNG")
    assert img_bgr is not None
    img_rgb = img_bgr[:, :, ::-1].copy()

    r_rgb = ColorEdgeEstimator(channel_order="rgb").detect(img_rgb)
    r_bgr = ColorEdgeEstimator(channel_order="bgr").detect(img_bgr)
    assert r_rgb is not None and r_bgr is not None
    assert r_rgb.angle_deg == pytest.approx(r_bgr.angle_deg, abs=1e-6)


def test_color_edge_estimator_validates_channel_order():
    with pytest.raises(ValueError):
        ColorEdgeEstimator(channel_order="hsv")
    assert ColorEdgeEstimator().channel_order == "rgb"


# ---------------------------------------------------------------------------
# Calibration resolution for arbitrary zoom levels (mouse-wheel zoom)
# ---------------------------------------------------------------------------

from chip_edge_detector import ChipEdgeDetector  # noqa: E402


def test_ppn_exact_preset_hit():
    cal = {1.0: 100.0, 5.0: 500.0}
    assert ChipEdgeDetector.pixels_per_nm_for_zoom(cal, 5.0) == 500.0


def test_ppn_arbitrary_zoom_scales_nearest():
    """0.9x must resolve from the nearest preset (1x) scaled linearly."""
    cal = {1.0: 100.0, 5.0: 500.0}
    assert ChipEdgeDetector.pixels_per_nm_for_zoom(cal, 0.9) == pytest.approx(90.0)
    # 7x sits between 5x and 10x -> nearest is 5x
    assert ChipEdgeDetector.pixels_per_nm_for_zoom(cal, 7.0) == pytest.approx(700.0)


def test_ppn_empty_calibration_falls_back_to_default():
    ppn = ChipEdgeDetector.pixels_per_nm_for_zoom({}, 2.5)
    assert ppn == pytest.approx(2.5 * 2650.0 / 1_000_000.0)


def test_capture_frame_retries_transient_failures():
    """A transient SmartCam timeout must not abort the alignment."""

    class FlakyCam:
        def __init__(self):
            self.calls = 0

        def capture(self):
            self.calls += 1
            if self.calls < 3:
                return None  # simulate two timeouts
            return "frame"

    stage = SimulatedStage(axes=("x", "y", "r"))
    stage.connect()
    cam = FlakyCam()
    ctrl = AutoAlignController(cam, stage)
    frame = ctrl.capture_frame(max_retries=3, retry_delay_s=0.0)
    assert frame == "frame"
    assert cam.calls == 3


def test_capture_frame_returns_none_after_exhausted_retries():
    class DeadCam:
        def __init__(self):
            self.calls = 0

        def capture(self):
            self.calls += 1
            return None

    stage = SimulatedStage(axes=("x", "y", "r"))
    stage.connect()
    cam = DeadCam()
    ctrl = AutoAlignController(cam, stage)
    assert ctrl.capture_frame(max_retries=3, retry_delay_s=0.0) is None
    assert cam.calls == 3