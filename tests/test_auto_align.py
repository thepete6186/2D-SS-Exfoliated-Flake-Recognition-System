"""Tests for auto_align.AutoAlignController (orchestration layer)."""

import pytest

from auto_align import AutoAlignController
from edge_estimation import SyntheticRotatingCamera
from rotation_controller import (
    DEFAULT_STEPS_PER_DEGREE,
    RotationController,
)
from stage.simulated import SimulatedStage

R_LIMIT = 36000  # Zolix R axis soft limit: +-45 deg at 800 steps/deg.


def build_stage_camera(start_angle_deg: float):
    """SimulatedStage parked at start_angle_deg, camera bound to stage R."""
    stage = SimulatedStage(
        axes=("x", "y", "r"), limits={"r": (-R_LIMIT, R_LIMIT)}
    )
    stage.connect()
    spd = DEFAULT_STEPS_PER_DEGREE
    stage.move_absolute("r", start_angle_deg * spd, wait=True, timeout=30.0)
    cam = SyntheticRotatingCamera(steps_per_degree=spd).bind_stage(stage, "r")
    return stage, cam


def test_rotation_controller_degree_to_pulse_conversion():
    stage = SimulatedStage()
    stage.connect()
    rot = RotationController(stage, steps_per_degree=DEFAULT_STEPS_PER_DEGREE)
    assert rot.degrees_to_steps(5.0) == 4000
    assert rot.degrees_to_steps(-2.5) == -2000
    assert rot.degrees_to_steps(0.0) == 0


def test_error_and_correction_canonical_math():
    rot = RotationController(SimulatedStage())
    # 20 deg canonical -> square to horizontal: error 20, correction -20.
    error, corr, axis = rot.error_and_correction(20.0)
    assert axis == "horizontal"
    assert error == pytest.approx(20.0)
    assert corr == pytest.approx(-20.0)
    # 70 deg canonical -> vertical: error 20, correction +20.
    error, corr, axis = rot.error_and_correction(70.0)
    assert axis == "vertical"
    assert corr == pytest.approx(20.0)
    # Perpendicular-edge switch (20 vs 110) is invariant in canonical space.
    _e, corr20, _axis = rot.error_and_correction(20.0)
    _e, corr110, axis110 = rot.error_and_correction(110.0)
    assert corr110 == pytest.approx(corr20)
    assert axis110 == "horizontal"
    # Locked target overrides the nearest-axis choice.
    _e, corr, _axis = rot.error_and_correction(20.0, target="vertical")
    assert corr == pytest.approx(70.0)


def test_closed_loop_converges():
    stage, cam = build_stage_camera(20.0)
    controller = AutoAlignController(cam, stage)
    ok, message = controller.align(max_iterations=12, tolerance_deg=2.0)
    assert ok, message
    assert controller.converged is True
    assert controller.current_error <= 2.0
    # The sample should be visually squared: stage came back near 0.
    assert abs(stage.get_position()["r"] / DEFAULT_STEPS_PER_DEGREE) < 2.0


def test_already_aligned_converges_immediately():
    stage, cam = build_stage_camera(1.0)
    controller = AutoAlignController(cam, stage)
    ok, message = controller.align(max_iterations=5, tolerance_deg=2.0)
    assert ok, message
    assert controller.iterations <= 2


def test_align_uses_blocking_rotations():
    stage, cam = build_stage_camera(20.0)
    controller = AutoAlignController(cam, stage)
    calls = []
    original = stage.move_relative

    def spy(axis, steps, wait=True, timeout=30.0):
        calls.append((axis, steps, wait))
        return original(axis, steps, wait=wait, timeout=timeout)

    stage.move_relative = spy
    controller.align(max_iterations=8, tolerance_deg=2.0)
    assert calls, "no stage moves issued"
    assert all(axis == "r" for axis, _steps, _wait in calls)
    assert all(wait is True for _axis, _steps, wait in calls)


def test_failed_rotation_reported():
    """A failing stage move must surface as a failed alignment."""
    stage, cam = build_stage_camera(20.0)

    def broken_move(axis, steps, wait=True, timeout=30.0):
        raise RuntimeError("hardware offline")

    stage.move_relative = broken_move
    controller = AutoAlignController(cam, stage)
    ok, message = controller.align(max_iterations=3, tolerance_deg=2.0)
    assert ok is False
    assert "Rotation failed" in message


def test_single_shot_correction_count():
    """On clean synthetic data only ONE full correction should be needed."""
    stage, cam = build_stage_camera(20.0)
    controller = AutoAlignController(cam, stage)

    calls = []
    original = stage.move_relative

    def spy(axis, steps, wait=True, timeout=30.0):
        calls.append((axis, steps, wait))
        return original(axis, steps, wait=wait, timeout=timeout)

    stage.move_relative = spy
    ok, message = controller.align(max_iterations=10, tolerance_deg=2.0)
    assert ok, message
    # Exactly one correction move (the initial single-shot).
    assert len(calls) == 1, f"expected 1 correction, got {len(calls)}: {calls}"


def test_no_microadjustments_with_noise():
    """Even with ±5 deg estimation noise, corrections stay bounded."""
    import random

    stage, cam = build_stage_camera(20.0)
    controller = AutoAlignController(cam, stage)

    rng = random.Random(42)

    orig_estimate = controller.estimate_edge_angle

    def noisy_estimate(frame):
        angle = orig_estimate(frame)
        if angle is not None:
            angle = angle + rng.gauss(0, 5.0)
        return angle

    controller.estimate_edge_angle = noisy_estimate

    calls = []
    original = stage.move_relative

    def spy(axis, steps, wait=True, timeout=30.0):
        calls.append((axis, steps, wait))
        return original(axis, steps, wait=wait, timeout=timeout)

    stage.move_relative = spy
    ok, message = controller.align(max_iterations=10, tolerance_deg=2.0)
    # May converge with 1 or 2 full corrections, but never many small ones.
    assert len(calls) <= 2, f"too many corrections ({len(calls)}): {calls}"
    assert controller.current_error <= 10.0, (
        f"residual {controller.current_error} deg too large after "
        f"{len(calls)} corrections"
    )


def test_first_measurement_always_corrects():
    """A first measurement that is misaligned by a few degrees (inside the
    post-correction noise band) must STILL rotate the stage. The old code
    treated the very first measurement as a "post-correction residual" and
    returned success without rotating whenever the error was <= 5x tolerance.
    """
    stage, cam = build_stage_camera(8.0)  # 8 deg off -> > 2 deg tolerance

    calls = []
    original = stage.move_relative

    def spy(axis, steps, wait=True, timeout=30.0):
        calls.append((axis, steps, wait))
        return original(axis, steps, wait=wait, timeout=timeout)

    stage.move_relative = spy
    controller = AutoAlignController(cam, stage)
    ok, message = controller.align(max_iterations=10, tolerance_deg=2.0)
    assert ok, message
    # The initial 8-deg misalignment must actually be corrected, not accepted.
    assert len(calls) >= 1, f"no rotation issued for 8-deg misalignment: {message}"
    assert abs(stage.get_position()["r"] / DEFAULT_STEPS_PER_DEGREE) < 2.0
    """If the first correction overshoots wildly, corrections stay bounded."""
    stage, cam = build_stage_camera(20.0)
    controller = AutoAlignController(cam, stage)

    # Make the estimator report a wildly wrong angle after any rotation.
    orig_estimate = controller.estimate_edge_angle

    def bad_estimate(frame):
        angle = orig_estimate(frame)
        if angle is not None:
            angle = angle + 100.0
        return angle

    controller.estimate_edge_angle = bad_estimate

    calls = []
    original = stage.move_relative

    def spy(axis, steps, wait=True, timeout=30.0):
        calls.append((axis, steps, wait))
        return original(axis, steps, wait=wait, timeout=timeout)

    stage.move_relative = spy
    ok, message = controller.align(max_iterations=10, tolerance_deg=2.0)
    # With bad data we may get 1 or 2 corrections, but never a long chain.
    assert len(calls) <= 2, f"expected <=2 corrections, got {len(calls)}: {calls}"
    assert ok is False or controller.current_error <= 10.0