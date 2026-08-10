"""Tests for SimulatedStage — the hardware-free Stage implementation."""

import pytest

from stage.base import (
    StageError,
    StageLimitError,
    StageNotConnectedError,
)
from stage.simulated import SimulatedStage


def make_stage(**kwargs):
    stage = SimulatedStage(**kwargs)
    stage.connect()
    return stage


def test_positions_update_on_relative_and_absolute_moves():
    stage = make_stage()

    stage.move_relative("x", 500)
    assert stage.get_position()["x"] == 500.0

    stage.move_relative("x", -200)
    assert stage.get_position()["x"] == 300.0

    stage.move_absolute("y", 1234.5)
    assert stage.get_position()["y"] == 1234.5
    assert stage.get_position()["r"] == 0.0


def test_home_zeroes_positions():
    stage = make_stage()
    stage.move_relative("x", 500)
    stage.move_relative("y", -300)

    stage.home("all")

    positions = stage.get_position()
    assert positions == {"x": 0.0, "y": 0.0, "r": 0.0}


def test_limits_clamp_and_raise():
    stage = make_stage(limits={"x": (-1000.0, 1000.0)})

    with pytest.raises(StageLimitError):
        stage.move_relative("x", 5000, wait=True)

    assert stage.get_position()["x"] == 1000.0
    assert stage.get_status()["limits"]["x+"] is True

    # Fire-and-forget clamps and flags without raising
    stage.home("x")
    assert stage.get_status()["limits"]["x+"] is False
    stage.move_relative("x", -5000, wait=False)
    assert stage.get_position()["x"] == -1000.0
    assert stage.get_status()["limits"]["x-"] is True


def test_rate_modeled_motion_reaches_target():
    stage = make_stage(speed_pps=10000.0)

    stage.move_relative("x", 100, wait=True, timeout=2.0)

    assert stage.get_position()["x"] == pytest.approx(100.0)
    assert stage.get_status()["moving"]["x"] is False


def test_typed_errors_match_driver():
    stage = SimulatedStage()
    with pytest.raises(StageNotConnectedError):
        stage.move_relative("x", 100)

    stage_xy = make_stage(axes=("x", "y"))
    with pytest.raises(StageError, match="not configured"):
        stage_xy.move_relative("r", 100)


def test_status_shape_parity_with_driver():
    stage = make_stage()

    status = stage.get_status()

    assert set(status.keys()) == {
        "position", "moving", "limits", "home_switch",
        "axis_alarms", "emergency_stop",
    }
    assert set(status["position"].keys()) == {"x", "y", "r"}
    assert set(status["limits"].keys()) == {"x+", "x-", "y+", "y-", "r+", "r-"}
    assert status["emergency_stop"] is False


def test_set_speed_updates_rate():
    stage = make_stage(speed_pps=1000.0)

    stage.set_speed("x", 5000.0)

    # A move should now take less time (higher speed)
    stage.move_relative("x", 100, wait=True, timeout=2.0)
    assert stage.get_position()["x"] == pytest.approx(100.0)


def test_set_speed_invalid_raises():
    stage = make_stage()

    with pytest.raises(StageError, match="positive"):
        stage.set_speed("x", 0)

    with pytest.raises(StageError, match="positive"):
        stage.set_speed("x", -100)
