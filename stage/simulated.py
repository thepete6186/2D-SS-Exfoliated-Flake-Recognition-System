"""
SimulatedStage — hardware-free Stage implementation.

Drop-in double for ZolixZC300: same interface, same typed errors, same
get_status() shape. Used by tests and for dry-running scan logic on
machines without the stage attached (e.g. `zc300_smoke --simulate`).

Motion is instant by default; pass speed_pps to model motion at a
constant rate so wait_until_stopped() and the 'moving' status behave
like real hardware.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional, Sequence, Tuple

from stage.base import (
    AXES,
    Stage,
    StageError,
    StageLimitError,
    StageNotConnectedError,
)

logger = logging.getLogger(__name__)


class _Motion:
    """An in-flight rate-modeled move."""

    def __init__(self, start_pos: float, target: float, speed_pps: float):
        self.start_pos = start_pos
        self.target = target
        self.start_time = time.monotonic()
        distance = abs(target - start_pos)
        self.arrival_time = self.start_time + (distance / speed_pps if speed_pps else 0.0)

    def position_at(self, now: float) -> float:
        if now >= self.arrival_time:
            return self.target
        span = self.arrival_time - self.start_time
        frac = (now - self.start_time) / span if span > 0 else 1.0
        return self.start_pos + (self.target - self.start_pos) * frac

    def done(self, now: float) -> bool:
        return now >= self.arrival_time


class SimulatedStage(Stage):
    """
    Simulated 3-axis stage.

    Parameters
    ----------
    axes : sequence of str
        Configured logical axes (subset of ("x", "y", "r")).
    speed_pps : float, optional
        None = instant motion (fast tests). A value models constant-rate
        motion so 'moving' status and blocking waits behave realistically.
    limits : dict, optional
        Per-axis (low, high) soft limits in pulses. A move past a bound
        clamps to it, sets the corresponding limit flag, and — like the
        real driver's blocking wait — raises StageLimitError when the
        move was issued with wait=True.
    """

    def __init__(
        self,
        axes: Sequence[str] = AXES,
        speed_pps: Optional[float] = None,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        for axis in axes:
            if str(axis).lower() not in AXES:
                raise StageError(f"Unknown axis {axis!r} (expected one of {AXES})")
        self._axes = tuple(str(a).lower() for a in axes)
        self._speed_pps = float(speed_pps) if speed_pps else None
        self._limits = dict(limits or {})

        self._connected = False
        self._lock = threading.RLock()
        self._positions: Dict[str, float] = {axis: 0.0 for axis in AXES}
        self._motions: Dict[str, _Motion] = {}
        self._tripped: Dict[str, bool] = {f"{a}{s}": False for a in AXES for s in "+-"}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True
        logger.info("SimulatedStage connected (axes: %s)", ",".join(self._axes))

    def disconnect(self) -> None:
        with self._lock:
            self._finalize_motions()
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_relative(self, axis: str, steps: int,
                      wait: bool = True, timeout: float = 30.0) -> None:
        ax = self._validate_axis(axis)
        self._require_connected()
        if int(steps) == 0:
            return
        target = self._current_position(ax) + float(int(steps))
        self._start_move(ax, target, wait, timeout)

    def move_absolute(self, axis: str, position: float,
                      wait: bool = True, timeout: float = 60.0) -> None:
        ax = self._validate_axis(axis)
        self._require_connected()
        self._start_move(ax, float(position), wait, timeout)

    def home(self, axis: str = "all",
             wait: bool = True, timeout: float = 120.0) -> None:
        axes = self._axes if axis == "all" else (self._validate_axis(axis),)
        self._require_connected()
        with self._lock:
            for ax in axes:
                self._motions.pop(ax, None)
                self._positions[ax] = 0.0
                self._tripped[f"{ax}+"] = False
                self._tripped[f"{ax}-"] = False

    def stop(self, axis: str = "all") -> None:
        self._require_connected()
        axes = self._axes if axis == "all" else (self._validate_axis(axis),)
        with self._lock:
            now = time.monotonic()
            for ax in axes:
                motion = self._motions.pop(ax, None)
                if motion is not None:
                    self._positions[ax] = motion.position_at(now)

    def set_speed(self, axis: str, speed_pps: float) -> None:
        """Set the constant speed for an axis (pulses/s)."""
        ax = self._validate_axis(axis)
        self._require_connected()
        if speed_pps <= 0:
            raise StageError(f"Speed must be positive, got {speed_pps}")
        with self._lock:
            self._speed_pps = float(speed_pps)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_position(self) -> Dict[str, float]:
        return self.get_status()["position"]

    def get_status(self) -> Dict[str, object]:
        self._require_connected()
        with self._lock:
            now = time.monotonic()
            positions = {}
            moving = {}
            for axis in AXES:
                motion = self._motions.get(axis)
                if motion is not None:
                    positions[axis] = motion.position_at(now)
                    moving[axis] = not motion.done(now)
                    if motion.done(now):
                        self._positions[axis] = motion.target
                        self._motions.pop(axis)
                else:
                    positions[axis] = self._positions[axis]
                    moving[axis] = False

            return {
                "position": positions,
                "moving": moving,
                "limits": dict(self._tripped),
                "home_switch": {axis: False for axis in AXES},
                "axis_alarms": {axis: False for axis in AXES},
                "emergency_stop": False,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise StageNotConnectedError("SimulatedStage is not connected")

    def _current_position(self, axis: str) -> float:
        with self._lock:
            motion = self._motions.get(axis)
            if motion is not None:
                return motion.position_at(time.monotonic())
            return self._positions[axis]

    def _finalize_motions(self) -> None:
        now = time.monotonic()
        for axis, motion in list(self._motions.items()):
            self._positions[axis] = motion.position_at(now)
        self._motions.clear()

    def _start_move(self, axis: str, target: float,
                    wait: bool, timeout: float) -> None:
        clamped_flag = None
        low, high = self._limits.get(axis, (None, None)) or (None, None)
        if high is not None and target > high:
            target, clamped_flag = float(high), f"{axis}+"
        elif low is not None and target < low:
            target, clamped_flag = float(low), f"{axis}-"

        with self._lock:
            # Finalize any in-flight motion on this axis first
            prior = self._motions.pop(axis, None)
            if prior is not None:
                self._positions[axis] = prior.position_at(time.monotonic())
            if clamped_flag is not None:
                self._tripped[clamped_flag] = True
            if self._speed_pps is None:
                self._positions[axis] = target
            else:
                self._motions[axis] = _Motion(
                    self._positions[axis], target, self._speed_pps
                )

        if wait:
            self.wait_until_stopped((axis,), timeout=timeout, settle_time=0.0)
            if clamped_flag is not None:
                raise StageLimitError(
                    f"Axis '{axis}' clamped at soft limit "
                    f"{clamped_flag[-1]}{target:g}"
                )
