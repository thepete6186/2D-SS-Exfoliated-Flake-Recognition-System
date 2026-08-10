"""
Stage interface and exception hierarchy.

The raster-scan loop (and any other consumer) depends on the Stage ABC,
not on a concrete driver — ZolixZC300 talks to the real controller,
SimulatedStage provides a hardware-free implementation for tests.

All positions and distances are in integer controller pulses; physical
unit conversion (nm/µm/deg) lives in camera/coordinate_mapper.py.
"""

from __future__ import annotations

import abc
import time
from typing import Dict, Optional, Sequence

AXES = ("x", "y", "r")


class StageError(Exception):
    """Base class for all stage errors."""


class ModbusError(StageError):
    """Malformed or unexpected MODBUS traffic (CRC mismatch, bad frame)."""


class StageNotConnectedError(StageError):
    """Operation attempted without an open, verified connection."""


class StageBusyError(StageError):
    """Motion command rejected because the axis is already moving (exc 0x06)."""


class StageNotEnabledError(StageError):
    """Motion command rejected because the axis is not enabled (exc 0x09)."""


class StageLimitError(StageError):
    """A limit switch blocked or aborted the motion (exc 0x07 / status bit)."""


class StageEstopError(StageError):
    """Emergency stop is active (exc 0x08 / status bit 9)."""


class StageAlarmError(StageError):
    """A driver alarm is active on the axis (status bits 10-12)."""


class StageTimeoutError(StageError):
    """wait_until_stopped() deadline expired while axes were still moving."""


class Stage(abc.ABC):
    """
    Abstract motion-stage interface.

    Axes are logical names from AXES ("x", "y", "r"); a concrete stage
    may be configured with a subset (e.g. no rotation hardware).
    """

    _axes: Sequence[str] = AXES

    # -- connection --------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Open and verify the connection."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Stop all axes (best effort) and close the connection."""

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        """True while the connection is open and verified."""

    # -- motion ------------------------------------------------------------

    @abc.abstractmethod
    def move_relative(self, axis: str, steps: int,
                      wait: bool = True, timeout: float = 30.0) -> None:
        """Move an axis by a signed number of pulses (fixed-length move)."""

    @abc.abstractmethod
    def move_absolute(self, axis: str, position: float,
                      wait: bool = True, timeout: float = 60.0) -> None:
        """Move an axis to an absolute position in pulses."""

    @abc.abstractmethod
    def home(self, axis: str = "all",
             wait: bool = True, timeout: float = 120.0) -> None:
        """Run the controller's homing routine for one axis or all."""

    @abc.abstractmethod
    def stop(self, axis: str = "all") -> None:
        """Stop one axis or all axes."""

    @abc.abstractmethod
    def set_speed(self, axis: str, speed_pps: float) -> None:
        """Set the constant speed for an axis (pulses/s)."""

    # -- status ------------------------------------------------------------

    @abc.abstractmethod
    def get_position(self) -> Dict[str, float]:
        """Current positions in pulses, keyed by logical axis name."""

    @abc.abstractmethod
    def get_status(self) -> Dict[str, object]:
        """
        Full stage status:

        {'position': {axis: pulses}, 'moving': {axis: bool},
         'limits': {'x+': bool, 'x-': bool, ...},
         'home_switch': {axis: bool}, 'axis_alarms': {axis: bool},
         'emergency_stop': bool}
        """

    # -- blocking wait (shared implementation) -----------------------------

    def wait_until_stopped(self, axes: Optional[Sequence[str]] = None,
                           timeout: float = 30.0, poll_interval: float = 0.05,
                           settle_time: float = 0.05) -> None:
        """
        Poll get_status() until every watched axis reports stopped.

        Raises StageEstopError / StageAlarmError immediately when those
        conditions appear, StageLimitError when a watched axis hits a
        limit switch while still moving, and StageTimeoutError at the
        deadline. An axis that stopped at a limit (e.g. after homing to
        the negative limit) returns normally — inspect get_status()
        when a truncated move matters.
        """
        watch = tuple(self._validate_axis(a) for a in (axes or self._axes))
        deadline = time.monotonic() + timeout

        while True:
            status = self.get_status()

            if status["emergency_stop"]:
                raise StageEstopError("Emergency stop is active")
            for axis in watch:
                if status["axis_alarms"][axis]:
                    raise StageAlarmError(f"Axis '{axis}' reports a driver alarm")

            moving = [a for a in watch if status["moving"][a]]
            if not moving:
                if settle_time > 0:
                    time.sleep(settle_time)
                return

            for axis in moving:
                if status["limits"][f"{axis}+"] or status["limits"][f"{axis}-"]:
                    raise StageLimitError(
                        f"Axis '{axis}' hit a limit switch during motion"
                    )

            if time.monotonic() >= deadline:
                raise StageTimeoutError(
                    f"Axes still moving after {timeout:.1f}s: {moving}"
                )
            time.sleep(poll_interval)

    # -- shared helpers ----------------------------------------------------

    def _validate_axis(self, axis: str) -> str:
        """Normalize an axis name and check it is configured on this stage."""
        ax = str(axis).lower()
        if ax not in AXES:
            raise StageError(f"Unknown axis {axis!r} (expected one of {AXES})")
        if ax not in self._axes:
            raise StageError(f"Axis '{ax}' not configured on this stage")
        return ax
