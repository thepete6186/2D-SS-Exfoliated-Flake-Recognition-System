#!/usr/bin/env python3
"""
RotationController - motion-control layer for sample auto-alignment.

This module wraps a `stage.Stage` and its rotation axis ("r" on the Zolix
ZC300) and is responsible ONLY for translating an angle correction (deg)
into a stage move. It shares no logic with the image-processing code in
`edge_estimation.py`.

The Zolix ZC300 R axis is calibrated at 0.00125 deg/step, i.e.

    steps_per_degree = 1 / 0.00125 = 800

The original `auto_align.py` used a hardcoded 100 pulses/degree, which
under-rotated every correction by 8x and is why the sample never converged.
This value is exposed as a first-class, configurable constant so it can be
re-verified against hardware.

Usage:
    from rotation_controller import RotationController

    rot = RotationController(stage, axis="r")
    error, correction = rot.error_and_correction(angle_deg=115.0)
    rot.rotate_relative(correction)   # blocks until the axis stops
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Zolix ZC300 R axis: 0.00125 deg/step.
STAGE_ROTATION_STEP_DEG = 0.00125
DEFAULT_STEPS_PER_DEGREE = 1.0 / STAGE_ROTATION_STEP_DEG  # 800.0

# A correction smaller than this (deg) is treated as "nothing to do" so the
# controller cannot spin indefinitely on sub-step noise.
MIN_CORRECTION_DEG = 1.0 / DEFAULT_STEPS_PER_DEGREE  # 0.00125 deg = 1 step


class RotationError(Exception):
    """Raised when a stage rotation cannot be carried out."""


class RotationController:
    """
    Converts degree corrections into rotation-axis stage moves.

    Parameters
    ----------
    stage : stage.Stage
        Any Stage implementation (ZolixZC300, SimulatedStage, ...).
    axis : str
        Logical rotation axis name (default "r" on the Zolix).
    steps_per_degree : float, optional
        Stage steps (pulses) per degree of rotation. Defaults to the Zolix
        ZC300 calibration (800). Pass `--steps-per-degree` to override.
    rotation_sign : float
        +1 or -1. Flip if a positive stage move rotates the sample the
        opposite way in the camera image (hardware calibration knob).
    """

    def __init__(
        self,
        stage,
        axis: str = "r",
        steps_per_degree: Optional[float] = None,
        rotation_sign: float = 1.0,
    ) -> None:
        self.stage = stage
        self.axis = axis
        self.steps_per_degree = (
            float(steps_per_degree) if steps_per_degree else DEFAULT_STEPS_PER_DEGREE
        )
        if self.steps_per_degree <= 0:
            raise ValueError("steps_per_degree must be positive")
        if rotation_sign not in (1.0, -1.0):
            raise ValueError("rotation_sign must be +1.0 or -1.0")
        self.rotation_sign = float(rotation_sign)
        # Feed-through for logging / status.
        self.last_correction_deg = 0.0
        self.last_correction_steps = 0

    # ------------------------------------------------------------------
    # Geometry / conversion
    # ------------------------------------------------------------------
    def degrees_to_steps(self, angle_deg: float) -> int:
        """Convert an angle (deg) into signed rotation-axis pulses."""
        return int(round(angle_deg * self.steps_per_degree))

    def error_and_correction(
        self, angle_deg: float, target: Optional[str] = None
    ) -> Tuple[float, float, str]:
        """
        Compute the error and rotation needed to square an edge angle.

        Works in "canonical" angle = angle mod 90 in [0, 90). This is
        invariant to the 180-deg line ambiguity AND to the 90-deg switch
        between a rectangle's perpendicular edges, so the returned
        correction is stable no matter which edge the estimator reports.

        ``target`` ("horizontal" | "vertical") locks which axis to align
        to; when None the nearest axis is chosen. Returns
        ``(error_deg, correction_deg, axis)`` where ``error == |correction|``
        (both in [0, 45]).
        """
        a = float(angle_deg) % 90.0  # canonical angle in [0, 90)

        if target == "vertical":
            correction = 90.0 - a
        elif target == "horizontal":
            correction = -a
        elif a <= 45.0:
            correction = -a
            target = "horizontal"
        else:
            correction = 90.0 - a
            target = "vertical"

        error = min(a, 90.0 - a)
        return error, correction, target

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def rotate_relative(
        self, angle_deg: float, wait: bool = True, timeout: float = 30.0
    ) -> int:
        """
        Rotate the sample by ``angle_deg`` (blocking by default).

        Returns the number of pulses moved (0 if the correction was too
        small to move even one step).
        """
        angle_deg = float(angle_deg) * self.rotation_sign
        steps = self.degrees_to_steps(angle_deg)
        self.last_correction_deg = angle_deg
        self.last_correction_steps = steps

        if abs(steps) < 1:
            logger.info(
                "Correction %+.4f deg is below one stage step "
                "(%.4f deg/step); skipping move",
                angle_deg,
                1.0 / self.steps_per_degree,
            )
            return 0

        logger.info(
            "Rotating %s axis by %+.3f deg (%d pulses)",
            self.axis,
            angle_deg,
            steps,
        )
        try:
            self.stage.move_relative(
                self.axis,
                int(steps),
                wait=wait,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - hardware path
            logger.error("Stage rotation failed: %s", exc)
            raise RotationError(f"Stage rotation failed: {exc}") from exc
        return int(steps)
