#!/usr/bin/env python3
"""
Auto-Align Controller - orchestration for sample edge alignment.

Responsibilities (kept deliberately thin):
            1. capture a frame from the camera,
    2. estimate the dominant edge angle (edge_estimation.EdgeAngleEstimator),
    3. compute the correction (rotation_controller.RotationController),
    4. move the rotation axis (blocking),
    5. repeat until the angle is within tolerance.

The image-processing details live in `edge_estimation.py` and the
motion-control details live in `rotation_controller.py`; this module only
ties them together, so neither is coupled to the other.

Camera-agnostic: any object with a ``capture()`` method returning an ndarray
works. `frame_is_rgb` tells the estimator the channel order (SmartCamCamera
returns RGB; OpenCV/webcam frames are BGR).

Usage:
    from auto_align import AutoAlignController

    controller = AutoAlignController(camera, stage)
    success, message = controller.align(max_iterations=10, tolerance_deg=2.0)

CLI:
    python auto_align.py --simulate --camera synthetic   # no hardware
    python auto_align.py --port COM3                     # real camera+stage
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

from edge_estimation import EdgeAngleEstimator
from rotation_controller import (
    DEFAULT_STEPS_PER_DEGREE,
    RotationController,
    RotationError,
)

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_TOLERANCE_DEG = 2.0
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_ROTATION_STEP_DEG = 45.0  # Safety cap (full R-axis travel). Each pass applies the FULL correction; the loop is only a verification/retry bound.

# After the first single-shot correction, a post-correction verification
# capture may still report a small residual angle due to estimation noise.
# Residuals within DEFAULT_VERIFY_TOLERANCE_FACTOR * tolerance_deg are
# accepted as converged (NOT re-corrected). Only if the residual exceeds
# this band does the system retry with another full correction.
DEFAULT_VERIFY_TOLERANCE_FACTOR = 5.0

# Maximum number of full corrections to attempt (1 initial + N retries).
# Prevents unbounded correction loops when estimation noise is high.
DEFAULT_MAX_CORRECTIONS = 2



class AutoAlignController:
    """
    Coordinates camera capture, edge estimation, and stage rotation to
    align the sample with the X/Y axes (coarse alignment).

    Parameters
    ----------
    camera : object
        Anything with a ``capture() -> np.ndarray`` method.
    stage : stage.Stage
        ZolixZC300, SimulatedStage, or compatible.
    rotation_axis : str
        Logical rotation axis name (default "r").
    steps_per_degree : float, optional
        Stage pulses per degree for the rotation axis. Default 800 (Zolix).
    frame_is_rgb : bool
        True if camera.capture() returns RGB (e.g. SmartCamCamera).
    estimator : EdgeAngleEstimator, optional
        Override the default estimator (tests can inject a stub).
    rotation : RotationController, optional
        Override the default rotation controller (tests can inject a stub).
    """

    def __init__(
        self,
        camera,
        stage,
        rotation_axis: str = "r",
        steps_per_degree: Optional[float] = None,
        frame_is_rgb: bool = True,
        estimator: Optional[EdgeAngleEstimator] = None,
        rotation: Optional[RotationController] = None,
    ) -> None:
        self.camera = camera
        self.stage = stage
        self.rotation_axis = rotation_axis
        self.frame_is_rgb = bool(frame_is_rgb)

        if estimator is None:
            estimator = EdgeAngleEstimator(
                channel_order="rgb" if frame_is_rgb else "bgr"
            )
        if rotation is None:
            rotation = RotationController(
                stage, axis=rotation_axis, steps_per_degree=steps_per_degree
            )
        self.estimator = estimator
        self.rotation = rotation

        # Alignment state
        self.current_angle: Optional[float] = None
        self.current_error: float = 0.0
        self.iterations = 0
        self.converged = False
        # Locked target axis ("horizontal" | "vertical"), chosen on the
        # first measurement to keep the correction sign stable.
        self._target_axis: Optional[str] = None

    # ------------------------------------------------------------------
    # Step 1: capture
    # ------------------------------------------------------------------
    def capture_frame(self):
        """Capture one frame from the camera (or None on failure)."""
        try:
            frame = self.camera.capture()
        except Exception as exc:  # pragma: no cover - hardware path
            logger.error("Frame capture failed: %s", exc)
            return None
        if frame is None:
            logger.error("Failed to capture frame from camera")
        return frame

    # ------------------------------------------------------------------
    # Step 2: estimate
    # ------------------------------------------------------------------
    def estimate_edge_angle(self, frame) -> Optional[float]:
        """Estimate the dominant edge angle (deg, [0, 180)) or None."""
        result = self.estimator.detect(frame)
        if result is None:
            logger.warning("No edge detected in frame")
            return None
        logger.info(
            "Detected edge: angle=%.2f deg, length=%.1f px, mode=%s",
            result.angle_deg,
            result.length_px,
            result.mode,
        )
        return result.angle_deg

    # ------------------------------------------------------------------
    # Step 3: compute correction
    # ------------------------------------------------------------------
    def compute_correction(self, current_angle: float) -> float:
        """Signed rotation (deg) to square the current edge angle.

        On the **first** call the nearest horizontal/vertical axis is chosen
        and LOCKED to ``self._target_axis``.  All subsequent calls reuse the
        locked target, preventing the correction sign from oscillating when
        the angle hovers near 45° (or near 135°).

        The correction is computed in full [0, 180) space (not canonical
        mod-90), so locking to "vertical" correctly handles angles like 92°
        without producing spurious 90°-wide corrections.
        """
        if self._target_axis is None:
            # First measurement: pick nearest axis in canonical space and lock.
            _, _, axis = self.rotation.error_and_correction(
                current_angle, target=None
            )
            self._target_axis = axis

        # Compute correction toward the locked target in canonical
        # (mod-90) space. This preserves the stability of the original
        # error_and_correction() while preventing the target-axis flip-flop
        # that causes oscillation near 45 degrees.
        a = float(current_angle) % 90.0

        if self._target_axis == "vertical":
            correction = 90.0 - a
        else:
            correction = -a

        error = min(a, 90.0 - a)
        self.current_error = error
        logger.info(
            "Current angle: %.1f deg, error to %s: %.2f deg, "
            "correction: %+.1f deg",
            current_angle,
            self._target_axis,
            error,
            correction,
        )
        return correction

    # ------------------------------------------------------------------
    # Steps 4-5: rotate + repeat
    # ------------------------------------------------------------------
    def apply_rotation(self, correction_deg: float) -> bool:
        """Apply a blocking rotation correction; True on success."""
        try:
            self.rotation.rotate_relative(correction_deg, wait=True, timeout=60.0)
            return True
        except RotationError as exc:
            logger.error("Rotation failed: %s", exc)
            return False

    def align(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
        rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
        progress_callback: Optional[Callable[[int, float, float, bool], None]] = None,
    ) -> Tuple[bool, str]:
        """
        Run the measure -> correct -> verify loop.

        This is a **single-shot** correction: the first iteration applies the
        FULL correction in one blocking move (no 5-deg incremental stepping).
        A subsequent verification capture confirms the result. If the
        post-correction residual is small (within a noise band), it is
        accepted as converged -- the system does NOT apply microadjustments.
        Only if the residual is very large (correction likely failed) does it
        retry with another full correction.

        Stops when the angle is within ``tolerance_deg`` of the nearest axis,
        after at most ``max_iterations`` passes. Returns ``(converged, message)``
        """
        logger.info("Starting auto-alignment (tolerance %.2f deg)", tolerance_deg)
        self.iterations = 0
        self.converged = False
        self.current_angle = None
        self.current_error = 0.0
        self._target_axis = None

        # Residuals within this band are treated as estimation noise, not as
        # alignment errors -- they are accepted without further correction.
        verify_tolerance_deg = tolerance_deg * DEFAULT_VERIFY_TOLERANCE_FACTOR
        corrections_applied = 0

        for i in range(int(max_iterations)):
            self.iterations = i + 1

            logger.info(
                "Iteration %d/%d: Capturing frame...",
                self.iterations,
                max_iterations,
            )
            frame = self.capture_frame()
            if frame is None:
                return False, f"Failed to capture frame at iteration {self.iterations}"

            logger.info("Iteration %d: Estimating edge angle...", self.iterations)
            angle = self.estimate_edge_angle(frame)
            if angle is None:
                return False, f"Failed to detect edge at iteration {self.iterations}"

            self.current_angle = angle
            correction = self.compute_correction(angle)

            # --- Check if within tight tolerance (converged) ---
            if self.current_error <= tolerance_deg:
                self.converged = True
                logger.info(
                    "Alignment converged after %d iterations "
                    "(angle %.1f deg, error %.2f deg)",
                    self.iterations,
                    angle,
                    self.current_error,
                )
                if progress_callback:
                    progress_callback(self.iterations, angle, 0.0, True)
                return (
                    True,
                    f"Aligned in {self.iterations} iterations "
                    f"(angle: {angle:.1f} deg)",
                )

            # --- Decide whether to apply a correction ---
            if corrections_applied == 0:
                # FIRST measurement: no correction has been applied yet, so a
                # residual above the tight tolerance is REAL misalignment, NOT
                # post-correction estimation noise. Always apply the full
                # correction here -- never "accept" the initial state as
                # aligned just because it is inside the noise band.
                should_correct = True
            elif self.current_error <= verify_tolerance_deg:
                # Small post-correction residual -- within the estimation-noise
                # band. Accept as converged. NO MICROADJUSTMENTS.
                self.converged = True
                logger.info(
                    "Alignment accepted (post-correction residual %.2f deg "
                    "within noise band of %.2f deg) after %d iterations",
                    self.current_error,
                    verify_tolerance_deg,
                    self.iterations,
                )
                if progress_callback:
                    progress_callback(self.iterations, angle, 0.0, True)
                return (
                    True,
                    f"Aligned in {self.iterations} iterations "
                    f"(angle: {angle:.1f} deg, residual: {self.current_error:.1f} deg)",
                )
            elif corrections_applied < DEFAULT_MAX_CORRECTIONS:
                # Large residual after correction: retry with another full correction.
                logger.warning(
                    "Post-correction error %.2f deg exceeds verification "
                    "threshold %.2f deg; retrying with full correction.",
                    self.current_error,
                    verify_tolerance_deg,
                )
                should_correct = True
            else:
                # Exhausted retries with large residual: not converged.
                logger.warning(
                    "Alignment failed: residual %.2f deg exceeds tolerance "
                    "after %d corrections.",
                    self.current_error,
                    corrections_applied,
                )
                self.converged = False
                return (
                    False,
                    f"Residual angle {self.current_error:.1f} deg too large "
                    f"after {corrections_applied} corrections "
                    f"(angle: {self.current_angle:.1f} deg)",
                )
            # --- Apply the full correction ---
            # Safety cap only. With single-shot alignment the full correction
            # (<= 45 deg) is normally applied in one move; rotation_step_deg
            # only guards against a pathological >45-deg correction.
            if abs(correction) > abs(rotation_step_deg):
                correction = (1.0 if correction > 0 else -1.0) * abs(rotation_step_deg)

            logger.info(
                "Iteration %d: Applying correction of %+.1f deg...",
                self.iterations,
                correction,
            )
            if not self.apply_rotation(correction):
                return False, f"Rotation failed at iteration {self.iterations}"

            corrections_applied += 1

            if progress_callback:
                progress_callback(self.iterations, angle, correction, False)

        logger.warning(
            "Alignment did not converge after %d iterations (angle: %s deg)",
            max_iterations,
            f"{self.current_angle:.1f}" if self.current_angle is not None else "n/a",
        )
        return (
            False,
            f"Did not converge after {max_iterations} iterations "
            f"(angle: {self.current_angle:.1f} deg)",
        )


    def get_status(self) -> dict:
        """Current alignment status."""
        return {
            "iterations": self.iterations,
            "current_angle": self.current_angle,
            "current_error": self.current_error,
            "converged": self.converged,
        }


def test_alignment(camera, stage, max_iterations: int = 5):
    """Run a headless alignment and print a per-iteration trace."""
    def progress_callback(iteration, angle, correction, converged):
        state = "CONVERGED" if converged else f"correction: {correction:.1f} deg"
        print(f"  Iteration {iteration}: angle={angle:.1f} deg, {state}")

    controller = AutoAlignController(camera, stage)

    print("Starting auto-alignment test...")
    print(f"  Tolerance: {DEFAULT_TOLERANCE_DEG} deg")
    print(f"  Max iterations: {max_iterations}")
    print()

    success, message = controller.align(
        max_iterations=max_iterations,
        tolerance_deg=DEFAULT_TOLERANCE_DEG,
        progress_callback=progress_callback,
    )

    print()
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Message: {message}")
    print(f"Status: {controller.get_status()}")
    return success, message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Align Controller")
    parser.add_argument("--port", default="COM3", help="Stage serial port")
    parser.add_argument("--simulate", action="store_true",
                        help="Use SimulatedStage (no hardware)")
    parser.add_argument("--camera", choices=["smartcam", "synthetic", "opencv"],
                        default="smartcam",
                        help="Camera backend (default smartcam)")
    parser.add_argument("--opencv-index", type=int, default=0,
                        help="Camera index for --camera opencv")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Max alignment iterations")
    parser.add_argument("--sample-angle", type=float, default=20.0,
                        help="Starting misalignment (deg) for synthetic mode")
    parser.add_argument("--steps-per-degree", type=float, default=None,
                        help=f"Rotation pulses per degree "
                             f"(default {DEFAULT_STEPS_PER_DEGREE:g})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from stage.simulated import SimulatedStage
    from stage.zolix_zc300 import ZolixZC300

    # Build stage.
    if args.simulate:
        stage = SimulatedStage(
            axes=("x", "y", "r"),
            speed_pps=2000.0,
            limits={
                "x": (-100000, 100000),
                "y": (-100000, 100000),
                "r": (-36000, 36000),
            },
        )
        stage.connect()
    else:
        stage = ZolixZC300(port=args.port, axes=("x", "y", "r"))
        stage.connect()

    # Build camera.
    if args.camera == "synthetic":
        from edge_estimation import SyntheticRotatingCamera

        spd = args.steps_per_degree or DEFAULT_STEPS_PER_DEGREE
        camera = SyntheticRotatingCamera(steps_per_degree=spd)
        camera.bind_stage(stage, axis="r")
        # Park the stage so the sample starts misaligned by sample-angle deg.
        stage.move_absolute("r", args.sample_angle * spd, wait=True, timeout=30.0)
    elif args.camera == "opencv":
        import cv2 as _cv2

        cap = _cv2.VideoCapture(args.opencv_index)

        class OpenCVCamera:
            def capture(self):
                ok, frame = cap.read()
                return frame if ok else None

        camera = OpenCVCamera()
    else:  # smartcam (lab camera)
        from camera_local.smartcam_camera import SmartCamCamera

        camera = SmartCamCamera(camera_index=0)
        if not camera.connect():
            print("Failed to connect to camera")
            if not args.simulate:
                stage.disconnect()
            exit(1)

    try:
        test_alignment(camera, stage, max_iterations=args.iterations)
    finally:
        if args.camera == "smartcam":
            camera.disconnect()
        elif args.camera == "opencv":
            cap.release()
        if not args.simulate:
            stage.disconnect()