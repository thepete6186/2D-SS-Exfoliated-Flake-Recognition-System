#!/usr/bin/env python3
"""
Auto-Align Controller - Automatic sample edge alignment using Canny + Hough.

This module provides automatic coarse alignment of the sample by:
1. Capturing a frame from the camera
2. Detecting edges using Canny edge detection
3. Estimating dominant edge angle using Hough line detection
4. Computing rotation correction needed
5. Moving the R (rotation) axis iteratively until aligned

The alignment is coarse (not final flake detection) and works with any
camera backend that provides frames and any stage with a rotation axis.

Usage:
    from auto_align import AutoAlignController

    controller = AutoAlignController(camera, stage)
    success = controller.align(max_iterations=10, tolerance_deg=2.0)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_CANNY_LOW = 50
DEFAULT_CANNY_HIGH = 150
DEFAULT_HOUGH_THRESHOLD = 50
DEFAULT_HOUGH_MIN_LINE_LENGTH = 100
DEFAULT_HOUGH_MAX_LINE_GAP = 10
DEFAULT_TOLERANCE_DEG = 2.0  # Stop when angle is within ±2°
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_ROTATION_STEP_DEG = 5.0  # Degrees per correction step


class AutoAlignController:
    """
    Automatic sample alignment controller.

    Coordinates camera capture, edge detection, angle estimation,
    and stage rotation to align the sample with X/Y axes.
    """

    def __init__(
        self,
        camera,
        stage,
        rotation_axis: str = "r",
        canny_low: int = DEFAULT_CANNY_LOW,
        canny_high: int = DEFAULT_CANNY_HIGH,
        hough_threshold: int = DEFAULT_HOUGH_THRESHOLD,
        hough_min_line_length: int = DEFAULT_HOUGH_MIN_LINE_LENGTH,
        hough_max_line_gap: int = DEFAULT_HOUGH_MAX_LINE_GAP,
    ):
        """
        Initialize the alignment controller.

        Parameters
        ----------
        camera : Camera object with capture() method
            Camera instance (SmartCamCamera, MMCoreCamera, etc.)
        stage : Stage object with move_relative() and get_position()
            Stage instance (ZolixZC300, SimulatedStage, etc.)
        rotation_axis : str
            Name of the rotation axis (default "r" for Zolix R axis)
        canny_low : int
            Lower threshold for Canny edge detection
        canny_high : int
            Upper threshold for Canny edge detection
        hough_threshold : int
            Minimum number of intersections for Hough line detection
        hough_min_line_length : int
            Minimum line length for Hough detection
        hough_max_line_gap : int
            Maximum gap between line segments for Hough detection
        """
        self.camera = camera
        self.stage = stage
        self.rotation_axis = rotation_axis

        # Edge detection parameters
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

        # Alignment state
        self.current_angle = None
        self.target_angle = 0.0
        self.iterations = 0
        self.converged = False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a frame from the camera.

        Returns
        -------
        frame : np.ndarray or None
            BGR frame from camera, or None if capture failed
        """
        try:
            frame = self.camera.capture()
            if frame is None:
                logger.error("Failed to capture frame from camera")
                return None

            # Convert RGB to BGR for OpenCV processing
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame

            return frame_bgr

        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None

    def estimate_edge_angle(self, frame: np.ndarray) -> Optional[float]:
        """
        Estimate the dominant edge angle using Canny + Hough line detection.

        Parameters
        ----------
        frame : np.ndarray
            BGR input frame

        Returns
        -------
        angle_deg : float or None
            Dominant edge angle in degrees (0-180), or None if detection failed
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Canny edge detection
            edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

            # Hough line detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=self.hough_threshold,
                minLineLength=self.hough_min_line_length,
                maxLineGap=self.hough_max_line_gap,
            )

            if lines is None or len(lines) == 0:
                logger.warning("No lines detected by Hough transform")
                return None

            # Calculate angles for all detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate line angle
                dx = x2 - x1
                dy = y2 - y1

                if dx == 0 and dy == 0:
                    continue

                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)

                # Normalize to 0-180 range
                if angle_deg < 0:
                    angle_deg += 180

                angles.append(angle_deg)

            if not angles:
                logger.warning("No valid angles calculated from lines")
                return None

            # Find the dominant angle using histogram
            # Use 10-degree bins
            hist, bins = np.histogram(angles, bins=18, range=(0, 180))
            dominant_bin = np.argmax(hist)
            dominant_angle = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2

            logger.info(f"Detected {len(angles)} lines, dominant angle: {dominant_angle:.1f}°")
            return dominant_angle

        except Exception as e:
            logger.error(f"Angle estimation failed: {e}")
            return None

    def compute_correction(self, current_angle: float) -> float:
        """
        Compute the rotation correction needed to align the sample.

        The goal is to align the dominant edge to either horizontal (0°/180°)
        or vertical (90°), whichever is closer.

        Parameters
        ----------
        current_angle : float
            Current detected edge angle in degrees (0-180)

        Returns
        -------
        correction_deg : float
            Rotation correction in degrees (positive = rotate CCW)
        """
        # Find nearest horizontal or vertical alignment
        # Horizontal: 0° or 180° (use 0°)
        # Vertical: 90°

        error_to_horizontal = min(current_angle, 180 - current_angle)
        error_to_vertical = abs(current_angle - 90)

        # Choose whichever is closer
        if error_to_horizontal < error_to_vertical:
            # Align to horizontal (0°)
            if current_angle < 90:
                correction = -current_angle  # Rotate clockwise
            else:
                correction = 180 - current_angle  # Rotate counter-clockwise
        else:
            # Align to vertical (90°)
            correction = 90 - current_angle

        # Normalize correction to [-90, 90]
        if correction > 90:
            correction -= 180
        elif correction < -90:
            correction += 180

        logger.info(
            f"Current angle: {current_angle:.1f}°, "
            f"error to horizontal: {error_to_horizontal:.1f}°, "
            f"error to vertical: {error_to_vertical:.1f}°, "
            f"correction: {correction:.1f}°"
        )

        return correction

    def apply_rotation(self, correction_deg: float) -> bool:
        """
        Apply rotation correction to the stage.

        Parameters
        ----------
        correction_deg : float
            Rotation correction in degrees

        Returns
        -------
        success : bool
            True if rotation was successful
        """
        try:
            # Convert degrees to pulses (assuming 360° = 36000 pulses, i.e., 100 pulses/degree)
            # This is a common configuration for stepper motors
            pulses_per_degree = 100.0
            correction_pulses = int(correction_deg * pulses_per_degree)

            logger.info(f"Moving {self.rotation_axis} axis by {correction_deg:.1f}° ({correction_pulses} pulses)")

            # Move the rotation axis
            self.stage.move_relative(
                self.rotation_axis,
                correction_pulses,
                wait=True,
                timeout=60.0
            )

            return True

        except Exception as e:
            logger.error(f"Rotation failed: {e}")
            return False

    def align(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance_deg: float = DEFAULT_TOLERANCE_DEG,
        rotation_step_deg: float = DEFAULT_ROTATION_STEP_DEG,
        progress_callback=None
    ) -> Tuple[bool, str]:
        """
        Perform automatic alignment of the sample.

        Parameters
        ----------
        max_iterations : int
            Maximum number of alignment iterations
        tolerance_deg : float
            Stop when angle is within ±tolerance_deg of target
        rotation_step_deg : float
            Maximum rotation per iteration (scales down as we converge)
        progress_callback : callable, optional
            Callback function called after each iteration with:
            (iteration, angle, correction, converged) -> None

        Returns
        -------
        success : bool
            True if alignment converged within tolerance
        message : str
            Status message describing the result
        """
        logger.info("Starting auto-alignment...")
        self.iterations = 0
        self.converged = False

        for i in range(max_iterations):
            self.iterations = i + 1

            # Step 1: Capture frame
            logger.info(f"Iteration {self.iterations}/{max_iterations}: Capturing frame...")
            frame = self.capture_frame()
            if frame is None:
                return False, f"Failed to capture frame at iteration {self.iterations}"

            # Step 2: Estimate angle
            logger.info(f"Iteration {self.iterations}: Estimating edge angle...")
            angle = self.estimate_edge_angle(frame)
            if angle is None:
                return False, f"Failed to detect edge at iteration {self.iterations}"

            self.current_angle = angle

            # Step 3: Check if converged
            error_to_nearest_axis = min(angle % 90, 90 - angle % 90)
            if error_to_nearest_axis < tolerance_deg:
                self.converged = True
                logger.info(
                    f"Alignment converged after {self.iterations} iterations. "
                    f"Final angle: {angle:.1f}°, error: {error_to_nearest_axis:.1f}°"
                )
                if progress_callback:
                    progress_callback(self.iterations, angle, 0.0, True)
                return True, f"Aligned in {self.iterations} iterations (angle: {angle:.1f}°)"

            # Step 4: Compute correction
            correction = self.compute_correction(angle)

            # Scale down correction as we get closer (proportional control)
            correction_magnitude = abs(correction)
            if correction_magnitude > rotation_step_deg:
                correction = np.sign(correction) * rotation_step_deg

            # Step 5: Apply rotation
            logger.info(f"Iteration {self.iterations}: Applying correction of {correction:.1f}°...")
            success = self.apply_rotation(correction)
            if not success:
                return False, f"Rotation failed at iteration {self.iterations}"

            # Call progress callback
            if progress_callback:
                progress_callback(self.iterations, angle, correction, False)

        # Max iterations reached without convergence
        logger.warning(
            f"Alignment did not converge after {max_iterations} iterations. "
            f"Final angle: {self.current_angle:.1f}°"
        )
        return False, f"Did not converge after {max_iterations} iterations (angle: {self.current_angle:.1f}°)"

    def get_status(self) -> dict:
        """
        Get current alignment status.

        Returns
        -------
        status : dict
            Dictionary with alignment status information
        """
        return {
            "iterations": self.iterations,
            "current_angle": self.current_angle,
            "target_angle": self.target_angle,
            "converged": self.converged,
        }


def test_alignment(camera, stage, max_iterations: int = 5):
    """
    Test the alignment controller with a camera and stage.

    Parameters
    ----------
    camera : Camera object
        Camera instance
    stage : Stage object
        Stage instance
    max_iterations : int
        Maximum alignment iterations for testing
    """
    def progress_callback(iteration, angle, correction, converged):
        status = "CONVERGED" if converged else f"correction: {correction:.1f}°"
        print(f"  Iteration {iteration}: angle={angle:.1f}°, {status}")

    controller = AutoAlignController(camera, stage)

    print("Starting auto-alignment test...")
    print(f"  Tolerance: {DEFAULT_TOLERANCE_DEG}°")
    print(f"  Max iterations: {max_iterations}")
    print()

    success, message = controller.align(
        max_iterations=max_iterations,
        tolerance_deg=DEFAULT_TOLERANCE_DEG,
        progress_callback=progress_callback
    )

    print()
    print(f"Result: {'SUCCESS' if success else 'FAILED'}")
    print(f"Message: {message}")
    print(f"Status: {controller.get_status()}")

    return success, message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Auto-Align Controller Test")
    parser.add_argument("--port", default="COM3", help="Stage serial port")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatedStage")
    parser.add_argument("--iterations", type=int, default=5, help="Max alignment iterations")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Import camera and stage
    from camera_local.smartcam_camera import SmartCamCamera
    from stage.zolix_zc300 import ZolixZC300
    from stage.simulated import SimulatedStage

    print("Initializing camera...")
    camera = SmartCamCamera(camera_index=0)
    if not camera.connect():
        print("Failed to connect to camera")
        exit(1)

    print("Initializing stage...")
    if args.simulate:
        stage = SimulatedStage(axes=("x", "y", "r"), speed_pps=2000.0)
    else:
        stage = ZolixZC300(port=args.port, axes=("x", "y", "r"))
        stage.connect()

    try:
        test_alignment(camera, stage, max_iterations=args.iterations)
    finally:
        camera.disconnect()
        if not args.simulate:
            stage.disconnect()