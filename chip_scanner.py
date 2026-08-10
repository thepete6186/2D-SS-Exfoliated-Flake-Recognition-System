#!/usr/bin/env python3
"""
Chip Scanner - Manual area scanning with snake pattern.

This module provides manual area scanning functionality:
1. User manually moves the stage to a start position
2. User manually moves the stage to an end position
3. System computes a snake pattern path covering the area
4. Stage moves frame-by-frame with 5% overlap between frames

The scan area is defined by two corner points (start and end) in
stage coordinates (pulses). The system computes a grid of positions
and moves the stage in a snake pattern, capturing a frame at each
position.

Usage:
    from chip_scanner import ChipScanner

    scanner = ChipScanner(stage, camera)
    scanner.set_start_position(x, y)
    scanner.set_end_position(x, y)
    scanner.start_scan(overlap_percent=5.0)
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Stage calibration constants
STAGE_STEP_SIZE_UM = 0.625  # micrometers per step for X, Y axes
STAGE_ROTATION_STEP_DEG = 0.00125  # degrees per step for R axis

# Camera field of view in pulses (approximate for 3840x2160 at 5x zoom)
# 1 mm ≈ 2650 pixels, so 3840 pixels ≈ 1.45 mm ≈ 2320 pulses
# 2160 pixels ≈ 0.815 mm ≈ 1304 pulses
DEFAULT_FOV_X_PULSES = 2320  # Camera X field of view in stage pulses
DEFAULT_FOV_Y_PULSES = 1304  # Camera Y field of view in stage pulses


class ChipScanner:
    """
    Manual area scanner with snake pattern movement.

    The user defines a rectangular scan area by setting start and end
    positions. The scanner computes a grid of positions with configurable
    overlap and moves the stage in a snake pattern, capturing a frame
    at each position.
    """

    def __init__(
        self,
        stage,
        camera,
        fov_x_pulses: float = DEFAULT_FOV_X_PULSES,
        fov_y_pulses: float = DEFAULT_FOV_Y_PULSES,
    ):
        """
        Initialize the scanner.

        Parameters
        ----------
        stage : Stage object
            Stage instance with move_absolute() and get_position()
        camera : Camera object
            Camera instance with capture() method
        fov_x_pulses : float
            Camera X field of view in stage pulses
        fov_y_pulses : float
            Camera Y field of view in stage pulses
        """
        self.stage = stage
        self.camera = camera
        self.fov_x_pulses = fov_x_pulses
        self.fov_y_pulses = fov_y_pulses

        # Scan area
        self.start_pos: Optional[Tuple[float, float]] = None  # (x, y) in pulses
        self.end_pos: Optional[Tuple[float, float]] = None  # (x, y) in pulses

        # Scan state
        self.is_scanning = False
        self._stop_event = threading.Event()
        self.current_position = 0
        self.total_positions = 0
        self.captured_frames: List[np.ndarray] = []
        self.captured_positions: List[Tuple[float, float]] = []

        # Progress callback
        self.progress_callback: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def set_start_position(self, x: float, y: float):
        """Set the start position (corner 1) in stage pulses."""
        self.start_pos = (x, y)
        logger.info(f"Start position set: ({x}, {y}) pulses")

    def set_end_position(self, x: float, y: float):
        """Set the end position (corner 2) in stage pulses."""
        self.end_pos = (x, y)
        logger.info(f"End position set: ({x}, {y}) pulses")

    def get_current_position(self) -> Tuple[float, float]:
        """Get current stage position in pulses."""
        status = self.stage.get_status()
        pos = status["position"]
        return (pos.get("x", 0.0), pos.get("y", 0.0))

    def clear_positions(self):
        """Clear start and end positions."""
        self.start_pos = None
        self.end_pos = None
        logger.info("Scan positions cleared")

    # ------------------------------------------------------------------
    # Path Planning
    # ------------------------------------------------------------------

    def compute_snake_path(
        self,
        overlap_percent: float = 5.0
    ) -> List[Tuple[float, float]]:
        """
        Compute a snake pattern path covering the scan area.

        Parameters
        ----------
        overlap_percent : float
            Percentage overlap between adjacent frames (default 5%)

        Returns
        -------
        path : list of (x, y) tuples
            List of stage positions in pulses forming a snake pattern
        """
        if self.start_pos is None or self.end_pos is None:
            raise ValueError("Both start and end positions must be set")

        x1, y1 = self.start_pos
        x2, y2 = self.end_pos

        # Ensure x1 < x2 and y1 < y2
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Calculate step sizes with overlap
        # Effective step = FOV * (1 - overlap/100)
        overlap_factor = 1.0 - overlap_percent / 100.0
        step_x = self.fov_x_pulses * overlap_factor
        step_y = self.fov_y_pulses * overlap_factor

        # Calculate number of steps
        range_x = x_max - x_min
        range_y = y_max - y_min

        num_steps_x = max(1, int(np.ceil(range_x / step_x)))
        num_steps_y = max(1, int(np.ceil(range_y / step_y)))

        # Generate grid positions
        x_positions = np.linspace(x_min, x_max, num_steps_x)
        y_positions = np.linspace(y_min, y_max, num_steps_y)

        # Build snake pattern
        path = []
        for i, y in enumerate(y_positions):
            if i % 2 == 0:
                # Even row: left to right
                row_x = x_positions
            else:
                # Odd row: right to left (snake)
                row_x = x_positions[::-1]

            for x in row_x:
                path.append((float(x), float(y)))

        logger.info(
            f"Snake path computed: {len(path)} positions "
            f"({num_steps_x} x {num_steps_y} grid, "
            f"{overlap_percent}% overlap)"
        )

        return path

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def start_scan(
        self,
        overlap_percent: float = 5.0,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Start the snake pattern scan in a background thread.

        Parameters
        ----------
        overlap_percent : float
            Percentage overlap between frames (default 5%)
        progress_callback : callable, optional
            Callback function called with (current, total, position) after
            each frame capture

        Returns
        -------
        success : bool
            True if scan started successfully
        """
        if self.start_pos is None or self.end_pos is None:
            logger.error("Cannot start scan: positions not set")
            return False

        if self.is_scanning:
            logger.warning("Scan already in progress")
            return False

        self.progress_callback = progress_callback
        self._stop_event.clear()
        self.captured_frames = []
        self.captured_positions = []

        # Compute path
        try:
            path = self.compute_snake_path(overlap_percent)
        except ValueError as e:
            logger.error(f"Path computation failed: {e}")
            return False

        self.total_positions = len(path)
        self.current_position = 0

        # Start scan in background thread
        thread = threading.Thread(
            target=self._scan_worker,
            args=(path,),
            daemon=True
        )
        thread.start()

        self.is_scanning = True
        logger.info(f"Scan started: {self.total_positions} positions")
        return True

    def stop_scan(self):
        """Stop the current scan."""
        self._stop_event.set()
        self.is_scanning = False
        logger.info("Scan stopped by user")

    def _scan_worker(self, path: List[Tuple[float, float]]):
        """
        Worker thread for scanning.

        Moves the stage to each position in the path and captures a frame.
        """
        try:
            for i, (x, y) in enumerate(path):
                if self._stop_event.is_set():
                    logger.info("Scan aborted by user")
                    break

                self.current_position = i + 1

                # Move stage to position
                logger.info(f"Moving to position {i+1}/{len(path)}: ({x:.0f}, {y:.0f})")
                self.stage.move_absolute("x", x, wait=True, timeout=60.0)
                self.stage.move_absolute("y", y, wait=True, timeout=60.0)

                # Small delay for stage to settle
                time.sleep(0.1)

                # Capture frame
                frame = self.camera.capture()
                if frame is not None:
                    self.captured_frames.append(frame)
                    self.captured_positions.append((x, y))
                    logger.info(f"Captured frame {i+1}/{len(path)}")
                else:
                    logger.warning(f"Failed to capture frame at position {i+1}")

                # Call progress callback
                if self.progress_callback:
                    self.progress_callback(i + 1, len(path), (x, y))

        except Exception as e:
            logger.error(f"Scan error: {e}", exc_info=True)
        finally:
            self.is_scanning = False
            logger.info(
                f"Scan complete: {len(self.captured_frames)} frames captured "
                f"out of {len(path)} positions"
            )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_results(self) -> Dict:
        """
        Get scan results.

        Returns
        -------
        results : dict
            Dictionary with scan results
        """
        return {
            "frames": self.captured_frames,
            "positions": self.captured_positions,
            "total_positions": self.total_positions,
            "captured_count": len(self.captured_frames),
            "is_scanning": self.is_scanning,
        }

    def save_frames(self, output_dir: str, prefix: str = "frame"):
        """
        Save all captured frames to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save frames to
        prefix : str
            Prefix for frame filenames

        Returns
        -------
        count : int
            Number of frames saved
        """
        import os
        import cv2

        os.makedirs(output_dir, exist_ok=True)

        for i, (frame, pos) in enumerate(zip(self.captured_frames, self.captured_positions)):
            # Convert RGB to BGR for saving
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                bgr = frame

            filename = f"{prefix}_{i:04d}_x{pos[0]:.0f}_y{pos[1]:.0f}.png"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, bgr)

        logger.info(f"Saved {len(self.captured_frames)} frames to {output_dir}")
        return len(self.captured_frames)


def test_scanner(stage, camera, overlap_percent: float = 5.0):
    """
    Test the scanner with a camera and stage.

    Parameters
    ----------
    stage : Stage object
        Stage instance
    camera : Camera object
        Camera instance
    overlap_percent : float
        Overlap percentage
    """
    scanner = ChipScanner(stage, camera)

    # Get current position as start
    start = scanner.get_current_position()
    scanner.set_start_position(*start)

    # Set end position 10mm away in both axes
    # 10mm = 10000 µm = 16000 pulses (at 0.625 µm/step)
    end = (start[0] + 16000, start[1] + 16000)
    scanner.set_end_position(*end)

    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Overlap: {overlap_percent}%")

    def progress_callback(current, total, position):
        print(f"  Position {current}/{total}: ({position[0]:.0f}, {position[1]:.0f})")

    print("Starting scan...")
    success = scanner.start_scan(
        overlap_percent=overlap_percent,
        progress_callback=progress_callback
    )

    if not success:
        print("Failed to start scan")
        return

    # Wait for scan to complete
    while scanner.is_scanning:
        time.sleep(0.5)

    results = scanner.get_results()
    print(f"\nScan complete: {results['captured_count']} frames captured")
    print(f"Total positions: {results['total_positions']}")

    return scanner


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chip Scanner Test")
    parser.add_argument("--port", default="COM3", help="Stage serial port")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatedStage")
    parser.add_argument("--overlap", type=float, default=5.0, help="Overlap percentage")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

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
        test_scanner(stage, camera, overlap_percent=args.overlap)
    finally:
        camera.disconnect()
        if not args.simulate:
            stage.disconnect()