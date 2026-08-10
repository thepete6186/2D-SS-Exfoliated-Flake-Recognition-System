#!/usr/bin/env python3
"""
Black Edge Detector - Glare-resistant sub-pixel edge detection.

Detects the dark/black physical transition border at the plate boundary
using brightness minima to locate the edge between metallic red surface
and substrate.

Features:
- 1D brightness profile scanning with Gaussian smoothing
- Multi-row sampling for robust line fitting
- Sub-pixel edge detection via local minimum interpolation
- Automatic outlier rejection
- Returns edge angle (theta) and anchor point for stage alignment

Usage:
    from black_edge_detector import detect_straight_black_edge

    # Detect edge in a BGR frame
    theta, anchor_point = detect_straight_black_edge(frame)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Constants
MAX_BLACK_VALUE = 40  # Maximum intensity for valid black edge (0-255)
MIN_VALID_POINTS = 4  # Minimum valid points required for line fitting
NUM_SCAN_ROWS = 10  # Number of parallel scans across the edge
GAUSSIAN_BLUR_SIZE = 5  # Kernel size for 1D Gaussian blur (must be odd)
SCAN_MARGIN = 50  # Pixels to skip at image edges during scanning


def find_black_edge_point(
    frame: np.ndarray,
    y: int,
    x_start: int,
    x_end: int,
    max_black_value: float = MAX_BLACK_VALUE
) -> Tuple[float, bool]:
    """
    Find the black edge point along a 1D horizontal scan at row y.

    Parameters
    ----------
    frame : np.ndarray
        BGR input frame (H, W, 3)
    y : int
        Row coordinate to scan
    x_start : int
        Starting X coordinate for scan
    x_end : int
        Ending X coordinate for scan
    max_black_value : float
        Maximum intensity value to consider as valid black edge

    Returns
    -------
    edge_x : float
        X coordinate of detected edge (sub-pixel if interpolated), -1 if not found
    valid : bool
        True if valid edge was detected
    """
    h, w = frame.shape[:2]

    # Clamp coordinates to valid image bounds
    y = max(0, min(h - 1, y))
    x_start = max(0, min(w - 1, x_start))
    x_end = max(0, min(w - 1, x_end))

    # Ensure x_start < x_end
    if x_start > x_end:
        x_start, x_end = x_end, x_start

    # Check for valid scan range
    if x_end - x_start < 3:
        return -1.0, False

    try:
        # Convert to grayscale/luma (Y channel from RGB conversion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract 1D profile along the scan line
        profile = gray[y, x_start:x_end].astype(np.float32)

        # Apply 1D Gaussian blur to reduce sensor noise
        if len(profile) > GAUSSIAN_BLUR_SIZE:
            profile_smooth = cv2.GaussianBlur(
                profile.reshape(1, -1),
                (GAUSSIAN_BLUR_SIZE, 1),
                0
            ).flatten()
        else:
            profile_smooth = profile

        # Find the darkest point (local minimum)
        min_idx = np.argmin(profile_smooth)
        min_value = profile_smooth[min_idx]

        # Validate that this is a valid black border
        if min_value > max_black_value:
            logger.debug(
                f"Row {y}: Minimum value {min_value:.1f} exceeds threshold {max_black_value}"
            )
            return -1.0, False

        # Convert to original image coordinates
        edge_x = float(x_start + min_idx)

        # Optional: Sub-pixel refinement using parabolic interpolation
        # around the minimum for better accuracy
        if 1 <= min_idx < len(profile_smooth) - 1:
            # Fit parabola to three points around minimum
            y0 = profile_smooth[min_idx - 1]
            y1 = profile_smooth[min_idx]
            y2 = profile_smooth[min_idx + 1]

            # Parabolic vertex formula: x_vertex = x0 - 0.5 * (y2 - y0) / (y2 - 2*y1 + y0)
            denominator = y2 - 2 * y1 + y0
            if abs(denominator) > 1e-6:  # Avoid division by zero
                delta = 0.5 * (y2 - y0) / denominator
                # Clamp to reasonable range (-0.5 to 0.5 pixels)
                delta = max(-0.5, min(0.5, delta))
                edge_x += delta

        logger.debug(f"Row {y}: Found black edge at x={edge_x:.2f}, value={min_value:.1f}")
        return edge_x, True

    except Exception as e:
        logger.error(f"Error scanning row {y}: {e}")
        return -1.0, False


def detect_straight_black_edge(
    frame: np.ndarray,
    scan_direction: str = "horizontal",
    num_rows: int = NUM_SCAN_ROWS,
    min_valid_points: int = MIN_VALID_POINTS,
    max_black_value: float = MAX_BLACK_VALUE
) -> Tuple[Optional[float], Optional[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Detect a straight black edge by multi-row sampling and line fitting.

    Parameters
    ----------
    frame : np.ndarray
        BGR input frame (H, W, 3)
    scan_direction : str
        "horizontal" for vertical edges (scan across X), or
        "vertical" for horizontal edges (scan across Y)
    num_rows : int
        Number of parallel scan lines (5-10 recommended)
    min_valid_points : int
        Minimum number of valid edge points required for line fitting
    max_black_value : float
        Maximum intensity for valid black edge detection

    Returns
    -------
    theta : float or None
        Edge orientation angle in degrees (0-180), None if detection failed
    anchor_point : tuple or None
        (x, y) anchor point on the fitted line, None if detection failed
    edge_points : list
        List of (x, y) tuples containing detected edge points (for debugging)
    """
    h, w = frame.shape[:2]
    edge_points = []

    try:
        if scan_direction == "horizontal":
            # Scan horizontally across X at different Y rows (detect vertical edges)
            y_positions = np.linspace(
                SCAN_MARGIN,
                h - SCAN_MARGIN,
                num_rows,
                dtype=int
            )

            for y in y_positions:
                edge_x, valid = find_black_edge_point(
                    frame, y, SCAN_MARGIN, w - SCAN_MARGIN, max_black_value
                )
                if valid:
                    edge_points.append((edge_x, float(y)))

        elif scan_direction == "vertical":
            # Scan vertically across Y at different X columns (detect horizontal edges)
            x_positions = np.linspace(
                SCAN_MARGIN,
                w - SCAN_MARGIN,
                num_rows,
                dtype=int
            )

            for x in x_positions:
                # For vertical scan, we need to swap axes in find_black_edge_point
                # Extract column and scan vertically
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                profile = gray[SCAN_MARGIN:h - SCAN_MARGIN, x].astype(np.float32)

                if len(profile) > GAUSSIAN_BLUR_SIZE:
                    profile_smooth = cv2.GaussianBlur(
                        profile.reshape(1, -1),
                        (GAUSSIAN_BLUR_SIZE, 1),
                        0
                    ).flatten()
                else:
                    profile_smooth = profile

                min_idx = np.argmin(profile_smooth)
                min_value = profile_smooth[min_idx]

                if min_value <= max_black_value:
                    edge_y = float(SCAN_MARGIN + min_idx)
                    edge_points.append((float(x), edge_y))
        else:
            raise ValueError(f"Invalid scan_direction: {scan_direction}. Use 'horizontal' or 'vertical'")

        # Check if we have enough valid points
        if len(edge_points) < min_valid_points:
            logger.warning(
                f"Insufficient valid edge points: {len(edge_points)}/{min_valid_points} required"
            )
            return None, None, edge_points

        # Convert to numpy array for fitting
        points_array = np.array(edge_points, dtype=np.float32)

        # Fit line using OpenCV (L2 distance = least squares)
        # cv2.fitLine returns: [vx, vy, x0, y0] where:
        #   (vx, vy) is unit direction vector
        #   (x0, y0) is a point on the line
        line_params = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)

        vx, vy, x0, y0 = line_params.flatten()

        # Calculate angle in degrees
        # Angle from horizontal axis: atan2(vy, vx) gives angle in radians
        theta_rad = np.arctan2(vy, vx)
        theta_deg = np.degrees(theta_rad)

        # Normalize to 0-180 degree range
        if theta_deg < 0:
            theta_deg += 180

        # Anchor point
        anchor_point = (float(x0), float(y0))

        logger.info(
            f"Edge detection successful: {len(edge_points)} points, "
            f"theta={theta_deg:.2f}°, anchor=({x0:.1f}, {y0:.1f})"
        )

        return theta_deg, anchor_point, edge_points

    except Exception as e:
        logger.error(f"Edge detection failed: {e}", exc_info=True)
        return None, None, edge_points


def detect_edge_robust(
    frame: np.ndarray,
    scan_direction: str = "horizontal",
    num_attempts: int = 3
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    """
    Robust edge detection with multiple attempts and parameter tuning.

    Attempts detection with progressively relaxed parameters if initial
    detection fails.

    Parameters
    ----------
    frame : np.ndarray
        BGR input frame (H, W, 3)
    scan_direction : str
        "horizontal" or "vertical"
    num_attempts : int
        Maximum number of detection attempts with different parameters

    Returns
    -------
    theta : float or None
        Edge orientation angle in degrees
    anchor_point : tuple or None
        (x, y) anchor point on the fitted line
    """
    # First attempt with standard parameters
    theta, anchor, points = detect_straight_black_edge(frame, scan_direction)

    if theta is not None:
        return theta, anchor

    # Second attempt: relax black threshold
    if num_attempts >= 2:
        logger.info("Retrying with relaxed black threshold...")
        theta, anchor, points = detect_straight_black_edge(
            frame,
            scan_direction,
            max_black_value=MAX_BLACK_VALUE * 1.5  # 60 instead of 40
        )
        if theta is not None:
            return theta, anchor

    # Third attempt: use more scan rows and lower minimum
    if num_attempts >= 3:
        logger.info("Retrying with more scan rows...")
        theta, anchor, points = detect_straight_black_edge(
            frame,
            scan_direction,
            num_rows=15,
            min_valid_points=3
        )
        if theta is not None:
            return theta, anchor

    logger.error("All edge detection attempts failed")
    return None, None


def visualize_edge_detection(
    frame: np.ndarray,
    edge_points: List[Tuple[float, float]],
    theta: Optional[float] = None,
    anchor_point: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Visualize edge detection results on the frame.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR frame
    edge_points : list
        List of (x, y) edge points
    theta : float or None
        Edge angle in degrees
    anchor_point : tuple or None
        (x, y) anchor point

    Returns
    -------
    vis_frame : np.ndarray
        Visualization frame with edge points and line drawn
    """
    vis = frame.copy()

    # Draw detected edge points
    for i, (x, y) in enumerate(edge_points):
        cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"{i}",
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Draw fitted line if available
    if theta is not None and anchor_point is not None:
        x0, y0 = anchor_point

        # Draw line extending across the image
        h, w = frame.shape[:2]
        line_length = max(h, w)

        # Calculate line endpoints
        theta_rad = np.radians(theta)
        dx = np.cos(theta_rad) * line_length
        dy = np.sin(theta_rad) * line_length

        pt1 = (int(x0 - dx), int(y0 - dy))
        pt2 = (int(x0 + dx), int(y0 + dy))

        cv2.line(vis, pt1, pt2, (0, 0, 255), 2)
        cv2.circle(vis, (int(x0), int(y0)), 8, (0, 0, 255), -1)

        # Draw angle text
        cv2.putText(
            vis,
            f"Theta: {theta:.1f}°",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

    return vis


# ---------------------------------------------------------------------------
# Test Routine
# ---------------------------------------------------------------------------

def test_with_camera(camera_index: int = 0):
    """
    Test edge detection with live camera feed.

    Press 'q' to quit, 's' to save frame.
    """
    from camera_local.smartcam_camera import SmartCamCamera

    print("Initializing camera...")
    cam = SmartCamCamera(camera_index=camera_index)

    if not cam.connect():
        print("Failed to connect to camera")
        return

    print("Camera connected. Press 'q' to quit, 's' to save frame")

    try:
        while True:
            frame = cam.capture()
            if frame is None:
                print("Failed to capture frame")
                break

            # Convert RGB to BGR for OpenCV processing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect edge
            theta, anchor, points = detect_edge_robust(frame_bgr)

            # Visualize
            vis = visualize_edge_detection(frame_bgr, points, theta, anchor)

            # Display
            cv2.imshow("Black Edge Detection", vis)

            # Print results
            if theta is not None:
                print(f"\rEdge: θ={theta:.2f}° anchor=({anchor[0]:.1f}, {anchor[1]:.1f}) "
                      f"points={len(points)}", end="")
            else:
                print(f"\rEdge: NOT DETECTED (points={len(points)})", end="")

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("edge_detection_result.png", vis)
                print("\nSaved edge_detection_result.png")

    finally:
        cam.disconnect()
        cv2.destroyAllWindows()


def test_with_synthetic_frame():
    """
    Test with a synthetic frame containing a black edge.
    """
    print("Testing with synthetic frame...")

    # Create synthetic frame: red background with black vertical line
    h, w = 480, 640
    frame = np.full((h, w, 3), (180, 50, 50), dtype=np.uint8)  # Red background (BGR)

    # Add black vertical edge at x=320 with some noise
    edge_x = 320
    for y in range(h):
        # Black line with slight variation
        noise = np.random.randint(-5, 6)
        width = 3 + noise
        frame[y, edge_x - width:edge_x + width] = (0, 0, 0)

    # Add some glare spots
    cv2.circle(frame, (200, 200), 30, (255, 255, 255), -1)
    cv2.circle(frame, (500, 300), 25, (255, 255, 255), -1)

    # Detect edge
    theta, anchor, points = detect_straight_black_edge(frame)

    print(f"Detected edge: theta={theta}, anchor={anchor}")
    print(f"Number of edge points: {len(points)}")

    # Visualize
    vis = visualize_edge_detection(frame, points, theta, anchor)
    cv2.imwrite("synthetic_edge_test.png", vis)
    print("Saved synthetic_edge_test.png")

    # Show result
    cv2.imshow("Synthetic Test", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Black Edge Detector Test")
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Test with live camera"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Test with synthetic frame"
    )
    args = parser.parse_args()

    if args.camera:
        test_with_camera()
    elif args.synthetic:
        test_with_synthetic_frame()
    else:
        print("Running synthetic test by default...")
        test_with_synthetic_frame()