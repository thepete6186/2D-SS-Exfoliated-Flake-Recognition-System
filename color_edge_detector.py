#!/usr/bin/env python3
"""
Color Edge Detector - Detects chip boundaries using color contrast in HSV space.

Unlike black edge detection which only works for dark edges, this detector
finds long continuous color boundaries between the substrate and stage.
The substrate and stage always have different colors, making this robust.

Usage:
    from color_edge_detector import detect_color_edge, visualize_color_edge
    theta, anchor, points = detect_color_edge(frame)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Detection parameters
HUE_CANNY_LOW = 50
HUE_CANNY_HIGH = 150
MIN_LINE_LENGTH = 100
HOUGH_THRESHOLD = 50

# Lines too close to the image border are usually the frame boundary
# (the camera frame edge), not the substrate-stage boundary. Reject them.
BORDER_MARGIN = 15


def _line_is_on_border(x1: int, y1: int, x2: int, y2: int,
                       w: int, h: int, margin: int = BORDER_MARGIN) -> bool:
    """Return True if both endpoints lie entirely within the border strip.

    A true substrate edge crosses toward the interior, so at least one
    endpoint should be a comfortable distance from the frame edge.
    """
    near_h = (x1 <= margin and x2 <= margin) or (x1 >= w - margin and x2 >= w - margin)
    near_v = (y1 <= margin and y2 <= margin) or (y1 >= h - margin and y2 >= h - margin)
    return near_h or near_v


def detect_color_edge(
    frame: np.ndarray,
    min_line_length: int = MIN_LINE_LENGTH,
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[List[Tuple[float, float]]]]:
    """
    Detect the longest continuous color edge in the image using HSV color space.
    
    Parameters
    ----------
    frame : np.ndarray
        BGR input frame (H, W, 3)
    min_line_length : int
        Minimum length for a valid edge (pixels)
    
    Returns
    -------
    theta : float or None
        Edge angle in degrees (0-180), or None if detection fails
    anchor : tuple or None
        (x, y) center point of the detected edge, or None
    edge_points : list or None
        List of (x, y) edge points along the line, or None
    """
    if frame is None or frame.size == 0:
        logger.error("Invalid frame provided")
        return None, None, None

    try:
        H, W = frame.shape[:2]

        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract Hue and Saturation channels (0-179 and 0-255 in OpenCV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        
        # Apply Gaussian blur to reduce noise
        hue_blur = cv2.GaussianBlur(hue, (5, 5), 0)
        sat_blur = cv2.GaussianBlur(saturation, (5, 5), 0)
        
        # Run Canny edge detection on Hue and Saturation channels
        edges_hue = cv2.Canny(hue_blur, HUE_CANNY_LOW, HUE_CANNY_HIGH)
        edges_sat = cv2.Canny(sat_blur, HUE_CANNY_LOW, HUE_CANNY_HIGH)
        edges = cv2.bitwise_or(edges_hue, edges_sat)
        
        # Use HoughLinesP to find straight line segments
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, HOUGH_THRESHOLD,
            minLineLength=min_line_length, maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            logger.error("No lines found in edge image")
            return None, None, None
        
        # DEBUG: Log the actual format of lines from this OpenCV version
        logger.info(f"Found {len(lines)} raw Hough lines")
        
        # Collect candidate lines, rejecting those stuck on the image border
        candidates = []
        for row in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(float(v)) for v in row)
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < min_line_length:
                continue
            if _line_is_on_border(x1, y1, x2, y2, W, H):
                continue
            candidates.append((length, x1, y1, x2, y2))
        
        if not candidates:
            logger.warning("No usable edge lines after border filtering")
            return None, None, None
        
        # Pick the longest interior-substrate candidate line
        candidates.sort(key=lambda c: c[0], reverse=True)
        max_length, x1, y1, x2, y2 = candidates[0]
        
        logger.info(f"Longest interior edge: ({x1},{y1})->({x2},{y2}) length={max_length:.0f}px")
        
        # Calculate edge angle
        if abs(x2 - x1) < 5:
            theta = 90.0
        elif abs(y2 - y1) < 5:
            theta = 0.0
        else:
            theta = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            theta = theta % 180.0
        
        # Anchor point (center of line)
        anchor = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        # Sample edge points along the line
        num_points = max(5, int(max_length / 20))
        edge_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            edge_points.append((float(px), float(py)))
        
        logger.info(f"Detected edge: theta={theta:.1f}deg at ({anchor[0]:.0f}, {anchor[1]:.0f}) "
                    f"length={max_length:.0f}px points={len(edge_points)}")
        
        return theta, anchor, edge_points

    except Exception as e:
        logger.error(f"Color edge detection failed: {e}", exc_info=True)
        return None, None, None


def visualize_color_edge(
    frame: np.ndarray,
    edge_points: List[Tuple[float, float]],
    theta: Optional[float] = None,
    anchor_point: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Visualize the detected edge on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR input frame
    edge_points : list of tuple
        (x, y) edge points to visualize
    theta : float or None
        Edge angle in degrees (for text label)
    anchor_point : tuple or None
        (x, y) center point of edge

    Returns
    -------
    vis : np.ndarray
        Visualized frame with edge overlay
    """
    vis = frame.copy()

    # Draw edge points as yellow dots
    for x, y in edge_points:
        cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 255), -1)

    # Draw fitted line in red
    if len(edge_points) >= 2:
        points_array = np.array(edge_points, dtype=np.float32)
        line_fit = cv2.fitLine(points_array, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = line_fit.flatten()

        length = max(frame.shape[0], frame.shape[1])
        x1 = int(x - length * vx)
        y1 = int(y - length * vy)
        x2 = int(x + length * vx)
        y2 = int(y + length * vy)

        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Draw angle text
        if theta is not None:
            angle_text = f"{theta:.1f} deg"
            cv2.putText(
                vis, angle_text, (int(x) - 60, int(y) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )

    return vis