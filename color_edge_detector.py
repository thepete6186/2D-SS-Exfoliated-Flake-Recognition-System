#!/usr/bin/env python3
"""Color Edge Detector - chip boundaries via color difference.

This detector finds edges where the COLOR changes rather than where a single
channel brightens. The edge map is the spatial gradient magnitude of the Lab
chromatic (a/b) channels - i.e. how fast the *color* changes pixel-to-pixel.
Uniform-color frames (no color difference anywhere) produce no edges, which is
what suppresses false positives on flat / textured images.

It then reports the longest *interior* color-difference line, which is the
overall chip/substrate boundary; shorter flake-fragment edges are naturally
excluded by the length filter.
"""
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Canny thresholds applied to the chroma-gradient magnitude edge map.
CANNY_LOW = 30
CANNY_HIGH = 90
MIN_LINE_LENGTH = 100
HOUGH_THRESHOLD = 50
# Hough gap (pixels) used when merging collinear edge segments.
HOUGH_MAX_LINE_GAP = 10

# Lines too close to the image border are usually the camera frame edge,
# not the substrate-stage boundary. Reject them.
BORDER_MARGIN = 15

# A real chip/substrate boundary spans a large fraction of the field of view,
# whereas flake-fragment edges are short. Require a candidate to be at least
# this fraction of the smaller image dimension (on top of the absolute
# MIN_LINE_LENGTH floor) so flake fragments are rejected and the overall edge
# is kept. Scales to live camera resolution.
MIN_LINE_FRACTION = 0.30


def _line_is_on_border(x1: int, y1: int, x2: int, y2: int,
                       w: int, h: int, margin: int = BORDER_MARGIN) -> bool:
    """True if both endpoints lie entirely within the border strip."""
    near_h = (x1 <= margin and x2 <= margin) or (x1 >= w - margin and x2 >= w - margin)
    near_v = (y1 <= margin and y2 <= margin) or (y1 >= h - margin and y2 >= h - margin)
    return near_h or near_v


def detect_color_edge(
    frame: np.ndarray,
    min_line_length: int = MIN_LINE_LENGTH,
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[List[Tuple[float, float]]]]:
    """
    Detect the longest continuous color edge in the image.

    The edge map is the spatial gradient magnitude of the Lab a/b (chromatic)
    channels - i.e. where the COLOR changes. The longest interior line is
    reported as the chip edge; shorter flake-fragment edges are excluded by
    the length filter.

    Parameters
    ----------
    frame : np.ndarray
        BGR input frame (H, W, 3)
    min_line_length : int
        Minimum edge length in pixels (absolute floor). An image-scale
        minimum (see MIN_LINE_FRACTION) is added on top automatically.

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

        # --- Color-difference edge map -------------------------------------
        # Spatial gradient magnitude of the Lab chromatic (a, b) channels:
        # large wherever the COLOR changes between neighbours, ~0 on uniform
        # color. Suppresses false positives on flat frames while keeping the
        # real chip/substrate (and flake/substrate) boundaries.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        a = lab[:, :, 1].astype(np.float32)
        b = lab[:, :, 2].astype(np.float32)
        ga = np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
        gb = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3),
                      cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))
        chroma_grad = np.hypot(ga, gb)
        edges = cv2.Canny(np.uint8(np.clip(chroma_grad, 0, 255)), CANNY_LOW, CANNY_HIGH)

        # Effective minimum length: absolute floor + image-scale fraction so
        # short flake-fragment edges are rejected and the long overall edge is
        # kept. Scales to live camera resolution.
        eff_min = int(max(min_line_length, MIN_LINE_FRACTION * min(H, W)))

        # HoughLinesP to find straight line segments
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, HOUGH_THRESHOLD,
            minLineLength=eff_min, maxLineGap=HOUGH_MAX_LINE_GAP
        )

        if lines is None or len(lines) == 0:
            logger.error("No lines found in edge image")
            return None, None, None

        logger.info(f"Found {len(lines)} raw Hough lines (min_length={eff_min})")

        # Collect interior candidate lines. The longest surviving line is the
        # chip edge; flake-fragment edges are shorter and get filtered by the
        # length requirement above.
        candidates = []
        for row in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(float(v)) for v in row)
            length = np.hypot(x2 - x1, y2 - y1)
            if length < eff_min:
                continue
            if _line_is_on_border(x1, y1, x2, y2, W, H):
                continue
            candidates.append((length, x1, y1, x2, y2))

        if not candidates:
            logger.warning("No usable edge lines after filtering")
            return None, None, None

        # Pick the longest interior color-difference candidate line.
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
