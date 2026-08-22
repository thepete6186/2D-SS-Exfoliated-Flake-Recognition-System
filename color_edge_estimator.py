#!/usr/bin/env python3
"""
Color Edge Estimator Wrapper - Makes color_edge_detector compatible with AutoAlignController.

AutoAlignController expects an estimator with a .detect() method that returns
an object with .angle_deg, .length_px, and .mode attributes (like EdgeAngleResult).
This wrapper adapts color_edge_detector to that interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from color_edge_detector import detect_color_edge

logger = logging.getLogger(__name__)


@dataclass
class ColorEdgeResult:
    """Result from color-based edge detection (compatible with EdgeAngleResult)."""
    angle_deg: float
    length_px: float
    endpoints: Tuple[float, float, float, float]
    edge_points: np.ndarray
    num_segments: int
    mode: str = "color_edge"


class ColorEdgeEstimator:
    """
    Edge angle estimator using HSV color-based detection.
    
    This wrapper makes color_edge_detector compatible with AutoAlignController
    by providing the same interface as EdgeAngleEstimator.
    
    Parameters
    ----------
    min_line_length : int
        Minimum edge length in pixels (passed to detect_color_edge)
    channel_order : str
        "rgb" (default; SmartCamCamera returns RGB) or "bgr" (OpenCV/webcam).
        The estimator converts to BGR internally because the HSV detector
        expects BGR input.
    """
    
    def __init__(self, min_line_length: int = 100, channel_order: str = "rgb"):
        if channel_order not in ("rgb", "bgr"):
            raise ValueError(
                f"channel_order must be 'rgb' or 'bgr', got {channel_order!r}"
            )
        self.min_line_length = min_line_length
        self.channel_order = channel_order
    
    def detect(self, frame: np.ndarray) -> Optional[ColorEdgeResult]:
        """
        Detect the dominant edge angle in the frame.
        
        Parameters
        ----------
        frame : np.ndarray
            BGR or RGB input frame (H, W, 3)
        
        Returns
        -------
        ColorEdgeResult or None
            Detected edge information, or None if detection fails
        """
        if frame is None or frame.size == 0:
            logger.warning("Invalid frame provided to ColorEdgeEstimator")
            return None
        
        # Convert to BGR as needed (color_edge_detector expects BGR).
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if self.channel_order == "rgb":
                # camera returns RGB (SmartCam) -> flip to BGR
                frame_bgr = frame[:, :, ::-1].copy()
            else:
                frame_bgr = frame
        else:
            frame_bgr = frame
        
        # Detect color edge
        theta, anchor, points = detect_color_edge(
            frame_bgr, 
            min_line_length=self.min_line_length
        )
        
        if theta is None or anchor is None or not points:
            logger.warning("Color edge detection failed")
            return None
        
        # Convert points to numpy array
        edge_points = np.array(points, dtype=np.float32)
        
        # Calculate endpoints from points
        if len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[-1]
            length = float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
        else:
            x1, y1 = anchor
            x2, y2 = anchor
            length = 0.0
        
        logger.info(
            "ColorEdgeEstimator detected: angle=%.2f deg, length=%.1f px, points=%d",
            theta, length, len(points)
        )
        
        return ColorEdgeResult(
            angle_deg=float(theta),
            length_px=length,
            endpoints=(float(x1), float(y1), float(x2), float(y2)),
            edge_points=edge_points,
            num_segments=len(points),
            mode="color_edge"
        )