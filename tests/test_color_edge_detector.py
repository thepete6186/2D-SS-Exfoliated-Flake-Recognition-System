#!/usr/bin/env python3
"""
Tests for color_edge_detector module.
"""

import cv2
import numpy as np
import pytest
from color_edge_detector import detect_color_edge, visualize_color_edge


def test_detect_edge_on_capture_image():
    """Test that the detector finds the edge in Capture.PNG."""
    img = cv2.imread('Capture.PNG')
    assert img is not None
    
    theta, anchor, points = detect_color_edge(img)
    
    # Should detect an edge (angle depends on image orientation)
    assert theta is not None
    assert 0 <= theta < 180  # Valid angle range
    assert anchor is not None
    assert points is not None
    assert len(points) > 5  # Should have multiple points


def test_blank_frame_returns_none():
    """Test that blank frames return None."""
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    theta, anchor, points = detect_color_edge(blank)
    assert theta is None
    assert anchor is None
    assert points is None


def test_none_frame_returns_none():
    """Test that None input returns None."""
    theta, anchor, points = detect_color_edge(None)
    assert theta is None
    assert anchor is None
    assert points is None


def test_visualize_does_not_crash():
    """Test that visualization doesn't crash with valid input."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    points = [(10, 10), (50, 50), (90, 90)]
    
    vis = visualize_color_edge(img, points, theta=45.0, anchor_point=(50, 50))
    assert vis is not None
    assert vis.shape == img.shape


def test_detector_returns_three_values():
    """Test that detector always returns a 3-tuple."""
    img = cv2.imread('Capture.PNG')
    result = detect_color_edge(img)
    assert isinstance(result, tuple)
    assert len(result) == 3