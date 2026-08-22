#!/usr/bin/env python3
"""Regression test for edge-detection false positives.

Images in ``edgepictures/``:
  * falsepositive.PNG  / falsepositive2.PNG  -> textured/uniform, MUST be rejected
  * nofalsepositive.PNG                     -> uniform, MUST be rejected
  * realedge.PNG                            -> real substrate-stage hue edge,
                                               MUST be detected
"""

import pathlib

import cv2

from color_edge_detector import detect_color_edge

EDG = pathlib.Path(__file__).resolve().parent.parent / "edgepictures"


def _load(name):
    img = cv2.imread(str(EDG / name))
    assert img is not None, f"missing edgepicture: {name}"
    return img


def test_false_positive_images_do_not_produce_an_edge():
    """Textures/dust must NEVER be reported as a chip edge."""
    for name in ("falsepositive.PNG", "falsepositive2.PNG"):
        theta, anchor, points = detect_color_edge(_load(name))
        assert theta is None, f"{name}: expected no edge, got theta={theta}"
        assert anchor is None and points is None


def test_uniform_image_does_not_produce_an_edge():
    theta, anchor, points = detect_color_edge(_load("nofalsepositive.PNG"))
    assert theta is None
    assert anchor is None and points is None


def test_real_edge_is_detected():
    theta, anchor, points = detect_color_edge(_load("realedge.PNG"))
    assert theta is not None
    assert 0 <= theta < 180
    assert anchor is not None
    assert points is not None and len(points) > 5
