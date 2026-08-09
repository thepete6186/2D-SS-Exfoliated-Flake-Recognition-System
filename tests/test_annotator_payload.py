"""Tests for the annotator's substrate-calibration payload helpers.

These exercise the tkinter-free module-level functions in
camera.camera_annotator (payload build/parse and calibration
aggregation) — no Tk objects are constructed.
"""

import numpy as np
import pytest

from camera.camera_annotator import (
    MIN_CALIB_PIXELS,
    Point,
    build_points_payload,
    compute_calibration_from_samples,
    parse_points_payload,
)


def test_point_label_roundtrip():
    p = Point(10, 20, hsv=(60, 80, 150), label="substrate")
    restored = Point.from_dict(p.to_dict())

    assert (restored.x, restored.y) == (10, 20)
    assert restored.hsv == (60, 80, 150)
    assert restored.label == "substrate"

    legacy = Point.from_dict({"x": 1, "y": 2, "hsv": [3, 4, 5]})
    assert legacy.label == "Point"


def test_build_parse_payload_roundtrip():
    points = [
        Point(10, 20, hsv=(60, 80, 150)),
        Point(30, 40, hsv=(100, 90, 120), label="substrate"),
    ]
    calibration = {
        "peak": [100.2, 90.7, 120.1],
        "std": [1.5, 2.5, 3.5],
        "n_pixels": 225,
        "n_samples": 1,
    }

    payload = build_points_payload(points, calibration, image_size=(640, 480))

    # Legacy key mirrors the calibration peak as ints for old readers.
    assert payload["substrate_hsv"] == [100, 91, 120]
    assert payload["substrate_calibration"] == calibration
    assert payload["image_size"] == (640, 480)

    restored, restored_cal = parse_points_payload(payload)
    assert restored_cal == calibration
    assert [(p.x, p.y, p.hsv, p.label) for p in restored] == [
        (10, 20, (60, 80, 150), "Point"),
        (30, 40, (100, 90, 120), "substrate"),
    ]


def test_build_payload_without_calibration_uses_fallback():
    points = [Point(1, 2, hsv=(3, 4, 5))]

    payload = build_points_payload(
        points, None, image_size=None, fallback_substrate_hsv=(90, 50, 100)
    )

    assert payload["substrate_hsv"] == [90, 50, 100]
    assert payload["substrate_calibration"] is None


def test_legacy_json_loads():
    legacy = {
        "points": [{"x": 5, "y": 6, "hsv": [7, 8, 9]}],
        "substrate_hsv": [90, 50, 100],
        "image_size": [640, 480],
    }

    points, calibration = parse_points_payload(legacy)

    assert calibration is None
    assert len(points) == 1
    assert points[0].label == "Point"
    assert points[0].hsv == (7, 8, 9)


def test_compute_calibration_min_pixel_fallback():
    small = [np.full((30, 3), (60.0, 80.0, 150.0), dtype=np.float32)]
    assert 30 < MIN_CALIB_PIXELS

    result = compute_calibration_from_samples(small)

    assert result is not None
    assert result["std"] is None  # too few pixels for a trustworthy std
    assert result["peak"] == pytest.approx([60.0, 80.0, 150.0])
    assert result["n_pixels"] == 30
    assert result["n_samples"] == 1

    enough = small * 2  # 60 pixels across 2 samples
    result = compute_calibration_from_samples(enough)
    assert result["std"] is not None
    assert result["n_pixels"] == 60
    assert result["n_samples"] == 2

    assert compute_calibration_from_samples([]) is None
