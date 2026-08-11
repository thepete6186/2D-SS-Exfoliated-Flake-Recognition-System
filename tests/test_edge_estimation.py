"""Tests for edge_estimation.EdgeAngleEstimator (image-processing layer)."""

import cv2
import numpy as np
import pytest

from edge_estimation import EdgeAngleEstimator, synthetic_rect_frame


def circular_dist(a: float, b: float, period: float = 90.0) -> float:
    """Shortest circular distance between two angles on a period."""
    d = abs(a - b) % period
    return min(d, period - d)


def test_estimator_recovers_synthetic_angle():
    est = EdgeAngleEstimator()
    for angle in (8, 20, 30, 44):
        res = est.detect(synthetic_rect_frame(angle))
        assert res is not None, f"failed to detect edge for angle {angle}"
        # Reported angle may be the perpendicular edge; canonical (mod 90)
        # must still match the true canonical angle.
        assert circular_dist(res.angle_deg % 90, angle % 90) < 3.0, (
            f"angle {angle}: estimated {res.angle_deg:.1f} (canonical "
            f"{res.angle_deg % 90:.1f})"
        )


def test_estimator_prefers_long_continuous_edge_over_short_noise():
    """A long continuous edge must win over many short random lines."""
    est = EdgeAngleEstimator()
    frame = synthetic_rect_frame(20.0)
    rng = np.random.default_rng(0)
    h, w = frame.shape[:2]
    for _ in range(40):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        ang = float(rng.uniform(0, 180))
        length = int(rng.integers(5, 25))
        x2 = int(x1 + length * np.cos(np.radians(ang)))
        y2 = int(y1 + length * np.sin(np.radians(ang)))
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    res = est.detect(frame)
    assert res is not None
    assert circular_dist(res.angle_deg % 90, 20.0) < 3.0, (
        f"noise overwhelmed the dominant edge: {res.angle_deg:.1f} deg"
    )


def test_estimator_returns_none_on_blank_frame():
    est = EdgeAngleEstimator()
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    assert est.detect(blank) is None


def test_grayscale_honors_channel_order():
    est_bgr = EdgeAngleEstimator(channel_order="bgr")
    est_rgb = EdgeAngleEstimator(channel_order="rgb")
    frame = np.full((4, 4, 3), (10, 200, 30), dtype=np.uint8)
    assert not np.array_equal(est_bgr.grayscale(frame), est_rgb.grayscale(frame))


def test_detect_accepts_grayscale_input():
    est = EdgeAngleEstimator()
    frame = synthetic_rect_frame(20.0)
    gray = est.grayscale(frame)
    assert est.detect(gray) is not None


@pytest.mark.parametrize("bad_order", ["bayer", "rgba", ""])
def test_bad_channel_order_rejected(bad_order):
    with pytest.raises(ValueError):
        EdgeAngleEstimator(channel_order=bad_order)