#!/usr/bin/env python3
"""
Edge angle estimation - long-continuous-edge selection via Canny + Hough.

This module is the *image-processing* layer of the sample auto-alignment
feature. It is deliberately pure OpenCV/numpy: it knows nothing about the
camera backend or the stage, so it can be reused with any camera that
exposes a ``capture()`` method returning an ndarray.

Core idea (matches the spec "if there's a long, continuous edge, then
it's an edge"):

1. Canny edge detection -> clean single-pixel edges.
2. HoughLinesP -> line *segments* each with an explicit length.
3. Orientation grouping -> cluster segments by angle (half-plane).
4. Continuous-edge merging -> within the dominant orientation, chain
   collinear segments whose endpoints are near/overlapping into a single
   "continuous edge"; score by total length.
5. The longest continuous edge is taken as THE edge; its angle is refined
   with a least-squares line fit (cv2.fitLine) for sub-degree accuracy.

Usage:
    from edge_estimation import EdgeAngleEstimator

    est = EdgeAngleEstimator()
    result = est.detect(frame_bgr)          # -> EdgeAngleResult or None
    print(result.angle_deg, result.length_px)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgeAngleResult:
    """Result of an edge-angle estimation."""

    #: Dominant edge angle in degrees, normalized to [0, 180) (undirected).
    angle_deg: float
    #: Total length (px) of the dominant edge.
    length_px: float
    #: (x1, y1, x2, y2) endpoints of the fitted dominant-edge line.
    endpoints: Tuple[float, float, float, float]
    #: Nx2 [ (x, y) ... ] points belonging to the dominant edge.
    edge_points: np.ndarray
    #: Number of raw Hough segments that contributed.
    num_segments: int
    #: "continuous_edge" (long single edge) or "histogram_fallback".
    mode: str


class EdgeAngleEstimator:
    """
    Estimates the dominant long-continuous edge angle in a frame.

    Parameters
    ----------
    canny_low, canny_high : int
        Canny hysteresis thresholds.
    hough_threshold : int
        HoughLinesP accumulator threshold.
    hough_min_line_length : int
        Minimum segment length (px) accepted from HoughLinesP.
    hough_max_line_gap : int
        Max gap allowed within a single Hough segment (px).
    angle_bin_width : float
        Width (deg) of the orientation grouping bins.
    rho_tol_px : float
        Max perpendicular distance (px) between segments to treat them as
        collinear when merging into a continuous edge.
    gap_tol_px : float
        Max end-gap (px) along the line direction to treat segments as
        continuous when merging.
    min_edge_length_ratio : float
        A continuous edge must be at least this fraction of the image
        diagonal to be trusted on its own; otherwise fall back to the
        length-weighted orientation histogram.
    channel_order : str
        "bgr" (OpenCV default) or "rgb" (e.g. SmartCamCamera returns RGB).
    """

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 40,
        hough_min_line_length: int = 60,
        hough_max_line_gap: int = 8,
        angle_bin_width: float = 5.0,
        rho_tol_px: float = 6.0,
        gap_tol_px: float = 12.0,
        min_edge_length_ratio: float = 0.20,
        channel_order: str = "bgr",
    ) -> None:
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.hough_threshold = int(hough_threshold)
        self.hough_min_line_length = int(hough_min_line_length)
        self.hough_max_line_gap = int(hough_max_line_gap)
        self.angle_bin_width = float(angle_bin_width)
        self.rho_tol_px = float(rho_tol_px)
        self.gap_tol_px = float(gap_tol_px)
        self.min_edge_length_ratio = float(min_edge_length_ratio)
        if channel_order not in ("bgr", "rgb"):
            raise ValueError(
                f"channel_order must be 'bgr' or 'rgb', got {channel_order!r}"
            )
        self.channel_order = channel_order

    # ------------------------------------------------------------------
    # Grayscale
    # ------------------------------------------------------------------
    def grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert a 3-channel frame to grayscale, honoring channel_order."""
        if self.channel_order == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _run_hough(
        self,
        blurred: np.ndarray,
        canny_low: int,
        canny_high: int,
        hough_threshold: int,
        min_line_length: int,
    ) -> Optional[np.ndarray]:
        """Run Canny + HoughLinesP with the given thresholds."""
        edges = cv2.Canny(blurred, int(canny_low), int(canny_high))
        return cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(hough_threshold),
            minLineLength=int(min_line_length),
            maxLineGap=self.hough_max_line_gap,
        )

    def _candidate_params(self, blurred: np.ndarray):
        """Yield (canny_low, canny_high, hough_threshold, min_len) tuples.

        Tries the configured thresholds first, then Otsu-adaptive thresholds
        (robust to dark/bright scenes), then a lenient low-threshold pass.
        """
        primary = (
            self.canny_low, self.canny_high,
            self.hough_threshold, self.hough_min_line_length,
        )
        yield primary

        # Otsu-adaptive thresholds from the frame's own intensity histogram.
        otsu_high, _ = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        if otsu_high > 0:
            otsu_low = max(1, int(otsu_high * 0.5))
            yield (
                otsu_low, int(otsu_high),
                self.hough_threshold, self.hough_min_line_length,
            )
            # Lenient fallback for faint/short edges.
            yield (
                max(1, otsu_low // 2), max(1, int(otsu_high * 0.75)),
                max(10, self.hough_threshold // 2),
                max(20, self.hough_min_line_length // 2),
            )

    def _candidate_images(self, blurred: np.ndarray):
        """Yield the raw blurred frame, then CLAHE-contrast-boosted copies.

        Low-contrast microscope frames (uniform substrate, faint flake edge)
        yield no Canny edges at default thresholds; CLAHE on the grayscale
        plane recovers those edges. Several clip limits are tried and Hough
        picks whichever reveals a real edge.
        """
        yield blurred
        for clip in (8.0, 15.0, 25.0):
            try:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                yield clahe.apply(blurred)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("CLAHE clip=%.1f failed: %s", clip, exc)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> Optional[EdgeAngleResult]:
        """Estimate the dominant edge angle of one frame (or None)."""
        if frame is None:
            return None
        if frame.ndim == 2:
            gray = frame.astype(np.uint8)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = self.grayscale(frame)
        else:
            logger.warning("Unsupported frame shape %s", frame.shape)
            return None

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diag = math.hypot(gray.shape[1], gray.shape[0])
        min_span = self.min_edge_length_ratio * diag

        # Scan candidate images (raw, then CLAHE-contrast-boosted copies)
        # across adaptive Hough thresholds. Real microscope frames are often
        # low-contrast uniform substrate; CLAHE on the grayscale plane
        # recovers edges that default thresholds miss. The first candidate
        # that yields a long, trusted continuous edge wins.
        fallback_segments: Optional[List[Dict]] = None
        best_span = 0.0
        for image in self._candidate_images(blurred):
            for params in self._candidate_params(image):
                lines = self._run_hough(image, *params)
                if lines is None or len(lines) == 0:
                    continue
                segments = self._extract_segments(lines)
                if not segments:
                    continue
                group = self._dominant_orientation_group(segments)
                edge = self._longest_continuous_edge(group)
                span = edge["span"] if edge else 0.0
                if span >= best_span:
                    best_span = span
                    fallback_segments = segments
                if edge is not None and span >= min_span:
                    logger.info(
                        "Detected edge (span %.0f px >= %.0f px) via %s image",
                        span, min_span,
                        "CLAHE" if image is not blurred else "raw",
                    )
                    return self._result_from_edge(edge, gray.shape)

        # No trusted long edge anywhere; last resort: length-weighted
        # orientation histogram over the best segment set we saw.
        if fallback_segments:
            logger.info(
                "No long continuous edge (best %.1f px < %.1f px); "
                "using length-weighted histogram fallback",
                best_span, min_span,
            )
            return self._result_from_fallback(fallback_segments, gray.shape)

        logger.warning("No lines detected by Hough transform")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_segments(self, lines: np.ndarray) -> List[Dict]:
        """Convert HoughLinesP output into normalized segment dicts.

        Each segment is stored with its oriented angle (in [-90, 90)),
        length, unit direction and signed distance from the origin (rho).
        Handles both (N, 1, 4) and (N, 4) array shapes across OpenCV
        versions.
        """
        segments: List[Dict] = []
        for row in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (int(float(v)) for v in row)
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 1e-3:
                continue
            # Oriented line angle in [-90, 90): no periodicity for a single
            # physical edge, so the orientation bins are wrap-safe.
            ang = math.degrees(math.atan2(dy, dx))
            oriented = (ang + 90.0) % 180.0 - 90.0
            ux = dx / length
            uy = dy / length
            nx, ny = -uy, ux
            segments.append(
                {
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "angle": oriented, "length": length,
                    "ux": ux, "uy": uy, "rho": x1 * nx + y1 * ny,
                }
            )
        return segments

    def _dominant_orientation_group(self, segments: List[Dict]) -> List[Dict]:
        """Return the orientation bin (by total length) with the most edge."""
        half_width = self.angle_bin_width / 2.0
        n_bins = int(math.ceil(180.0 / self.angle_bin_width))
        hist = [0.0] * n_bins
        for s in segments:
            idx = int((s["angle"] + 90.0) / self.angle_bin_width)
            idx = max(0, min(n_bins - 1, idx))
            # Weight by segment length so a long edge dominates short clutter.
            hist[idx] += s["length"]

        best_idx = max(range(n_bins), key=lambda i: hist[i])
        lo = -90.0 + best_idx * self.angle_bin_width - half_width
        hi = lo + 2 * half_width
        group = [s for s in segments if lo <= s["angle"] < hi]
        # Robust mean direction of the winning bin.
        vx = sum(s["ux"] * s["length"] for s in group)
        vy = sum(s["uy"] * s["length"] for s in group)
        for s in group:
            # Re-orient each segment so signs agree with the group direction.
            if s["ux"] * vx + s["uy"] * vy < 0:
                s["ux"], s["uy"], s["rho"] = -s["ux"], -s["uy"], -s["rho"]
        return group

    def _longest_continuous_edge(self, group: List[Dict]) -> Optional[Dict]:
        """Merge collinear, near/overlapping segments into continuous edges.

        Returns the edge with the greatest span (max projected offset along
        the line direction minus the min), or None if the group is empty.
        """
        if not group:
            return None

        vx = sum(s["ux"] * s["length"] for s in group)
        vy = sum(s["uy"] * s["length"] for s in group)
        norm = math.hypot(vx, vy) or 1.0
        vx, vy = vx / norm, vy / norm
        nx, ny = -vy, vx

        items = []
        for s in group:
            mid_x = (s["x1"] + s["x2"]) / 2.0
            mid_y = (s["y1"] + s["y2"]) / 2.0
            rho = mid_x * nx + mid_y * ny
            p1 = s["x1"] * vx + s["y1"] * vy
            p2 = s["x2"] * vx + s["y2"] * vy
            items.append((s, rho, min(p1, p2), max(p1, p2)))

        items.sort(key=lambda it: it[2])  # by t_min

        chains: List[Dict] = []
        for s, rho, tmin, tmax in items:
            best_chain = None
            best_gap = None
            for c in chains:
                if abs(rho - c["bar_rho"]) <= self.rho_tol_px:
                    gap = tmin - c["t_max"]
                    if gap <= self.gap_tol_px and (best_gap is None or gap < best_gap):
                        best_chain = c
                        best_gap = gap
            if best_chain is not None:
                c = best_chain
                c["segs"].append(s)
                c["n"] += 1
                c["rho_sum"] += rho
                c["bar_rho"] = c["rho_sum"] / c["n"]
                c["t_min"] = min(c["t_min"], tmin)
                c["t_max"] = max(c["t_max"], tmax)
            else:
                chains.append(
                    {"segs": [s], "n": 1, "rho_sum": rho, "bar_rho": rho,
                     "t_min": tmin, "t_max": tmax}
                )

        if not chains:
            return None

        for c in chains:
            c["span"] = c["t_max"] - c["t_min"]
            c["length"] = sum(s["length"] for s in c["segs"])
        return max(chains, key=lambda c: c["span"])

    def _fit_edge_angle(
        self, points: np.ndarray
    ) -> Tuple[float, Tuple[float, float, float, float]]:
        """Least-squares fit through edge points -> (angle_deg, endpoints)."""
        if points.shape[0] < 2:
            raise ValueError("Not enough points to fit a line")
        vec = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = (float(a[0]) for a in vec)  # unit dir + point on line
        ang = math.degrees(math.atan2(vy, vx))
        if ang < 0:
            ang += 180.0

        proj = points[:, 0] * vx + points[:, 1] * vy
        t_min, t_max = float(proj.min()), float(proj.max())
        endpoints = (
            x0 + vx * t_min, y0 + vy * t_min,
            x0 + vx * t_max, y0 + vy * t_max,
        )
        return ang, endpoints

    def _result_from_edge(
        self, edge: Dict, shape: Tuple[int, int]
    ) -> EdgeAngleResult:
        segs = edge["segs"]
        pts = np.vstack(
            [[[s["x1"], s["y1"]], [s["x2"], s["y2"]]] for s in segs]
        ).reshape(-1, 2).astype(np.float32)
        angle_deg, endpoints = self._fit_edge_angle(pts)
        return EdgeAngleResult(
            angle_deg=angle_deg,
            length_px=edge["length"],
            endpoints=endpoints,
            edge_points=pts,
            num_segments=len(segs),
            mode="continuous_edge",
        )

    def _result_from_fallback(
        self, segments: List[Dict], shape: Tuple[int, int]
    ) -> EdgeAngleResult:
        # Coarse orientation histogram weighted by length (10 deg bins).
        bin_w = 10.0
        n_bins = 18
        hist = [0.0] * n_bins
        for s in segments:
            idx = int((s["angle"] + 90.0) / bin_w)
            idx = max(0, min(n_bins - 1, idx))
            hist[idx] += s["length"]
        best_idx = max(range(n_bins), key=lambda i: hist[i])
        lo = -90.0 + best_idx * bin_w
        win = [s for s in segments if lo <= s["angle"] < lo + bin_w]
        total = sum(s["length"] for s in win) or 1.0
        vx = sum(s["ux"] * s["length"] for s in win) / total
        vy = sum(s["uy"] * s["length"] for s in win) / total
        angle_deg = math.degrees(math.atan2(vy, vx))
        if angle_deg < 0:
            angle_deg += 180.0
        pts = np.vstack(
            [[[s["x1"], s["y1"]], [s["x2"], s["y2"]]] for s in win]
        ).reshape(-1, 2).astype(np.float32)
        _a, endpoints = self._fit_edge_angle(pts)
        return EdgeAngleResult(
            angle_deg=angle_deg,
            length_px=sum(s["length"] for s in win),
            endpoints=endpoints,
            edge_points=pts,
            num_segments=len(win),
            mode="histogram_fallback",
        )


# ----------------------------------------------------------------------
# Synthetic helpers (hardware-free path)
# ----------------------------------------------------------------------

def synthetic_rect_frame(
    angle_deg: float,
    size: Tuple[int, int] = (640, 480),
    rect_w: int = 420,
    rect_h: int = 240,
    background: Tuple[int, int, int] = (20, 25, 30),
    sample: Tuple[int, int, int] = (235, 235, 235),
) -> np.ndarray:
    """Render a rectangular sample on a contrasting background (BGR).

    ``angle_deg`` is a counter-clockwise rotation of the sample rectangle;
    its long axis sits at ``angle_deg`` from horizontal, which is what
    EdgeAngleEstimator should recover.
    """
    h, w = size
    img = np.full((h, w, 3), background, dtype=np.uint8)
    x0 = (w - rect_w) // 2
    y0 = (h - rect_h) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (x0, y0), (x0 + rect_w, y0 + rect_h), 255, thickness=-1)
    # Negative sign so that a positive angle_deg rotates the sample
    # counter-clockwise in image coordinates (y-down), matching the
    # convention "positive stage rotation increases the image angle".
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -float(angle_deg), 1.0)
    mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    img[mask_rot > 0] = sample
    return img


class SyntheticRotatingCamera:
    """
    Fake camera that renders a rotated rectangle.

    Angles come from a manual ``set_angle`` call or, when bound to an
    actual stage, from the stage's current R-axis position. This gives a
    true closed loop for tests / dry runs: rotating the stage rotates the
    rendered sample.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (640, 480),
        rect_w: int = 420,
        rect_h: int = 240,
        background: Tuple[int, int, int] = (20, 25, 30),
        sample: Tuple[int, int, int] = (235, 235, 235),
        steps_per_degree: float = 800.0,
        initial_angle: float = 20.0,
    ) -> None:
        self.size = tuple(size)
        self.rect_w = int(rect_w)
        self.rect_h = int(rect_h)
        self.background = tuple(int(v) for v in background)
        self.sample = tuple(int(v) for v in sample)
        self.steps_per_degree = float(steps_per_degree)
        self._angle = float(initial_angle)
        self._bound: Optional[Tuple[object, str]] = None

    def bind_stage(self, stage, axis: str = "r") -> "SyntheticRotatingCamera":
        """Read the stage R position each capture instead of a fixed angle."""
        self._bound = (stage, axis)
        return self

    def set_angle(self, angle_deg: float) -> None:
        self._angle = float(angle_deg)

    def get_angle(self) -> float:
        if self._bound is not None:
            stage, axis = self._bound
            position = stage.get_position().get(axis, 0.0)
            return position / self.steps_per_degree
        return self._angle

    def capture(self) -> np.ndarray:
        return synthetic_rect_frame(
            self.get_angle(),
            size=self.size,
            rect_w=self.rect_w,
            rect_h=self.rect_h,
            background=self.background,
            sample=self.sample,
        )
