"""
Decode tests for the Axiocam 208 SmartCamApi buffer.

The camera's live buffer is 16,588,800 bytes. That number factors as
3840 x 2160 x 2 — a single 4K frame at 2 bytes/pixel (the Axiocam 208
is a 4K/8.3 MP camera). These tests encode a known ground-truth image
into that buffer layout (12-bit Bayer in 16-bit words, and YUY2) and
assert that SmartCamCamera._decode reconstructs the image.

They also cover the GenTL path: harvesters components must be
demosaiced according to their pixel format, not treated as grayscale.
"""

import numpy as np
import cv2
import pytest

from camera.smartcam_camera import SmartCamCamera

W4K, H4K = 3840, 2160
BUFFER_BYTES = W4K * H4K * 2  # 16,588,800 — the observed buffer size

_CH = {"r": 0, "g": 1, "b": 2}


# ---------------------------------------------------------------------------
# Fixtures: synthetic microscope scene + encoders
# ---------------------------------------------------------------------------

def make_scene(w=W4K, h=H4K, seed=0):
    """Smooth pink-substrate scene with yellow flakes (like the real feed)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 185 + 30 * np.sin(xx / w * 3.1) + 15 * (yy / h)
    g = 150 + 25 * np.cos(yy / h * 2.5)
    b = 195 + 20 * np.sin((xx + yy) / (w + h) * 4)
    img = np.stack([r, g, b], axis=-1)
    for _ in range(12):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        ax, ay = rng.uniform(60, 250), rng.uniform(40, 180)
        mask = (((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2) < 1.0
        img[mask] = (210.0, 200.0, 90.0)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def mosaic(rgb, pattern):
    """Build a Bayer CFA; `pattern` is the literal top-left 2x2, e.g. 'rggb'."""
    p = pattern.lower()
    cfa = np.zeros(rgb.shape[:2], np.float32)
    cfa[0::2, 0::2] = rgb[0::2, 0::2, _CH[p[0]]]
    cfa[0::2, 1::2] = rgb[0::2, 1::2, _CH[p[1]]]
    cfa[1::2, 0::2] = rgb[1::2, 0::2, _CH[p[2]]]
    cfa[1::2, 1::2] = rgb[1::2, 1::2, _CH[p[3]]]
    return cfa


def bayer16_buffer(rgb, pattern="rggb", left_aligned=False):
    """Encode as 12-bit Bayer stored in little-endian 16-bit words."""
    cfa = mosaic(rgb, pattern)
    v12 = np.round(cfa / 255.0 * 4095.0).astype(np.uint16)
    if left_aligned:
        v12 = v12 << 4
    raw = v12.astype("<u2").tobytes()
    assert len(raw) == BUFFER_BYTES
    return np.frombuffer(raw, dtype=np.uint8)


def yuy2_buffer(rgb):
    """Encode as YUY2 (Y0 U Y1 V), 2 bytes/pixel, video-range BT.601
    (the convention cv2.COLOR_YUV2RGB_YUY2 inverts)."""
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    y = 16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    u = 128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    v = 128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    u2 = (u[:, 0::2] + u[:, 1::2]) / 2.0
    v2 = (v[:, 0::2] + v[:, 1::2]) / 2.0
    h, w = y.shape
    out = np.empty((h, w * 2), np.uint8)
    out[:, 0::4] = np.clip(y[:, 0::2], 0, 255).astype(np.uint8)
    out[:, 1::4] = np.clip(u2, 0, 255).astype(np.uint8)
    out[:, 2::4] = np.clip(y[:, 1::2], 0, 255).astype(np.uint8)
    out[:, 3::4] = np.clip(v2, 0, 255).astype(np.uint8)
    raw = out.tobytes()
    assert len(raw) == BUFFER_BYTES
    # Fixture sanity: OpenCV must be able to invert our encoding.
    check = cv2.cvtColor(
        np.frombuffer(raw, np.uint8).reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUY2
    )
    assert mae(check, rgb) < 8.0, "YUY2 fixture does not round-trip via cv2"
    return np.frombuffer(raw, dtype=np.uint8)


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def lum(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def lum_corr(a, b):
    return float(np.corrcoef(lum(a).ravel(), lum(b).ravel())[0, 1])


@pytest.fixture(scope="module")
def scene():
    return make_scene()


# ---------------------------------------------------------------------------
# SmartCam decode: the 16,588,800-byte buffer is ONE 4K frame at 2 B/px
# ---------------------------------------------------------------------------

class TestSmartCamDecode:
    def test_bayer16_4k_shape_and_structure(self, scene):
        """Auto-detection must decode the buffer as a single 4K frame."""
        cam = SmartCamCamera()
        rgb = cam._decode(bayer16_buffer(scene, "rggb"))
        assert rgb is not None
        assert rgb.shape == (H4K, W4K, 3), (
            f"Decoded shape {rgb.shape} — buffer misinterpreted "
            f"(expected one {W4K}x{H4K} frame at 2 bytes/pixel)"
        )
        assert lum_corr(rgb, scene) > 0.97
        assert mae(lum(rgb), lum(scene)) < 8.0

    def test_bayer16_forced_pattern_recovers_colors(self, scene):
        """Explicit pixel_format must reconstruct full color, not just structure."""
        cam = SmartCamCamera(pixel_format="bayer16:rggb")
        rgb = cam._decode(bayer16_buffer(scene, "rggb"))
        assert rgb is not None
        assert rgb.shape == (H4K, W4K, 3)
        assert mae(rgb, scene) < 6.0

    def test_bayer16_left_aligned_values(self, scene):
        """12-bit data left-aligned in the 16-bit word must also decode."""
        cam = SmartCamCamera(pixel_format="bayer16:rggb")
        rgb = cam._decode(bayer16_buffer(scene, "rggb", left_aligned=True))
        assert rgb is not None
        assert rgb.shape == (H4K, W4K, 3)
        assert mae(rgb, scene) < 6.0

    def test_yuy2_auto_detected(self, scene):
        """A YUY2 buffer must be recognized and decoded to correct colors."""
        cam = SmartCamCamera()
        rgb = cam._decode(yuy2_buffer(scene))
        assert rgb is not None
        assert rgb.shape == (H4K, W4K, 3)
        assert mae(rgb, scene) < 10.0

    def test_decode_stable_across_frames(self, scene):
        """Consecutive (slightly different) frames must decode with the SAME
        cached format — no per-frame re-searching that changes the shape."""
        rng = np.random.default_rng(1)
        noisy = np.clip(
            scene.astype(np.float32) + rng.normal(0, 2, scene.shape), 0, 255
        ).astype(np.uint8)
        cam = SmartCamCamera()
        first = cam._decode(bayer16_buffer(scene, "rggb"))
        fmt_after_first = cam._fmt
        second = cam._decode(bayer16_buffer(noisy, "rggb"))
        assert first is not None and second is not None
        assert first.shape == second.shape == (H4K, W4K, 3)
        assert cam._fmt is fmt_after_first, "format cache must be reused"


class TestFormatDetectionRobustness:
    """Regressions for defects found in adversarial review of the fix."""

    @staticmethod
    def _dim_bayer16_buffer(scene, rng):
        """A realistic 'operator still focusing / lamp low' first frame:
        12-bit codes <= ~140 with ~6-code read noise."""
        cfa = mosaic(scene, "rggb")
        v12 = (cfa / 255.0 * 140.0) + rng.normal(0, 6, cfa.shape)
        v12 = np.clip(v12, 0, 4095).astype("<u2")
        return np.frombuffer(v12.tobytes(), dtype=np.uint8)

    def test_dim_first_frame_does_not_cache_wrong_format(self, scene):
        """A dim first frame must NOT commit to a wrong format (e.g. uyvy);
        detection defers until a contrasty frame arrives."""
        rng = np.random.default_rng(2)
        cam = SmartCamCamera()
        cam._decode(self._dim_bayer16_buffer(scene, rng))
        assert cam._fmt is None or cam._fmt["kind"] == "bayer16", (
            f"dim frame cached wrong format: {cam._fmt}"
        )
        # A later bright frame must be detected and decoded correctly
        rgb = cam._decode(bayer16_buffer(scene, "rggb"))
        # detection may be throttled after a deferral; decode a few frames
        for _ in range(6):
            if cam._fmt is not None:
                break
            rgb = cam._decode(bayer16_buffer(scene, "rggb"))
        assert cam._fmt is not None and cam._fmt["kind"] == "bayer16"
        assert rgb.shape == (H4K, W4K, 3)
        assert mae(lum(rgb), lum(scene)) < 8.0

    def test_dim_then_bright_left_aligned_no_wraparound(self, scene):
        """The 12-bit shift must not be frozen from a dim first frame:
        left-aligned data seen dim-first must still decode a later bright
        frame without wrap-around banding."""
        dim = np.clip(scene.astype(np.float32) * 0.04, 0, 255).astype(np.uint8)
        cam = SmartCamCamera(pixel_format="bayer16:rggb")
        cam._decode(bayer16_buffer(dim, "rggb", left_aligned=True))
        bright = cam._decode(bayer16_buffer(scene, "rggb", left_aligned=True))
        assert bright is not None
        assert mae(bright, scene) < 6.0, (
            "bright frame corrupted by a stale cached shift"
        )

    def test_bggr_auto_detect_recovers_structure_and_warns(self, scene, caplog):
        """R/B orientation is not blindly determinable; auto-detect must
        still recover the image structure and must WARN about the
        ambiguity so the operator knows to check colors."""
        import logging as _logging
        cam = SmartCamCamera()
        with caplog.at_level(_logging.WARNING, logger="camera.smartcam_camera"):
            rgb = cam._decode(bayer16_buffer(scene, "bggr"))
        assert rgb is not None
        assert rgb.shape == (H4K, W4K, 3)
        # The green plane is identical under an R/B orientation swap, so
        # it must be recovered regardless of which orientation was chosen
        g_corr = float(np.corrcoef(
            rgb[..., 1].ravel().astype(np.float32),
            scene[..., 1].ravel().astype(np.float32),
        )[0, 1])
        assert g_corr > 0.97
        assert any("SMARTCAM_PIXEL_FORMAT" in r.message for r in caplog.records), (
            "no ambiguity warning pointing at the override was logged"
        )


# ---------------------------------------------------------------------------
# GenTL path: components must be demosaiced per their pixel format
# ---------------------------------------------------------------------------

class TestGenTLComponentToRGB:
    def test_bayer8_component_demosaiced(self, scene):
        from camera.zeiss_camera import _component_to_rgb

        small = cv2.resize(scene, (640, 480), interpolation=cv2.INTER_AREA)
        cfa = mosaic(small, "rggb").astype(np.uint8).ravel()
        rgb = _component_to_rgb(cfa, 640, 480, "BayerRG8")
        assert rgb.shape == (480, 640, 3)
        assert mae(rgb, small) < 6.0

    def test_rgb8_component_passthrough(self, scene):
        from camera.zeiss_camera import _component_to_rgb

        small = cv2.resize(scene, (640, 480), interpolation=cv2.INTER_AREA)
        rgb = _component_to_rgb(small.ravel(), 640, 480, "RGB8")
        assert rgb.shape == (480, 640, 3)
        assert mae(rgb, small) < 1.0

    def test_mono8_component_gray(self, scene):
        from camera.zeiss_camera import _component_to_rgb

        small = cv2.resize(scene, (640, 480), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        rgb = _component_to_rgb(gray.ravel(), 640, 480, "Mono8")
        assert rgb.shape == (480, 640, 3)
        assert np.array_equal(rgb[..., 0], gray)

    def test_packed12_component_unpacked(self, scene):
        """BayerRG12p arrives as raw uint8 bytes (1.5 B/px) — must be
        unpacked, not truncated to the first npix bytes."""
        from camera.zeiss_camera import _component_to_rgb

        small = cv2.resize(scene, (640, 480), interpolation=cv2.INTER_AREA)
        v12 = np.round(mosaic(small, "rggb") / 255.0 * 4095.0).astype(np.uint16)
        flat = v12.ravel()
        p0, p1 = flat[0::2], flat[1::2]
        packed = np.empty(flat.size // 2 * 3, np.uint8)
        packed[0::3] = (p0 >> 4).astype(np.uint8)
        packed[1::3] = (((p0 & 0x0F) << 4) | (p1 >> 8)).astype(np.uint8)
        packed[2::3] = (p1 & 0xFF).astype(np.uint8)
        rgb = _component_to_rgb(packed, 640, 480, "BayerRG12p")
        assert rgb.shape == (480, 640, 3)
        assert mae(rgb, small) < 6.0

    def test_u16_bayer_component_delivered_as_bytes(self, scene):
        """A 12-bit-in-16-bit component handed back as uint8 (2 B/px)
        must be viewed as u16, not decoded as an 8-bit image."""
        from camera.zeiss_camera import _component_to_rgb

        small = cv2.resize(scene, (640, 480), interpolation=cv2.INTER_AREA)
        v12 = np.round(mosaic(small, "rggb") / 255.0 * 4095.0).astype("<u2")
        as_bytes = np.frombuffer(v12.tobytes(), dtype=np.uint8)
        rgb = _component_to_rgb(as_bytes, 640, 480, "BayerRG12")
        assert rgb.shape == (480, 640, 3)
        assert mae(rgb, small) < 6.0

    def test_component_bad_size_raises(self):
        """A component whose size fits no known layout must raise loudly,
        not return a plausible-looking garbled frame."""
        from camera.zeiss_camera import CameraError, _component_to_rgb

        bad = np.zeros(int(640 * 480 * 1.2), np.uint8)
        with pytest.raises(CameraError):
            _component_to_rgb(bad, 640, 480, "BayerRG8")
