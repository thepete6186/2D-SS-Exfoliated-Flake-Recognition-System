"""
SmartCamApi.dll camera driver for Zeiss Axiocam 208 Color
==========================================================

Native camera access via Zeiss's proprietary SmartCamApi.dll.

The Axiocam 208 Color is a 4K (3840x2160, 8.3 MP) color camera.
GetAcquisitionBufferSize returns 16,588,800 bytes = 3840*2160*2 —
a single 4K frame at 2 bytes/pixel (12-bit Bayer data in 16-bit
words, or possibly YUY2 depending on firmware). The buffer is
decoded/demosaiced to full-color RGB.

NOTE: 16,588,800 also factors as 4608*3600*1 and as 6*(1920*960*1.5);
earlier revisions picked those interpretations. The 1920x960 packed
read consumed only the first 2,764,800 bytes (the top ~360 of 2160
sensor rows) with the wrong row stride, and its 3-byte unpack beat
against the true 2-byte words with a 6-byte period — producing
sheared, rainbow-striped garbage whose shape also changed frame to
frame because the format search re-ran per frame. The format is now
auto-detected once on the first judgeable frame by scoring every
plausible decode for spatial coherence, then cached. Override with
the SMARTCAM_PIXEL_FORMAT env var or the pixel_format constructor
argument, e.g. "bayer16:rggb", "bayer16:bggr", "yuy2".

Performance notes:
  - Format detection runs ONCE, then every frame uses the cached
    format (no per-frame searching).
  - capture() returns a full-resolution HxWx3 uint8 RGB array.

Requirements:
    - SmartCamApi.dll (ships with Labscope)
    - libusb0.dll (the driver bound to the camera)
    - Camera connected via USB 3.0
    - Labscope / ZEN closed (exclusive USB access)

Usage:
    cam = SmartCamCamera()
    cam.connect()
    rgb = cam.capture()   # np.ndarray (2160, 3840, 3) uint8 RGB, or None
    cam.disconnect()
"""

import ctypes
import logging
import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SmartCamApi.dll location
# ---------------------------------------------------------------------------
_SMARTCAM_DLL_PATHS = [
    # Labscope driver installer location
    r"C:\Program Files\Carl Zeiss\Labscope\DriverInstaller\Primostar3HD_Axiocam202_208\SmartCamApi.dll",
    # Common Labscope install locations
    r"C:\Program Files\Carl Zeiss\Labscope\SmartCamApi.dll",
    r"C:\Program Files (x86)\Carl Zeiss\Labscope\SmartCamApi.dll",
]


def _find_smartcam_dll() -> Optional[str]:
    """Locate SmartCamApi.dll."""
    for p in _SMARTCAM_DLL_PATHS:
        if os.path.isfile(p):
            return p
    for root in [r"C:\Program Files\Carl Zeiss", r"C:\Program Files (x86)\Carl Zeiss"]:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if fn.lower() == "smartcamapi.dll":
                        return os.path.join(dirpath, fn)
    return None


# Literal CFA layout (top-left 2x2, row-major) -> OpenCV demosaic code.
# OpenCV names its Bayer codes by the SECOND row/column, so literal RGGB
# needs COLOR_BayerBG2RGB (verified empirically against synthetic CFAs).
_LITERAL_BAYER_TO_CV2 = {
    "rggb": cv2.COLOR_BayerBG2RGB,
    "bggr": cv2.COLOR_BayerRG2RGB,
    "grbg": cv2.COLOR_BayerGB2RGB,
    "gbrg": cv2.COLOR_BayerGR2RGB,
}

# Bayer patterns that decode with identical spatial statistics but with
# red and blue exchanged — indistinguishable without a scene prior.
_RB_SWAPPED = {"rggb": "bggr", "bggr": "rggb", "grbg": "gbrg", "gbrg": "grbg"}


def _unpack12(packed: np.ndarray, w: int, h: int) -> np.ndarray:
    """Unpack 12-bit packed data (3 bytes -> 2 pixels) to uint16 (h, w)."""
    b = packed.reshape(-1, 3)
    p0 = (b[:, 0].astype(np.uint16) << 4) | (b[:, 1].astype(np.uint16) >> 4)
    p1 = ((b[:, 1].astype(np.uint16) & 0x0F) << 8) | b[:, 2].astype(np.uint16)
    pixels = np.empty(b.shape[0] * 2, dtype=np.uint16)
    pixels[0::2] = p0
    pixels[1::2] = p1
    return pixels.reshape(h, w)


class SmartCamError(Exception):
    """SmartCamApi errors."""
    pass


class SmartCamCamera:
    """
    Camera driver using Zeiss SmartCamApi.dll.

    Provides direct native access to the Axiocam 208 via Zeiss's
    proprietary protocol (libusb0). No screen capture, no Labscope
    dependency, full sensor resolution.
    """

    # Native 4K live resolution of the Axiocam 208 Color
    EXPECTED_RESOLUTION = (3840, 2160)

    # Live buffer: one 3840x2160 frame at 2 bytes/pixel
    LIVE_RESOLUTION = (3840, 2160)
    LIVE_BYTES = 3840 * 2160 * 2  # 16,588,800

    # Frames that are too flat to score before we give up and cache
    # the default format anyway
    _MAX_DETECT_ATTEMPTS = 30

    def __init__(self, camera_index: int = 0,
                 pixel_format: Optional[str] = None):
        self.camera_index = camera_index
        self.connected = False
        self.backend = "smartcam"

        # Optional forced pixel format ("bayer16:rggb", "yuy2", ...).
        # Falls back to the SMARTCAM_PIXEL_FORMAT environment variable.
        self._forced_format_str = (
            pixel_format or os.environ.get("SMARTCAM_PIXEL_FORMAT") or None
        )
        self._detect_attempts = 0
        self._skip_detect = 0

        # DLL state
        self._dll = None
        self._camera_handle = None
        self._lib_initialized = False

        # Last known raw buffer size from the API (bytes)
        self._last_buffer_size = 0

        # Continuous acquisition state
        self._acquisition_started = False

        # Cached decode format: dict or None until first frame detected
        self._fmt = None

        # Camera settings
        self.settings: Dict[str, Any] = {
            "exposure_ms": 10.0,
            "gain": 0.0,
            "resolution": list(self.EXPECTED_RESOLUTION),
        }

        # Find and load the DLL
        self._dll_path = _find_smartcam_dll()
        if self._dll_path:
            try:
                self._dll = ctypes.WinDLL(self._dll_path)
                self._setup_prototypes()
                logger.info("Loaded SmartCamApi.dll via WinDLL from %s", self._dll_path)
            except Exception as e:
                logger.warning("WinDLL load/setup failed (%s), trying CDLL...", e)
                try:
                    self._dll = ctypes.CDLL(self._dll_path)
                    self._setup_prototypes()
                    logger.info("Loaded SmartCamApi.dll via CDLL from %s", self._dll_path)
                except Exception as err:
                    logger.error("Failed to load SmartCamApi.dll: %s", err)
                    self._dll = None
        else:
            logger.error("SmartCamApi.dll not found")

    # ------------------------------------------------------------------
    # DLL function prototypes
    # ------------------------------------------------------------------
    def _setup_prototypes(self):
        """Set up ctypes function prototypes before any API calls."""
        dll = self._dll
        if dll is None:
            return

        dll.ApiLib_InitializeLibrary.restype = ctypes.c_int
        dll.ApiLib_InitializeLibrary.argtypes = []

        dll.ApiLib_FinalizeLibrary.restype = ctypes.c_int
        dll.ApiLib_FinalizeLibrary.argtypes = []

        dll.ApiLib_GetCameraCount.restype = ctypes.c_int
        dll.ApiLib_GetCameraCount.argtypes = [ctypes.POINTER(ctypes.c_int)]

        dll.ApiCam_OpenCamera.restype = ctypes.c_int
        dll.ApiCam_OpenCamera.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]

        dll.ApiCam_CloseCamera.restype = ctypes.c_int
        dll.ApiCam_CloseCamera.argtypes = [ctypes.c_void_p]

        dll.ApiCam_GetAcquisitionBufferSize.restype = ctypes.c_int
        dll.ApiCam_GetAcquisitionBufferSize.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]

        dll.ApiCam_StartSequenceAcquisition.restype = ctypes.c_int
        dll.ApiCam_StartSequenceAcquisition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

        dll.ApiCam_StartContinuousAcquisition.restype = ctypes.c_int
        dll.ApiCam_StartContinuousAcquisition.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]

        dll.ApiCam_GetSequenceImage.restype = ctypes.c_int
        dll.ApiCam_GetSequenceImage.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]

        dll.ApiCam_AbortAcquisition.restype = ctypes.c_int
        dll.ApiCam_AbortAcquisition.argtypes = [ctypes.c_void_p]

        dll.ApiLib_GetErrorDescription.restype = ctypes.c_int
        dll.ApiLib_GetErrorDescription.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]

        # Optional export — set_exposure/set_gain call this; without a
        # prototype ctypes would marshal the c_double argument wrongly
        try:
            dll.ApiCam_SetParameterValue.restype = ctypes.c_int
            dll.ApiCam_SetParameterValue.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_double,
            ]
        except AttributeError:
            pass

    def _error_string(self, code: int) -> str:
        """Get error description for a code."""
        if self._dll is None:
            return f"Error {code}"
        try:
            buf = ctypes.create_string_buffer(256)
            self._dll.ApiLib_GetErrorDescription(code, buf, 256)
            return buf.value.decode("ascii", "replace")
        except Exception:
            return f"Error {code}"

    def _handle_value(self) -> Optional[int]:
        """Return the raw handle value, or None."""
        if self._camera_handle is None:
            return None
        return self._camera_handle.value

    # ------------------------------------------------------------------
    # Format detection (once, cached) + deterministic per-frame decode
    # ------------------------------------------------------------------

    # A decode is accepted only if its high-frequency ratio is below this.
    # Correct decodes of real scenes score ~0.01-0.3 (structure is much
    # larger than pixel-to-pixel noise); misinterpretations score >~1
    # because their "detail" IS pixel-to-pixel noise.
    _ACCEPT_RATIO = 0.5

    @staticmethod
    def _u16_shift(arr16: np.ndarray) -> int:
        """Bits to right-shift a 16-bit container down to 8 bits for THIS
        frame. 12-bit right-aligned data (max 4095) needs >>4; left-aligned
        or true 16-bit data needs >>8. Computed per frame — never cached —
        so a dim first frame cannot corrupt later bright frames."""
        return 4 if int(arr16.max()) <= 4095 else 8

    @staticmethod
    def _coherence_score(rgb: Optional[np.ndarray]) -> Optional[float]:
        """Contrast-normalized high-frequency ratio on a center crop;
        LOWER = more image-like. Returns None when the crop is too flat
        to judge (e.g. lens cap on) or the decode failed. Normalizing by
        the crop's std keeps dim mis-decodes (whose byte-interleave
        "texture" scales down with the signal) from out-scoring a
        correct-but-flat decode."""
        if rgb is None or rgb.size == 0:
            return None
        h, w = rgb.shape[:2]
        ch, cw = min(h, 512), min(w, 512)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        f = rgb[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
        std = float(f.std())
        if std < 2.0:
            return None
        row_diff = float(np.mean(np.abs(f[1:] - f[:-1])))
        col_diff = float(np.mean(np.abs(f[:, 1:] - f[:, :-1])))
        return (row_diff + col_diff) / std

    @staticmethod
    def _plausible_yuv422(raw: np.ndarray, fmt: Dict[str, Any]) -> bool:
        """Sanity-check a YUV422 candidate via its chroma bytes. Real
        YUV422 chroma sits near 128 (neutral); Bayer/gray 16-bit data
        misread as YUV422 puts its near-empty high bytes in the chroma
        slots (mean ~0), which decodes to a smooth but violently
        color-cast image that can out-score the truth on dim frames."""
        n = fmt["w"] * fmt["h"] * 2
        if raw.size < n:
            return False
        chroma_offsets = (1, 3) if fmt["kind"] == "yuy2" else (0, 2)
        luma_offsets = (0, 2) if fmt["kind"] == "yuy2" else (1, 3)
        for off in chroma_offsets:
            mean = float(np.mean(raw[off:n:4]))
            if abs(mean - 128.0) > 64.0:
                return False
        # Real video also has non-degenerate luma; 16-bit data misread
        # as YUV422 leaves the near-empty high bytes in the luma slots
        luma = np.concatenate([raw[off:n:4][::16] for off in luma_offsets])
        if float(np.mean(luma)) < 16.0 and float(np.std(luma)) < 2.0:
            return False
        return True

    @staticmethod
    def _parse_pixel_format(spec: str) -> Optional[Dict[str, Any]]:
        """Parse "bayer16:rggb", "yuy2", "packed12:grbg@1920x960", ..."""
        try:
            spec = spec.strip().lower()
            res = None
            if "@" in spec:
                spec, res_s = spec.split("@", 1)
                w_s, h_s = res_s.split("x")
                res = (int(w_s), int(h_s))
            parts = spec.split(":")
            kind = parts[0]
            pattern = parts[1] if len(parts) > 1 else "rggb"
            if pattern not in _LITERAL_BAYER_TO_CV2:
                logger.warning("Unknown Bayer pattern %r; using rggb", pattern)
                pattern = "rggb"
            w, h = res or ((1920, 960) if kind == "packed12" else (3840, 2160))
            if kind in ("bayer16", "yuy2", "uyvy", "gray16", "gray8",
                        "bayer8", "packed12", "nv12"):
                fmt: Dict[str, Any] = {"kind": kind, "w": w, "h": h}
                if kind in ("bayer16", "bayer8", "packed12"):
                    fmt["pattern"] = pattern
                return fmt
            logger.warning("Unknown pixel format kind %r", kind)
            return None
        except Exception as e:
            logger.warning("Could not parse pixel format %r: %s", spec, e)
            return None

    def _candidate_formats(self, n: int) -> list:
        """Plausible format interpretations for an n-byte buffer, in
        priority order (ties in scoring resolve to the earliest).

        Byte analysis of the Axiocam 208 shows the data is 8-bit per
        pixel (even/odd bytes have identical statistics). The buffer may
        also contain multiple copies of a smaller frame (e.g. 8 copies
        of 1920x1080 in a 16,588,800-byte buffer), so modulo is used to
        match sub-frame sizes.

        Channel analysis of decoded images shows the data is GRAYSCALE
        (all Bayer patterns produce identical R=G=B channels), so gray8
        is prioritized over bayer8.
        """
        cands = []
        patterns = list(_LITERAL_BAYER_TO_CV2)

        def add(kind, w, h, pattern=None):
            fmt = {"kind": kind, "w": w, "h": h}
            if pattern:
                fmt["pattern"] = pattern
            cands.append(fmt)

        # 8-bit grayscale — channel analysis shows the data is grayscale.
        # The Axiocam 208 live view is 1920x1080 grayscale.
        # Use >= so a buffer with extra padding still matches.
        for w, h in [(1920, 1080), (3840, 2160), (4608, 3600)]:
            if n >= w * h:
                add("gray8", w, h)

        # 8-bit Bayer — kept as fallback in case a color mode is enabled.
        for w, h in [(1920, 1080), (3840, 2160), (4608, 3600)]:
            if n % (w * h) == 0:
                for p in patterns:  # rggb first (typical for Sony sensors)
                    add("bayer8", w, h, p)

        # 16-bit interpretations (less likely given byte analysis)
        for w, h in [(3840, 2160), (1920, 1080)]:
            if n % (w * h * 2) == 0:
                for p in patterns:
                    add("bayer16", w, h, p)
                add("yuy2", w, h)
                add("uyvy", w, h)
                add("gray16", w, h)

        # NV12 (YUV420 semi-planar) — 3,110,400 bytes = 1920x1080x1.5.
        # User confirmed this decode looks accurate.
        for w, h in [(1920, 1080), (3840, 2160)]:
            if n == w * h * 3 // 2:
                add("nv12", w, h)

        # 12-bit packed interpretations (3 bytes per 2 pixels).
        # Byte analysis showed the actual data extent is ~3,110,400 bytes
        # = 1920x1080 12-bit packed. Only add when the buffer size EXACTLY
        # matches, so a 16,588,800-byte buffer doesn't get misinterpreted.
        for w, h in [(1920, 1080), (1920, 960)]:
            if n == w * h * 3 // 2:
                for p in patterns:
                    add("packed12", w, h, p)
        return cands

    def _decode_with(self, raw: np.ndarray,
                     fmt: Dict[str, Any]) -> Optional[np.ndarray]:
        """Decode raw uint8 buffer -> HxWx3 uint8 RGB per fmt. No searching."""
        try:
            kind, w, h = fmt["kind"], fmt["w"], fmt["h"]
            if kind == "bayer16":
                arr16 = raw[: w * h * 2].view("<u2").reshape(h, w)
                bayer8 = (arr16 >> self._u16_shift(arr16)).astype(np.uint8)
                return cv2.cvtColor(
                    bayer8, _LITERAL_BAYER_TO_CV2[fmt["pattern"]]
                )
            if kind in ("yuy2", "uyvy"):
                arr = raw[: w * h * 2].reshape(h, w, 2)
                code = (cv2.COLOR_YUV2RGB_YUY2 if kind == "yuy2"
                        else cv2.COLOR_YUV2RGB_UYVY)
                return cv2.cvtColor(arr, code)
            if kind == "gray16":
                arr16 = raw[: w * h * 2].view("<u2").reshape(h, w)
                gray = (arr16 >> self._u16_shift(arr16)).astype(np.uint8)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if kind == "bayer8":
                bayer = raw[: w * h].reshape(h, w)
                return cv2.cvtColor(
                    bayer, _LITERAL_BAYER_TO_CV2[fmt["pattern"]]
                )
            if kind == "gray8":
                gray = raw[: w * h].reshape(h, w)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            if kind == "nv12":
                # NV12: Y plane (h*w) + interleaved UV (h*w/2)
                nv12 = raw[: w * h * 3 // 2].reshape(h * 3 // 2, w)
                return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)
            if kind == "packed12":
                img12 = _unpack12(raw[: w * h * 3 // 2], w, h)
                bayer8 = (img12 >> 4).astype(np.uint8)
                # Allow manual override via SMARTCAM_PIXEL_FORMAT
                if self._forced_format_str and "packed12" in self._forced_format_str:
                    pattern = fmt.get("pattern", "rggb")
                    code = _LITERAL_BAYER_TO_CV2.get(pattern)
                    if code:
                        return cv2.cvtColor(bayer8, code)
                # Try all 4 Bayer patterns, pick the one with most
                # natural-looking colors (highest R/G/B variance)
                best_rgb, best_var = None, -1
                for lit, code in _LITERAL_BAYER_TO_CV2.items():
                    try:
                        rgb = cv2.cvtColor(bayer8, code)
                        r, g, b = cv2.split(rgb)
                        var = float(np.var([r.mean(), g.mean(), b.mean()]))
                        if var > best_var:
                            best_var = var
                            best_rgb = rgb
                    except Exception:
                        pass
                if best_rgb is not None:
                    return best_rgb
                return None
            logger.error("Unknown format kind %r", kind)
            return None
        except Exception as e:
            logger.error("Decode failed for %s: %s", fmt, e)
            return None

    def _resolve_format(self, raw: np.ndarray) -> Optional[Dict[str, Any]]:
        """Pick the buffer format. Returns None to retry on a later
        (more contrasty) frame; otherwise returns the format to cache."""
        if self._forced_format_str:
            fmt = self._parse_pixel_format(self._forced_format_str)
            if fmt is not None:
                logger.info("Using forced pixel format: %s", fmt)
                return fmt

        n = int(raw.size)

        # KNOWN-CORRECT FORMAT: The Axiocam 208 live view is 1920x1080
        # NV12 (YUV420 semi-planar) — user confirmed this looks accurate.
        # The buffer is trimmed to 3,110,400 bytes in _decode().
        if n == 1920 * 1080 * 3 // 2:
            logger.info("Using known format: nv12 1920x1080")
            return {"kind": "nv12", "w": 1920, "h": 1080}

        # Fallback: grayscale 8-bit for any other size
        if n >= 1920 * 1080:
            logger.info("Using fallback format: gray8 1920x1080")
            return {"kind": "gray8", "w": 1920, "h": 1080}

        cands = self._candidate_formats(n)
        if not cands:
            logger.warning("No known format for buffer size=%s; "
                           "falling back to gray8", n)
            side = int(np.sqrt(n))
            return {"kind": "gray8", "w": side, "h": n // side}

        best_fmt, best_ratio = None, None
        for fmt in cands:
            if fmt["kind"] in ("yuy2", "uyvy") and not \
                    self._plausible_yuv422(raw, fmt):
                continue
            rgb = self._decode_with(raw, dict(fmt))
            ratio = self._coherence_score(rgb)
            if ratio is None:
                continue
            if best_ratio is None or ratio < best_ratio:
                best_fmt, best_ratio = fmt, ratio

        # Accept only a decode that actually looks like an image. A frame
        # where every candidate is flat OR noisy (dim, defocused, lens
        # cap) must NOT commit a format — a mis-decode of a dim frame can
        # otherwise out-score the correct-but-flat decode and get cached
        # for the whole session.
        if best_fmt is None or best_ratio >= self._ACCEPT_RATIO:
            self._detect_attempts += 1
            if self._detect_attempts < self._MAX_DETECT_ATTEMPTS:
                logger.info(
                    "Frame not judgeable for format detection (best "
                    "ratio=%s, attempt %s); using default this frame",
                    "n/a" if best_ratio is None else f"{best_ratio:.2f}",
                    self._detect_attempts,
                )
                return None
            best_fmt = cands[0]
            logger.warning("Format detection inconclusive after %s "
                           "frames; committing to default %s",
                           self._detect_attempts, best_fmt)
            return best_fmt

        logger.info("Detected pixel format %s (ratio=%.3f)",
                    best_fmt, best_ratio)
        if best_fmt["kind"] in ("bayer16", "bayer8", "packed12"):
            # The R/B-swapped pattern scores identically by construction —
            # orientation cannot be determined from image statistics alone.
            pattern = best_fmt["pattern"]
            partner = _RB_SWAPPED[pattern]
            logger.warning(
                "Bayer R/B orientation is ambiguous (chose %s over %s). "
                "If red and blue look swapped in the live view, set "
                "SMARTCAM_PIXEL_FORMAT=%s:%s (or pass pixel_format=) to "
                "flip it.",
                pattern, partner, best_fmt["kind"], partner,
            )
        return best_fmt

    def _decode(self, raw: np.ndarray) -> Optional[np.ndarray]:
        """Decode raw buffer -> HxWx3 uint8 RGB using the cached format.

        The SmartCamApi buffer is a fixed 16,588,800-byte allocation, but
        the camera only fills the portion containing the actual frame —
        the rest is zero padding (byte analysis showed 81.2% zeros).
        Trim to the last non-zero byte so format detection sees the real
        frame size (e.g. 3,110,400 bytes = 1920x1080 12-bit packed).
        """
        raw = np.ascontiguousarray(raw)
        if raw.size == 0:
            return None

        # Trim trailing zero padding to find the actual data extent
        nz = np.nonzero(raw)[0]
        if nz.size > 0:
            actual = int(nz[-1]) + 1
            if actual < raw.size:
                raw = raw[:actual]

        if self._fmt is None:
            if self._skip_detect > 0:
                self._skip_detect -= 1
            else:
                fmt = self._resolve_format(raw)
                if fmt is not None:
                    self._fmt = fmt
                    self.settings["resolution"] = (fmt["w"], fmt["h"])
                else:
                    # Scoring all candidates is expensive; while frames
                    # stay unjudgeable, only retry every few frames.
                    self._skip_detect = 4
            if self._fmt is None:
                cands = self._candidate_formats(int(raw.size))
                if not cands:
                    return None
                return self._decode_with(raw, cands[0])
        return self._decode_with(raw, self._fmt)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """Open the camera device via SmartCamApi.dll."""
        if self._dll is None:
            logger.error("SmartCamApi.dll not loaded")
            return False

        if self.is_connected:
            return True

        try:
            rc = self._dll.ApiLib_InitializeLibrary()
            if rc != 0:
                logger.error("ApiLib_InitializeLibrary failed: %s", self._error_string(rc))
                return False
            self._lib_initialized = True

            count = ctypes.c_int(0)
            rc = self._dll.ApiLib_GetCameraCount(ctypes.byref(count))
            if rc != 0:
                logger.error("ApiLib_GetCameraCount failed: %s", self._error_string(rc))
                self._cleanup()
                return False
            logger.info("SmartCamApi: %s camera(s) found", count.value)

            if count.value == 0:
                logger.error("No cameras found (close Labscope/ZEN; check USB3 + libusb0)")
                self._cleanup()
                return False

            if self.camera_index >= count.value:
                logger.error("camera_index %s out of range (%s camera(s))",
                             self.camera_index, count.value)
                self._cleanup()
                return False

            handle = ctypes.c_void_p()
            rc = self._dll.ApiCam_OpenCamera(self.camera_index, ctypes.byref(handle))
            if rc != 0 or not handle.value:
                logger.error("ApiCam_OpenCamera failed: rc=%s (%s), handle=%s",
                             rc, self._error_string(rc), handle.value)
                self._cleanup()
                return False

            self._camera_handle = handle
            self.connected = True
            self._acquisition_started = False
            self._fmt = None
            self._detect_attempts = 0
            self._skip_detect = 0
            logger.info("Connected to camera via SmartCamApi (handle=%s, index=%s)",
                        handle.value, self.camera_index)

            time.sleep(0.15)
            return True

        except Exception as e:
            logger.error("SmartCamApi connect failed: %s", e)
            self._cleanup()
            return False

    def _cleanup(self):
        """Internal cleanup. Always safe to call."""
        if self._dll is not None and self._camera_handle is not None:
            try:
                if self._camera_handle.value:
                    self._dll.ApiCam_AbortAcquisition(self._camera_handle)
            except Exception:
                pass
            try:
                if self._camera_handle.value:
                    self._dll.ApiCam_CloseCamera(self._camera_handle)
            except Exception:
                pass
        self._camera_handle = None
        self._acquisition_started = False
        self._fmt = None

        if self._lib_initialized and self._dll is not None:
            try:
                self._dll.ApiLib_FinalizeLibrary()
            except Exception:
                pass
            self._lib_initialized = False

        self.connected = False

    # ------------------------------------------------------------------
    # Camera Settings
    # ------------------------------------------------------------------
    def set_exposure(self, exposure_us: float) -> None:
        """Set exposure time in microseconds via SmartCamApi SetParameterValue."""
        self.settings["exposure_ms"] = exposure_us / 1000.0
        if not self.is_connected or self._dll is None:
            return
        try:
            # Try common SmartCamApi exposure parameter IDs
            # These are typical Zeiss SmartCam parameter IDs
            for param_id in [9, 10, 11, 12, 13, 14]:
                rc = self._dll.ApiCam_SetParameterValue(
                    self._camera_handle, param_id, ctypes.c_double(exposure_us / 1000.0)
                )
                if rc == 0:
                    logger.info(f"Set exposure to {exposure_us} us via param {param_id}")
                    return
                elif rc not in (5, 13):  # 5=not readable, 13=not writable
                    logger.debug(f"SetParameterValue({param_id}) rc={rc}")
            logger.warning("SmartCamApi: could not set exposure (parameter ID unknown)")
        except Exception as e:
            logger.warning(f"SmartCamApi: set_exposure failed: {e}")

    def set_gain(self, gain_db: float) -> None:
        """Set camera gain in decibels via SmartCamApi SetParameterValue."""
        self.settings["gain"] = gain_db
        if not self.is_connected or self._dll is None:
            return
        try:
            # Try common SmartCamApi gain parameter IDs
            for param_id in [2, 3, 4, 5, 6, 7, 8]:
                rc = self._dll.ApiCam_SetParameterValue(
                    self._camera_handle, param_id, ctypes.c_double(gain_db)
                )
                if rc == 0:
                    logger.info(f"Set gain to {gain_db} dB via param {param_id}")
                    return
                elif rc not in (5, 13):
                    logger.debug(f"SetParameterValue({param_id}) rc={rc}")
            logger.warning("SmartCamApi: could not set gain (parameter ID unknown)")
        except Exception as e:
            logger.warning(f"SmartCamApi: set_gain failed: {e}")

    def disconnect(self) -> None:
        """Release the camera device."""
        self._cleanup()
        logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
        return (
            self.connected
            and self._camera_handle is not None
            and self._camera_handle.value is not None
        )

    # ------------------------------------------------------------------
    # Capture (sequence acquisition per frame, cached buffer)
    # ------------------------------------------------------------------
    def capture(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        """
        Capture a single frame as HxWx3 uint8 RGB (or None on failure).

        Uses the original working sequence-acquisition flow:
        1. Start sequence acquisition for 1 image (with our cached buffer)
        2. Poll GetSequenceImage(handle, 0, buffer, size) — this call both
           checks readiness AND fills the buffer when the image is ready.

        The 16.5 MB buffer is allocated ONCE and reused, and the Bayer
        format is detected once and cached, so later frames are fast.
        """
        if not self.is_connected:
            logger.error("Camera not connected")
            return None

        handle = self._camera_handle
        index_as_void = ctypes.c_void_p(self.camera_index)

        try:
            # --- Determine buffer size (queried once, then cached) ---
            if self._last_buffer_size > 0:
                size = self._last_buffer_size
            else:
                buf_size = ctypes.c_int(0)
                rc = self._dll.ApiCam_GetAcquisitionBufferSize(
                    handle, 1, ctypes.byref(buf_size)
                )
                if rc != 0 or buf_size.value <= 0:
                    # 2 bytes/pixel — an undersized buffer here would let
                    # the DLL write past the end of it
                    buf_size = ctypes.c_int(self.LIVE_BYTES)
                size = int(buf_size.value)
                self._last_buffer_size = size

            # Allocate the reusable 16.5 MB buffer ONCE
            if not hasattr(self, "_acq_buffer") or self._acq_buffer_size != size:
                self._acq_buffer = ctypes.create_string_buffer(size)
                self._acq_buffer_size = size

            # --- Start sequence acquisition for 1 image ---
            rc = self._dll.ApiCam_StartSequenceAcquisition(
                handle, 1, size, self._acq_buffer
            )
            if rc != 0:
                logger.debug(
                    "StartSequenceAcquisition(handle) failed (%s); try index",
                    self._error_string(rc),
                )
                rc = self._dll.ApiCam_StartSequenceAcquisition(
                    index_as_void, 1, size, self._acq_buffer
                )
            if rc != 0:
                logger.error("ApiCam_StartSequenceAcquisition failed: %s",
                             self._error_string(rc))
                try:
                    self._dll.ApiCam_AbortAcquisition(handle)
                except Exception:
                    pass
                return None

            # --- Poll for the image (fills buffer when ready) ---
            start = time.time()
            while True:
                rc = self._dll.ApiCam_GetSequenceImage(
                    handle, 0, self._acq_buffer, size
                )
                if rc == 0:
                    break
                if rc == 20:  # image not ready
                    if (time.time() - start) * 1000 > timeout_ms:
                        logger.error("Timeout waiting for image")
                        try:
                            self._dll.ApiCam_AbortAcquisition(handle)
                        except Exception:
                            pass
                        return None
                    time.sleep(0.005)
                    continue
                logger.error("ApiCam_GetSequenceImage failed: %s",
                             self._error_string(rc))
                try:
                    self._dll.ApiCam_AbortAcquisition(handle)
                except Exception:
                    pass
                return None

            raw = np.frombuffer(
                self._acq_buffer, dtype=np.uint8, count=size
            ).copy()
            rgb = self._decode(raw)
            if rgb is None:
                logger.error("Failed to decode raw frame (%s bytes)", raw.size)
                return None
            return rgb

        except Exception as e:
            logger.error("SmartCamApi capture failed: %s", e, exc_info=True)
            try:
                if self._camera_handle is not None and self._camera_handle.value:
                    self._dll.ApiCam_AbortAcquisition(self._camera_handle)
            except Exception:
                pass
            return None

    # ------------------------------------------------------------------
    # Camera Info
    # ------------------------------------------------------------------
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        return {
            "backend": self.backend,
            "connected": self.connected,
            "resolution": self.settings["resolution"],
            "name": "Zeiss Axiocam 208 (SmartCamApi)",
            "model": "Axiocam 208",
            "handle": self._handle_value(),
            "index": self.camera_index,
            "last_buffer_size": self._last_buffer_size,
        }

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return f"SmartCamCamera(index={self.camera_index}, {status})"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_smartcam():
    """Test SmartCamApi camera access."""
    logging.basicConfig(level=logging.DEBUG)
    cam = SmartCamCamera()
    print(f"Backend: {cam.backend}")
    print(f"DLL path: {cam._dll_path}")

    if cam.connect():
        print("[OK] Connected!")
        info = cam.get_camera_info()
        print(f"  Info: {info}")

        print("Capturing 3 frames to measure speed...")
        start = time.time()
        for i in range(3):
            data = cam.capture()
            if data is not None:
                print(f"  Frame {i}: shape={data.shape} dtype={data.dtype} "
                      f"min/max={data.min()}/{data.max()}")
            else:
                print(f"  Frame {i}: FAILED")
        elapsed = time.time() - start
        print(f"  3 frames in {elapsed:.2f}s -> {3.0 / elapsed:.1f} fps")
        try:
            bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            cv2.imwrite("_debug_frame.png", bgr)
            print("  Saved _debug_frame.png")
        except Exception as e:
            print(f"  Could not save debug PNG: {e}")

        cam.disconnect()
        print("[OK] Disconnected")
    else:
        print("[FAIL] Connection failed")
        print("  (Is Labscope running and holding the camera?)")


if __name__ == "__main__":
    test_smartcam()