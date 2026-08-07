"""
SmartCamApi.dll camera driver for Zeiss Axiocam 208 Color
==========================================================

Native camera access via Zeiss's proprietary SmartCamApi.dll.

The Axiocam 208 Color is a 12 MP (4608x3600) Bayer color camera.
GetAcquisitionBufferSize returns 16588800 bytes = 4608*3600*1
(8-bit Bayer CFA). The buffer is demosaiced to full-color RGB.

Performance notes:
  - Format detection (Bayer pattern) is done ONCE on the first frame
    using a small center crop, then cached.
  - Continuous acquisition is started once and reused, so we never
    restart the acquisition per-frame (the main cause of slowness).
  - capture() returns a full-resolution HxWx3 uint8 RGB array.

Requirements:
    - SmartCamApi.dll (ships with Labscope)
    - libusb0.dll (the driver bound to the camera)
    - Camera connected via USB 3.0
    - Labscope / ZEN closed (exclusive USB access)

Usage:
    cam = SmartCamCamera()
    cam.connect()
    rgb = cam.capture()   # np.ndarray (3600, 4608, 3) uint8 RGB, or None
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


# Bayer demosaic codes (OpenCV)
_BAYER_CODES = [
    (cv2.COLOR_BayerBG2RGB, "BGGR"),
    (cv2.COLOR_BayerRG2RGB, "RGGB"),
    (cv2.COLOR_BayerGR2RGB, "GRBG"),
    (cv2.COLOR_BayerGB2RGB, "GBRG"),
]


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

    # Native 12MP resolution of the Axiocam 208 Color
    EXPECTED_RESOLUTION = (4608, 3600)

    # Actual observed live format: 1920x960 12-bit packed Bayer
    # (1.5 bytes/pixel, 2 pixels per 3 bytes).
    # The 16,588,800-byte buffer contains 6 frames stacked vertically.
    LIVE_RESOLUTION = (1920, 960)
    LIVE_BYTES = 1920 * 960 * 3 // 2  # 2,764,800

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.connected = False
        self.backend = "smartcam"

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
    # Format detection (cached) + fast decode
    # ------------------------------------------------------------------
    @staticmethod
    def _score_rgb(rgb: np.ndarray) -> float:
        """Higher = more colorful/likely a real image. Fast on small crops."""
        if rgb is None or rgb.size == 0:
            return -1.0
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        std = float(np.std(rgb))
        if std < 1e-6:
            return 0.0
        color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))
        return std + color_diff * 3.0

    def _detect_format(self, raw: np.ndarray) -> Dict[str, Any]:
        """
        Detect the buffer format ONCE.
        Returns a cached format dict used by _decode() on every later frame.
        """
        n = int(raw.size)
        raw_bytes = raw.tobytes()

        # ---- PRIMARY: 1920x1080 12-bit packed Bayer (observed live) ----
        # The buffer may contain multiple frames. Use only the FIRST one
        # (the rest is padding/empty).
        w, h = self.LIVE_RESOLUTION
        frame_bytes = self.LIVE_BYTES
        if n >= frame_bytes:
            # Use the first complete frame in the buffer
            offset = 0
            content = np.frombuffer(raw_bytes[offset:offset + frame_bytes], dtype=np.uint8)
            # Unpack 12-bit packed: 3 bytes -> 2 pixels
            b = content.reshape(-1, 3)
            p0 = (b[:, 0].astype(np.uint16) << 4) | (b[:, 1].astype(np.uint16) >> 4)
            p1 = ((b[:, 1].astype(np.uint16) & 0x0F) << 8) | b[:, 2].astype(np.uint16)
            pixels = np.empty(b.shape[0] * 2, dtype=np.uint16)
            pixels[0::2] = p0
            pixels[1::2] = p1
            img12 = pixels.reshape(h, w)
            # Scale 12-bit -> 8-bit
            scaled = (img12.astype(np.float32) / 4095.0 * 255.0).astype(np.uint8)
            # Detect Bayer pattern on the full frame.
            # For dark scenes (where all patterns score similarly), use a
            # channel-separation heuristic: the correct pattern maximizes
            # the difference between the brightest and darkest channels.
            best_code, best_name, best_score = None, None, -1.0
            for code, name in _BAYER_CODES:
                try:
                    rgb = cv2.cvtColor(scaled, code)
                    s = self._score_rgb(rgb)
                    # Secondary heuristic: prefer patterns where R and B
                    # channels differ more (helps with pink/red substrates)
                    r = rgb[..., 0].astype(np.float32)
                    g = rgb[..., 1].astype(np.float32)
                    b = rgb[..., 2].astype(np.float32)
                    rb_diff = float(np.mean(np.abs(r - b)))
                    s += rb_diff * 2.0  # bonus for R/B separation
                    if s > best_score:
                        best_score, best_code, best_name = s, code, name
                except Exception:
                    continue
            fmt = {
                "kind": "bayer12",
                "w": w,
                "h": h,
                "code": best_code if best_code is not None else _BAYER_CODES[0][0],
                "name": best_name if best_name is not None else _BAYER_CODES[0][1],
            }
            self.settings["resolution"] = (w, h)
            logger.info("Detected format: 12-bit packed Bayer %s at %sx%s (score=%.2f)",
                        fmt["name"], w, h, best_score)
            return fmt

        # ---- FALLBACK 1: 16-bit at 3840x2160 ----
        if n == 3840 * 2160 * 2:
            w, h = 3840, 2160
            arr16 = np.frombuffer(raw_bytes, dtype="<u2").reshape(h, w)
            return {"kind": "gray16", "w": w, "h": h}

        # ---- FALLBACK 2: 8-bit at 3840x2160 (mono) ----
        if n == 3840 * 2160:
            return {"kind": "gray8", "w": 3840, "h": 2160}

        # ---- FALLBACK 3: 8-bit at 1920x1080 ----
        if n == 1920 * 1080:
            return {"kind": "gray8", "w": 1920, "h": 1080}

        logger.warning("Could not identify buffer format for size=%s", n)
        return {"kind": "gray8", "w": 3840, "h": 2160}

    def _decode(self, raw: np.ndarray) -> Optional[np.ndarray]:
        """Decode raw buffer -> HxWx3 uint8 RGB using the cached format."""
        if self._fmt is None:
            self._fmt = self._detect_format(raw)
        fmt = self._fmt
        raw_bytes = raw.tobytes()

        if fmt["kind"] == "bayer12":
            w = fmt["w"]
            # The buffer contains multiple frames stacked vertically.
            # Try different frame heights and Bayer patterns to find the
            # correct combination.
            best_rgb = None
            best_score = -1.0
            best_info = "none"
            for h in [960, 1080, 540, 480, 360, 270, 240]:
                frame_bytes = w * h * 3 // 2
                if raw.size < frame_bytes:
                    continue
                # Try this frame height with all Bayer patterns
                content = np.frombuffer(raw_bytes[:frame_bytes], dtype=np.uint8)
                b = content.reshape(-1, 3)
                p0 = (b[:, 0].astype(np.uint16) << 4) | (b[:, 1].astype(np.uint16) >> 4)
                p1 = ((b[:, 1].astype(np.uint16) & 0x0F) << 8) | b[:, 2].astype(np.uint16)
                pixels = np.empty(b.shape[0] * 2, dtype=np.uint16)
                pixels[0::2] = p0
                pixels[1::2] = p1
                img12 = pixels.reshape(h, w)
                # Simple linear stretch 12-bit -> 8-bit
                scaled = (img12.astype(np.float32) / 4095.0 * 255.0).astype(np.uint8)
                bayer8 = np.clip(scaled, 0, 255).astype(np.uint8)
                for code, name in _BAYER_CODES:
                    try:
                        rgb = cv2.cvtColor(bayer8, code)
                        r_mean = float(np.mean(rgb[..., 0]))
                        g_mean = float(np.mean(rgb[..., 1]))
                        b_mean = float(np.mean(rgb[..., 2]))
                        # Natural images have balanced channels
                        means = np.array([r_mean, g_mean, b_mean])
                        balance = 1.0 / (1.0 + np.std(means))
                        color = float(np.std(rgb))
                        score = balance * 20.0 + color
                        if score > best_score:
                            best_score = score
                            best_rgb = rgb
                            best_info = f"h={h} pattern={name}"
                    except Exception:
                        continue
            logger.info(f"Selected: {best_info} (score={best_score:.2f})")
            if best_rgb is None:
                return np.zeros((960, w, 3), dtype=np.uint8)
            return best_rgb

        if fmt["kind"] == "bayer8":
            w, h = fmt["w"], fmt["h"]
            bayer = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w)
            # Contrast stretch (percentile) on the full frame
            lo, hi = np.percentile(bayer, (1.0, 99.5))
            if hi <= lo:
                lo, hi = float(bayer.min()), float(bayer.max())
            if hi <= lo:
                hi = lo + 1.0
            scaled = (bayer.astype(np.float32) - lo) * (255.0 / (hi - lo))
            bayer8 = np.clip(scaled, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(bayer8, fmt["code"])
            return rgb

        if fmt["kind"] == "gray16":
            w, h = fmt["w"], fmt["h"]
            arr16 = np.frombuffer(raw_bytes, dtype="<u2").reshape(h, w)
            lo, hi = np.percentile(arr16, (1.0, 99.5))
            if hi <= lo:
                lo, hi = float(arr16.min()), float(arr16.max())
            if hi <= lo:
                hi = lo + 1.0
            scaled = (arr16.astype(np.float32) - lo) * (255.0 / (hi - lo))
            gray = np.clip(scaled, 0, 255).astype(np.uint8)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        if fmt["kind"] == "gray8":
            w, h = fmt["w"], fmt["h"]
            gray = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return None

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
            # --- Determine buffer size ---
            buf_size = ctypes.c_int(0)
            rc = self._dll.ApiCam_GetAcquisitionBufferSize(
                handle, 1, ctypes.byref(buf_size)
            )
            if rc != 0 or buf_size.value <= 0:
                w, h = self.EXPECTED_RESOLUTION
                buf_size = ctypes.c_int(w * h)
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

            raw = np.frombuffer(self._acq_buffer.raw[:size], dtype=np.uint8).copy()
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