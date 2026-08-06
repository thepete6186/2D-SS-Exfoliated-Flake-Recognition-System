#!/usr/bin/env python3
"""
Camera Driver for Zeiss Axiocam 208 Color (USB3 Vision, VID_0758/PID_6002)
==========================================================================

The Axiocam 208 is a **USB3 Vision** camera (Vendor ID 0x0758, Product ID 0x6002).
It is **not** a UVC webcam — OpenCV's ``VideoCapture`` cannot open it by index.

This driver communicates with the camera through the **GenICam GenTL** interface
using the `harvesters <https://github.com/genicam/harvesters>`_ library together
with the Zeiss USB3 Vision GenTL producer (``zeiss_u3vgentlk.cti``).

The camera is bound to the ``libusb0`` driver (required by Zeiss Labscope).
The GenTL producer talks to the camera through libusb0 — **no driver rebinding
or pnputil operations are performed**. The libusb0 binding is left untouched.

Backend selection (tried in order):
1. **Harvesters / GenTL**  — works with the Axiocam 208 over USB3 Vision via libusb0
2. **OpenCV / UVC**        — fallback only if a UVC webcam is available

Usage:
    cam = ZeissCamera()
    cam.connect()
    rgb = cam.capture()
    cam.disconnect()
"""

from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USB identifiers for the Zeiss Axiocam 208 Color
# ---------------------------------------------------------------------------
ZEISS_VID = 0x0758
ZEISS_PID = 0x6002

# Known Zeiss GenTL producer locations
_ZEISS_GENTL_PATHS = [
    r"C:\Program Files\Zeiss\ZeissVisionSuite_Compact\zeiss_u3vgentl\64",
    r"C:\Program Files\Zeiss\ZeissVisionSuite_Compact\zeiss_u3vgentl\32",
    r"C:\Program Files (x86)\Zeiss\ZeissVisionSuite_Compact\zeiss_u3vgentl\64",
    r"C:\Program Files (x86)\Zeiss\ZeissVisionSuite_Compact\zeiss_u3vgentl\32",
]
_ZEISS_CTI_NAME = "zeiss_u3vgentlk.cti"


class CameraError(Exception):
    """Camera-related errors."""
    pass


# ---------------------------------------------------------------------------
# USB device detection (pyusb) – informational only
# ---------------------------------------------------------------------------


def _find_usb_device(vid: int = ZEISS_VID, pid: int = ZEISS_PID) -> Optional[Any]:
    """
    Locate the Zeiss Axiocam 208 on the USB bus using pyusb.

    Returns the pyusb Device object or ``None`` if not found.
    This is informational only; the actual acquisition uses GenTL.
    """
    try:
        import usb.core
    except ImportError:
        logger.debug("pyusb not installed — skipping USB device enumeration")
        return None

    try:
        devs = usb.core.find(find_all=True)
        for d in devs:
            if d.idVendor == vid and d.idProduct == pid:
                logger.info(
                    f"USB device found: VID={hex(vid)} PID={hex(pid)} "
                    f"({d.manufacturer or '?'} / {d.product or '?'})"
                )
                return d
    except Exception as e:
        logger.warning(f"pyusb enumeration error: {e}")
    return None


def is_camera_present() -> bool:
    """Return True if the Axiocam 208 USB device is physically connected."""
    return _find_usb_device() is not None


# ---------------------------------------------------------------------------
# GenTL producer discovery
# ---------------------------------------------------------------------------


def _find_gentl_producers() -> List[str]:
    """
    Discover candidate GenTL producer (.cti) files.

    Looks in:
    - ``GENICAM_GENTL64_PATH`` / ``GENICAM_GENTL32_PATH`` env vars
    - Known Zeiss install locations

    Only returns producers matching the current Python interpreter's
    architecture (64-bit vs 32-bit). Loading both architectures into
    a single Harvester causes the producer to fail enumeration.
    """
    import struct

    is_64bit = struct.calcsize("P") == 8
    candidates: List[str] = []

    # 1) Environment variables (may contain one or more paths separated by ';')
    #    Only use the path matching our architecture (GENICAM_GENTL64_PATH for
    #    64-bit Python, GENICAM_GENTL32_PATH for 32-bit Python).
    env_vars = (
        ("GENICAM_GENTL64_PATH", "64")
        if is_64bit
        else ("GENICAM_GENTL32_PATH", "32")
    )
    for var, arch in [env_vars]:
        val = os.environ.get(var, "")
        for p in val.split(os.pathsep):
            p = p.strip()
            if p and os.path.isdir(p):
                candidates.extend(glob.glob(os.path.join(p, "*.cti")))
            elif p and os.path.isfile(p) and p.lower().endswith(".cti"):
                candidates.append(p)

    # 2) Known Zeiss install locations - only include the matching architecture
    for root in _ZEISS_GENTL_PATHS:
        # Skip paths for the wrong architecture entirely (e.g. skip \32\ for 64-bit)
        if not is_64bit and r"\64" in root:
            continue
        if is_64bit and r"\32" in root:
            continue
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "*.cti")))

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for c in candidates:
        lc = c.lower()
        if lc not in seen:
            seen.add(lc)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Optional harvesters import
# ---------------------------------------------------------------------------

try:
    from harvesters.core import Harvester
    _HARVESTERS_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on environment
    _HARVESTERS_AVAILABLE = False
    logger.debug(f"harvesters not available: {e}")


class ZeissCamera:
    """
    Driver for the Zeiss Axiocam 208 Color camera.

    Tries the GenICam/GenTL backend (via ``harvesters``) first, since the
    Axiocam 208 is a USB3 Vision device that OpenCV cannot open. Falls back
    to OpenCV ``VideoCapture`` for UVC webcams.

    Parameters
    ----------
    camera_index : int
        Device index (0-based) among devices of the chosen backend.
    backend : str, optional
        Force a backend: ``'gentl'`` or ``'opencv'``. If ``None``, tries
        gentl first then opencv.
    gentl_file : str, optional
        Explicit path to a GenTL producer ``.cti`` file. If ``None``,
        auto-discovers via :func:`_find_gentl_producers`.
    """

    def __init__(
        self,
        camera_index: int = 0,
        backend: Optional[str] = None,
        gentl_file: Optional[str] = None,
    ):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.connected = False
        self.backend = "none"

        # GenTL / harvesters state
        self._harvester: Optional[Harvester] = None
        self._ia: Optional[Any] = None  # image acquirer
        self._gentl_file: Optional[str] = gentl_file
        self._acquisition_started = False

        # Frame-drop / overflow tracking
        self._frame_drops = 0
        self._buffer_overflows = 0
        self._last_frame_time = 0.0

        # Backend preference
        if backend is not None:
            backend = backend.lower()
            if backend == "gentl":
                self._try_gentl = True
                self._try_opencv = False
            elif backend in ("opencv", "uvc"):
                self._try_gentl = False
                self._try_opencv = True
            else:
                raise ValueError(f"Unknown backend: {backend}")
        else:
            self._try_gentl = True
            self._try_opencv = True

        # OpenCV backends to try (Windows-friendly)
        self._opencv_backends: List[int] = [
            cv2.CAP_MSMF,
            cv2.CAP_DSHOW,
            cv2.CAP_ANY,
        ]

        # Default camera settings
        self.settings: Dict[str, Any] = {
            "exposure_us": 10000.0,  # 10 ms default
            "gain_db": 0.0,
            "white_balance": "auto",
            "resolution": (5472, 3648),  # Max resolution, adjustable
            "pixel_format": "RGB",
        }

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Open the camera device.

        Returns
        -------
        bool
            True if connected successfully, False otherwise.
        """
        # 1) Try GenTL / harvesters backend (correct for USB3 Vision)
        if self._try_gentl and _HARVESTERS_AVAILABLE:
            if self._connect_gentl():
                return True

        # 2) Try OpenCV / UVC backend
        if self._try_opencv:
            if self._connect_opencv():
                return True

        return False

    def _connect_gentl(self) -> bool:
        """Connect via GenICam GenTL using the harvesters library."""
        try:
            cti_files = (
                [self._gentl_file]
                if self._gentl_file
                else _find_gentl_producers()
            )
            cti_files = [f for f in cti_files if f and os.path.isfile(f)]

            if not cti_files:
                logger.warning(
                    "GenTL backend: no .cti producer file found. "
                    "Set GENICAM_GENTL64_PATH or pass gentl_file=."
                )
                return False

            logger.info(f"GenTL: candidate CTI files: {cti_files}")

            h = Harvester()
            for cti in cti_files:
                try:
                    h.add_file(cti)
                    logger.info(f"GenTL: loaded producer {cti}")
                except Exception as e:
                    logger.debug(f"Could not add GenTL file {cti}: {e}")

            h.update()
            n = len(h.device_info_list)
            logger.info(f"GenTL: found {n} device(s)")
            if n == 0:
                logger.warning(
                    "GenTL: no devices enumerated. Check the camera is "
                    "plugged into a USB 3 port, powered, and not held by "
                    "another application (e.g. Zeiss ZEN)."
                )
                try:
                    h.reset()
                except Exception:
                    pass
                return False

            if self.camera_index >= n:
                logger.error(
                    f"GenTL: camera_index {self.camera_index} out of range "
                    f"({n} device(s) found)"
                )
                try:
                    h.reset()
                except Exception:
                    pass
                return False

            dev_info = h.device_info_list[self.camera_index]
            logger.info(
                f"GenTL: opening device {self.camera_index}: "
                f"{dev_info.vendor} {dev_info.model} "
                f"(SN {dev_info.serial_number})"
            )

            self._ia = h.create_image_acquirer(
                h.device_info_list[self.camera_index]
            )
            self._harvester = h

            # Try to apply settings (exposure/gain) via GenICam features
            self._apply_gentl_settings()

            # Read back resolution
            try:
                w = int(self._ia.remote_device.node_map.Width.value)
                hgt = int(self._ia.remote_device.node_map.Height.value)
                self.settings["resolution"] = (w, hgt)
                logger.info(f"GenTL resolution: {w}x{hgt}")
            except Exception as e:
                logger.debug(f"Could not read resolution via GenTL: {e}")

            self.backend = "gentl"
            self.connected = True
            logger.info(f"Connected to camera {self.camera_index} via GenTL")
            return True

        except Exception as e:
            logger.error(f"GenTL connection failed: {e}")
            self._ia = None
            self._harvester = None
            return False

    def _connect_opencv(self) -> bool:
        """Connect via OpenCV VideoCapture (UVC webcam)."""
        try:
            self.cap = None
            for backend in self._opencv_backends:
                cap = cv2.VideoCapture(self.camera_index, backend)
                if cap is not None and cap.isOpened():
                    self.cap = cap
                    self.backend = {
                        cv2.CAP_MSMF: "opencv-msmf",
                        cv2.CAP_DSHOW: "opencv-dshow",
                        cv2.CAP_ANY: "opencv-any",
                    }.get(backend, f"opencv-{backend}")
                    logger.info(
                        f"Opened camera {self.camera_index} via {self.backend}"
                    )
                    break
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

            if self.cap is None:
                logger.error(
                    f"OpenCV: could not open camera at index "
                    f"{self.camera_index} with any backend"
                )
                return False

            self._set_resolution()
            self._apply_settings()

            self.connected = True
            logger.info(f"Connected to camera {self.camera_index} via OpenCV")
            return True

        except Exception as e:
            logger.error(f"OpenCV connection failed: {e}")
            self.cap = None
            return False

    def disconnect(self) -> None:
        """Release the camera device."""
        # GenTL
        if self._ia is not None:
            try:
                if self._acquisition_started:
                    self._ia.stop_acquisition()
                    self._acquisition_started = False
            except Exception:
                pass
            try:
                self._ia.destroy()
            except Exception:
                pass
            self._ia = None
        if self._harvester is not None:
            try:
                self._harvester.reset()
            except Exception:
                pass
            self._harvester = None

        # OpenCV
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.connected = False
        self._frame_drops = 0
        self._buffer_overflows = 0
        logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
        if self.backend.startswith("gentl"):
            return self.connected and self._ia is not None
        return (
            self.connected
            and self.cap is not None
            and self.cap.isOpened()
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.

        Returns
        -------
        np.ndarray or None
            RGB image as numpy array (H, W, 3), or None on failure.
        """
        if not self.is_connected:
            logger.error("Camera not connected")
            return None

        if self.backend.startswith("gentl"):
            return self._capture_gentl()
        return self._capture_opencv()

    def _capture_gentl(self) -> Optional[np.ndarray]:
        """Capture a frame via GenTL/harvesters with robust error handling."""
        try:
            # Start acquisition if not already running
            if not self._acquisition_started:
                self._ia.start_acquisition()
                self._acquisition_started = True

            with self._ia.fetch_buffer(timeout=5.0) as buffer:
                component = buffer.payload.components[0]
                # component.data is a 1-D numpy array; reshape to 2D/3D
                data = component.data
                w = component.width
                h = component.height

                # Number of channels
                if data.ndim == 1:
                    nb = data.size // (w * h)
                    data = (
                        data.reshape(h, w, nb)
                        if nb > 1
                        else data.reshape(h, w)
                    )

                # Convert to RGB
                if data.ndim == 2:
                    # Grayscale -> RGB
                    rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
                elif data.shape[2] == 1:
                    rgb = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
                elif data.shape[2] == 3:
                    # Assume BGR (common) -> RGB
                    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                elif data.shape[2] == 4:
                    rgb = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
                else:
                    rgb = data[..., :3]

                self._last_frame_time = time.time()
                return np.ascontiguousarray(rgb)

        except TimeoutError:
            self._frame_drops += 1
            logger.warning(
                f"GenTL capture timeout (drops={self._frame_drops})"
            )
            return None
        except Exception as e:
            # Detect buffer overflow conditions
            msg = str(e).lower()
            if "overflow" in msg or "buffer" in msg:
                self._buffer_overflows += 1
                logger.warning(
                    f"GenTL buffer overflow (overflows={self._buffer_overflows}): {e}"
                )
            else:
                logger.error(f"GenTL capture failed: {e}")
            return None

    def _capture_opencv(self) -> Optional[np.ndarray]:
        """Capture a frame via OpenCV."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                self._frame_drops += 1
                logger.error(
                    f"Failed to capture frame (drops={self._frame_drops})"
                )
                return None
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._last_frame_time = time.time()
            return rgb
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None

    def capture_to_file(
        self, output_path: Path, format: str = "png"
    ) -> bool:
        """
        Capture image and save to file.

        Parameters
        ----------
        output_path : Path
            Output file path.
        format : str
            Image format: 'png' (lossless) or 'jpg' (compressed).

        Returns
        -------
        bool
            True on success, False on failure.
        """
        rgb = self.capture()
        if rgb is None:
            return False

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if format.lower() in ("jpg", "jpeg"):
            cv2.imwrite(
                str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
        else:
            cv2.imwrite(str(output_path), bgr)

        logger.info(f"Saved image to {output_path}")
        return True

    # ------------------------------------------------------------------
    # Continuous streaming
    # ------------------------------------------------------------------

    def capture_stream(
        self, max_fps: float = 30.0, drop_if_slow: bool = True
    ) -> Iterator[np.ndarray]:
        """
        Generator that yields frames continuously.

        Parameters
        ----------
        max_fps : float
            Target maximum frame rate. If the camera produces frames faster,
            excess frames are dropped (when ``drop_if_slow`` is True).
        drop_if_slow : bool
            If True, skip frames when the consumer is slower than the camera.

        Yields
        ------
        np.ndarray
            RGB image as numpy array (H, W, 3).
        """
        if not self.is_connected:
            raise CameraError("Camera not connected")

        min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        last_yield = 0.0

        while self.is_connected:
            frame = self.capture()
            if frame is None:
                # Brief pause to avoid busy-looping on errors
                time.sleep(0.001)
                continue

            now = time.time()
            if drop_if_slow and (now - last_yield) < min_interval:
                self._frame_drops += 1
                continue

            last_yield = now
            yield frame

    # ------------------------------------------------------------------
    # Camera Settings
    # ------------------------------------------------------------------

    def set_exposure(self, exposure_us: float) -> None:
        """
        Set exposure time in microseconds.
        """
        self.settings["exposure_us"] = exposure_us

        if not self.is_connected:
            return

        if self.backend.startswith("gentl"):
            self._set_gentl_feature("ExposureTime", float(exposure_us))
            return

        try:
            exposure_val = exposure_us / 10.0
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        except Exception as e:
            logger.warning(f"Failed to set exposure: {e}")

    def set_gain(self, gain_db: float) -> None:
        """
        Set camera gain in decibels (dB).
        """
        self.settings["gain_db"] = gain_db

        if not self.is_connected:
            return

        if self.backend.startswith("gentl"):
            # GenICam GainRaw / Gain features are often in dB or linear
            self._set_gentl_feature("Gain", float(gain_db))
            return

        try:
            self.cap.set(cv2.CAP_PROP_GAIN, gain_db)
        except Exception as e:
            logger.warning(f"Failed to set gain: {e}")

    def set_resolution(self, width: int, height: int) -> None:
        """Set camera resolution."""
        self.settings["resolution"] = (width, height)

        if not self.is_connected:
            return

        if self.backend.startswith("gentl"):
            self._set_gentl_feature("Width", int(width))
            self._set_gentl_feature("Height", int(height))
            return

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._set_resolution()
        except Exception as e:
            logger.warning(f"Failed to set resolution: {e}")

    def _set_gentl_feature(self, name: str, value: Any) -> None:
        """Set a GenICam feature on the remote device node map."""
        try:
            node_map = self._ia.remote_device.node_map
            node = getattr(node_map, name, None)
            if node is None:
                logger.debug(f"GenTL feature {name} not found")
                return
            node.value = value
        except Exception as e:
            logger.warning(f"Failed to set GenTL feature {name}: {e}")

    def _apply_gentl_settings(self) -> None:
        """Apply current settings via GenICam features."""
        try:
            self._set_gentl_feature(
                "ExposureTime", float(self.settings["exposure_us"])
            )
            self._set_gentl_feature(
                "Gain", float(self.settings["gain_db"])
            )
        except Exception as e:
            logger.warning(f"Failed to apply GenTL settings: {e}")

    def _set_resolution(self) -> None:
        """Apply resolution settings via OpenCV and read back actual values."""
        if self.cap is None:
            return

        width, height = self.settings["resolution"]
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.settings["resolution"] = (actual_width, actual_height)
        logger.info(f"Camera resolution: {actual_width}x{actual_height}")

    def _apply_settings(self) -> None:
        """Apply current settings to OpenCV camera."""
        if not self.is_connected:
            return
        try:
            self.set_exposure(self.settings["exposure_us"])
            self.set_gain(self.settings["gain_db"])
        except Exception as e:
            logger.warning(f"Failed to apply settings: {e}")

    # ------------------------------------------------------------------
    # Camera Info
    # ------------------------------------------------------------------

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        info: Dict[str, Any] = {
            "backend": self.backend,
            "connected": self.connected,
            "resolution": self.settings["resolution"],
        }

        if self.is_connected:
            if (
                self.backend.startswith("gentl")
                and self._harvester is not None
            ):
                try:
                    dev = self._harvester.device_info_list[
                        self.camera_index
                    ]
                    info["name"] = f"{dev.vendor} {dev.model}".strip()
                    info["model"] = dev.model or "Axiocam 208"
                    info["serial"] = dev.serial_number
                except Exception:
                    info["name"] = f"USB3 Vision Camera {self.camera_index}"
                    info["model"] = "Axiocam 208 (USB3 Vision)"
            else:
                info["name"] = f"USB Camera {self.camera_index}"
                info["model"] = "Axiocam 208 (USB)"
        else:
            info["name"] = "Unknown"
            info["model"] = "Unknown"

        return info

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return runtime diagnostics (frame drops, overflows, etc.)."""
        return {
            "frame_drops": self._frame_drops,
            "buffer_overflows": self._buffer_overflows,
            "last_frame_time": self._last_frame_time,
            "backend": self.backend,
            "connected": self.connected,
        }

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return (
            f"ZeissCamera(index={self.camera_index}, "
            f"backend={self.backend}, {status})"
        )