"""
Camera Driver for USB Camera (Zeiss XiCam 208 via OpenCV)
=========================================================

Simple USB camera driver using OpenCV's VideoCapture.
The Zeiss XiCam 208 appears as a standard USB video device.

Usage:
    cam = Camera()
    cam.connect()
    rgb = cam.capture()
    cam.disconnect()
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Camera-related errors."""
    pass


class ZeissCamera:
    """
    USB camera driver using OpenCV VideoCapture.

    The Zeiss XiCam 208 is a USB3 Vision camera that appears as a
    standard USB video device. OpenCV can access it directly.

    Parameters
    ----------
    camera_index : int
        Camera device index (default: 0). Try 0, 1, 2 if not found.
    """

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.connected = False
        self.backend = "opencv"

        # Default camera settings
        self.settings: Dict[str, Any] = {
            'exposure_us': 10000.0,  # 10ms default
            'gain_db': 0.0,
            'white_balance': 'auto',
            'resolution': (5472, 3648),  # Max resolution, adjustable
            'pixel_format': 'RGB',
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
        try:
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                logger.error(f"Could not open camera at index {self.camera_index}")
                self.cap = None
                return False

            # Try to set resolution
            self._set_resolution()

            # Try to set exposure (may not work for all cameras)
            self._apply_settings()

            self.connected = True
            logger.info(f"Connected to camera {self.camera_index}")
            return True

        except Exception as e:
            logger.error(f"Camera connection failed: {e}")
            self.cap = None
            return False

    def disconnect(self) -> None:
        """Release the camera device."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.connected = False
        logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
        return self.connected and self.cap is not None and self.cap.isOpened()

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

        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                return None

            # Convert BGR (OpenCV default) to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return rgb

        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None

    def capture_to_file(self, output_path: Path, format: str = "png") -> bool:
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

        # Convert RGB to BGR for OpenCV imwrite
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if format.lower() in ('jpg', 'jpeg'):
            cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(output_path), bgr)

        logger.info(f"Saved image to {output_path}")
        return True

    # ------------------------------------------------------------------
    # Camera Settings
    # ------------------------------------------------------------------

    def set_exposure(self, exposure_us: float) -> None:
        """
        Set exposure time.

        Parameters
        ----------
        exposure_us : float
            Exposure time in microseconds.
        """
        self.settings['exposure_us'] = exposure_us

        if not self.is_connected:
            return

        try:
            # OpenCV exposure is in some camera-specific units
            # For many cameras, it's in 1/100000 seconds (10 µs units)
            # So exposure_us / 10 gives the OpenCV value
            exposure_val = exposure_us / 10.0
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        except Exception as e:
            logger.warning(f"Failed to set exposure: {e}")

    def set_gain(self, gain_db: float) -> None:
        """
        Set camera gain.

        Parameters
        ----------
        gain_db : float
            Gain in decibels (dB).
        """
        self.settings['gain_db'] = gain_db

        if not self.is_connected:
            return

        try:
            self.cap.set(cv2.CAP_PROP_GAIN, gain_db)
        except Exception as e:
            logger.warning(f"Failed to set gain: {e}")

    def set_resolution(self, width: int, height: int) -> None:
        """
        Set camera resolution.

        Parameters
        ----------
        width : int
            Image width in pixels.
        height : int
            Image height in pixels.
        """
        self.settings['resolution'] = (width, height)

        if not self.is_connected:
            return

        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._set_resolution()
        except Exception as e:
            logger.warning(f"Failed to set resolution: {e}")

    def _set_resolution(self) -> None:
        """Apply resolution settings and read back actual values."""
        if self.cap is None:
            return

        width, height = self.settings['resolution']
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.settings['resolution'] = (actual_width, actual_height)
        logger.info(f"Camera resolution: {actual_width}x{actual_height}")

    def _apply_settings(self) -> None:
        """Apply current settings to camera."""
        if not self.is_connected:
            return

        try:
            self.set_exposure(self.settings['exposure_us'])
            self.set_gain(self.settings['gain_db'])
        except Exception as e:
            logger.warning(f"Failed to apply settings: {e}")

    # ------------------------------------------------------------------
    # Camera Info
    # ------------------------------------------------------------------

    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information.

        Returns
        -------
        dict
            Camera name, model, resolution, backend.
        """
        info = {
            'backend': self.backend,
            'connected': self.connected,
            'resolution': self.settings['resolution'],
        }

        if self.is_connected:
            info['name'] = f"USB Camera {self.camera_index}"
            info['model'] = 'XiCam 208 (USB)'
            info['backend'] = 'opencv'
        else:
            info['name'] = 'Unknown'
            info['model'] = 'Unknown'

        return info

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return f"ZeissCamera(index={self.camera_index}, {status})"
