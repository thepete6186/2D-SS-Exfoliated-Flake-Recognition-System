"""
Zeiss XiCam 208 Color Camera Driver
=====================================

Camera interface for Zeiss XiCam 208 Color scientific camera.

Tries Zeiss SDK first, falls back to OpenCV VideoCapture if SDK not available.
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
    Camera interface for Zeiss XiCam 208 Color.
    
    Parameters
    ----------
    use_sdk : bool
        Try to use Zeiss SDK first (default: True). Falls back to OpenCV if not available.
    camera_index : int
        Camera device index for OpenCV fallback (default: 0).
    """
    
    def __init__(self, use_sdk: bool = True, camera_index: int = 0):
        self.use_sdk = use_sdk
        self.camera_index = camera_index
        self.camera = None
        self.sdk_camera = None
        self.connected = False
        self.backend = "none"
        
        # Default camera settings
        self.settings: Dict[str, Any] = {
            'exposure_us': 10000.0,  # 10ms default
            'gain_db': 0.0,
            'white_balance': 'auto',
            'resolution': (5472, 3648),  # Max resolution, adjustable
            'pixel_format': 'RGB',
        }
        
        # Try to import Zeiss SDK
        self._zeiss_sdk_available = False
        if self.use_sdk:
            try:
                # Try common Zeiss SDK module names
                # Actual import depends on SDK version
                import zeiss.microscopy as zm
                self._zeiss_sdk = zm
                self._zeiss_sdk_available = True
                logger.info("Zeiss SDK found")
            except ImportError:
                try:
                    # Alternative: try pyzeiss or other names
                    import pyzeiss
                    self._zeiss_sdk = pyzeiss
                    self._zeiss_sdk_available = True
                    logger.info("Zeiss SDK found (pyzeiss)")
                except ImportError:
                    logger.warning("Zeiss SDK not found, will use OpenCV fallback")
                    self._zeiss_sdk_available = False
    
    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    
    def connect(self) -> bool:
        """
        Initialize camera connection.
        
        Returns
        -------
        bool
            True if connected successfully, False otherwise.
        """
        if self._zeiss_sdk_available:
            return self._connect_sdk()
        else:
            logger.info("Using OpenCV fallback")
            return self._connect_opencv()
    
    def _connect_sdk(self) -> bool:
        """Connect using Zeiss Microscopy SDK."""
        try:
            # Find available cameras
            cameras = self._zeiss_sdk.Camera.list_cameras()
            if not cameras:
                logger.error("No Zeiss cameras found")
                return False
            
            # Use first available camera
            self.sdk_camera = cameras[0]
            self.sdk_camera.open()
            
            # Apply default settings
            self._apply_sdk_settings()
            
            self.connected = True
            self.backend = "zeiss_sdk"
            logger.info(f"Connected to Zeiss camera: {self.sdk_camera.name}")
            return True
            
        except Exception as e:
            logger.error(f"Zeiss SDK connection failed: {e}")
            logger.info("Falling back to OpenCV")
            self._zeiss_sdk_available = False
            return self._connect_opencv()
    
    def _connect_opencv(self) -> bool:
        """Fallback: connect using OpenCV VideoCapture."""
        try:
            # Try multiple camera indices
            for idx in range(5):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Try to read a frame to verify it works
                    ret, frame = cap.read()
                    if ret:
                        # Check if this looks like a high-res camera
                        height = frame.shape[0]
                        logger.info(f"Camera {idx}: {frame.shape}")
                        
                        self.camera = cap
                        self.connected = True
                        self.backend = "opencv"
                        self.camera_index = idx
                        
                        # Try to set resolution
                        self._set_opencv_resolution()
                        
                        logger.info(f"Connected to camera {idx} via OpenCV")
                        return True
                    cap.release()
            
            logger.error("No suitable camera found")
            return False
            
        except Exception as e:
            logger.error(f"OpenCV camera connection failed: {e}")
            return False
    
    def _set_opencv_resolution(self):
        """Try to set camera resolution via OpenCV."""
        if self.camera is None:
            return
        
        width, height = self.settings['resolution']
        
        # Try to set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read actual resolution
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Camera resolution: {actual_width}x{actual_height}")
        self.settings['resolution'] = (actual_width, actual_height)
    
    def disconnect(self) -> None:
        """Release camera resources."""
        if self.sdk_camera is not None:
            try:
                self.sdk_camera.close()
            except Exception:
                pass
            self.sdk_camera = None
        
        if self.camera is not None:
            try:
                self.camera.release()
            except Exception:
                pass
            self.camera = None
        
        self.connected = False
        self.backend = "none"
        logger.info("Camera disconnected")
    
    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture single image.
        
        Returns
        -------
        np.ndarray or None
            RGB image as numpy array (H, W, 3), or None on failure.
        """
        if not self.connected:
            logger.error("Camera not connected")
            return None
        
        try:
            if self.backend == "zeiss_sdk":
                return self._capture_sdk()
            else:
                return self._capture_opencv()
        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None
    
    def _capture_sdk(self) -> np.ndarray:
        """Capture using Zeiss SDK."""
        # Actual SDK call - adjust based on actual API
        image = self.sdk_camera.capture_image()
        
        # Convert to numpy array (adjust based on SDK)
        # Typically returns RGB already
        rgb = np.array(image)
        
        return rgb
    
    def _capture_opencv(self) -> np.ndarray:
        """Capture using OpenCV."""
        if self.camera is None:
            raise CameraError("Camera not initialized")
        
        ret, frame = self.camera.read()
        if not ret:
            raise CameraError("Failed to capture frame")
        
        # Convert BGR (OpenCV default) to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb
    
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
        
        # Set compression quality for JPEG
        if format.lower() == 'jpg' or format.lower() == 'jpeg':
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
        
        if not self.connected:
            return
        
        try:
            if self.backend == "zeiss_sdk":
                self.sdk_camera.exposure = exposure_us
            else:
                # OpenCV: exposure control varies by camera
                # Often in camera-specific units (not microseconds)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_us / 1000.0)
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
        
        if not self.connected:
            return
        
        try:
            if self.backend == "zeiss_sdk":
                self.sdk_camera.gain = gain_db
            else:
                self.camera.set(cv2.CAP_PROP_GAIN, gain_db)
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
        
        if not self.connected:
            return
        
        try:
            if self.backend == "opencv":
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._set_opencv_resolution()
            else:
                logger.warning("Resolution setting not implemented for Zeiss SDK")
        except Exception as e:
            logger.warning(f"Failed to set resolution: {e}")
    
    def get_settings(self) -> Dict[str, Any]:
        """
        Get current camera settings.
        
        Returns
        -------
        dict
            Dictionary of current settings.
        """
        return dict(self.settings)
    
    def _apply_sdk_settings(self) -> None:
        """Apply current settings to Zeiss SDK camera."""
        if self.backend != "zeiss_sdk" or not self.connected:
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
            Camera name, model, resolution, etc.
        """
        info = {
            'backend': self.backend,
            'connected': self.connected,
            'resolution': self.settings['resolution'],
        }
        
        if self.backend == "zeiss_sdk" and self.sdk_camera:
            try:
                info['name'] = self.sdk_camera.name
                info['model'] = getattr(self.sdk_camera, 'model', 'Unknown')
            except Exception:
                info['name'] = 'Zeiss Camera'
                info['model'] = 'Unknown'
        else:
            info['name'] = f"Camera {self.camera_index}"
            info['model'] = 'Unknown'
        
        return info
    
    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return f"ZeissCamera(backend={self.backend}, {status})"