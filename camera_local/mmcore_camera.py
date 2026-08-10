"""
Camera Driver for Zeiss Axiocam 208 Color using Micro-Manager (pymmcore)
==========================================================================

Driver for the Zeiss Axiocam 208 Color scientific camera using the
open-source Micro-Manager SDK via pymmcore.

The Axiocam 208 is a USB3 Vision camera that is not accessible via
standard OpenCV VideoCapture. This driver uses Micro-Manager's
pymmcore library which provides a unified interface for scientific
cameras including Zeiss Axiocam series.

Requirements:
    - pymmcore>=2.0.0
    - Micro-Manager device adapter for Axiocam 208 (Zeiss)
    - Camera connected via USB 3.0

Usage:
    cam = MMCoreCamera()
    cam.connect()
    rgb = cam.capture()
    cam.disconnect()
"""

import os
import glob
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Camera-related errors."""
    pass


@dataclass
class CameraInfo:
    """Camera information structure."""
    name: str
    model: str
    serial: str
    backend: str
    connected: bool
    resolution: Tuple[int, int]


class MMCoreCamera:
    """
    Camera driver using Micro-Manager Core (pymmcore).

    This driver provides access to the Zeiss Axiocam 208 Color camera
    through the Micro-Manager SDK, which supports USB3 Vision cameras
    that are not accessible via standard OpenCV.

    Parameters
    ----------
    camera_index : int
        Device index (0-based) among available cameras.
    mm_config_path : str, optional
        Path to Micro-Manager configuration file. If None, will attempt
        to auto-configure the camera.
    """

    def __init__(self, camera_index: int = 0,
                 mm_config_path: Optional[str] = None):
        self.camera_index = camera_index
        self.mm_config_path = mm_config_path
        self.connected = False
        self.backend = "none"

        # pymmcore objects
        self._mmc = None
        self._camera_name = ""

        # Camera settings
        self.settings: Dict[str, Any] = {
            'exposure_us': 10000.0,  # 10 ms default
            'gain_db': 0.0,
            'white_balance': 'auto',
            'resolution': (5472, 3648),  # Max resolution for Axiocam 208
            'pixel_format': 'RGB',
        }

        # Try to import pymmcore
        self._mmcore_available = self._check_mmcore()

    def _check_mmcore(self) -> bool:
        """Check if pymmcore is available."""
        try:
            import pymmcore
            logger.info(f"pymmcore version: {pymmcore.__version__}")
            return True
        except ImportError as e:
            logger.warning(f"pymmcore not available: {e}")
            return False

    def _check_mmcore_plus(self) -> bool:
        """Check if pymmcore-plus is available."""
        try:
            import pymmcore_plus
            logger.info(f"pymmcore-plus available")
            return True
        except ImportError as e:
            logger.warning(f"pymmcore-plus not available: {e}")
            return False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Open the camera device using Micro-Manager.

        Returns
        -------
        bool
            True if connected successfully, False otherwise.
        """
        if not self._mmcore_available:
            logger.error(
                "pymmcore is not installed. Install with: pip install pymmcore"
            )
            return False

        # Try pymmcore-plus first (modern API), fall back to pymmcore
        try:
            from pymmcore_plus import CMMCorePlus
            logger.info("Using pymmcore-plus (CMMCorePlus)")
        except ImportError:
            try:
                import pymmcore
                logger.info(f"Using pymmcore {pymmcore.__version__}")
                # For pymmcore without plus, we need to use the base class
                # Create a wrapper or use pymmcore.CMMCore directly
                return False  # TODO: Implement pymmcore-only support if needed
            except ImportError:
                logger.error("Neither pymmcore nor pymmcore-plus available")
                return False

        try:
            # Initialize CMMCore
            self._mmc = CMMCorePlus()
            mmc = self._mmc

            # Try to load configuration if provided
            if self.mm_config_path and Path(self.mm_config_path).exists():
                try:
                    mmc.loadSystemConfiguration(str(self.mm_config_path))
                    logger.info(f"Loaded MM config from {self.mm_config_path}")
                except Exception as e:
                    logger.warning(f"Failed to load config file: {e}")

            # Auto-configure if no config file
            if not mmc.getLoadedDevices():
                self._auto_configure(mmc)

            # Get available cameras (exclude "Core" which is always present)
            devices = mmc.getLoadedDevices()
            logger.info(f"MMCore: loaded devices: {devices}")

            # Find camera devices (exclude Core)
            camera_devices = [
                d for d in devices
                if d != "Core" and (
                    'cam' in d.lower() or 'camera' in d.lower()
                )
            ]
            if not camera_devices:
                # Try to find any device that might be a camera (exclude Core)
                camera_devices = [d for d in devices if d != "Core"]

            # If no devices loaded, try to load from AxioCam adapter
            if not camera_devices:
                logger.info("No devices loaded, trying AxioCam adapter...")
                try:
                    # Try to find available AxioCam devices
                    axio_devs = mmc.getAvailableDevices("AxioCam")
                    if axio_devs:
                        dev_name = axio_devs[0]
                        logger.info(f"Found AxioCam device: {dev_name}")
                        mmc.loadDevice(dev_name, "AxioCam", dev_name)
                        camera_devices = [dev_name]
                    else:
                        # Try generic USB3 Vision adapter
                        try:
                            u3v_devs = mmc.getAvailableDevices("Usb3CamHS")
                            if u3v_devs:
                                dev_name = u3v_devs[0]
                                logger.info(
                                    f"Found USB3 Vision device: {dev_name}"
                                )
                                mmc.loadDevice(
                                    dev_name, "Usb3CamHS", dev_name
                                )
                                camera_devices = [dev_name]
                        except Exception as e:
                            logger.debug(
                                f"Could not load Usb3CamHS adapter: {e}"
                            )
                except Exception as e:
                    logger.debug(f"Could not load AxioCam device: {e}")

            if not camera_devices:
                logger.error("MMCore: no camera devices found")
                return False

            if self.camera_index >= len(camera_devices):
                logger.error(
                    f"MMCore: camera_index {self.camera_index} out of range "
                    f"({len(camera_devices)} device(s) found)"
                )
                return False

            # Select camera
            self._camera_name = camera_devices[self.camera_index]
            logger.info(f"MMCore: selecting camera: {self._camera_name}")

            # Initialize camera
            mmc.initializeDevice(self._camera_name)

            # Set camera as the active device
            mmc.setCameraDevice(self._camera_name)

            # Apply settings
            self._apply_mmcore_settings()

            # Read back resolution
            try:
                width = int(mmc.getImageWidth())
                height = int(mmc.getImageHeight())
                self.settings['resolution'] = (width, height)
                logger.info(f"MMCore resolution: {width}x{height}")
            except Exception as e:
                logger.debug(f"Could not read resolution: {e}")

            self.backend = "mmcore"
            self.connected = True
            logger.info("Connected to camera via Micro-Manager (pymmcore)")
            return True

        except Exception as e:
            logger.error(f"MMCore connection failed: {e}")
            self._mmc = None
            self._camera_name = ""
            return False

    def _auto_configure(self, mmc) -> None:
        """
        Auto-configure Micro-Manager for Axiocam 208.

        Attempts to find and load the Zeiss Axiocam 208 device adapter
        from the Micro-Manager installation.
        """
        try:
            # Common Micro-Manager installation paths
            mm_paths = [
                r"C:\Program Files\Micro-Manager-2.0",
                r"C:\Program Files\Micro-Manager-1.4",
                r"C:\Program Files (x86)\Micro-Manager-2.0",
                r"C:\Program Files (x86)\Micro-Manager-1.4",
            ]

            # Also check pymmcore-plus managed installation
            mm_plus_path = os.path.join(
                os.path.expanduser("~"),
                "AppData", "Local", "pymmcore-plus", "pymmcore-plus", "mm"
            )
            if os.path.isdir(mm_plus_path):
                for entry in os.listdir(mm_plus_path):
                    full = os.path.join(mm_plus_path, entry)
                    if os.path.isdir(full):
                        mm_paths.append(full)

            # Also check environment variable
            mm_path = os.environ.get("MICRO_MANAGER_PATH", "")
            if mm_path:
                mm_paths.insert(0, mm_path)

            # Try to find Micro-Manager config files (.cfg)
            cfg_found = False
            for mm_path in mm_paths:
                if not os.path.isdir(mm_path):
                    continue

                # Look for *.cfg files (system configurations)
                cfg_files = glob.glob(os.path.join(mm_path, "*.cfg"))
                for cfg in cfg_files:
                    try:
                        logger.info(f"Loading MM config: {cfg}")
                        mmc.loadSystemConfiguration(cfg)
                        cfg_found = True
                        break
                    except Exception as e:
                        logger.debug(
                            f"Could not load config {cfg}: {e}"
                        )
                if cfg_found:
                    break

            if not cfg_found:
                # Try loading device adapter search paths
                for mm_path in mm_paths:
                    if os.path.isdir(mm_path):
                        # Check both DeviceAdapters subdir and root
                        search_paths = []
                        da_dir = os.path.join(mm_path, "DeviceAdapters")
                        if os.path.isdir(da_dir):
                            search_paths.append(da_dir)
                        # Root dir may also contain adapter DLLs
                        search_paths.append(mm_path)

                        try:
                            mmc.setDeviceAdapterSearchPaths(search_paths)
                            logger.info(
                                f"Set DeviceAdapterSearchPaths: {search_paths}"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Could not set adapter search path: {e}"
                            )

                # Check if AxioCam adapter is available
                try:
                    axio_devs = mmc.getAvailableDevices("AxioCam")
                    if axio_devs:
                        logger.info(
                            f"Found AxioCam devices: {axio_devs}"
                        )
                except Exception as e:
                    logger.debug(f"Could not query AxioCam adapter: {e}")

                logger.warning(
                    "Could not auto-configure Axiocam 208. "
                    "No .cfg file found. Please provide a Micro-Manager "
                    "config file via mm_config_path, or set "
                    "MICRO_MANAGER_PATH env var."
                )

        except Exception as e:
            logger.warning(f"Auto-configuration failed: {e}")

    def disconnect(self) -> None:
        """Release the camera device."""
        if self._mmc is not None:
            try:
                if self._camera_name:
                    self._mmc.stopSequenceAcquisition()
                    self._mmc.reset()
            except Exception:
                pass
            self._mmc = None

        self._camera_name = ""
        self.connected = False
        logger.info("Camera disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if camera is connected and ready."""
        if not self.connected or self._mmc is None:
            return False
        try:
            return self._mmc.getCameraDevice() == self._camera_name
        except Exception:
            return False

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
            # Capture image
            img = self._mmc.snapImage()

            # Convert to numpy array if needed
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            # Ensure correct shape (H, W, C)
            if img.ndim == 2:
                # Grayscale -> RGB
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 3:
                # Assume BGR from Micro-Manager -> RGB
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                rgb = img[..., :3]

            return np.ascontiguousarray(rgb)

        except Exception as e:
            logger.error(f"Capture failed: {e}")
            return None

    def capture_sequence(self, num_frames: int) -> Optional[np.ndarray]:
        """
        Capture a sequence of frames.

        Parameters
        ----------
        num_frames : int
            Number of frames to capture.

        Returns
        -------
        np.ndarray or None
            Array of images (N, H, W, 3), or None on failure.
        """
        if not self.is_connected:
            logger.error("Camera not connected")
            return None

        try:
            frames = []
            for i in range(num_frames):
                img = self.capture()
                if img is None:
                    logger.warning(f"Failed to capture frame {i}")
                    break
                frames.append(img)

            if not frames:
                return None

            return np.stack(frames, axis=0)

        except Exception as e:
            logger.error(f"Sequence capture failed: {e}")
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
        Set exposure time in microseconds.
        """
        self.settings['exposure_us'] = exposure_us

        if not self.is_connected:
            return

        try:
            # Micro-Manager typically uses milliseconds for exposure
            exposure_ms = exposure_us / 1000.0
            self._mmc.setExposureTime(exposure_ms)
            logger.info(
                f"Set exposure to {exposure_us} us ({exposure_ms} ms)"
            )
        except Exception as e:
            logger.warning(f"Failed to set exposure: {e}")

    def set_gain(self, gain_db: float) -> None:
        """
        Set camera gain in decibels (dB).
        """
        self.settings['gain_db'] = gain_db

        if not self.is_connected:
            return

        try:
            # Try common gain property names
            for prop in ['Gain', 'GainDB', 'Gain(dB)']:
                try:
                    self._mmc.setProperty(
                        self._camera_name, prop, str(gain_db)
                    )
                    logger.info(f"Set gain to {gain_db} dB via {prop}")
                    return
                except Exception:
                    continue

            logger.warning("Could not set gain - property not found")
        except Exception as e:
            logger.warning(f"Failed to set gain: {e}")

    def set_resolution(self, width: int, height: int) -> None:
        """Set camera resolution."""
        self.settings['resolution'] = (width, height)

        if not self.is_connected:
            return

        try:
            self._mmc.setImageWidth(width)
            self._mmc.setImageHeight(height)

            # Read back actual resolution
            actual_width = int(self._mmc.getImageWidth())
            actual_height = int(self._mmc.getImageHeight())
            self.settings['resolution'] = (actual_width, actual_height)
            logger.info(f"Set resolution to {actual_width}x{actual_height}")
        except Exception as e:
            logger.warning(f"Failed to set resolution: {e}")

    def _apply_mmcore_settings(self) -> None:
        """Apply current settings to Micro-Manager camera."""
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
        """Get camera information."""
        info: Dict[str, Any] = {
            'backend': self.backend,
            'connected': self.connected,
            'resolution': self.settings['resolution'],
        }

        if self.is_connected and self._mmc is not None:
            try:
                info['name'] = self._camera_name
                info['model'] = 'Axiocam 208 (Micro-Manager)'
                info['serial'] = self._mmc.getProperty(
                    self._camera_name, 'SerialNumber'
                )
            except Exception:
                info['name'] = self._camera_name or 'Unknown'
                info['model'] = 'Axiocam 208 (Micro-Manager)'
                info['serial'] = 'Unknown'
        else:
            info['name'] = 'Unknown'
            info['model'] = 'Unknown'
            info['serial'] = 'Unknown'

        return info

    def get_available_properties(self) -> List[str]:
        """
        Get list of available camera properties.

        Returns
        -------
        List[str]
            List of property names that can be configured.
        """
        if not self.is_connected or self._mmc is None:
            return []

        try:
            props = self._mmc.getDevicePropertyNames(self._camera_name)
            return list(props)
        except Exception as e:
            logger.warning(f"Failed to get properties: {e}")
            return []

    def set_property(self, property_name: str, value: Any) -> bool:
        """
        Set a camera property by name.

        Parameters
        ----------
        property_name : str
            Name of the property to set.
        value : Any
            Value to set.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        if not self.is_connected or self._mmc is None:
            logger.error("Camera not connected")
            return False

        try:
            self._mmc.setProperty(
                self._camera_name, property_name, str(value)
            )
            logger.info(f"Set {property_name} = {value}")
            return True
        except Exception as e:
            logger.warning(f"Failed to set property {property_name}: {e}")
            return False

    def get_property(self, property_name: str) -> Optional[Any]:
        """
        Get a camera property value.

        Parameters
        ----------
        property_name : str
            Name of the property to get.

        Returns
        -------
        Any or None
            Property value, or None if not available.
        """
        if not self.is_connected or self._mmc is None:
            return None

        try:
            return self._mmc.getProperty(self._camera_name, property_name)
        except Exception as e:
            logger.debug(f"Failed to get property {property_name}: {e}")
            return None

    def __repr__(self) -> str:
        status = "connected" if self.connected else "disconnected"
        return (
            f"MMCoreCamera(index={self.camera_index}, "
            f"backend={self.backend}, {status})"
        )