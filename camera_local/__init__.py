"""
Camera module for Zeiss Axiocam 208 Color camera.

Provides camera capture functionality through both OpenCV/GenTL
(ZeissCamera) and Micro-Manager (MMCoreCamera), plus calibration
system for the 2D material transfer system.
"""

from .zeiss_camera import ZeissCamera, CameraError
from .mmcore_camera import MMCoreCamera, CameraInfo
from .calibration import CalibrationManager, CalibrationError
from .coordinate_mapper import CoordinateMapper

__all__ = [
    "ZeissCamera",
    "MMCoreCamera",
    "CameraInfo",
    "CameraError",
    "CalibrationManager",
    "CalibrationError",
    "CoordinateMapper",
]
