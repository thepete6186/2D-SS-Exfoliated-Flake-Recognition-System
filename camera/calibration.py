"""
Calibration system for pixel-to-nanometer conversion.

Handles variable magnification by storing calibration presets for
different zoom/magnification settings.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Calibration-related errors."""
    pass


class CalibrationManager:
    """
    Manages pixel-to-nanometer calibrations for different magnifications.
    
    Stores calibration presets and handles coordinate conversions.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize calibration manager.
        
        Parameters
        ----------
        config_path : Path, optional
            Path to calibration config file. Defaults to camera/calibration.json
        """
        if config_path is None:
            config_path = Path(__file__).parent / "calibration.json"
        
        self.config_path = config_path
        self.calibrations: Dict[str, dict] = {}
        self.active_name: str = "default"
        
        # Load existing calibrations
        self._load_calibrations()
    
    # ------------------------------------------------------------------
    # Calibration Management
    # ------------------------------------------------------------------
    
    def add_calibration(self, name: str, nm_per_pixel: float,
                        metadata: Optional[dict] = None) -> None:
        """
        Add a new calibration preset.
        
        Parameters
        ----------
        name : str
            Calibration name (e.g., "10x", "20x", "zoom_2x")
        nm_per_pixel : float
            Nanometers per pixel (e.g., 345.0 means 1 pixel = 345 nm)
        metadata : dict, optional
            Additional metadata (objective_mag, zoom_level, binning, etc.)
        """
        if metadata is None:
            metadata = {}
        
        self.calibrations[name] = {
            'nm_per_pixel': float(nm_per_pixel),
            'metadata': metadata
        }
        
        self._save_calibrations()
        logger.info(f"Added calibration '{name}': {nm_per_pixel:.2f} nm/pixel")
    
    def remove_calibration(self, name: str) -> bool:
        """
        Remove a calibration preset.
        
        Returns
        -------
        bool
            True if removed, False if not found.
        """
        if name in self.calibrations:
            del self.calibrations[name]
            
            # If we removed the active calibration, switch to default
            if self.active_name == name:
                self.active_name = "default" if "default" in self.calibrations else list(self.calibrations.keys())[0] if self.calibrations else "default"
            
            self._save_calibrations()
            logger.info(f"Removed calibration '{name}'")
            return True
        return False
    
    def get_calibration(self, name: str) -> Optional[dict]:
        """
        Get a calibration preset by name.
        
        Returns
        -------
        dict or None
            Calibration dict with 'nm_per_pixel' and 'metadata', or None if not found.
        """
        return self.calibrations.get(name)
    
    def get_active(self) -> float:
        """
        Get current nm/pixel for the active calibration.
        
        Returns
        -------
        float
            Nanometers per pixel (default: 1.0 if no calibration)
        """
        cal = self.calibrations.get(self.active_name, {})
        return cal.get('nm_per_pixel', 1.0)
    
    def get_active_metadata(self) -> dict:
        """
        Get metadata for the active calibration.
        
        Returns
        -------
        dict
            Metadata dict (empty if no calibration)
        """
        cal = self.calibrations.get(self.active_name, {})
        return cal.get('metadata', {})
    
    def set_active(self, name: str) -> bool:
        """
        Switch to a different calibration preset.
        
        Parameters
        ----------
        name : str
            Calibration name to activate.
        
        Returns
        -------
        bool
            True if successful, False if calibration not found.
        """
        if name in self.calibrations:
            self.active_name = name
            self._save_calibrations()
            logger.info(f"Active calibration: {name}")
            return True
        
        logger.warning(f"Calibration '{name}' not found")
        return False
    
    def list_calibrations(self) -> list:
        """
        List all available calibration names.
        
        Returns
        -------
        list
            List of calibration names.
        """
        return list(self.calibrations.keys())
    
    # ------------------------------------------------------------------
    # Calibration Routine
    # ------------------------------------------------------------------
    
    def calibrate_from_points(self, point1: Tuple[int, int],
                               point2: Tuple[int, int],
                               known_distance_um: float) -> float:
        """
        Calculate nm/pixel from two points with known distance.
        
        This is the core calibration calculation. Use this when the user
        clicks two points on a stage micrometer.
        
        Parameters
        ----------
        point1, point2 : tuple
            Pixel coordinates (x, y) of the two points.
        known_distance_um : float
            Known physical distance between the points (in µm).
        
        Returns
        -------
        float
            Calculated nm/pixel.
        
        Example
        -------
        >>> # User clicks two points 100 µm apart on a stage micrometer
        >>> nm_per_pixel = calibrate_from_points(
        ...     point1=(100, 200),
        ...     point2=(400, 200),
        ...     known_distance_um=100.0
        ... )
        >>> print(f"{nm_per_pixel:.2f} nm/pixel")
        333.33 nm/pixel
        """
        # Calculate pixel distance
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        pixel_distance = np.sqrt(dx**2 + dy**2)
        
        if pixel_distance < 1.0:
            raise CalibrationError("Points are too close together (pixel distance < 1)")
        
        # Calculate nm/pixel
        nm_per_pixel = (known_distance_um * 1000.0) / pixel_distance
        
        logger.info(f"Calibration: {nm_per_pixel:.2f} nm/pixel "
                    f"({known_distance_um} µm over {pixel_distance:.1f} pixels)")
        
        return float(nm_per_pixel)
    
    # ------------------------------------------------------------------
    # Coordinate Conversion
    # ------------------------------------------------------------------
    
    def pixel_to_nm(self, px: int, py: int,
                    image_center: Optional[Tuple[int, int]] = None) -> Tuple[float, float]:
        """
        Convert pixel coordinates to nanometers (relative to image center).
        
        Parameters
        ----------
        px, py : int
            Pixel coordinates.
        image_center : tuple, optional
            Image center (width/2, height/2). If None, assumes (0, 0).
        
        Returns
        -------
        tuple
            (dx_nm, dy_nm) - position in nanometers relative to center.
        """
        nm_per_pixel = self.get_active()
        
        if image_center is not None:
            cx, cy = image_center
            dx_nm = (px - cx) * nm_per_pixel
            dy_nm = (py - cy) * nm_per_pixel
        else:
            dx_nm = px * nm_per_pixel
            dy_nm = py * nm_per_pixel
        
        return (dx_nm, dy_nm)
    
    def nm_to_pixel(self, dx_nm: float, dy_nm: float,
                    image_center: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Convert nanometers to pixel coordinates.
        
        Parameters
        ----------
        dx_nm, dy_nm : float
            Position in nanometers.
        image_center : tuple, optional
            Image center (width/2, height/2). If None, assumes (0, 0).
        
        Returns
        -------
        tuple
            (px, py) - pixel coordinates.
        """
        nm_per_pixel = self.get_active()
        
        if nm_per_pixel == 0:
            raise CalibrationError("No active calibration (nm_per_pixel = 0)")
        
        if image_center is not None:
            cx, cy = image_center
            px = int(round(dx_nm / nm_per_pixel + cx))
            py = int(round(dy_nm / nm_per_pixel + cy))
        else:
            px = int(round(dx_nm / nm_per_pixel))
            py = int(round(dy_nm / nm_per_pixel))
        
        return (px, py)
    
    def pixel_distance_to_nm(self, pixel_distance: float) -> float:
        """
        Convert a pixel distance to nanometers.
        
        Parameters
        ----------
        pixel_distance : float
            Distance in pixels.
        
        Returns
        -------
        float
            Distance in nanometers.
        """
        nm_per_pixel = self.get_active()
        return pixel_distance * nm_per_pixel
    
    def nm_distance_to_pixel(self, nm_distance: float) -> float:
        """
        Convert a nanometer distance to pixels.
        
        Parameters
        ----------
        nm_distance : float
            Distance in nanometers.
        
        Returns
        -------
        float
            Distance in pixels.
        """
        nm_per_pixel = self.get_active()
        if nm_per_pixel == 0:
            raise CalibrationError("No active calibration (nm_per_pixel = 0)")
        return nm_distance / nm_per_pixel
    
    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    
    def _load_calibrations(self) -> None:
        """Load calibrations from config file."""
        if not self.config_path.exists():
            logger.info(f"No calibration file found at {self.config_path}")
            self._create_default_calibration()
            return
        
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
            
            self.calibrations = data.get('calibrations', {})
            self.active_name = data.get('active_calibration', 'default')
            
            logger.info(f"Loaded {len(self.calibrations)} calibrations from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load calibrations: {e}")
            self._create_default_calibration()
    
    def _save_calibrations(self) -> None:
        """Save calibrations to config file."""
        try:
            data = {
                'calibrations': self.calibrations,
                'active_calibration': self.active_name
            }
            
            # Create parent directory if needed
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved calibrations to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save calibrations: {e}")
    
    def _create_default_calibration(self) -> None:
        """Create a default calibration if none exists."""
        self.calibrations = {
            'default': {
                'nm_per_pixel': 1.0,
                'metadata': {
                    'objective_mag': 1,
                    'zoom_level': 1.0,
                    'notes': 'Default 1:1 calibration (1 nm/pixel)'
                }
            }
        }
        self.active_name = 'default'
        self._save_calibrations()
        logger.info("Created default calibration")
    
    def __repr__(self) -> str:
        return f"CalibrationManager(active={self.active_name}, calibrations={len(self.calibrations)})"