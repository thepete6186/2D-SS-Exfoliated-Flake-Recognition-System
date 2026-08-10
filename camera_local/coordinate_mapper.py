"""
Coordinate conversion utilities for camera-to-stage mapping.

Converts between:
- Pixel coordinates (camera image)
- Nanometers (physical distance)
- Stage steps/µm (motor controller units)
"""

import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CoordinateMapper:
    """
    Converts coordinates between camera pixels, nanometers, and stage units.
    
    Handles:
    - Pixel ↔ nanometer conversion (via calibration)
    - Nanometer ↔ stage step conversion (via stage config)
    - Stage orientation/inversion
    - Image center offset
    """
    
    def __init__(self, calibration_manager, stage_config: Optional[dict] = None):
        """
        Initialize coordinate mapper.
        
        Parameters
        ----------
        calibration_manager : CalibrationManager
            Calibration manager for pixel-to-nm conversion.
        stage_config : dict, optional
            Stage configuration with µm/step values:
            {
                'sigmakoki': {'xy_um_per_step': 0.5, 'z_um_per_step': 0.1},
                'zolix': {'xy_um_per_step': 0.5, 'r_deg_per_step': 0.01}
            }
        """
        self.calibration = calibration_manager
        self.stage_config = stage_config or {}
    
    # ------------------------------------------------------------------
    # Pixel ↔ Nanometer
    # ------------------------------------------------------------------
    
    def pixel_to_nm(self, px: int, py: int,
                    image_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Convert pixel coordinates to nanometers (relative to image center).
        
        Parameters
        ----------
        px, py : int
            Pixel coordinates.
        image_size : tuple
            Image (width, height) in pixels.
        
        Returns
        -------
        tuple
            (dx_nm, dy_nm) - position in nanometers relative to image center.
        """
        width, height = image_size
        center = (width // 2, height // 2)
        return self.calibration.pixel_to_nm(px, py, center)
    
    def nm_to_pixel(self, dx_nm: float, dy_nm: float,
                    image_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert nanometers to pixel coordinates.
        
        Parameters
        ----------
        dx_nm, dy_nm : float
            Position in nanometers (relative to image center).
        image_size : tuple
            Image (width, height) in pixels.
        
        Returns
        -------
        tuple
            (px, py) - pixel coordinates.
        """
        width, height = image_size
        center = (width // 2, height // 2)
        return self.calibration.nm_to_pixel(dx_nm, dy_nm, center)
    
    def pixel_distance_to_nm(self, pixel_distance: float) -> float:
        """Convert pixel distance to nanometers."""
        return self.calibration.pixel_distance_to_nm(pixel_distance)
    
    def nm_distance_to_pixel(self, nm_distance: float) -> float:
        """Convert nanometer distance to pixels."""
        return self.calibration.nm_distance_to_pixel(nm_distance)
    
    # ------------------------------------------------------------------
    # Nanometer ↔ Stage Steps
    # ------------------------------------------------------------------
    
    def nm_to_sigmakoki_steps(self, dx_nm: float, dy_nm: float,
                               axis: str = "xy") -> Tuple[int, int]:
        """
        Convert nanometers to SigmaKoki stage steps.
        
        Parameters
        ----------
        dx_nm, dy_nm : float
            Distance in nanometers.
        axis : str
            Axis type: "xy" (linear) or "z" (Z axis).
        
        Returns
        -------
        tuple
            (steps_x, steps_y) or (steps_x, steps_z) depending on axis.
        """
        config = self.stage_config.get('sigmakoki', {})
        
        if axis == "z":
            um_per_step = config.get('z_um_per_step', 0.1)
            # Only convert the relevant axis
            dx_um = dx_nm / 1000.0
            steps = int(round(dx_um / um_per_step))
            return (steps, 0)
        else:
            um_per_step = config.get('xy_um_per_step', 0.5)
            dx_um = dx_nm / 1000.0
            dy_um = dy_nm / 1000.0
            steps_x = int(round(dx_um / um_per_step))
            steps_y = int(round(dy_um / um_per_step))
            return (steps_x, steps_y)
    
    def nm_to_zolix_steps(self, dx_nm: float, dy_nm: float,
                          axis: str = "xy") -> Tuple[int, int]:
        """
        Convert nanometers to Zolix stage steps.
        
        Parameters
        ----------
        dx_nm, dy_nm : float
            Distance in nanometers.
        axis : str
            Axis type: "xy" (linear) or "r" (rotation).
        
        Returns
        -------
        tuple
            (steps_x, steps_y) or (steps_x, steps_r) depending on axis.
        """
        config = self.stage_config.get('zolix', {})
        
        if axis == "r":
            deg_per_step = config.get('r_deg_per_step', 0.01)
            # For rotation, convert nm to degrees (assuming circular motion)
            # This is approximate - adjust based on your setup
            dx_deg = dx_nm / 1000.0  # Simplified: 1 µm ≈ 0.001°
            steps = int(round(dx_deg / deg_per_step))
            return (steps, 0)
        else:
            um_per_step = config.get('xy_um_per_step', 0.5)
            dx_um = dx_nm / 1000.0
            dy_um = dy_nm / 1000.0
            steps_x = int(round(dx_um / um_per_step))
            steps_y = int(round(dy_um / um_per_step))
            return (steps_x, steps_y)
    
    def stage_steps_to_nm(self, steps_x: int, steps_y: int,
                          stage: str = "sigmakoki") -> Tuple[float, float]:
        """
        Convert stage steps to nanometers.
        
        Parameters
        ----------
        steps_x, steps_y : int
            Step counts.
        stage : str
            Stage name: "sigmakoki" or "zolix".
        
        Returns
        -------
        tuple
            (dx_nm, dy_nm) in nanometers.
        """
        config = self.stage_config.get(stage, {})
        um_per_step = config.get('xy_um_per_step', 0.5)
        
        dx_um = steps_x * um_per_step
        dy_um = steps_y * um_per_step
        dx_nm = dx_um * 1000.0
        dy_nm = dy_um * 1000.0
        
        return (dx_nm, dy_nm)
    # ------------------------------------------------------------------
    # Full Pipeline: Pixel → Stage Commands
    # ------------------------------------------------------------------
    
    def pixel_to_stage_commands(self, px: int, py: int,
                                 image_size: Tuple[int, int],
                                 stage: str = "sigmakoki",
                                 invert_x: bool = False,
                                 invert_y: bool = False,
                                 flip_xy: bool = False) -> Dict[str, Any]:
        """
        Convert pixel coordinates to stage movement commands.
        
        This is the main entry point for the pick-and-place workflow.
        
        Parameters
        ----------
        px, py : int
            Pixel coordinates (from flake detection).
        image_size : tuple
            Image (width, height) in pixels.
        stage : str
            Target stage: "sigmakoki" or "zolix".
        invert_x, invert_y : bool
            Invert direction for each axis.
        flip_xy : bool
            Swap X and Y axes.
        
        Returns
        -------
        dict
            Stage command with keys:
            - 'stage_id': str
            - 'axis': str
            - 'steps_x': int
            - 'steps_y': int
            - 'dx_nm': float
            - 'dy_nm': float
        """
        # Step 1: Pixel → nanometers (relative to image center)
        dx_nm, dy_nm = self.pixel_to_nm(px, py, image_size)
        
        # Step 2: Nanometers → stage steps
        if stage == "sigmakoki":
            steps_x, steps_y = self.nm_to_sigmakoki_steps(dx_nm, dy_nm)
        elif stage == "zolix":
            steps_x, steps_y = self.nm_to_zolix_steps(dx_nm, dy_nm)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        # Step 3: Apply axis transformations
        if flip_xy:
            steps_x, steps_y = steps_y, steps_x
            dx_nm, dy_nm = dy_nm, dx_nm
        
        if invert_x:
            steps_x = -steps_x
            dx_nm = -dx_nm
        
        if invert_y:
            steps_y = -steps_y
            dy_nm = -dy_nm
        
        return {
            'stage_id': stage,
            'steps_x': steps_x,
            'steps_y': steps_y,
            'dx_nm': dx_nm,
            'dy_nm': dy_nm,
            'pixel_x': px,
            'pixel_y': py,
        }
    
    def flake_centroid_to_stage(self, centroid: Tuple[int, int],
                                 image_size: Tuple[int, int],
                                 stage: str = "sigmakoki",
                                 **kwargs) -> Dict[str, Any]:
        """
        Convert a flake centroid (from pipeline output) to stage commands.
        
        Parameters
        ----------
        centroid : tuple
            (x, y) pixel coordinates of flake centroid.
        image_size : tuple
            Image (width, height) in pixels.
        stage : str
            Target stage.
        **kwargs
            Additional arguments passed to pixel_to_stage_commands().
        
        Returns
        -------
        dict
            Stage command dict.
        """
        px, py = centroid
        return self.pixel_to_stage_commands(px, py, image_size, stage, **kwargs)
    
    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    
    def get_current_nm_per_pixel(self) -> float:
        """Get current nm/pixel from active calibration."""
        return self.calibration.get_active()
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get current calibration information."""
        return {
            'nm_per_pixel': self.calibration.get_active(),
            'calibration_name': self.calibration.active_name,
            'metadata': self.calibration.get_active_metadata(),
        }
    
    def __repr__(self) -> str:
        return (f"CoordinateMapper(calibration={self.calibration.active_name}, "
                f"nm_per_pixel={self.calibration.get_active():.2f})")