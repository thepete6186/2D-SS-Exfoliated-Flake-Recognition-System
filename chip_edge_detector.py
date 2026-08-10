#!/usr/bin/env python3
"""
Chip Edge Detector - Automated chip boundary detection interface.

Features:
- Live video feed from SmartCam camera
- Zolix ZC300 stage control
- Automated black edge detection with directional scanning
- Real-time scan visualization
- Stops automatically when edge is found

Usage:
    python chip_edge_detector.py                    # GUI
    python chip_edge_detector.py --port COM3        # specify stage port
    python chip_edge_detector.py --simulate         # use SimulatedStage
"""

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from camera_local.smartcam_camera import SmartCamCamera
from stage.zolix_zc300 import ZolixZC300
from stage.simulated import SimulatedStage
from stage.base import Stage
from black_edge_detector import (
    find_black_edge_point,
    detect_edge_robust,
    detect_straight_black_edge,
    visualize_edge_detection
)
from auto_align import AutoAlignController
from chip_scanner import ChipScanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
AXES = ("x", "y", "r")
AXIS_NAMES = {"x": "X", "y": "Y", "r": "R (Z)"}
SCAN_MARGIN = 50  # Pixels to skip at image edges

# Stage calibration (Zolix ZC300)
# Linear axes (X, Y): 0.625 µm/step
# Rotation axis (R): 0.00125 degrees/step
STAGE_STEP_SIZE_UM = 0.625  # micrometers per step for X, Y axes
STAGE_ROTATION_STEP_DEG = 0.00125  # degrees per step for R axis

# Default camera calibration (Zeiss Axiocam 208 at 3840x2160)
# 1 mm ≈ 2650 pixels (approximate, adjust based on actual measurement)
DEFAULT_PIXELS_PER_MM = 2650.0
DEFAULT_PIXELS_PER_NM = DEFAULT_PIXELS_PER_MM / 1_000_000.0  # Convert to nm


class ChipEdgeDetector:
    """
    Integrated chip edge detection with stage control.
    """

    def __init__(self, root: tk.Tk, port: str, simulate: bool):
        self.root = root
        self.root.title("Chip Edge Detector - Automated Boundary Detection")
        self.root.geometry("1400x800")
        self.root.state("zoomed")

        # Stage control
        self.port = port
        self.simulate = simulate
        self.stage: Optional[Stage] = None
        self.stage_connected = False
        self.rel_steps: Dict[str, tk.StringVar] = {}
        self.abs_pos: Dict[str, tk.StringVar] = {}
        self.speed_var = tk.StringVar(value="1000")

        # Camera
        self.camera: Optional[SmartCamCamera] = None
        self.camera_connected = False
        self._capture_thread = None
        self._capture_stop = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_fps = 0.0

        # Display
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.current_image = None
        self.display_image = None

        # Calibration
        self.calibration = {}  # {zoom_level: pixels_per_nm}
        self.current_zoom_preset = tk.StringVar(value="5x")  # Default to 5x zoom
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_line = None

        # Apply default calibration (1 mm ≈ 2650 pixels at 5x zoom)
        self._apply_default_calibration()

        # Edge detection
        self.edge_detected = False
        self.edge_theta = None
        self.edge_anchor = None
        self.edge_points = []
        self.scan_direction = tk.StringVar(value="left-to-right")
        self.is_scanning = False

        # FPS
        self.frame_count = 0
        self.last_fps_time = 0
        self.fps = 0.0

        # Build GUI
        self._build_gui()

        # Start frame update loop
        self._update_frame()

    # ------------------------------------------------------------------
    # GUI Building
    # ------------------------------------------------------------------

    def _build_gui(self):
        """Build the GUI layout."""
        pad = {"padx": 5, "pady": 5}

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side: video display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side: controls
        right_panel = ttk.Frame(main_container, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Video canvas
        self.canvas = tk.Canvas(left_panel, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)

        # Right panel contents
        self._build_right_panel(right_panel)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Status: Disconnected", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_right_panel(self, parent):
        """Build the right control panel."""
        pad = {"padx": 5, "pady": 5}

        # Stage Connection section
        stage_conn_frame = ttk.LabelFrame(parent, text="Stage Control (Zolix ZC300)")
        stage_conn_frame.pack(fill=tk.X, **pad)

        # Connection controls
        conn_row = ttk.Frame(stage_conn_frame)
        conn_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(conn_row, text="Port:").pack(side=tk.LEFT)
        self.port_var = tk.StringVar(value=self.port)
        ttk.Entry(conn_row, textvariable=self.port_var, width=10).pack(side=tk.LEFT, padx=5)
        self.sim_chk = tk.BooleanVar(value=self.simulate)
        ttk.Checkbutton(conn_row, text="Simulate", variable=self.sim_chk).pack(side=tk.LEFT, padx=5)
        self.btn_stage_connect = ttk.Button(
            conn_row, text="Connect Stage", command=self._toggle_stage_connection
        )
        self.btn_stage_connect.pack(side=tk.LEFT, padx=5)

        self.stage_conn_label = ttk.Label(conn_row, text="Disconnected", foreground="gray")
        self.stage_conn_label.pack(side=tk.LEFT, padx=5)

        # Motion frame (per-axis rows)
        motion = ttk.LabelFrame(parent, text="Stage Motion (pulses)")
        motion.pack(fill=tk.X, **pad)

        # Header
        ttk.Label(motion, text="Axis", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=4)
        ttk.Label(motion, text="(-)", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=4)
        ttk.Label(motion, text="Step", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=4)
        ttk.Label(motion, text="(+)", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=4)
        ttk.Label(motion, text="Abs pos", font=("Arial", 10, "bold")).grid(row=0, column=4, padx=4)
        ttk.Label(motion, text="Go", font=("Arial", 10, "bold")).grid(row=0, column=5, padx=4)
        ttk.Label(motion, text="Home", font=("Arial", 10, "bold")).grid(row=0, column=6, padx=4)

        for i, axis in enumerate(AXES, start=1):
            name = AXIS_NAMES[axis]
            ttk.Label(motion, text=name, font=("Arial", 10, "bold")).grid(
                row=i, column=0, padx=4, pady=2
            )

            self.rel_steps[axis] = tk.StringVar(value="1000")
            ttk.Button(
                motion, text="◀ -", width=6,
                command=lambda a=axis: self._move_stage(a, -1),
            ).grid(row=i, column=1, padx=2, pady=2)
            ttk.Entry(motion, textvariable=self.rel_steps[axis], width=8).grid(
                row=i, column=2, padx=2
            )
            ttk.Button(
                motion, text="+ ▶", width=6,
                command=lambda a=axis: self._move_stage(a, +1),
            ).grid(row=i, column=3, padx=2, pady=2)

            self.abs_pos[axis] = tk.StringVar(value="0")
            ttk.Entry(motion, textvariable=self.abs_pos[axis], width=10).grid(
                row=i, column=4, padx=2
            )
            ttk.Button(
                motion, text="Go", width=6,
                command=lambda a=axis: self._move_stage_abs(a),
            ).grid(row=i, column=5, padx=2)
            ttk.Button(
                motion, text="Home", width=6,
                command=lambda a=axis: self._home_stage(a),
            ).grid(row=i, column=6, padx=2)

        # Speed + global controls
        ctrl = ttk.LabelFrame(parent, text="Stage Controls")
        ctrl.pack(fill=tk.X, **pad)

        ttk.Label(ctrl, text="Speed (pulses/s):").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(ctrl, textvariable=self.speed_var, width=10).grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Button(ctrl, text="Apply Speed", command=self._apply_stage_speed).grid(
            row=0, column=2, padx=8
        )
        ttk.Button(ctrl, text="Home All", command=lambda: self._home_stage("all")).grid(
            row=0, column=3, padx=8
        )
        ttk.Button(ctrl, text="Stop All", command=lambda: self._stop_stage("all")).grid(
            row=0, column=4, padx=8
        )

        # Stage status
        stage_status = ttk.LabelFrame(parent, text="Stage Status")
        stage_status.pack(fill=tk.BOTH, expand=True, **pad)

        self.stage_status_text = tk.Text(
            stage_status, height=6, state=tk.DISABLED, font=("Consolas", 9)
        )
        self.stage_status_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Stage calibration info
        cal_info_frame = ttk.LabelFrame(parent, text="Stage Calibration")
        cal_info_frame.pack(fill=tk.X, **pad)

        cal_info_text = (
            f"X, Y: {STAGE_STEP_SIZE_UM} µm/step\n"
            f"R: {STAGE_ROTATION_STEP_DEG}°/step"
        )
        ttk.Label(cal_info_frame, text=cal_info_text, font=("Consolas", 9)).pack(
            padx=5, pady=5, anchor=tk.W
        )

        # Camera section
        cam_frame = ttk.LabelFrame(parent, text="Camera (SmartCam)")
        cam_frame.pack(fill=tk.X, **pad)

        self.btn_cam_connect = ttk.Button(
            cam_frame, text="Connect Camera", command=self._toggle_camera_connection
        )
        self.btn_cam_connect.pack(pady=5, padx=5, fill=tk.X)

        self.cam_conn_label = ttk.Label(cam_frame, text="Disconnected", foreground="gray")
        self.cam_conn_label.pack(pady=2)

        # Zoom controls
        zoom_frame = ttk.LabelFrame(parent, text="Zoom Control")
        zoom_frame.pack(fill=tk.X, **pad)

        zoom_row = ttk.Frame(zoom_frame)
        zoom_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(zoom_row, text="Zoom Preset:").pack(side=tk.LEFT)
        self.zoom_preset_combo = ttk.Combobox(
            zoom_row,
            textvariable=self.current_zoom_preset,
            values=["1x", "5x", "10x", "20x", "50x"],
            state="readonly",
            width=10
        )
        self.zoom_preset_combo.pack(side=tk.LEFT, padx=5)
        self.zoom_preset_combo.bind("<<ComboboxSelected>>", self._on_zoom_preset_changed)

        ttk.Button(zoom_row, text="Calibrate", command=self._open_calibration_dialog).pack(
            side=tk.LEFT, padx=5
        )

        # Edge Detection section
        edge_frame = ttk.LabelFrame(parent, text="Chip Edge Detection")
        edge_frame.pack(fill=tk.X, **pad)

        # Scan direction
        dir_row = ttk.Frame(edge_frame)
        dir_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(dir_row, text="Scan Direction:").pack(side=tk.LEFT)
        self.scan_direction = tk.StringVar(value="left-to-right")
        ttk.Radiobutton(
            dir_row, text="Left → Right", variable=self.scan_direction,
            value="left-to-right"
        ).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(
            dir_row, text="Right → Left", variable=self.scan_direction,
            value="right-to-left"
        ).pack(side=tk.LEFT, padx=10)

        # Edge detection controls
        edge_btn_row = ttk.Frame(edge_frame)
        edge_btn_row.pack(fill=tk.X, padx=5, pady=5)

        self.btn_detect_edge = ttk.Button(
            edge_btn_row,
            text="Detect Chip Edge",
            command=self._start_edge_detection,
            style="Accent.TButton"
        )
        self.btn_detect_edge.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.btn_stop_scan = ttk.Button(
            edge_btn_row,
            text="Stop Scan",
            command=self._stop_scanning,
            state=tk.DISABLED
        )
        self.btn_stop_scan.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Auto-alignment controls
        align_btn_row = ttk.Frame(edge_frame)
        align_btn_row.pack(fill=tk.X, padx=5, pady=5)

        self.btn_align = ttk.Button(
            align_btn_row,
            text="Auto-Align Sample",
            command=self._start_auto_align
        )
        self.btn_align.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.btn_stop_align = ttk.Button(
            align_btn_row,
            text="Stop Alignment",
            command=self._stop_alignment,
            state=tk.DISABLED
        )
        self.btn_stop_align.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Edge detection results
        self.edge_result_label = ttk.Label(edge_frame, text="Edge: Not detected")
        self.edge_result_label.pack(pady=5)

        # Manual Area Scan section
        scan_frame = ttk.LabelFrame(parent, text="Manual Area Scan")
        scan_frame.pack(fill=tk.X, **pad)

        # Scan instructions
        scan_info = (
            "1. Move stage to start corner\n"
            "2. Click 'Set Start'\n"
            "3. Move stage to end corner\n"
            "4. Click 'Set End'\n"
            "5. Click 'Start Scan'"
        )
        ttk.Label(scan_frame, text=scan_info, justify=tk.LEFT).pack(
            padx=5, pady=5, anchor=tk.W
        )

        # Position display
        pos_row = ttk.Frame(scan_frame)
        pos_row.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(pos_row, text="Start:").pack(side=tk.LEFT)
        self.scan_start_label = ttk.Label(pos_row, text="Not set", width=20)
        self.scan_start_label.pack(side=tk.LEFT, padx=5)

        ttk.Label(pos_row, text="End:").pack(side=tk.LEFT)
        self.scan_end_label = ttk.Label(pos_row, text="Not set", width=20)
        self.scan_end_label.pack(side=tk.LEFT, padx=5)

        # Overlap control
        overlap_row = ttk.Frame(scan_frame)
        overlap_row.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(overlap_row, text="Overlap (%):").pack(side=tk.LEFT)
        self.overlap_var = tk.StringVar(value="5.0")
        ttk.Entry(overlap_row, textvariable=self.overlap_var, width=8).pack(
            side=tk.LEFT, padx=5
        )

        # Scan buttons
        scan_btn_row = ttk.Frame(scan_frame)
        scan_btn_row.pack(fill=tk.X, padx=5, pady=5)

        self.btn_set_start = ttk.Button(
            scan_btn_row, text="Set Start", command=self._set_scan_start
        )
        self.btn_set_start.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.btn_set_end = ttk.Button(
            scan_btn_row, text="Set End", command=self._set_scan_end
        )
        self.btn_set_end.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.btn_start_scan = ttk.Button(
            scan_btn_row, text="Start Scan", command=self._start_manual_scan
        )
        self.btn_start_scan.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        self.btn_stop_manual_scan = ttk.Button(
            scan_btn_row, text="Stop", command=self._stop_manual_scan,
            state=tk.DISABLED
        )
        self.btn_stop_manual_scan.pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)

        # Scan progress
        self.scan_progress_label = ttk.Label(scan_frame, text="Ready")
        self.scan_progress_label.pack(pady=2)

        # Instructions
        instructions_frame = ttk.LabelFrame(parent, text="Instructions")
        instructions_frame.pack(fill=tk.X, **pad)

        instructions = (
            "1. Connect camera and stage\n"
            "2. Position stage near chip edge\n"
            "3. Select scan direction (L→R or R→L)\n"
            "4. Click 'Detect Chip Edge'\n"
            "5. System scans and stops at edge\n"
            "6. Edge angle and position displayed\n\n"
            "Mouse wheel: Zoom | Middle-drag: Pan"
        )
        ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT).pack(
            padx=5, pady=5, anchor=tk.W
        )

    # ------------------------------------------------------------------
    # Stage Connection
    # ------------------------------------------------------------------

    def _toggle_stage_connection(self):
        """Connect or disconnect stage."""
        if self.stage_connected:
            self._disconnect_stage()
        else:
            self._connect_stage()

    def _connect_stage(self):
        """Connect to stage."""
        port = self.port_var.get().strip()
        simulate = self.sim_chk.get()
        try:
            self.stage = self._build_stage(port, simulate)
            threading.Thread(target=self._connect_stage_worker, daemon=True).start()
            self._set_status("Connecting to stage...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create stage: {e}")

    def _build_stage(self, port: str, simulate: bool) -> Stage:
        """Construct the stage driver."""
        if simulate:
            return SimulatedStage(
                axes=AXES,
                speed_pps=2000.0,
                limits={"x": (-100000, 100000), "y": (-100000, 100000), "r": (-36000, 36000)},
            )
        return ZolixZC300(
            port=port,
            axes=AXES,
            default_speed_pps=1000.0,
            acceleration=10_000_000.0,
        )

    def _connect_stage_worker(self):
        """Worker thread for stage connection."""
        try:
            self.stage.connect()
            self.stage_connected = True
            self.root.after(0, self._on_stage_connected)
        except Exception as e:
            self.stage_connected = False
            self.stage = None
            self.root.after(0, lambda: messagebox.showerror("Connect Failed", str(e)))
            self.root.after(0, lambda: self._set_status("Stage disconnected"))

    def _on_stage_connected(self):
        """Handle successful stage connection."""
        self.btn_stage_connect.config(text="Disconnect Stage")
        port = getattr(self.stage, '_port', 'Simulated')
        self.stage_conn_label.config(text=port, foreground="green")
        self._set_status("Stage connected")
        self._apply_stage_speed()
        self._update_stage_status()

    def _disconnect_stage(self):
        """Disconnect stage."""
        try:
            if self.stage is not None:
                self.stage.stop("all")
                self.stage.disconnect()
        except Exception:
            pass
        self.stage = None
        self.stage_connected = False
        self.btn_stage_connect.config(text="Connect Stage")
        self.stage_conn_label.config(text="Disconnected", foreground="gray")
        self._set_status("Stage disconnected")

    # ------------------------------------------------------------------
    # Stage Motion
    # ------------------------------------------------------------------

    def _move_stage(self, axis: str, sign: int):
        """Move stage relative."""
        if not self._require_stage_connected():
            return
        try:
            steps = int(float(self.rel_steps[axis].get()))
            self._set_status(f"Moving {AXIS_NAMES[axis]} {sign * steps} pulses...")
            threading.Thread(
                target=self._motion_worker,
                args=(lambda: self.stage.move_relative(axis, sign * steps, wait=True, timeout=60.0),),
                daemon=True,
            ).start()
        except ValueError:
            messagebox.showwarning("Invalid step", "Step must be a number")

    def _move_stage_abs(self, axis: str):
        """Move stage absolute."""
        if not self._require_stage_connected():
            return
        try:
            pos = float(self.abs_pos[axis].get())
            self._set_status(f"Moving {AXIS_NAMES[axis]} to {pos} pulses...")
            threading.Thread(
                target=self._motion_worker,
                args=(lambda: self.stage.move_absolute(axis, pos, wait=True, timeout=60.0),),
                daemon=True,
            ).start()
        except ValueError:
            messagebox.showwarning("Invalid position", "Absolute position must be a number")

    def _home_stage(self, axis: str):
        """Home stage."""
        if not self._require_stage_connected():
            return
        self._set_status(f"Homing {'all axes' if axis == 'all' else AXIS_NAMES[axis]}...")
        threading.Thread(
            target=self._motion_worker,
            args=(lambda: self.stage.home(axis, wait=True, timeout=120.0),),
            daemon=True,
        ).start()

    def _stop_stage(self, axis: str = "all"):
        """Stop stage."""
        if not self._require_stage_connected():
            return
        try:
            self.stage.stop(axis)
            self._set_status(f"Stopped {'all axes' if axis == 'all' else AXIS_NAMES[axis]}")
        except Exception as e:
            self._set_status(f"Stop error: {e}")

    def _apply_stage_speed(self):
        """Apply speed to stage."""
        if not self.stage_connected or self.stage is None:
            return
        try:
            speed = float(self.speed_var.get())
            for axis in AXES:
                try:
                    self.stage.set_speed(axis, speed)
                except Exception:
                    pass
            self._set_status(f"Speed set to {speed} pulses/s")
        except ValueError:
            messagebox.showwarning("Invalid speed", "Speed must be a number")

    def _require_stage_connected(self) -> bool:
        """Check if stage is connected."""
        if not self.stage_connected or self.stage is None:
            messagebox.showwarning("Not connected", "Connect the stage first")
            return False
        return True

    def _motion_worker(self, fn):
        """Worker for stage motion."""
        try:
            fn()
            self.root.after(0, lambda: self._set_status("Done"))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"Error: {e}"))

    def _update_stage_status(self):
        """Poll stage status."""
        if self.stage_connected and self.stage is not None:
            try:
                status = self.stage.get_status()
                pos = status["position"]
                moving = status["moving"]
                lines = []
                for axis in AXES:
                    lines.append(
                        f"{AXIS_NAMES[axis]:>6}: {pos.get(axis, 0):>12.1f} pulses"
                        f"   {'MOVING' if moving.get(axis) else 'stopped'}"
                    )
                if status.get("emergency_stop"):
                    lines.append("!! E-STOP ACTIVE !!")
                for key, flag in status.get("limits", {}).items():
                    if flag:
                        lines.append(f"Limit: {key}")
                self._render_stage_status("\n".join(lines))
            except Exception as e:
                self._render_stage_status(f"Status error: {e}")
        self.root.after(500, self._update_stage_status)

    def _render_stage_status(self, text: str):
        """Render stage status text."""
        self.stage_status_text.config(state=tk.NORMAL)
        self.stage_status_text.delete("1.0", tk.END)
        self.stage_status_text.insert(tk.END, text)
        self.stage_status_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Camera Connection
    # ------------------------------------------------------------------

    def _toggle_camera_connection(self):
        """Connect or disconnect camera."""
        if self.camera_connected:
            self._disconnect_camera()
        else:
            self._connect_camera()

    def _connect_camera(self):
        """Connect to SmartCam camera."""
        try:
            self.camera = SmartCamCamera(camera_index=0)
            if self.camera.connect():
                self.camera_connected = True
                self.btn_cam_connect.config(text="Disconnect Camera")
                info = self.camera.get_camera_info()
                self.cam_conn_label.config(
                    text=f"{info.get('name', 'Unknown')} ({info.get('backend', '')})",
                    foreground="green"
                )
                self._set_status("Camera connected")
                self._start_capture_thread()
            else:
                messagebox.showerror(
                    "Camera Connection Failed",
                    "Could not connect to SmartCam camera.\n\n"
                    "Troubleshooting:\n"
                    "  1. Ensure SmartCamApi.dll is installed (comes with Labscope)\n"
                    "  2. Ensure libusb0 driver is installed for the camera\n"
                    "  3. Make sure camera is not held by another app\n"
                    "  4. Try unplugging and replugging the camera"
                )
        except Exception as e:
            messagebox.showerror("Error", f"Camera connection failed: {e}")

    def _disconnect_camera(self):
        """Disconnect camera."""
        self._capture_stop.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        self._capture_thread = None
        self._latest_frame = None

        if self.camera:
            try:
                self.camera.disconnect()
            except Exception:
                pass
            self.camera = None

        self.camera_connected = False
        self.btn_cam_connect.config(text="Connect Camera")
        self.cam_conn_label.config(text="Disconnected", foreground="gray")
        self._set_status("Camera disconnected")

    def _start_capture_thread(self):
        """Start background capture thread."""
        self._capture_stop = threading.Event()
        self.frame_count = 0
        self.fps = 0.0
        self._capture_fps = 0.0
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            args=(self._capture_stop, self.camera),
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self, stop_event, camera):
        """Continuously capture frames."""
        frame_count = 0
        last_time = time.time()
        while not stop_event.is_set():
            try:
                data = camera.capture()
                if data is not None:
                    with self._frame_lock:
                        self._latest_frame = data
                    frame_count += 1
                    now = time.time()
                    if now - last_time >= 1.0:
                        self._capture_fps = frame_count / (now - last_time)
                        frame_count = 0
                        last_time = now
            except Exception as e:
                logger.error(f"Capture thread error: {e}")
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Frame Update Loop
    # ------------------------------------------------------------------

    def _update_frame(self):
        """Update camera frame."""
        if self.camera_connected and self.camera:
            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None:
                self.current_image = frame

                # Update FPS
                self.frame_count += 1
                current_time = cv2.getTickCount()
                if self.frame_count == 1:
                    self.last_fps_time = current_time
                elif current_time - self.last_fps_time >= cv2.getTickFrequency():
                    self.fps = self.frame_count / ((current_time - self.last_fps_time) / cv2.getTickFrequency())
                    self.frame_count = 0

                self._update_display()

        self.root.after(33, self._update_frame)

    def _update_display(self):
        """Update the canvas with current image."""
        if self.current_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        img = self.current_image
        img_h, img_w = img.shape[:2]

        # Convert RGB to BGR for display and processing
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Apply zoom
        if self.zoom_level != 1.0:
            target_w = int(img_w * self.zoom_level)
            target_h = int(img_h * self.zoom_level)
        else:
            target_w, target_h = img_w, img_h

        # Fit to canvas
        img_ratio = target_w / target_h
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            display_width = canvas_width
            display_height = int(canvas_width / img_ratio)
        else:
            display_height = canvas_height
            display_width = int(canvas_height * img_ratio)

        # Downscale
        if display_width < img_w or display_height < img_h:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        display_img = cv2.resize(img_bgr, (display_width, display_height), interpolation=interp)

        # Convert to PIL
        pil_img = Image.fromarray(display_img)
        self.display_image = ImageTk.PhotoImage(pil_img)

        # Clear and draw
        self.canvas.delete("all")
        x = (canvas_width - display_width) // 2 + self.pan_x
        y = (canvas_height - display_height) // 2 + self.pan_y
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image)

        # Store display parameters
        self.display_params = {
            "x": x,
            "y": y,
            "width": display_width,
            "height": display_height,
            "orig_width": img_w,
            "orig_height": img_h,
        }

        # Draw edge detection results if available
        if self.edge_points:
            self._draw_edge_results()

        # Draw scale bar if calibrated
        self._draw_scale_bar()

        # Update status
        status_text = f"Camera | {img_w}x{img_h} | {self.fps:.1f} fps | "
        if self.edge_detected and self.edge_theta is not None:
            status_text += f"Edge: θ={self.edge_theta:.1f}° | "
        status_text += f"Stage: {'Connected' if self.stage_connected else 'Disconnected'}"
        self.status_bar.config(text=status_text)

    def _draw_edge_results(self):
        """Draw edge detection results on canvas."""
        if not self.edge_points or not hasattr(self, 'display_params'):
            return

        params = self.display_params
        scale_x = params["width"] / params["orig_width"]
        scale_y = params["height"] / params["orig_height"]

        # Draw edge points
        for i, (ex, ey) in enumerate(self.edge_points):
            disp_x = int(ex * scale_x) + params["x"]
            disp_y = int(ey * scale_y) + params["y"]
            self.canvas.create_oval(
                disp_x - 4, disp_y - 4, disp_x + 4, disp_y + 4,
                fill="yellow", outline="white", width=2
            )

        # Draw fitted line if available
        if self.edge_theta is not None and self.edge_anchor is not None:
            x0, y0 = self.edge_anchor
            disp_x0 = int(x0 * scale_x) + params["x"]
            disp_y0 = int(y0 * scale_y) + params["y"]

            # Draw line extending across canvas
            theta_rad = np.radians(self.edge_theta)
            line_len = 2000
            dx = np.cos(theta_rad) * line_len
            dy = np.sin(theta_rad) * line_len

            self.canvas.create_line(
                disp_x0 - dx, disp_y0 - dy,
                disp_x0 + dx, disp_y0 + dy,
                fill="red", width=3
            )
            self.canvas.create_oval(
                disp_x0 - 6, disp_y0 - 6, disp_x0 + 6, disp_y0 + 6,
                fill="red", outline="white", width=2
            )

    def _draw_scale_bar(self):
        """Draw scale bar on canvas if calibration exists."""
        if not hasattr(self, 'display_params'):
            return

        scale_info = self.get_scale_bar_info()
        if scale_info is None:
            return

        pixels_per_nm, label = scale_info
        params = self.display_params

        # Calculate scale bar length in display pixels
        # We want ~100 pixels display length
        target_display_pixels = 100
        nm_for_100px = target_display_pixels / pixels_per_nm

        # Find a nice round number close to this
        nice_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        best_nm = min(nice_values, key=lambda x: abs(x - nm_for_100px))
        scale_bar_pixels = best_nm * pixels_per_nm

        # Convert to display coordinates
        scale_x = params["width"] / params["orig_width"]
        display_pixels = scale_bar_pixels * scale_x

        # Position: bottom-left corner with margin
        margin = 20
        x1 = params["x"] + margin
        y1 = params["y"] + params["height"] - margin
        x2 = x1 + display_pixels
        y2 = y1

        # Draw scale bar
        self.canvas.create_line(x1, y1, x2, y2, fill="white", width=3)
        self.canvas.create_line(x1, y1 - 5, x1, y1 + 5, fill="white", width=2)
        self.canvas.create_line(x2, y2 - 5, x2, y2 + 5, fill="white", width=2)

        # Draw label
        self.canvas.create_text(
            (x1 + x2) / 2, y1 - 15,
            text=label,
            fill="white",
            font=("Arial", 12, "bold")
        )

    # ------------------------------------------------------------------
    # Edge Detection
    # ------------------------------------------------------------------

    def _start_edge_detection(self):
        """Start automated edge detection in a background thread."""
        if not self.camera_connected or self.current_image is None:
            messagebox.showwarning("Warning", "Camera not connected")
            return

        self.is_scanning = True
        self.edge_detected = False
        self.edge_theta = None
        self.edge_anchor = None
        self.edge_points = []

        self.btn_detect_edge.config(state=tk.DISABLED)
        self.btn_stop_scan.config(state=tk.NORMAL)

        # Start edge detection in background thread
        threading.Thread(target=self._edge_detection_worker, daemon=True).start()

    def _stop_scanning(self):
        """Stop the edge detection scan."""
        self.is_scanning = False
        self._set_status("Scan stopped by user")
        self.btn_detect_edge.config(state=tk.NORMAL)
        self.btn_stop_scan.config(state=tk.DISABLED)

    def _edge_detection_worker(self):
        """Worker thread for edge detection."""
        error_msg = None
        theta = None
        anchor = None
        points = []

        try:
            self._set_status("Scanning for chip edge...")

            # Get current frame
            frame_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)

            # Perform edge detection (returns theta, anchor only)
            theta, anchor = detect_edge_robust(frame_bgr, scan_direction="horizontal")

            if theta is not None:
                # Edge found - get points separately
                theta2, anchor2, points = detect_straight_black_edge(frame_bgr, scan_direction="horizontal")
                self.edge_detected = True
                self.edge_theta = theta
                self.edge_anchor = anchor
                self.edge_points = points if points else []

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Edge detection error: {e}", exc_info=True)
        finally:
            self.is_scanning = False

            # Update UI in main thread - avoid lambda closure issues
            if error_msg:
                self.root.after(0, self._set_status, f"Edge detection error: {error_msg}")
            elif theta is not None:
                self.root.after(0, self._set_status,
                    f"Edge found! θ={theta:.1f}° at ({anchor[0]:.0f}, {anchor[1]:.0f})"
                )
                self.root.after(0, self.edge_result_label.config,
                    text=f"Edge: θ={theta:.1f}°\nPos: ({anchor[0]:.0f}, {anchor[1]:.0f})\nPoints: {len(self.edge_points)}"
                )
            else:
                self.root.after(0, self._set_status, "Edge not found")
                self.root.after(0, self.edge_result_label.config, text="Edge: NOT FOUND")

            self.root.after(0, lambda: self.btn_detect_edge.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_stop_scan.config(state=tk.DISABLED))

    # ------------------------------------------------------------------
    # Auto-Alignment
    # ------------------------------------------------------------------

    def _start_auto_align(self):
        """Start automatic sample alignment in a background thread."""
        if not self.camera_connected or self.current_image is None:
            messagebox.showwarning("Warning", "Camera not connected")
            return

        if not self._require_stage_connected():
            return

        self.is_scanning = True
        self.btn_align.config(state=tk.DISABLED)
        self.btn_stop_align.config(state=tk.NORMAL)
        self.btn_detect_edge.config(state=tk.DISABLED)

        # Start alignment in background thread
        threading.Thread(target=self._auto_align_worker, daemon=True).start()

    def _stop_alignment(self):
        """Stop the auto-alignment process."""
        self.is_scanning = False
        self._set_status("Alignment stopped by user")
        self.btn_align.config(state=tk.NORMAL)
        self.btn_stop_align.config(state=tk.DISABLED)
        self.btn_detect_edge.config(state=tk.NORMAL)

    def _auto_align_worker(self):
        """Worker thread for auto-alignment."""
        try:
            self._set_status("Starting auto-alignment...")

            # Create alignment controller
            controller = AutoAlignController(
                camera=self.camera,
                stage=self.stage,
                rotation_axis="r"
            )

            # Progress callback
            def progress_callback(iteration, angle, correction, converged):
                if converged:
                    msg = f"Iteration {iteration}: CONVERGED at {angle:.1f}°"
                else:
                    msg = f"Iteration {iteration}: angle={angle:.1f}°, correction={correction:.1f}°"
                self.root.after(0, lambda: self._set_status(msg))

            # Run alignment
            success, message = controller.align(
                max_iterations=10,
                tolerance_deg=2.0,
                rotation_step_deg=5.0,
                progress_callback=progress_callback
            )

            # Update UI with final result
            if success:
                self.root.after(0, lambda: self._set_status(f"Alignment successful: {message}"))
                self.root.after(0, lambda: messagebox.showinfo("Alignment Complete", message))
            else:
                self.root.after(0, lambda: self._set_status(f"Alignment failed: {message}"))
                self.root.after(0, lambda: messagebox.showwarning("Alignment Failed", message))

        except Exception as e:
            logger.error(f"Auto-alignment error: {e}", exc_info=True)
            self.root.after(0, lambda: self._set_status(f"Alignment error: {e}"))
        finally:
            self.is_scanning = False
            self.root.after(0, lambda: self.btn_align.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.btn_stop_align.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.btn_detect_edge.config(state=tk.NORMAL))

    # ------------------------------------------------------------------
    # Manual Area Scan
    # ------------------------------------------------------------------

    def _set_scan_start(self):
        """Set the current stage position as scan start."""
        if not self._require_stage_connected():
            return
        try:
            status = self.stage.get_status()
            pos = status["position"]
            x = pos.get("x", 0.0)
            y = pos.get("y", 0.0)
            self.scan_start = (x, y)
            self.scan_start_label.config(text=f"({x:.0f}, {y:.0f})")
            self._set_status(f"Scan start set: ({x:.0f}, {y:.0f})")
        except Exception as e:
            self._set_status(f"Error setting start: {e}")

    def _set_scan_end(self):
        """Set the current stage position as scan end."""
        if not self._require_stage_connected():
            return
        try:
            status = self.stage.get_status()
            pos = status["position"]
            x = pos.get("x", 0.0)
            y = pos.get("y", 0.0)
            self.scan_end = (x, y)
            self.scan_end_label.config(text=f"({x:.0f}, {y:.0f})")
            self._set_status(f"Scan end set: ({x:.0f}, {y:.0f})")
        except Exception as e:
            self._set_status(f"Error setting end: {e}")

    def _start_manual_scan(self):
        """Start the manual area scan."""
        if not self._require_stage_connected():
            return
        if not self.camera_connected:
            messagebox.showwarning("Warning", "Camera not connected")
            return

        if not hasattr(self, 'scan_start') or not hasattr(self, 'scan_end'):
            messagebox.showwarning("Warning", "Set both start and end positions first")
            return

        try:
            overlap = float(self.overlap_var.get())
        except ValueError:
            messagebox.showwarning("Invalid overlap", "Overlap must be a number")
            return

        # Create scanner
        self.scanner = ChipScanner(
            stage=self.stage,
            camera=self.camera
        )
        self.scanner.set_start_position(*self.scan_start)
        self.scanner.set_end_position(*self.scan_end)

        # Disable buttons during scan
        self.btn_set_start.config(state=tk.DISABLED)
        self.btn_set_end.config(state=tk.DISABLED)
        self.btn_start_scan.config(state=tk.DISABLED)
        self.btn_stop_manual_scan.config(state=tk.NORMAL)

        # Start scan
        success = self.scanner.start_scan(
            overlap_percent=overlap,
            progress_callback=self._scan_progress_callback
        )

        if not success:
            messagebox.showerror("Scan Failed", "Could not start scan")
            self.btn_set_start.config(state=tk.NORMAL)
            self.btn_set_end.config(state=tk.NORMAL)
            self.btn_start_scan.config(state=tk.NORMAL)
            self.btn_stop_manual_scan.config(state=tk.DISABLED)

    def _stop_manual_scan(self):
        """Stop the manual area scan."""
        if hasattr(self, 'scanner') and self.scanner:
            self.scanner.stop_scan()
        self._set_status("Manual scan stopped")

    def _scan_progress_callback(self, current, total, position):
        """Update scan progress in UI."""
        self.root.after(0, lambda: self.scan_progress_label.config(
            text=f"Position {current}/{total}: ({position[0]:.0f}, {position[1]:.0f})"
        ))
        self.root.after(0, lambda: self._set_status(
            f"Scanning: {current}/{total} positions"
        ))

        # Check if scan is complete
        if current >= total:
            self.root.after(0, self._on_scan_complete)

    def _on_scan_complete(self):
        """Handle scan completion."""
        self.btn_set_start.config(state=tk.NORMAL)
        self.btn_set_end.config(state=tk.NORMAL)
        self.btn_start_scan.config(state=tk.NORMAL)
        self.btn_stop_manual_scan.config(state=tk.DISABLED)

        if hasattr(self, 'scanner') and self.scanner:
            results = self.scanner.get_results()
            count = results['captured_count']
            self.scan_progress_label.config(text=f"Complete: {count} frames captured")
            self._set_status(f"Scan complete: {count} frames captured")

            # Ask to save frames
            if count > 0:
                if messagebox.askyesno("Save Frames", f"Save {count} captured frames?"):
                    from tkinter import filedialog
                    output_dir = filedialog.askdirectory(title="Select output directory")
                    if output_dir:
                        self.scanner.save_frames(output_dir)

    # ------------------------------------------------------------------
    # Zoom and Calibration
    # ------------------------------------------------------------------

    def _on_zoom_preset_changed(self, event):
        """Handle zoom preset selection."""
        preset = self.current_zoom_preset.get()
        zoom_map = {"1x": 1.0, "5x": 5.0, "10x": 10.0, "20x": 20.0, "50x": 50.0}
        self.zoom_level = zoom_map.get(preset, 1.0)
        self._update_display()

    def _apply_default_calibration(self):
        """Apply default calibration (1 mm ≈ 2650 pixels)."""
        # Apply default calibration to all zoom levels
        base_calibration = DEFAULT_PIXELS_PER_NM  # pixels per nm at 1x
        for zoom in [1.0, 5.0, 10.0, 20.0, 50.0]:
            self.calibration[zoom] = base_calibration * zoom
        logger.info(f"Applied default calibration: {DEFAULT_PIXELS_PER_MM} pixels/mm")

    def _open_calibration_dialog(self):
        """Open the calibration dialog."""
        CalibrationDialog(self.root, self)

    def start_calibration_mode(self):
        """Enter calibration mode."""
        self.calibration_mode = True
        self.calibration_points = []
        self.calibration_line = None
        self._set_status("Calibration mode: Click two points to draw a reference line")

    def exit_calibration_mode(self):
        """Exit calibration mode."""
        self.calibration_mode = False
        self.calibration_points = []
        self.calibration_line = None
        self._set_status("Calibration mode exited")

    def add_calibration_point(self, x, y):
        """Add a calibration point."""
        if not self.calibration_mode:
            return

        self.calibration_points.append((x, y))

        if len(self.calibration_points) == 2:
            # Two points selected - show dialog to enter length
            self._show_calibration_length_dialog()

    def _show_calibration_length_dialog(self):
        """Show dialog to enter the real-world length."""
        if len(self.calibration_points) != 2:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Reference Length")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Enter length in nanometers:").pack(pady=10)

        length_var = tk.StringVar()
        ttk.Entry(dialog, textvariable=length_var, width=20).pack(pady=5)

        def save_calibration():
            try:
                length_nm = float(length_var.get())
                if length_nm <= 0:
                    raise ValueError("Length must be positive")

                # Calculate pixels per nm for current zoom
                p1, p2 = self.calibration_points
                pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                pixels_per_nm = pixel_distance / length_nm

                # Store calibration for current zoom level
                self.calibration[self.zoom_level] = pixels_per_nm

                # Calculate calibrations for other zoom levels
                # pixels_per_nm scales linearly with zoom
                base_calibration = pixels_per_nm / self.zoom_level
                for zoom in [1.0, 5.0, 10.0, 20.0, 50.0]:
                    self.calibration[zoom] = base_calibration * zoom

                logger.info(f"Calibration saved: {pixels_per_nm:.2f} pixels/nm at {self.zoom_level}x zoom")
                self._set_status(f"Calibration saved: {pixels_per_nm:.2f} px/nm")

                dialog.destroy()
                self.exit_calibration_mode()

            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please enter a valid number: {e}")

        ttk.Button(dialog, text="Save", command=save_calibration).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()

    def get_scale_bar_info(self) -> Optional[Tuple[float, str]]:
        """
        Get scale bar information for current zoom.

        Returns
        -------
        (pixels_per_nm, label) or None
            Pixels per nanometer and label text for scale bar
        """
        if self.zoom_level not in self.calibration:
            return None

        pixels_per_nm = self.calibration[self.zoom_level]

        # Choose a nice round number for the scale bar
        target_nm_values = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        target_pixels = 100  # We want ~100 pixel scale bar

        best_nm = None
        for nm in target_nm_values:
            pixels = nm * pixels_per_nm
            if 50 <= pixels <= 200:
                best_nm = nm
                break

        if best_nm is None:
            # Fallback to closest
            best_nm = target_nm_values[0]
            best_pixels = best_nm * pixels_per_nm
            for nm in target_nm_values:
                pixels = nm * pixels_per_nm
                if abs(pixels - target_pixels) < abs(best_pixels - target_pixels):
                    best_nm = nm
                    best_pixels = pixels

        label = f"{best_nm} nm"
        return pixels_per_nm, label

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        """Handle canvas click."""
        if self.calibration_mode:
            # Convert canvas coordinates to image coordinates
            if hasattr(self, 'display_params'):
                params = self.display_params
                img_x = int((event.x - params["x"]) / (params["width"] / params["orig_width"]))
                img_y = int((event.y - params["y"]) / (params["height"] / params["orig_height"]))
                self.add_calibration_point(img_x, img_y)

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom."""
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1

        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
        self._update_display()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self._update_display()

    def _on_pan_drag(self, event):
        """Handle middle mouse drag for panning."""
        if not hasattr(self, '_pan_start_x'):
            self._pan_start_x = event.x
            self._pan_start_y = event.y
            self._pan_start_pan_x = self.pan_x
            self._pan_start_pan_y = self.pan_y
        else:
            dx = event.x - self._pan_start_x
            dy = event.y - self._pan_start_y
            self.pan_x = self._pan_start_pan_x + dx
            self.pan_y = self._pan_start_pan_y + dy
            self._update_display()

    def _on_pan_release(self, event):
        """Handle middle mouse button release."""
        if hasattr(self, '_pan_start_x'):
            delattr(self, '_pan_start_x')
            delattr(self, '_pan_start_y')
            delattr(self, '_pan_start_pan_x')
            delattr(self, '_pan_start_pan_y')

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        """Set status bar text."""
        self.status_bar.config(text=f"Status: {text}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Cleanup resources."""
        self._disconnect_camera()
        self._disconnect_stage()
        self.root.destroy()


class CalibrationDialog:
    """Dialog for managing calibration."""

    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Calibration Manager")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._build_gui()

    def _build_gui(self):
        """Build the calibration dialog GUI."""
        pad = {"padx": 10, "pady": 10}

        # Instructions
        ttk.Label(
            self.dialog,
            text="Calibration allows you to convert pixels to nanometers.\n"
                 "Click 'Start Calibration', then click two points on the image\n"
                 "to define a reference line of known length.",
            justify=tk.LEFT
        ).pack(pady=10, padx=10)

        # Current calibrations
        cal_frame = ttk.LabelFrame(self.dialog, text="Current Calibrations")
        cal_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.cal_text = tk.Text(cal_frame, height=8, width=50, font=("Consolas", 9))
        self.cal_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._update_calibration_display()

        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            btn_frame,
            text="Start Calibration",
            command=self._start_calibration
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        ttk.Button(
            btn_frame,
            text="Clear All",
            command=self._clear_calibrations
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        ttk.Button(
            btn_frame,
            text="Close",
            command=self.dialog.destroy
        ).pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

    def _update_calibration_display(self):
        """Update the calibration display."""
        self.cal_text.config(state=tk.NORMAL)
        self.cal_text.delete("1.0", tk.END)

        if not self.app.calibration:
            self.cal_text.insert(tk.END, "No calibrations saved yet.\n")
        else:
            self.cal_text.insert(tk.END, "Zoom Level | Pixels/nm\n")
            self.cal_text.insert(tk.END, "-" * 40 + "\n")
            for zoom in sorted(self.app.calibration.keys()):
                px_per_nm = self.app.calibration[zoom]
                self.cal_text.insert(tk.END, f"{zoom:6.1f}x    | {px_per_nm:10.2f}\n")

        self.cal_text.config(state=tk.DISABLED)

    def _start_calibration(self):
        """Start calibration mode."""
        self.dialog.destroy()
        self.app.start_calibration_mode()

    def _clear_calibrations(self):
        """Clear all calibrations."""
        self.app.calibration.clear()
        self._update_calibration_display()
        self.app._set_status("Calibrations cleared")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chip Edge Detector")
    parser.add_argument("--port", default="COM3", help="Stage serial port (default COM3)")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatedStage")
    args = parser.parse_args()

    root = tk.Tk()
    app = ChipEdgeDetector(root, port=args.port, simulate=args.simulate)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()


if __name__ == "__main__":
    main()
