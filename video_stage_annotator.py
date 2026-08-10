#!/usr/bin/env python3
"""
Video Stage Annotator - Integrated camera annotation + Zolix stage control.

Features:
- Live video feed from SmartCam camera (only SmartCam backend)
- Click to place points with HSV recording
- Zolix ZC300 stage control interface (X/Y/R axes)
- Video playback controls
- Save/load points with HSV to JSON

Run:
    python video_stage_annotator.py                    # GUI
    python video_stage_annotator.py --port COM3        # specify stage port
    python video_stage_annotator.py --simulate         # use SimulatedStage
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

# Import HSV pipeline helpers
sys.path.insert(0, str(Path(__file__).parent / "hsv-pipeline-semi"))
from semi_supervised_pipeline import extract_hsv_patch, substrate_stats_from_pixels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SUBSTRATE_PATCH_RADIUS = 7
AXES = ("x", "y", "r")
AXIS_NAMES = {"x": "X", "y": "Y", "r": "R (Z)"}


class Point:
    """Represents a clicked point on the image with HSV values."""

    def __init__(self, x: int, y: int, hsv: Optional[Tuple[int, int, int]] = None,
                 label: str = ""):
        self.x = x
        self.y = y
        self.hsv = hsv  # (H, S, V) tuple or None
        self.label = label or "Point"

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "hsv": list(self.hsv) if self.hsv else None,
            "label": self.label
        }

    @classmethod
    def from_dict(cls, data):
        hsv = tuple(data["hsv"]) if data.get("hsv") else None
        return cls(data["x"], data["y"], hsv, data.get("label", ""))


def compute_calibration_from_samples(
    pixel_arrays: List[np.ndarray],
    min_pixels: int = 50,
) -> Optional[dict]:
    """Aggregate cached per-click HSV patch pixels into a substrate calibration."""
    arrays = [np.asarray(a, dtype=np.float32).reshape(-1, 3) for a in pixel_arrays]
    arrays = [a for a in arrays if a.shape[0] > 0]
    if not arrays:
        return None

    all_px = np.concatenate(arrays, axis=0)
    peak, std = substrate_stats_from_pixels(all_px)
    return {
        "peak": [float(c) for c in peak],
        "std": [float(c) for c in std] if all_px.shape[0] >= min_pixels else None,
        "n_pixels": int(all_px.shape[0]),
        "n_samples": len(arrays),
    }


def build_points_payload(
    points: List[Point],
    substrate_calibration: Optional[dict],
    image_size: Optional[Tuple[int, int]],
    fallback_substrate_hsv: Optional[Tuple[int, int, int]] = None,
) -> dict:
    """Build the points-JSON payload."""
    if substrate_calibration:
        legacy = [int(round(c)) for c in substrate_calibration["peak"]]
    elif fallback_substrate_hsv:
        legacy = [int(round(c)) for c in fallback_substrate_hsv]
    else:
        legacy = None

    return {
        "points": [p.to_dict() for p in points],
        "substrate_hsv": legacy,
        "substrate_calibration": substrate_calibration,
        "image_size": image_size,
    }


def parse_points_payload(data: dict) -> Tuple[List[Point], Optional[dict]]:
    """Parse a points-JSON payload."""
    points = [Point.from_dict(p) for p in data.get("points", [])]
    return points, data.get("substrate_calibration")


class VideoStageAnnotator:
    """
    Integrated video annotation + Zolix stage control.

    Left panel: Video feed with point annotation
    Right panel: Stage controls + point management
    """

    def __init__(self, root: tk.Tk, port: str, simulate: bool):
        self.root = root
        self.root.title("Video Stage Annotator - SmartCam + Zolix ZC300")
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

        # Points
        self.points: List[Point] = []

        # Substrate HSV
        self.substrate_hsv: Optional[Tuple[int, int, int]] = None
        self.calibrate_mode = False
        self.substrate_calibration: Optional[dict] = None
        self.substrate_source = "none"
        self._substrate_pixels: Dict[int, np.ndarray] = {}

        # Display
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.current_image = None
        self.display_image = None

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
        self.canvas.bind("<Shift-Button-1>", self._on_substrate_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_mouse_move)
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

        # Camera section
        cam_frame = ttk.LabelFrame(parent, text="Camera (SmartCam)")
        cam_frame.pack(fill=tk.X, **pad)

        self.btn_cam_connect = ttk.Button(
            cam_frame, text="Connect Camera", command=self._toggle_camera_connection
        )
        self.btn_cam_connect.pack(pady=5, padx=5, fill=tk.X)

        self.cam_conn_label = ttk.Label(cam_frame, text="Disconnected", foreground="gray")
        self.cam_conn_label.pack(pady=2)

        # Substrate HSV section
        substrate_frame = ttk.LabelFrame(parent, text="Substrate HSV")
        substrate_frame.pack(fill=tk.X, **pad)

        self.btn_detect_substrate = ttk.Button(
            substrate_frame, text="Auto-Detect Substrate HSV",
            command=self._detect_substrate_hsv
        )
        self.btn_detect_substrate.pack(pady=5, padx=5, fill=tk.X)

        self.btn_calibrate_substrate = ttk.Button(
            substrate_frame, text="Calibrate Substrate (click mode)",
            command=self._toggle_calibrate_mode
        )
        self.btn_calibrate_substrate.pack(pady=2, padx=5, fill=tk.X)

        self.substrate_hsv_label = ttk.Label(substrate_frame, text="Not detected")
        self.substrate_hsv_label.pack(pady=2)

        self.substrate_count_label = ttk.Label(substrate_frame, text="")
        self.substrate_count_label.pack(pady=(0, 5))

        # Points section
        points_frame = ttk.LabelFrame(parent, text="Clicked Points (with HSV)")
        points_frame.pack(fill=tk.BOTH, expand=True, **pad)

        # Points list
        self.points_listbox = tk.Listbox(points_frame, height=10)
        self.points_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Points buttons
        points_btn_frame = ttk.Frame(points_frame)
        points_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(points_btn_frame, text="Save Points", command=self._save_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(points_btn_frame, text="Load Points", command=self._load_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(points_btn_frame, text="Clear All", command=self._clear_points).pack(side=tk.LEFT, padx=2)

        # Instructions
        instructions_frame = ttk.LabelFrame(parent, text="Instructions")
        instructions_frame.pack(fill=tk.X, **pad)

        instructions = (
            "Left-click: Add point (records HSV)\n"
            "Shift-click: Add substrate sample\n"
            "Right-click: Remove last point\n"
            "Mouse wheel: Zoom in/out\n"
            "Calibrate or Auto-Detect substrate first!"
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
        display_img = cv2.resize(img, (display_width, display_height), interpolation=interp)

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

        # Draw points
        self._draw_points()

        # Update status
        self.status_bar.config(
            text=f"Camera | {img_w}x{img_h} | {self.fps:.1f} fps | "
                 f"Stage: {'Connected' if self.stage_connected else 'Disconnected'} | "
                 f"Points: {len(self.points)}"
        )

    def _draw_points(self):
        """Draw click markers on canvas."""
        if not hasattr(self, 'display_params') or not self.points:
            return

        params = self.display_params
        scale_x = params["width"] / params["orig_width"]
        scale_y = params["height"] / params["orig_height"]

        flake_idx = 0
        substrate_idx = 0
        for point in self.points:
            disp_x = int(point.x * scale_x) + params["x"]
            disp_y = int(point.y * scale_y) + params["y"]

            if point.label == "substrate":
                substrate_idx += 1
                fill, text, text_fill = "cyan", f"S{substrate_idx}", "cyan"
            else:
                flake_idx += 1
                fill, text, text_fill = "red", f"#{flake_idx}", "yellow"

            self.canvas.create_oval(
                disp_x - 5, disp_y - 5, disp_x + 5, disp_y + 5,
                fill=fill, outline="white", width=2
            )
            self.canvas.create_text(
                disp_x + 10, disp_y - 10,
                text=text, fill=text_fill, font=("Arial", 10, "bold")
            )

    # ------------------------------------------------------------------
    # Substrate HSV Detection
    # ------------------------------------------------------------------

    def _detect_substrate_hsv(self):
        """Auto-detect substrate HSV from current image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image captured yet")
            return

        try:
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)
            H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

            h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
            s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
            v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

            h_sub = int(np.argmax(h_hist))
            s_sub = int(np.argmax(s_hist))
            v_sub = int(np.argmax(v_hist))

            self.substrate_hsv = (h_sub, s_sub, v_sub)
            self.substrate_source = "auto"
            self._update_substrate_labels()

            logger.info(f"Substrate HSV detected: H={h_sub}, S={s_sub}, V={v_sub}")

        except Exception as e:
            logger.error(f"Substrate HSV detection failed: {e}")
            messagebox.showerror("Error", f"Failed to detect substrate HSV: {e}")

    def _toggle_calibrate_mode(self):
        """Toggle click-to-calibrate substrate mode."""
        self.calibrate_mode = not self.calibrate_mode
        if self.calibrate_mode:
            self.btn_calibrate_substrate.config(
                text="Stop Calibrating (clicks = substrate)"
            )
            self._set_status("Calibrate mode — click bare substrate regions")
        else:
            self.btn_calibrate_substrate.config(
                text="Calibrate Substrate (click mode)"
            )
            self._set_status("Calibrate mode off")
        logger.info(f"Calibrate mode: {'on' if self.calibrate_mode else 'off'}")

    def _on_substrate_click(self, event):
        """Handle substrate-calibration click."""
        if self.current_image is None:
            return

        orig_x, orig_y = self._display_to_original(event.x, event.y)

        if not (0 <= orig_x < self.current_image.shape[1]
                and 0 <= orig_y < self.current_image.shape[0]):
            return

        self._add_substrate_sample(orig_x, orig_y)

    def _add_substrate_sample(self, orig_x: int, orig_y: int):
        """Sample a substrate patch."""
        pixels = extract_hsv_patch(
            self.current_image, orig_x, orig_y, radius=SUBSTRATE_PATCH_RADIUS
        )
        median_hsv = tuple(int(v) for v in np.median(pixels, axis=0))

        point = Point(orig_x, orig_y, hsv=median_hsv, label="substrate")
        self.points.append(point)
        self._substrate_pixels[id(point)] = pixels

        self._recompute_substrate_calibration()
        self._update_points_list()
        self._update_display()

        logger.info(f"Added substrate sample: ({orig_x}, {orig_y}) HSV: {median_hsv}")

    def _recompute_substrate_calibration(self):
        """Rebuild substrate calibration from all samples."""
        arrays = []
        for point in self.points:
            if point.label != "substrate":
                continue
            cached = self._substrate_pixels.get(id(point))
            if cached is not None:
                arrays.append(cached)
            elif point.hsv:
                arrays.append(np.array([point.hsv], dtype=np.float32))

        self.substrate_calibration = compute_calibration_from_samples(arrays)

        if self.substrate_calibration:
            self.substrate_source = "clicks"
            peak = self.substrate_calibration["peak"]
            self.substrate_hsv = tuple(int(round(c)) for c in peak)
        elif self.substrate_source == "clicks":
            self.substrate_source = "none"
            self.substrate_hsv = None

        self._update_substrate_labels()

    def _update_substrate_labels(self):
        """Refresh substrate HSV labels."""
        if self.substrate_hsv:
            h, s, v = self.substrate_hsv
            self.substrate_hsv_label.config(text=f"H={h}, S={s}, V={v}")
        else:
            self.substrate_hsv_label.config(text="Not detected")

        if self.substrate_source == "clicks" and self.substrate_calibration:
            n = self.substrate_calibration["n_samples"]
            self.substrate_count_label.config(text=f"Calibrated ({n} clicks)")
        elif self.substrate_source == "auto":
            self.substrate_count_label.config(text="Auto-detected")
        else:
            self.substrate_count_label.config(text="")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        """Handle left-click to add point."""
        if self.calibrate_mode:
            return self._on_substrate_click(event)
        if self.current_image is None:
            return

        orig_x, orig_y = self._display_to_original(event.x, event.y)

        if not (0 <= orig_x < self.current_image.shape[1]
                and 0 <= orig_y < self.current_image.shape[0]):
            return

        pixel = self.current_image[orig_y:orig_y + 1, orig_x:orig_x + 1]
        hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
        h_val = int(hsv[0, 0, 0])
        s_val = int(hsv[0, 0, 1])
        v_val = int(hsv[0, 0, 2])
        hsv_value = (h_val, s_val, v_val)

        point = Point(orig_x, orig_y, hsv=hsv_value)
        self.points.append(point)

        self._update_points_list()
        self._update_display()
        logger.info(f"Added point: ({orig_x}, {orig_y}) HSV: {hsv_value}")

    def _on_right_click(self, event):
        """Handle right-click to remove last point."""
        if self.points:
            removed = self.points.pop()
            self._substrate_pixels.pop(id(removed), None)
            if removed.label == "substrate":
                self._recompute_substrate_calibration()
            self._update_points_list()
            self._update_display()
            logger.info(f"Removed point: ({removed.x}, {removed.y})")

    def _on_mouse_move(self, event):
        """Handle mouse movement for HSV cursor readout."""
        if self.current_image is None or not hasattr(self, 'display_params'):
            return

        orig_x, orig_y = self._display_to_original(event.x, event.y)

        if 0 <= orig_x < self.current_image.shape[1] and 0 <= orig_y < self.current_image.shape[0]:
            pixel = self.current_image[orig_y:orig_y + 1, orig_x:orig_x + 1]
            hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)
            h_val = int(hsv[0, 0, 0])
            s_val = int(hsv[0, 0, 1])
            v_val = int(hsv[0, 0, 2])

            self.status_bar.config(
                text=f"Cursor: ({orig_x}, {orig_y}) HSV: ({h_val}, {s_val}, {v_val}) | "
                     f"Points: {len(self.points)}"
            )

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zoom."""
        if event.delta > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1

        self.zoom_level = max(0.1, min(10.0, self.zoom_level))
        self._update_display()

    def _on_pan_drag(self, event):
        """Handle middle mouse drag for panning."""
        if not hasattr(self, '_pan_start_x'):
            # Initialize pan drag
            self._pan_start_x = event.x
            self._pan_start_y = event.y
            self._pan_start_pan_x = self.pan_x
            self._pan_start_pan_y = self.pan_y
        else:
            # Calculate delta and update pan
            dx = event.x - self._pan_start_x
            dy = event.y - self._pan_start_y
            self.pan_x = self._pan_start_pan_x + dx
            self.pan_y = self._pan_start_pan_y + dy
            self._update_display()

    def _on_pan_release(self, event):
        """Handle middle mouse button release."""
        # Clean up pan drag state
        if hasattr(self, '_pan_start_x'):
            delattr(self, '_pan_start_x')
            delattr(self, '_pan_start_y')
            delattr(self, '_pan_start_pan_x')
            delattr(self, '_pan_start_pan_y')

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self._update_display()

    def _display_to_original(self, disp_x: int, disp_y: int) -> Tuple[int, int]:
        """Convert display coordinates to original image coordinates."""
        if not hasattr(self, 'display_params'):
            return (disp_x, disp_y)

        params = self.display_params
        scale_x = params["orig_width"] / params["width"]
        scale_y = params["orig_height"] / params["height"]

        orig_x = int((disp_x - params["x"]) * scale_x)
        orig_y = int((disp_y - params["y"]) * scale_y)

        return (orig_x, orig_y)

    # ------------------------------------------------------------------
    # Points Management
    # ------------------------------------------------------------------

    def _update_points_list(self):
        """Update the points listbox."""
        self.points_listbox.delete(0, tk.END)
        flake_idx = 0
        substrate_idx = 0
        for point in self.points:
            if point.label == "substrate":
                substrate_idx += 1
                prefix, suffix = f"S{substrate_idx}", "  [substrate]"
            else:
                flake_idx += 1
                prefix, suffix = f"#{flake_idx}", ""

            if point.hsv:
                h, s, v = point.hsv
                self.points_listbox.insert(
                    tk.END,
                    f"{prefix}: ({point.x}, {point.y})  HSV: ({h}, {s}, {v}){suffix}"
                )
            else:
                self.points_listbox.insert(
                    tk.END,
                    f"{prefix}: ({point.x}, {point.y}){suffix}"
                )

    def _save_points(self):
        """Save points to JSON."""
        if not self.points:
            messagebox.showwarning("Warning", "No points to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                data = build_points_payload(
                    self.points,
                    self.substrate_calibration,
                    image_size=(
                        self.current_image.shape[1],
                        self.current_image.shape[0]
                    ) if self.current_image is not None else None,
                    fallback_substrate_hsv=self.substrate_hsv,
                )

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                messagebox.showinfo("Success", f"Saved {len(self.points)} points to {file_path}")
                logger.info(f"Saved points to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save points: {e}")

    def _load_points(self):
        """Load points from JSON."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                self.points, self.substrate_calibration = parse_points_payload(data)
                self._substrate_pixels.clear()

                if self.substrate_calibration:
                    self.substrate_source = "clicks"
                    peak = self.substrate_calibration["peak"]
                    self.substrate_hsv = tuple(int(round(c)) for c in peak)
                else:
                    substrate = data.get("substrate_hsv")
                    if substrate:
                        self.substrate_hsv = tuple(substrate)
                        self.substrate_source = "auto"
                self._update_substrate_labels()

                self._update_points_list()
                self._update_display()

                messagebox.showinfo("Success", f"Loaded {len(self.points)} points")
                logger.info(f"Loaded points from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load points: {e}")

    def _clear_points(self):
        """Clear all points."""
        if self.points:
            if messagebox.askyesno("Confirm", "Clear all points?"):
                self.points.clear()
                self._substrate_pixels.clear()
                self._recompute_substrate_calibration()
                self._update_points_list()
                self._update_display()
                logger.info("Cleared all points")

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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Video Stage Annotator")
    parser.add_argument("--port", default="COM3", help="Stage serial port (default COM3)")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatedStage")
    args = parser.parse_args()

    root = tk.Tk()
    app = VideoStageAnnotator(root, port=args.port, simulate=args.simulate)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()


if __name__ == "__main__":
    main()