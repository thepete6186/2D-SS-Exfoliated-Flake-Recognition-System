#!/usr/bin/env python3
"""
Camera Annotator - Live camera view with click annotation and HSV recording.

Standalone tkinter app for:
- Live camera feed from Zeiss XiCam 208 (USB camera via OpenCV)
- Click to mark points (records pixel coordinates + HSV values)
- Auto-detect substrate HSV
- Save/load points with HSV to JSON
- Adjust camera settings (exposure, gain)

Performance:
  - Camera capture runs on a background thread (never blocks the GUI).
  - Frames are downscaled with cv2 (fast) before display.
  - The window opens maximized.
"""

import sys
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from camera.zeiss_camera import ZeissCamera, CameraError
from camera.mmcore_camera import MMCoreCamera
from camera.smartcam_camera import SmartCamCamera

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class CameraAnnotator:
    """
    Live camera viewer with point annotation and HSV recording.

    Features:
    - Live camera feed (background capture thread)
    - Click to add points (records HSV at click location)
    - Right-click to remove last point
    - Auto-detect substrate HSV
    - Save/load points with HSV to JSON
    - Exposure/gain controls
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Camera Annotator - XiCam 208")
        self.root.geometry("1200x800")
        self.root.state("zoomed")  # maximize on Windows

        # Camera
        self.camera = None
        self.camera_connected = False
        self.camera_backend = "auto"  # "auto", "gentl", "opencv", "mmcore", "smartcam"
        self.mm_config_path = None  # Path to Micro-Manager config

        # Background capture thread state
        self._capture_thread = None
        self._capture_stop = threading.Event()
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._capture_fps = 0.0

        # Points
        self.points: List[Point] = []

        # Substrate HSV (auto-detected)
        self.substrate_hsv: Optional[Tuple[int, int, int]] = None

        # Display settings
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.current_image = None  # Current frame (numpy array)
        self.display_image = None  # Resized for display (PIL Image)

        # FPS tracking
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
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left side: image display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side: controls
        right_panel = ttk.Frame(main_container, width=320)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # Image canvas
        self.canvas = tk.Canvas(left_panel, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Canvas events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        # Right panel contents
        self._build_right_panel(right_panel)

        # Status bar
        self.status_bar = ttk.Label(self.root, text="Status: Disconnected", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_right_panel(self, parent):
        """Build the right control panel."""
        # Title
        title = ttk.Label(parent, text="Camera Annotator", font=("Arial", 14, "bold"))
        title.pack(pady=10)

        # Connection section
        conn_frame = ttk.LabelFrame(parent, text="Camera")
        conn_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_connect = ttk.Button(conn_frame, text="Connect Camera", command=self._toggle_connection)
        self.btn_connect.pack(pady=5, padx=5, fill=tk.X)

        # Backend selection
        backend_frame = ttk.Frame(conn_frame)
        backend_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(backend_frame, text="Backend:").pack(side=tk.LEFT)
        self.backend_var = tk.StringVar(value="auto")
        backend_combo = ttk.Combobox(
            backend_frame, textvariable=self.backend_var, state="readonly",
            values=("auto", "gentl", "opencv", "micro-manager", "smartcam"),
            width=18
        )
        backend_combo.pack(side=tk.LEFT, padx=5)
        backend_combo.bind("<<ComboboxSelected>>", self._on_backend_change)

        self.btn_load_image = ttk.Button(conn_frame, text="Load Image from File", command=self._load_image_file)
        self.btn_load_image.pack(pady=5, padx=5, fill=tk.X)

        # Watch folder mode for Labscope integration
        self.watch_folder = None
        self.watch_interval = 1000  # Check every 1 second
        self.btn_watch_folder = ttk.Button(conn_frame, text="Watch Labscope Folder", command=self._toggle_watch_folder)
        self.btn_watch_folder.pack(pady=5, padx=5, fill=tk.X)

        # Substrate HSV section
        substrate_frame = ttk.LabelFrame(parent, text="Substrate HSV")
        substrate_frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_detect_substrate = ttk.Button(
            substrate_frame, text="Auto-Detect Substrate HSV",
            command=self._detect_substrate_hsv
        )
        self.btn_detect_substrate.pack(pady=5, padx=5, fill=tk.X)

        self.substrate_hsv_label = ttk.Label(substrate_frame, text="Not detected")
        self.substrate_hsv_label.pack(pady=5)

        # Camera settings
        settings_frame = ttk.LabelFrame(parent, text="Camera Settings")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)

        # Exposure
        ttk.Label(settings_frame, text="Exposure (us):").pack(anchor=tk.W, padx=5)
        self.exposure_var = tk.DoubleVar(value=10000.0)
        exposure_slider = ttk.Scale(
            settings_frame, from_=100, to=100000,
            variable=self.exposure_var, orient=tk.HORIZONTAL,
            command=self._on_exposure_change
        )
        exposure_slider.pack(padx=5, pady=2, fill=tk.X)
        self.exposure_label = ttk.Label(settings_frame, text="10000 us")
        self.exposure_label.pack(anchor=tk.E, padx=5)

        # Gain
        ttk.Label(settings_frame, text="Gain (dB):").pack(anchor=tk.W, padx=5)
        self.gain_var = tk.DoubleVar(value=0.0)
        gain_slider = ttk.Scale(
            settings_frame, from_=0, to=24,
            variable=self.gain_var, orient=tk.HORIZONTAL,
            command=self._on_gain_change
        )
        gain_slider.pack(padx=5, pady=2, fill=tk.X)
        self.gain_label = ttk.Label(settings_frame, text="0.0 dB")
        self.gain_label.pack(anchor=tk.E, padx=5)

        # Points section
        points_frame = ttk.LabelFrame(parent, text="Clicked Points (with HSV)")
        points_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Points list
        self.points_listbox = tk.Listbox(points_frame, height=15)
        self.points_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Points buttons
        points_btn_frame = ttk.Frame(points_frame)
        points_btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(points_btn_frame, text="Save Points", command=self._save_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(points_btn_frame, text="Load Points", command=self._load_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(points_btn_frame, text="Clear All", command=self._clear_points).pack(side=tk.LEFT, padx=2)

        # Instructions
        instructions_frame = ttk.LabelFrame(parent, text="Instructions")
        instructions_frame.pack(fill=tk.X, padx=5, pady=5)

        instructions = (
            "Left-click: Add point (records HSV)\n"
            "Right-click: Remove last point\n"
            "Mouse wheel: Zoom in/out\n"
            "Middle mouse drag: Pan\n"
            "Auto-Detect Substrate HSV first!"
        )
        ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT).pack(padx=5, pady=5, anchor=tk.W)

    # ------------------------------------------------------------------
    # Camera Connection
    # ------------------------------------------------------------------

    def _toggle_connection(self):
        """Connect or disconnect camera."""
        if self.camera_connected:
            self._disconnect_camera()
        else:
            self._connect_camera()

    def _on_backend_change(self, event=None):
        """Handle backend selection change."""
        backend = self.backend_var.get()
        if backend == "auto":
            self.camera_backend = "auto"
        elif backend == "gentl":
            self.camera_backend = "gentl"
        elif backend == "opencv":
            self.camera_backend = "opencv"
        elif backend == "micro-manager":
            self.camera_backend = "mmcore"
        elif backend == "smartcam":
            self.camera_backend = "smartcam"

        # If camera is currently connected, reconnect with new backend
        if self.camera_connected:
            self._disconnect_camera()
            self.status_bar.config(
                text=f"Backend set to {self.camera_backend}. Camera disconnected."
            )

    def _connect_camera(self):
        """Connect to camera."""
        backend = self.camera_backend
        try:
            self.camera = None

            # Try GenTL first (correct backend for Axiocam 208)
            if backend in ("auto", "gentl"):
                for idx in range(3):
                    cam = ZeissCamera(camera_index=idx, backend="gentl")
                    if cam.connect():
                        self.camera = cam
                        self.camera_connected = True
                        self.btn_connect.config(text="Disconnect")
                        info = self.camera.get_camera_info()
                        self.status_bar.config(
                            text=f"Connected: {info.get('name', 'Unknown')} "
                                 f"({info.get('backend', '')})"
                        )
                        logger.info("Camera connected via GenTL")
                        self._start_capture_thread()
                        return
                    if backend == "gentl":
                        break

            # Try SmartCamApi third (native Zeiss SDK - works without GenTL)
            if backend in ("auto", "smartcam"):
                cam = SmartCamCamera(camera_index=0)
                if cam.connect():
                    self.camera = cam
                    self.camera_connected = True
                    self.btn_connect.config(text="Disconnect")
                    info = self.camera.get_camera_info()
                    self.status_bar.config(
                        text=f"Connected: {info.get('name', 'Unknown')} "
                             f"(SmartCamApi: {info.get('backend', '')})"
                    )
                    logger.info("Camera connected via SmartCamApi")
                    self._start_capture_thread()
                    return
                elif backend == "smartcam":
                    messagebox.showerror(
                        "SmartCamApi Connection Failed",
                        "Could not connect to camera via SmartCamApi.\n\n"
                        "Troubleshooting:\n"
                        "  1. Ensure SmartCamApi.dll is installed (comes with Labscope)\n"
                        "  2. Ensure libusb0 driver is installed for the camera\n"
                        "  3. Make sure camera is not held by another app\n"
                        "  4. Try unplugging and replugging the camera"
                    )
                    return

            # Try Micro-Manager/pymmcore fourth
            if backend in ("auto", "mmcore"):
                cam = MMCoreCamera(camera_index=0, mm_config_path=self.mm_config_path)
                if cam.connect():
                    self.camera = cam
                    self.camera_connected = True
                    self.btn_connect.config(text="Disconnect")
                    info = self.camera.get_camera_info()
                    self.status_bar.config(
                        text=f"Connected: {info.get('name', 'Unknown')} "
                             f"(Micro-Manager: {info.get('backend', '')})"
                    )
                    logger.info("Camera connected via Micro-Manager")
                    self._start_capture_thread()
                    return
                elif backend == "mmcore":
                    messagebox.showerror(
                        "Micro-Manager Connection Failed",
                        "Could not connect to camera via Micro-Manager.\n\n"
                        "Troubleshooting:\n"
                        "  1. pip install pymmcore pymmcore-plus\n"
                        "  2. Install Micro-Manager 2.0 (https://micro-manager.org)\n"
                        "  3. Ensure Micro-Manager has Zeiss Axiocam adapter\n"
                        "  4. Set MICRO_MANAGER_PATH environment variable:\n"
                        "     setx MICRO_MANAGER_PATH \"C:\\Program Files\\Micro-Manager-2.0\"\n"
                        "  5. Make sure camera is plugged into USB 3 and not used\n"
                        "     by another app (e.g. Zeiss ZEN, Labscope)"
                    )
                    return

            # Try OpenCV last (won't work for Axiocam 208 but keep for compatibility)
            if backend in ("auto", "opencv"):
                for idx in range(3):
                    cam = ZeissCamera(camera_index=idx, backend="opencv")
                    if cam.connect():
                        self.camera = cam
                        self.camera_connected = True
                        self.btn_connect.config(text="Disconnect")
                        info = self.camera.get_camera_info()
                        self.status_bar.config(
                            text=f"Connected: {info.get('name', 'Unknown')} "
                                 f"({info.get('backend', '')})"
                        )
                        logger.info("Camera connected via OpenCV")
                        self._start_capture_thread()
                        return

            # If we got here, connection failed
            messagebox.showerror(
                "Camera Connection Failed",
                "Could not connect to the Zeiss Axiocam 208.\n\n"
                "REQUIRED: Physical USB reconnection\n"
                "The camera driver has a stale handle that can only be fixed by:\n"
                "  1. Unplug the camera USB cable\n"
                "  2. Wait 2 seconds\n"
                "  3. Plug it back in\n"
                "  4. Wait 5 seconds for Windows to detect it\n"
                "  5. Click 'Connect Camera' again\n\n"
                "Note: Software reset cannot fix this - physical reconnection is required.\n\n"
                "For offline work, use 'Load Image from File' instead."
            )
            self.camera = None
            self.camera_connected = False
        except Exception as e:
            messagebox.showerror("Error", f"Camera connection failed: {e}")
            self.camera = None
            self.camera_connected = False

    def _disconnect_camera(self):
        """Disconnect camera."""
        # Stop capture thread
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
        self.btn_connect.config(text="Connect Camera")
        self.status_bar.config(text="Status: Disconnected")
        logger.info("Camera disconnected")

    def _load_image_file(self):
        """Load a still image from disk for offline annotation."""
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return

        try:
            # Disconnect live camera if connected
            if self.camera_connected:
                self._disconnect_camera()

            # Read with OpenCV (BGR) and convert to RGB
            bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if bgr is None:
                messagebox.showerror("Error", f"Could not read image: {file_path}")
                return

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.current_image = rgb
            self.fps = 0.0

            # Update display once
            self._update_display()
            self.status_bar.config(
                text=f"Image: {Path(file_path).name} | "
                     f"{rgb.shape[1]}x{rgb.shape[0]} | Points: {len(self.points)}"
            )
            logger.info(f"Loaded image from {file_path} ({rgb.shape[1]}x{rgb.shape[0]})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def _toggle_watch_folder(self):
        """Toggle folder watch mode for Labscope integration."""
        if self.watch_folder:
            # Stop watching
            self.watch_folder = None
            self.btn_watch_folder.config(text="Watch Labscope Folder")
            self.status_bar.config(text="Folder watch stopped")
            return

        # Select folder
        folder = filedialog.askdirectory(title="Select Labscope Save Folder")
        if not folder:
            return

        self.watch_folder = folder
        self.btn_watch_folder.config(text="Stop Watching Folder")
        self.status_bar.config(text=f"Watching: {folder}")
        logger.info(f"Watching folder: {folder}")

        # Start watching
        self._watch_folder()

    def _watch_folder(self):
        """Check folder for new images and load them."""
        if not self.watch_folder:
            return

        try:
            # Look for image files
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            files = []
            for ext in extensions:
                files.extend(Path(self.watch_folder).glob(ext))

            # Sort by modification time, get most recent
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)

                # Check if it's different from current image
                if not hasattr(self, '_last_watched_file') or self._last_watched_file != str(latest):
                    self._last_watched_file = str(latest)

                    # Load the image
                    bgr = cv2.imread(str(latest), cv2.IMREAD_COLOR)
                    if bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        self.current_image = rgb
                        self._update_display()
                        self.status_bar.config(
                            text=f"Labscope: {latest.name} | "
                                 f"{rgb.shape[1]}x{rgb.shape[0]} | Points: {len(self.points)}"
                        )
                        logger.info(f"Loaded from Labscope: {latest.name}")
        except Exception as e:
            logger.error(f"Folder watch error: {e}")

        # Schedule next check
        self.root.after(self.watch_interval, self._watch_folder)

    # ------------------------------------------------------------------
    # Background Capture Thread
    # ------------------------------------------------------------------

    def _start_capture_thread(self):
        """Start the background capture thread."""
        self._capture_stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        logger.info("Capture thread started")

    def _capture_loop(self):
        """Continuously capture frames in the background."""
        frame_count = 0
        last_time = time.time()
        while not self._capture_stop.is_set():
            try:
                data = self.camera.capture()
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
        """Update camera frame (called at ~30 fps)."""
        if self.camera_connected and self.camera:
            # Grab the latest frame from the background thread
            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None:
                self.current_image = frame

                # Update FPS (display fps)
                self.frame_count += 1
                current_time = cv2.getTickCount()
                if self.frame_count == 1:
                    self.last_fps_time = current_time
                elif current_time - self.last_fps_time >= cv2.getTickFrequency():
                    self.fps = self.frame_count / ((current_time - self.last_fps_time) / cv2.getTickFrequency())
                    self.frame_count = 0

                # Update display
                self._update_display()

        # Schedule next frame (~30 fps)
        self.root.after(33, self._update_frame)

    def _update_display(self):
        """Update the canvas with current image."""
        if self.current_image is None:
            return

        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        img = self.current_image

        # Fast downscale with cv2 (much faster than PIL for large frames)
        img_h, img_w = img.shape[:2]

        # Apply zoom
        if self.zoom_level != 1.0:
            target_w = int(img_w * self.zoom_level)
            target_h = int(img_h * self.zoom_level)
        else:
            target_w, target_h = img_w, img_h

        # Fit to canvas (maintain aspect ratio)
        img_ratio = target_w / target_h
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            display_width = canvas_width
            display_height = int(canvas_width / img_ratio)
        else:
            display_height = canvas_height
            display_width = int(canvas_height * img_ratio)

        # Downscale with cv2 (fast, INTER_AREA for downscale)
        if display_width < img_w or display_height < img_h:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        display_img = cv2.resize(img, (display_width, display_height), interpolation=interp)

        # Convert to PIL PhotoImage
        pil_img = Image.fromarray(display_img)
        self.display_image = ImageTk.PhotoImage(pil_img)

        # Clear canvas and draw image
        self.canvas.delete("all")

        # Calculate position (centered with pan offset)
        x = (canvas_width - display_width) // 2 + self.pan_x
        y = (canvas_height - display_height) // 2 + self.pan_y

        self.canvas.create_image(x, y, anchor=tk.NW, image=self.display_image)

        # Store display parameters for coordinate conversion
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

        # Update status bar
        self.status_bar.config(
            text=f"Connected | {img_w}x{img_h} | "
                 f"{self.fps:.1f} fps (capture {self._capture_fps:.1f}) | Points: {len(self.points)}"
        )

    def _draw_points(self):
        """Draw click markers on canvas."""
        if not hasattr(self, 'display_params') or not self.points:
            return

        params = self.display_params
        scale_x = params["width"] / params["orig_width"]
        scale_y = params["height"] / params["orig_height"]

        for i, point in enumerate(self.points):
            # Convert original coordinates to display coordinates
            disp_x = int(point.x * scale_x) + params["x"]
            disp_y = int(point.y * scale_y) + params["y"]

            # Draw marker
            self.canvas.create_oval(
                disp_x - 5, disp_y - 5, disp_x + 5, disp_y + 5,
                fill="red", outline="white", width=2
            )

            # Draw label
            self.canvas.create_text(
                disp_x + 10, disp_y - 10,
                text=f"#{i+1}",
                fill="yellow", font=("Arial", 10, "bold")
            )

    # ------------------------------------------------------------------
    # Substrate HSV Detection
    # ------------------------------------------------------------------

    def _detect_substrate_hsv(self):
        """Auto-detect substrate HSV from current image using histogram mode."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image captured yet")
            return

        try:
            # Convert to HSV
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)
            H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

            # Find histogram mode (argmax) for each channel
            h_hist = cv2.calcHist([H], [0], None, [180], [0, 180]).flatten()
            s_hist = cv2.calcHist([S], [0], None, [256], [0, 256]).flatten()
            v_hist = cv2.calcHist([V], [0], None, [256], [0, 256]).flatten()

            h_sub = int(np.argmax(h_hist))
            s_sub = int(np.argmax(s_hist))
            v_sub = int(np.argmax(v_hist))

            self.substrate_hsv = (h_sub, s_sub, v_sub)

            # Update label
            self.substrate_hsv_label.config(
                text=f"H={h_sub}, S={s_sub}, V={v_sub}"
            )

            logger.info(f"Substrate HSV detected: H={h_sub}, S={s_sub}, V={v_sub}")

        except Exception as e:
            logger.error(f"Substrate HSV detection failed: {e}")
            messagebox.showerror("Error", f"Failed to detect substrate HSV: {e}")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event):
        """Handle left-click to add point with HSV."""
        if self.current_image is None:
            return

        # Convert display coordinates to original image coordinates
        orig_x, orig_y = self._display_to_original(event.x, event.y)

        # Extract HSV at click position
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)
        h_val = int(hsv[orig_y, orig_x, 0])
        s_val = int(hsv[orig_y, orig_x, 1])
        v_val = int(hsv[orig_y, orig_x, 2])
        hsv_value = (h_val, s_val, v_val)

        # Add point
        point = Point(orig_x, orig_y, hsv=hsv_value)
        self.points.append(point)

        # Update points list
        self._update_points_list()

        # Redraw
        self._update_display()

        logger.info(f"Added point: ({orig_x}, {orig_y}) HSV: {hsv_value}")

    def _on_right_click(self, event):
        """Handle right-click to remove last point."""
        if self.points:
            removed = self.points.pop()
            self._update_points_list()
            self._update_display()
            logger.info(f"Removed point: ({removed.x}, {removed.y})")

    def _on_mouse_move(self, event):
        """Handle mouse movement for HSV cursor readout."""
        if self.current_image is None or not hasattr(self, 'display_params'):
            return

        orig_x, orig_y = self._display_to_original(event.x, event.y)

        if 0 <= orig_x < self.current_image.shape[1] and 0 <= orig_y < self.current_image.shape[0]:
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2HSV)
            h_val = int(hsv[orig_y, orig_x, 0])
            s_val = int(hsv[orig_y, orig_x, 1])
            v_val = int(hsv[orig_y, orig_x, 2])

            # Update status bar with cursor position and HSV
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

        # Clamp zoom
        self.zoom_level = max(0.1, min(10.0, self.zoom_level))

        self._update_display()

    def _on_pan_drag(self, event):
        """Handle middle mouse drag for panning."""
        # Simplified pan implementation
        pass

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
    # Camera Settings
    # ------------------------------------------------------------------

    def _on_exposure_change(self, value):
        """Handle exposure slider change."""
        exposure_us = float(value)
        self.exposure_label.config(text=f"{exposure_us:.0f} us")

        if self.camera:
            self.camera.set_exposure(exposure_us)

    def _on_gain_change(self, value):
        """Handle gain slider change."""
        gain_db = float(value)
        self.gain_label.config(text=f"{gain_db:.1f} dB")

        if self.camera:
            self.camera.set_gain(gain_db)

    # ------------------------------------------------------------------
    # Points Management
    # ------------------------------------------------------------------

    def _update_points_list(self):
        """Update the points listbox with HSV values."""
        self.points_listbox.delete(0, tk.END)
        for i, point in enumerate(self.points):
            if point.hsv:
                h, s, v = point.hsv
                self.points_listbox.insert(
                    tk.END,
                    f"#{i+1}: ({point.x}, {point.y})  HSV: ({h}, {s}, {v})"
                )
            else:
                self.points_listbox.insert(
                    tk.END,
                    f"#{i+1}: ({point.x}, {point.y})"
                )

    def _save_points(self):
        """Save points with HSV to JSON file."""
        if not self.points:
            messagebox.showwarning("Warning", "No points to save")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                data = {
                    "points": [p.to_dict() for p in self.points],
                    "substrate_hsv": list(self.substrate_hsv) if self.substrate_hsv else None,
                    "image_size": (
                        self.current_image.shape[1],
                        self.current_image.shape[0]
                    ) if self.current_image is not None else None,
                }

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                messagebox.showinfo("Success", f"Saved {len(self.points)} points to {file_path}")
                logger.info(f"Saved points to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save points: {e}")

    def _load_points(self):
        """Load points from JSON file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                self.points = [Point.from_dict(p) for p in data.get("points", [])]

                # Load substrate HSV if present
                substrate = data.get("substrate_hsv")
                if substrate:
                    self.substrate_hsv = tuple(substrate)
                    self.substrate_hsv_label.config(
                        text=f"H={substrate[0]}, S={substrate[1]}, V={substrate[2]}"
                    )

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
                self._update_points_list()
                self._update_display()
                logger.info("Cleared all points")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Cleanup resources."""
        self._disconnect_camera()
        self.root.destroy()


def main():
    """Main entry point."""
    root = tk.Tk()
    app = CameraAnnotator(root)

    # Handle window close
    root.protocol("WM_DELETE_WINDOW", app.cleanup)

    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()