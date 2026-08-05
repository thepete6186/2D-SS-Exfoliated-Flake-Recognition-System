#!/usr/bin/env python3
"""
Camera Annotator - Live camera view with click annotation and HSV recording.

Standalone tkinter app for:
- Live camera feed from Zeiss XiCam 208 (USB camera via OpenCV)
- Click to mark points (records pixel coordinates + HSV values)
- Auto-detect substrate HSV
- Save/load points with HSV to JSON
- Adjust camera settings (exposure, gain)
"""

import sys
import json
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Point:
    """Represents a clicked point on the image with HSV values."""

    def __init__(self, x: int, y: int, hsv: Optional[Tuple[int, int, int]] = None,
                 label: str = ""):
        self.x = x
        self.y = y
        self.hsv = hsv  # (H, S, V) tuple or None
        self.label = label or f"Point {len(self.points) + 1}"

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
    - Live camera feed
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

        # Camera
        self.camera: Optional[ZeissCamera] = None
        self.camera_connected = False

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

    def _connect_camera(self):
        """Connect to camera."""
        try:
            # Try camera indices 0, 1, 2 to find the USB camera
            self.camera = None
            for idx in range(3):
                cam = ZeissCamera(camera_index=idx)
                if cam.connect():
                    self.camera = cam
                    break

            if self.camera is not None:
                self.camera_connected = True
                self.btn_connect.config(text="Disconnect")
                info = self.camera.get_camera_info()
                self.status_bar.config(text=f"Connected: {info.get('name', 'Unknown')}")
                logger.info("Camera connected")
            else:
                messagebox.showerror("Error", "Failed to connect to camera (tried indices 0-2)")
                self.camera = None
        except Exception as e:
            messagebox.showerror("Error", f"Camera connection failed: {e}")
            self.camera = None

    def _disconnect_camera(self):
        """Disconnect camera."""
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

    # ------------------------------------------------------------------
    # Frame Update Loop
    # ------------------------------------------------------------------

    def _update_frame(self):
        """Update camera frame (called at ~30 fps)."""
        if self.camera_connected and self.camera:
            try:
                # Capture frame
                rgb = self.camera.capture()
                if rgb is not None:
                    self.current_image = rgb

                    # Update FPS
                    self.frame_count += 1
                    current_time = cv2.getTickCount()
                    if self.frame_count == 1:
                        self.last_fps_time = current_time
                    elif current_time - self.last_fps_time >= cv2.getTickFrequency():
                        self.fps = self.frame_count / ((current_time - self.last_fps_time) / cv2.getTickFrequency())
                        self.frame_count = 0

                    # Update display
                    self._update_display()
            except Exception as e:
                logger.error(f"Frame update error: {e}")

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

        # Convert numpy array to PIL Image
        img = Image.fromarray(self.current_image)

        # Apply zoom
        if self.zoom_level != 1.0:
            new_width = int(img.width * self.zoom_level)
            new_height = int(img.height * self.zoom_level)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Fit to canvas (maintain aspect ratio)
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            display_width = canvas_width
            display_height = int(canvas_width / img_ratio)
        else:
            display_height = canvas_height
            display_width = int(canvas_height * img_ratio)

        # Resize for display
        display_img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        self.display_image = ImageTk.PhotoImage(display_img)

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
            "orig_width": self.current_image.shape[1],
            "orig_height": self.current_image.shape[0],
        }

        # Draw points
        self._draw_points()

        # Update status bar
        self.status_bar.config(
            text=f"Connected | {self.current_image.shape[1]}x{self.current_image.shape[0]} | "
                 f"{self.fps:.1f} fps | Points: {len(self.points)}"
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
