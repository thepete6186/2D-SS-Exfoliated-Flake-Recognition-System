#!/usr/bin/env python3
"""
Stage Control - tkinter UI for the Zolix ZC300 motion controller.

Controls X/Y/R axes via the stage.ZolixZC300 driver (MODBUS RTU):
- Connect/disconnect on a COM port
- Relative moves (fixed-length) with speed control
- Absolute moves
- Home axis / all
- Stop / E-stop
- Live position + status readout

Run:
    python stage_control.py                  # GUI
    python stage_control.py --port COM3      # specify port at launch
    python stage_control.py --simulate       # use SimulatedStage (no hardware)
"""

import argparse
import logging
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, Optional

from stage.base import (
    Stage,
    StageAlarmError,
    StageBusyError,
    StageError,
    StageEstopError,
    StageLimitError,
    StageNotConnectedError,
    StageTimeoutError,
)
from stage.simulated import SimulatedStage
from stage.zolix_zc300 import ZolixZC300

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AXES = ("x", "y", "r")
AXIS_NAMES = {"x": "X", "y": "Y", "r": "R (Z)"}


def build_stage(port: str, simulate: bool) -> Stage:
    """Construct the stage driver (real or simulated)."""
    if simulate:
        # Model motion at a visible rate so the UI status feels real
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


class StageControlApp:
    def __init__(self, root: tk.Tk, port: str, simulate: bool):
        self.root = root
        self.root.title("Stage Control - Zolix ZC300")
        self.root.geometry("640x560")
        self.root.resizable(False, False)

        self.port = port
        self.simulate = simulate
        self.stage: Optional[Stage] = None
        self.connected = False

        # Per-axis move parameters (steps for relative, pulses for absolute)
        self.rel_steps: Dict[str, tk.StringVar] = {}
        self.abs_pos: Dict[str, tk.StringVar] = {}
        self.speed_var = tk.StringVar(value="1000")  # pulses/s

        self._build_gui()
        self._update_status()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_gui(self):
        pad = {"padx": 8, "pady": 4}

        # --- Connection frame ---
        conn = ttk.LabelFrame(self.root, text="Connection")
        conn.pack(fill=tk.X, **pad)

        ttk.Label(conn, text="Port:").grid(row=0, column=0, sticky="w", padx=4)
        self.port_var = tk.StringVar(value=self.port)
        ttk.Entry(conn, textvariable=self.port_var, width=12).grid(
            row=0, column=1, padx=4
        )
        self.sim_chk = tk.BooleanVar(value=self.simulate)
        ttk.Checkbutton(conn, text="Simulate", variable=self.sim_chk).grid(
            row=0, column=2, padx=8
        )
        self.btn_connect = ttk.Button(
            conn, text="Connect", command=self._toggle_connection
        )
        self.btn_connect.grid(row=0, column=3, padx=8)

        self.conn_label = ttk.Label(conn, text="Disconnected", foreground="gray")
        self.conn_label.grid(row=0, column=4, sticky="e", padx=8)

        # --- Motion frame (per-axis rows) ---
        motion = ttk.LabelFrame(self.root, text="Motion (pulses)")
        motion.pack(fill=tk.X, **pad)

        # Header
        ttk.Label(motion, text="Axis", font=("Arial", 10, "bold")).grid(
            row=0, column=0, padx=4
        )
        ttk.Label(motion, text="(-)", font=("Arial", 10, "bold")).grid(
            row=0, column=1, padx=4
        )
        ttk.Label(motion, text="Step", font=("Arial", 10, "bold")).grid(
            row=0, column=2, padx=4
        )
        ttk.Label(motion, text="(+)", font=("Arial", 10, "bold")).grid(
            row=0, column=3, padx=4
        )
        ttk.Label(motion, text="Abs position", font=("Arial", 10, "bold")).grid(
            row=0, column=4, padx=4
        )
        ttk.Label(motion, text="Go abs", font=("Arial", 10, "bold")).grid(
            row=0, column=5, padx=4
        )
        ttk.Label(motion, text="Home", font=("Arial", 10, "bold")).grid(
            row=0, column=6, padx=4
        )

        for i, axis in enumerate(AXES, start=1):
            name = AXIS_NAMES[axis]
            ttk.Label(motion, text=name, font=("Arial", 10, "bold")).grid(
                row=i, column=0, padx=4, pady=2
            )

            self.rel_steps[axis] = tk.StringVar(value="1000")
            ttk.Button(
                motion, text="\u25c0 -", width=6,
                command=lambda a=axis: self._move(a, -1),
            ).grid(row=i, column=1, padx=2, pady=2)
            ttk.Entry(motion, textvariable=self.rel_steps[axis], width=8).grid(
                row=i, column=2, padx=2
            )
            ttk.Button(
                motion, text="+ \u25b6", width=6,
                command=lambda a=axis: self._move(a, +1),
            ).grid(row=i, column=3, padx=2, pady=2)

            self.abs_pos[axis] = tk.StringVar(value="0")
            ttk.Entry(motion, textvariable=self.abs_pos[axis], width=10).grid(
                row=i, column=4, padx=2
            )
            ttk.Button(
                motion, text="Go", width=6,
                command=lambda a=axis: self._move_abs(a),
            ).grid(row=i, column=5, padx=2)

            ttk.Button(
                motion, text="Home", width=6,
                command=lambda a=axis: self._home(a),
            ).grid(row=i, column=6, padx=2)

        # --- Speed + global controls ---
        ctrl = ttk.LabelFrame(self.root, text="Controls")
        ctrl.pack(fill=tk.X, **pad)

        ttk.Label(ctrl, text="Speed (pulses/s):").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Entry(ctrl, textvariable=self.speed_var, width=10).grid(
            row=0, column=1, sticky="w", padx=4
        )
        ttk.Button(ctrl, text="Apply Speed", command=self._apply_speed).grid(
            row=0, column=2, padx=8
        )
        ttk.Button(ctrl, text="Home All", command=lambda: self._home("all")).grid(
            row=0, column=3, padx=8
        )
        ttk.Button(ctrl, text="Stop All", command=lambda: self._stop("all")).grid(
            row=0, column=4, padx=8
        )
        ttk.Button(ctrl, text="E-STOP", command=self._estop).grid(
            row=0, column=5, padx=8
        )

        # --- Status readout ---
        status = ttk.LabelFrame(self.root, text="Status")
        status.pack(fill=tk.BOTH, expand=True, **pad)

        self.status_text = tk.Text(status, height=8, state=tk.DISABLED, font=("Consolas", 10))
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.status_var = tk.StringVar(value="Ready: disconnected")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _toggle_connection(self):
        if self.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        port = self.port_var.get().strip()
        simulate = self.sim_chk.get()
        try:
            self.stage = build_stage(port, simulate)
            # Run connect on a thread so the GUI doesn't freeze on a slow serial port
            threading.Thread(target=self._connect_worker, daemon=True).start()
            self._set_status("Connecting...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create stage: {e}")

    def _connect_worker(self):
        try:
            self.stage.connect()
            self.connected = True
            self.root.after(0, self._on_connected)
        except Exception as e:
            self.connected = False
            self.stage = None
            self.root.after(0, lambda: messagebox.showerror("Connect Failed", str(e)))
            self.root.after(0, lambda: self._set_status("Disconnected"))

    def _on_connected(self):
        self.btn_connect.config(text="Disconnect")
        self.conn_label.config(text=self.stage._port if hasattr(self.stage, "_port") else "Simulated", foreground="green")
        self._set_status("Connected")
        self._apply_speed()

    def _disconnect(self):
        try:
            if self.stage is not None:
                self.stage.stop("all")
                self.stage.disconnect()
        except Exception:
            pass
        self.stage = None
        self.connected = False
        self.btn_connect.config(text="Connect")
        self.conn_label.config(text="Disconnected", foreground="gray")
        self._set_status("Disconnected")

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def _move(self, axis: str, sign: int):
        if not self._require_connected():
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

    def _move_abs(self, axis: str):
        if not self._require_connected():
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

    def _home(self, axis: str):
        if not self._require_connected():
            return
        self._set_status(f"Homing {'all axes' if axis == 'all' else AXIS_NAMES[axis]}...")
        threading.Thread(
            target=self._motion_worker,
            args=(lambda: self.stage.home(axis, wait=True, timeout=120.0),),
            daemon=True,
        ).start()

    def _stop(self, axis: str = "all"):
        if not self._require_connected():
            return
        try:
            self.stage.stop(axis)
            self._set_status(f"Stopped {'all axes' if axis == 'all' else AXIS_NAMES[axis]}")
        except Exception as e:
            self._set_status(f"Stop error: {e}")

    def _estop(self):
        try:
            if self.stage is not None:
                self.stage.stop("all")
            self._set_status("E-STOP pressed: all axes stopped")
        except Exception as e:
            self._set_status(f"E-STOP error: {e}")

    def _apply_speed(self):
        if not self.connected or self.stage is None:
            return
        try:
            speed = float(self.speed_var.get())
            for axis in AXES:
                try:
                    self.stage.set_speed(axis, speed)
                except Exception:
                    pass  # some stages may not support per-axis speed
            self._set_status(f"Speed set to {speed} pulses/s")
        except ValueError:
            messagebox.showwarning("Invalid speed", "Speed must be a number")

    # ------------------------------------------------------------------
    # Workers / status polls
    # ------------------------------------------------------------------
    def _require_connected(self) -> bool:
        if not self.connected or self.stage is None:
            messagebox.showwarning("Not connected", "Connect the stage first")
            return False
        return True

    def _motion_worker(self, fn):
        try:
            fn()
            self.root.after(0, lambda: self._set_status("Done"))
        except StageLimitError as e:
            self.root.after(0, lambda: self._set_status(f"Limit: {e}"))
        except StageEstopError as e:
            self.root.after(0, lambda: self._set_status(f"E-STOP: {e}"))
        except StageAlarmError as e:
            self.root.after(0, lambda: self._set_status(f"Alarm: {e}"))
        except StageBusyError as e:
            self.root.after(0, lambda: self._set_status(f"Busy: {e}"))
        except StageTimeoutError as e:
            self.root.after(0, lambda: self._set_status(f"Timeout: {e}"))
        except StageNotConnectedError as e:
            self.root.after(0, lambda: self._set_status(f"Not connected: {e}"))
        except StageError as e:
            self.root.after(0, lambda: self._set_status(f"Stage error: {e}"))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"Error: {e}"))

    def _update_status(self):
        """Poll stage status every 500ms."""
        if self.connected and self.stage is not None:
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
                self._render_status("\n".join(lines))
            except Exception as e:
                self._render_status(f"Status error: {e}")
        self.root.after(500, self._update_status)

    def _render_status(self, text: str):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete("1.0", tk.END)
        self.status_text.insert(tk.END, text)
        self.status_text.config(state=tk.DISABLED)

    def _set_status(self, text: str):
        self.status_var.set(f"Status: {text}")

    def cleanup(self):
        self._disconnect()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Zolix ZC300 stage control UI")
    parser.add_argument("--port", default="COM3", help="Serial port (default COM3)")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatedStage (no hardware)")
    args = parser.parse_args()

    root = tk.Tk()
    app = StageControlApp(root, port=args.port, simulate=args.simulate)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()


if __name__ == "__main__":
    main()