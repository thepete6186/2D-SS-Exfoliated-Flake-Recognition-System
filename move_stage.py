#!/usr/bin/env python3
"""
Minimal CLI for the Zolix ZC300 motion controller.

Provides quick relative/absolute moves, home, stop, and status via the
stage.ZolixZC300 driver (MODBUS RTU). Intended for scripting and quick
hardware checks without the GUI.

Run:
    python move_stage.py --port COM3 status
    python move_stage.py --port COM3 rel x 1000
    python move_stage.py --port COM3 rel y -500
    python move_stage.py --port COM3 abs x 25000
    python move_stage.py --port COM3 home x
    python move_stage.py --port COM3 home all
    python move_stage.py --port COM3 stop
    python move_stage.py --port COM3 speed x 2000
    python move_stage.py --simulate status        # no hardware
"""

import argparse
import logging
import sys
from typing import Optional

from stage.base import Stage, StageError
from stage.simulated import SimulatedStage
from stage.zolix_zc300 import ZolixZC300

logging.basicConfig(level=logging.WARNING)

AXES = ("x", "y", "r")


def build_stage(port: str, simulate: bool) -> Stage:
    if simulate:
        return SimulatedStage(
            axes=AXES,
            speed_pps=2000.0,
            limits={"x": (-100000, 100000), "y": (-100000, 100000), "r": (-36000, 36000)},
        )
    return ZolixZC300(port=port, axes=AXES, default_speed_pps=1000.0)


def print_status(stage: Stage) -> None:
    status = stage.get_status()
    pos = status["position"]
    moving = status["moving"]
    print("Position (pulses):")
    for axis in AXES:
        print(f"  {axis}: {pos.get(axis, 0):>12.1f}  {'MOVING' if moving.get(axis) else 'stopped'}")
    if status.get("emergency_stop"):
        print("  !! E-STOP ACTIVE !!")
    for key, flag in status.get("limits", {}).items():
        if flag:
            print(f"  LIMIT: {key}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Zolix ZC300 CLI")
    parser.add_argument("--port", default="COM3", help="Serial port (default COM3)")
    parser.add_argument("--simulate", action="store_true", help="Use SimulatedStage (no hardware)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show current position/status")

    p_rel = sub.add_parser("rel", help="Relative move")
    p_rel.add_argument("axis", choices=AXES)
    p_rel.add_argument("steps", type=int)
    p_rel.add_argument("--no-wait", action="store_true")

    p_abs = sub.add_parser("abs", help="Absolute move")
    p_abs.add_argument("axis", choices=AXES)
    p_abs.add_argument("position", type=float)
    p_abs.add_argument("--no-wait", action="store_true")

    p_home = sub.add_parser("home", help="Home an axis or all")
    p_home.add_argument("axis", nargs="?", default="all", choices=AXES + ("all",))

    sub.add_parser("stop", help="Stop all axes")

    p_speed = sub.add_parser("speed", help="Set speed for an axis")
    p_speed.add_argument("axis", choices=AXES)
    p_speed.add_argument("pps", type=float)

    args = parser.parse_args(argv)

    stage = build_stage(args.port, args.simulate)
    try:
        stage.connect()
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1

    try:
        if args.command == "status":
            print_status(stage)
        elif args.command == "rel":
            stage.move_relative(args.axis, args.steps, wait=not args.no_wait)
            print(f"Moved {args.axis} by {args.steps} pulses")
        elif args.command == "abs":
            stage.move_absolute(args.axis, args.position, wait=not args.no_wait)
            print(f"Moved {args.axis} to {args.position} pulses")
        elif args.command == "home":
            stage.home(args.axis)
            print(f"Homed {args.axis}")
        elif args.command == "stop":
            stage.stop("all")
            print("Stopped all axes")
        elif args.command == "speed":
            stage.set_speed(args.axis, args.pps)
            print(f"Set {args.axis} speed to {args.pps} pulses/s")
        return 0
    except StageError as e:
        print(f"Stage error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            stage.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())