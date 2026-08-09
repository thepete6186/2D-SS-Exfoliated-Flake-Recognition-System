"""
Zolix ZC300 smoke-test CLI for the lab PC.

Validates the driver against real hardware — in particular the
absolute-move (0x0064) and home (0x0069) opcodes, which the reference
project never exercised.

Usage (Windows lab PC):
    python -m stage.zc300_smoke --list-ports
    python -m stage.zc300_smoke --port COM3 --identify --status
    python -m stage.zc300_smoke --port COM3 --jog x 1000
    python -m stage.zc300_smoke --port COM3 --abs x 5000
    python -m stage.zc300_smoke --port COM3 --home x

Dry run without hardware (any machine):
    python -m stage.zc300_smoke --simulate --jog x 500
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from stage.base import Stage, StageError
from stage.simulated import SimulatedStage
from stage.zolix_zc300 import ZolixZC300


def list_ports() -> None:
    """Enumerate serial ports (needs pyserial; no hardware access)."""
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("pyserial is not installed — run: pip install pyserial")
        return

    ports = lp.comports()
    if not ports:
        print("No serial ports found.")
        return
    print("Serial ports:")
    for port in ports:
        print(f"  {port.device:12s}  {port.description}")


def print_status(stage: Stage) -> None:
    status = stage.get_status()
    print("Status:")
    print(f"  position : {status['position']}")
    print(f"  moving   : {status['moving']}")
    print(f"  limits   : { {k: v for k, v in status['limits'].items() if v} or 'none tripped' }")
    print(f"  alarms   : { {k: v for k, v in status['axis_alarms'].items() if v} or 'none' }")
    print(f"  estop    : {status['emergency_stop']}")


def identify(stage: Stage) -> None:
    if isinstance(stage, ZolixZC300):
        print(f"Device model : {stage.read_device_model()}")
        # Diagnostic-only raw reads: serial number (30008, LONG) and
        # software version (30010, value/10)
        regs = stage._read_input_registers(30008, 3)
        serial_no = (regs[0] & 0xFFFF) << 16 | (regs[1] & 0xFFFF)
        print(f"Serial number: {serial_no}")
        print(f"Firmware     : v{(regs[2] & 0xFFFF) / 10:.1f}")
    else:
        print("Device model : SimulatedStage (no hardware)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stage.zc300_smoke",
        description="Zolix ZC300 stage driver smoke test.",
    )
    parser.add_argument("--list-ports", action="store_true",
                        help="Enumerate serial ports and exit.")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port of the ZC300 (e.g. COM3).")
    parser.add_argument("--slave", type=int, default=1,
                        help="MODBUS slave address (default 1).")
    parser.add_argument("--simulate", action="store_true",
                        help="Run against SimulatedStage instead of hardware.")
    parser.add_argument("--identify", action="store_true",
                        help="Print device model / serial / firmware.")
    parser.add_argument("--status", action="store_true",
                        help="Print full stage status.")
    parser.add_argument("--jog", nargs=2, metavar=("AXIS", "STEPS"), default=None,
                        help="Relative move: axis (x/y/r) and signed pulses.")
    parser.add_argument("--abs", dest="absolute", nargs=2, metavar=("AXIS", "POS"),
                        default=None, help="Absolute move to a position in pulses.")
    parser.add_argument("--home", nargs="?", const="all", default=None,
                        metavar="AXIS", help="Home one axis or all (default all).")
    args = parser.parse_args(argv)

    if args.list_ports:
        list_ports()
        return 0

    if args.simulate:
        stage: Stage = SimulatedStage()
    elif args.port:
        stage = ZolixZC300(port=args.port, slave_address=args.slave)
    else:
        parser.error("--port COMx is required (or --simulate / --list-ports)")

    print(f"Connecting ({'simulated' if args.simulate else args.port})...")
    try:
        stage.connect()
    except StageError as exc:
        print(f"CONNECT FAILED: {exc}")
        return 1
    print("Connected.")

    try:
        if args.identify:
            identify(stage)

        if args.jog:
            axis, steps = args.jog[0], int(args.jog[1])
            before = stage.get_position()
            print(f"Jog {axis} by {steps:+d} pulses (before: {before[axis.lower()]})")
            stage.move_relative(axis, steps, wait=True)
            after = stage.get_position()
            print(f"  done (after: {after[axis.lower()]})")

        if args.absolute:
            axis, pos = args.absolute[0], float(args.absolute[1])
            print(f"Absolute move {axis} -> {pos} pulses "
                  "(hardware-unvalidated opcode 0x0064 — watch the stage)")
            stage.move_absolute(axis, pos, wait=True)
            print(f"  done (position: {stage.get_position()[axis.lower()]})")

        if args.home is not None:
            print(f"Homing {args.home} (opcode 0x0069, hardware-unvalidated)")
            stage.home(args.home, wait=True)
            print(f"  done (position: {stage.get_position()})")

        if args.status or not any((args.identify, args.jog, args.absolute,
                                   args.home is not None)):
            print_status(stage)
    except StageError as exc:
        print(f"STAGE ERROR: {exc}")
        return 1
    finally:
        stage.disconnect()
        print("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
