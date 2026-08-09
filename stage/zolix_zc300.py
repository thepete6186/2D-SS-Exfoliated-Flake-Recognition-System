"""
Zolix ZC300 stage driver (MODBUS RTU over a USB virtual COM port).

Protocol facts (see docs in the transfer-stage-control reference repo,
MIT License, Copyright (c) 2026 Lewbert, whose driver this adapts):

- Fixed 115200 baud, 8 data bits, no parity, 1 stop bit; slave 1-255.
- PDU register address = manual register number - 1 (handled inside
  stage.modbus_rtu — all registers here are the manual's 1-based numbers).
- Floats are IEEE 754 big-endian across 2 consecutive registers.
- Opcode commands MUST be written with function 0x10 at register 30050
  with the exact register count for the opcode's arity (moves = 3,
  stops = 2, save-params = 1); a wrong count is rejected with exc 0x03.
- A motion command to an already-moving axis is rejected with exc 0x06.

Axis mapping: the ZC300 is a 3-axis (X/Y/Z) controller; the stage's
rotation axis is wired to the Z channel, exposed here as logical "r".

The driver works in integer pulses end-to-end and assumes the
controller's per-axis unit registers are set to pulse mode (checked on
connect). Physical-unit conversion lives in camera/coordinate_mapper.py.

Absolute-move (0x0064) and home (0x0069) opcodes are implemented from
the register map but were never exercised by the reference project —
validate them on real hardware with `python -m stage.zc300_smoke`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence

from stage.base import (
    AXES,
    ModbusError,
    Stage,
    StageBusyError,
    StageError,
    StageEstopError,
    StageLimitError,
    StageNotConnectedError,
    StageNotEnabledError,
)
from stage.modbus_rtu import (
    ModbusExceptionResponse,
    build_read_frame,
    build_write_multiple_floats,
    build_write_multiple_frame,
    build_write_single_frame,
    parse_float_pair,
    parse_multi_read_response,
    registers_to_ascii,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ZC300 register map (1-based MODBUS register numbers, as in the manual)
# ---------------------------------------------------------------------------

# --- Input registers (function code 0x04) ---
REG_DEVICE_MODEL  = 30001  # Device model string (7 regs, 2 ASCII chars each)
REG_MOTION_STATE  = 30012  # X/Y/Z motion state (3 regs: 0=stopped, 1=moving)
REG_STATUS        = 30015  # Limit/home/alarm/estop bitmask
REG_POS_X         = 30016  # X position (float, 2 regs); Y=30018, Z/R=30020
REG_UNIT_X        = 30022  # X unit (0=pulse, 1=mm, 2=deg); Y=30023, Z/R=30024

# --- Holding registers (function codes 0x03/0x06/0x10) ---
REG_OPCODE        = 30050  # Opcode + params (30051 axis, 30052 direction)
REG_TARGET_X      = 30059  # Absolute-move target (float, 2 regs); Y=30061, Z/R=30063
REG_ENABLE_X      = 30066  # X enable (0x01=enabled); Y=30067, Z/R=30068
REG_HOME_MODE_X   = 30105  # Homing mode (0x01 user / 0x02 neg limit / 0x03 origin switch)
REG_DIST_X        = 30114  # Fixed-length distance (float, 2 regs); Y=30116, Z/R=30118
REG_SPEED_CONST_X = 30129  # Constant speed (float, 2 regs); Y=30131, Z/R=30133
REG_ACC_X         = 30135  # Acceleration (float, 2 regs); Y=30137, Z/R=30139

# --- Opcodes ---
OP_ABSOLUTE       = 0x0064
OP_FIXED_LENGTH   = 0x0065
OP_CONTINUOUS     = 0x0066
OP_DECEL_STOP     = 0x0067
OP_IMMEDIATE_STOP = 0x0068
OP_HOME           = 0x0069
OP_SAVE_PARAMS    = 0x006D

# --- Axis selectors / directions (opcode params) ---
AXIS_SEL = {"x": 0x31, "y": 0x32, "r": 0x33}   # ZC300 Z channel = logical R
AXIS_ALL = 0x30
DIR_POS = 0x50
DIR_NEG = 0x4E

# --- Status bitmask (register 30015) ---
_LIMIT_BITS = {"x+": 0, "x-": 1, "y+": 3, "y-": 4, "r+": 6, "r-": 7}
_HOME_BITS = {"x": 2, "y": 5, "r": 8}
_ESTOP_BIT = 9
_ALARM_BITS = {"x": 10, "y": 11, "r": 12}

# --- MODBUS exception code -> typed error ---
_EXCEPTION_ERRORS = {
    0x06: StageBusyError,
    0x07: StageLimitError,
    0x08: StageEstopError,
    0x09: StageNotEnabledError,
}

_AXIS_IDX = {"x": 0, "y": 1, "r": 2}


class ZolixZC300(Stage):
    """
    MODBUS-RTU driver for the Zolix ZC300 motion controller.

    Parameters
    ----------
    port : str
        Serial port name (e.g. "COM3" on the lab PC).
    slave_address : int
        MODBUS slave address (1-255, default 1).
    timeout : float
        Serial read timeout in seconds.
    stop_mode : str
        "decel" (0x0067, smooth) or "immediate" (0x0068, emergency-style).
    axes : sequence of str
        Configured logical axes; omit "r" when no rotation hardware is
        attached.
    unit_check : str
        On connect the per-axis unit registers are read; "warn" logs a
        warning when an axis is not in pulse mode, "strict" raises,
        "off" skips the check.
    default_speed_pps : float
        Constant speed (pulses/s) written lazily before the first move
        on each axis; change per axis with set_speed().
    acceleration : float
        Acceleration written once per axis on connect (pulses/s^2).
    serial_factory : callable, optional
        Injection point for tests: called with the pyserial keyword
        arguments, must return a serial-port-like object. When None,
        pyserial is imported lazily inside connect().
    """

    def __init__(
        self,
        port: str,
        slave_address: int = 1,
        timeout: float = 0.05,
        stop_mode: str = "decel",
        axes: Sequence[str] = AXES,
        unit_check: str = "warn",
        default_speed_pps: float = 1000.0,
        acceleration: float = 10_000_000.0,
        serial_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._port = port
        self._slave = int(slave_address)
        self._timeout = float(timeout)
        self._stop_opcode = OP_DECEL_STOP if stop_mode == "decel" else OP_IMMEDIATE_STOP
        self._axes = tuple(self._normalize_axes(axes))
        self._unit_check = unit_check
        self._default_speed_pps = float(default_speed_pps)
        self._acceleration = float(acceleration)
        self._serial_factory = serial_factory

        self._ser: Optional[Any] = None
        self._connected = False
        self._lock = threading.RLock()
        self._device_model: Optional[str] = None

        # Register-write caches: skip redundant speed/distance writes
        self._speed_written: Dict[str, float] = {}
        self._distance_written: Dict[str, float] = {}

    @staticmethod
    def _normalize_axes(axes: Sequence[str]) -> Sequence[str]:
        normalized = []
        for axis in axes:
            ax = str(axis).lower()
            if ax not in AXES:
                raise StageError(f"Unknown axis {axis!r} (expected one of {AXES})")
            normalized.append(ax)
        return normalized

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the port, verify the controller, and configure axes.

        Sequence: liveness read (motion states) -> device model ->
        per-axis unit check -> axis enables -> acceleration writes.
        """
        factory = self._serial_factory
        if factory is None:
            import serial  # lazy: tests and SimulatedStage users never need pyserial
            factory = serial.Serial

        logger.info("Opening %s at 115200 baud (slave=%d)", self._port, self._slave)
        self._ser = factory(
            port=self._port,
            baudrate=115200,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self._timeout,
        )

        try:
            # Liveness probe: motion-state registers must answer
            self._read_input_registers(REG_MOTION_STATE, 3)

            model_regs = self._read_input_registers(REG_DEVICE_MODEL, 7)
            self._device_model = registers_to_ascii(model_regs)
            logger.info("Device model: %s", self._device_model)

            if self._unit_check != "off":
                units = self._read_input_registers(REG_UNIT_X, 3)
                self._check_units(units)

            for axis in self._axes:
                self._write_single(REG_ENABLE_X + _AXIS_IDX[axis], 0x01)

            for axis in self._axes:
                self._write_floats(
                    REG_ACC_X + 2 * _AXIS_IDX[axis], [self._acceleration]
                )
        except Exception:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            raise

        self._connected = True
        logger.info("Connected to Zolix ZC300 on %s (axes: %s)",
                    self._port, ",".join(self._axes))

    def _check_units(self, units) -> None:
        """All configured axes must be in pulse mode — every conversion here
        and in CoordinateMapper assumes pulses."""
        unit_names = {0: "pulse", 1: "mm", 2: "deg"}
        for axis in self._axes:
            value = units[_AXIS_IDX[axis]] if len(units) > _AXIS_IDX[axis] else 0
            if value != 0:
                msg = (
                    f"Axis '{axis}' is configured in "
                    f"{unit_names.get(value, hex(value))} mode, not pulse mode — "
                    "all step math assumes pulses (fix via the controller LCD panel)"
                )
                if self._unit_check == "strict":
                    raise StageError(msg)
                logger.warning(msg)

    def disconnect(self) -> None:
        """Stop all axes (best effort) and close the serial port."""
        try:
            if self.is_connected:
                self.stop("all")
        except Exception:
            pass
        with self._lock:
            self._connected = False
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ser is not None and getattr(self._ser, "is_open", False)

    def read_device_model(self) -> str:
        """Device model string (cached from connect)."""
        if self._device_model is None:
            with self._lock:
                self._require_connected()
                regs = self._read_input_registers(REG_DEVICE_MODEL, 7)
                self._device_model = registers_to_ascii(regs)
        return self._device_model

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_relative(self, axis: str, steps: int,
                      wait: bool = True, timeout: float = 30.0) -> None:
        """Fixed-length move by a signed number of pulses.

        Unlike the reference project's fire-and-forget single_step, this
        genuinely blocks (wait=True) until the axis reports stopped.
        """
        ax = self._validate_axis(axis)
        self._require_connected()
        steps = int(steps)
        if steps == 0:
            return

        with self._lock:
            self._ensure_speed(ax)
            distance = float(abs(steps))
            if self._distance_written.get(ax) != distance:
                self._write_floats(REG_DIST_X + 2 * _AXIS_IDX[ax], [distance])
                self._distance_written[ax] = distance
            direction = DIR_POS if steps > 0 else DIR_NEG
            self._write_opcode_block(OP_FIXED_LENGTH, AXIS_SEL[ax], direction)

        if wait:
            self.wait_until_stopped((ax,), timeout=timeout)

    def move_absolute(self, axis: str, position: float,
                      wait: bool = True, timeout: float = 60.0) -> None:
        """Absolute move to a position in pulses.

        Hardware-unvalidated opcode (the reference never used 0x0064) —
        verify on the lab PC with the smoke script before relying on it.
        """
        ax = self._validate_axis(axis)
        self._require_connected()

        with self._lock:
            self._ensure_speed(ax)
            self._write_floats(REG_TARGET_X + 2 * _AXIS_IDX[ax], [float(position)])
            # Direction param is a placeholder to satisfy the 3-register
            # arity; the controller derives direction from the target.
            self._write_opcode_block(OP_ABSOLUTE, AXIS_SEL[ax], DIR_POS)

        if wait:
            self.wait_until_stopped((ax,), timeout=timeout)

    def home(self, axis: str = "all", mode: Optional[int] = None,
             wait: bool = True, timeout: float = 120.0) -> None:
        """Run the controller homing routine (opcode 0x0069).

        Parameters
        ----------
        axis : str
            One logical axis or "all".
        mode : int, optional
            Homing mode written to 30105-30107 first: 0x01 = user zero
            (current position becomes origin), 0x02 = negative limit,
            0x03 = origin switch. None leaves the controller's setting.
        """
        axes = self._axes if axis == "all" else (self._validate_axis(axis),)
        self._require_connected()

        with self._lock:
            if mode is not None:
                for ax in axes:
                    self._write_single(REG_HOME_MODE_X + _AXIS_IDX[ax], int(mode))
            selector = AXIS_ALL if axis == "all" else AXIS_SEL[axes[0]]
            self._write_opcode_block(OP_HOME, selector, DIR_NEG)

        if wait:
            self.wait_until_stopped(axes, timeout=timeout)

    def stop(self, axis: str = "all") -> None:
        """Stop one axis or all axes (stop opcodes take 2 registers only)."""
        self._require_connected()
        selector = AXIS_ALL if axis == "all" else AXIS_SEL[self._validate_axis(axis)]
        with self._lock:
            self._write_opcode_block(self._stop_opcode, selector)

    def set_speed(self, axis: str, speed_pps: float) -> None:
        """Set the constant speed for an axis (pulses/s)."""
        ax = self._validate_axis(axis)
        self._require_connected()
        if speed_pps <= 0:
            raise StageError(f"Speed must be positive, got {speed_pps}")
        with self._lock:
            self._write_floats(
                REG_SPEED_CONST_X + 2 * _AXIS_IDX[ax], [float(speed_pps)]
            )
            self._speed_written[ax] = float(speed_pps)

    def save_parameters(self) -> bool:
        """Persist controller parameters to flash (all axes must be stopped)."""
        self._require_connected()
        with self._lock:
            states = self._read_input_registers(REG_MOTION_STATE, 3)
            if any(v == 1 for v in states):
                logger.warning("Cannot save parameters: axes are moving")
                return False
            self._write_opcode_block(OP_SAVE_PARAMS)
            return True

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, object]:
        """Read motion states, status bitmask, and positions in ONE
        10-register block read (30012-30021)."""
        self._require_connected()
        with self._lock:
            data = self._read_input_registers(REG_MOTION_STATE, 10)

        # Per-axis offsets are 0-based indices (the reference project's
        # settle-wait accidentally added the ASCII selector here).
        moving = {
            axis: len(data) > idx and data[idx] == 1
            for axis, idx in _AXIS_IDX.items()
        }
        raw = data[3] if len(data) > 3 else 0

        def bit(n: int) -> bool:
            return bool(raw & (1 << n))

        positions = {}
        for axis, idx in _AXIS_IDX.items():
            hi_i = 4 + 2 * idx
            if len(data) > hi_i + 1:
                positions[axis] = parse_float_pair(data[hi_i], data[hi_i + 1])
            else:
                positions[axis] = 0.0

        return {
            "position": positions,
            "moving": moving,
            "limits": {key: bit(n) for key, n in _LIMIT_BITS.items()},
            "home_switch": {axis: bit(n) for axis, n in _HOME_BITS.items()},
            "axis_alarms": {axis: bit(n) for axis, n in _ALARM_BITS.items()},
            "emergency_stop": bit(_ESTOP_BIT),
        }

    def get_position(self) -> Dict[str, float]:
        return self.get_status()["position"]

    # ------------------------------------------------------------------
    # MODBUS I/O internals
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise StageNotConnectedError(
                f"Not connected to the ZC300 on {self._port!r}"
            )

    def _ensure_speed(self, axis: str) -> None:
        """Write the constant speed lazily (skip when unchanged)."""
        if self._speed_written.get(axis) != self._default_speed_pps:
            self._write_floats(
                REG_SPEED_CONST_X + 2 * _AXIS_IDX[axis], [self._default_speed_pps]
            )
            self._speed_written[axis] = self._default_speed_pps

    def _send_frame(self, frame: bytes) -> bytes:
        """Send one frame and read the reply (reference serial rhythm)."""
        if self._ser is None:
            raise StageNotConnectedError("Serial port is not open")
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            self._ser.write(frame)
            self._ser.flush()
            time.sleep(0.002)
            return self._ser.read(256)
        except OSError as exc:
            self._connected = False
            raise StageNotConnectedError(f"Serial error: {exc}") from exc

    def _raise_for_exception(self, code: int, context: str) -> None:
        error_cls = _EXCEPTION_ERRORS.get(code, ModbusError)
        raise error_cls(f"{context}: MODBUS exception 0x{code:02X}")

    def _write_opcode_block(self, opcode: int, *params: int) -> None:
        """Write an opcode via fc 0x10 with the EXACT register count for
        its arity (the ZC300 rejects padded frames with exc 0x03).
        Caller must hold the lock."""
        values = [opcode] + list(params)
        response = self._send_frame(
            build_write_multiple_frame(self._slave, REG_OPCODE, values)
        )
        if response and len(response) >= 3 and response[1] == (0x10 | 0x80):
            self._raise_for_exception(response[2], f"Opcode 0x{opcode:04X}")

    def _write_single(self, register: int, value: int) -> None:
        response = self._send_frame(
            build_write_single_frame(self._slave, register, value)
        )
        if response and len(response) >= 3 and response[1] == (0x06 | 0x80):
            self._raise_for_exception(response[2], f"Write register {register}")

    def _write_floats(self, register: int, values) -> None:
        response = self._send_frame(
            build_write_multiple_floats(self._slave, register, list(values))
        )
        if response and len(response) >= 3 and response[1] == (0x10 | 0x80):
            self._raise_for_exception(response[2], f"Write register {register}")

    def _read_input_registers(self, register: int, count: int):
        response = self._send_frame(
            build_read_frame(self._slave, 0x04, register, count=count)
        )
        try:
            return parse_multi_read_response(response, 0x04)
        except ModbusExceptionResponse as exc:
            self._raise_for_exception(exc.code, f"Read register {register}")
        except ValueError as exc:
            raise ModbusError(f"Read register {register}: {exc}") from exc

    def __repr__(self) -> str:
        state = "connected" if self.is_connected else "disconnected"
        return (f"ZolixZC300({self._port!r}, slave={self._slave}, "
                f"axes={self._axes}, {state})")
