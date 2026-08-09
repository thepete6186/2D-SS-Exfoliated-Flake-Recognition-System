"""Tests for the ZolixZC300 driver against a scripted fake serial port.

FakeZC300 emulates the controller's Modbus RTU behavior: register reads
served from a value table, fc 0x06/0x10 acks, opcode handling with a
motion-state countdown, and scriptable exception responses. No pyserial
or hardware involved (the driver takes an injected serial_factory).
"""

import logging
import struct

import pytest

from stage.base import (
    StageAlarmError,
    StageBusyError,
    StageError,
    StageEstopError,
    StageLimitError,
    StageNotConnectedError,
    StageTimeoutError,
)
from stage.modbus_rtu import crc16
from stage.zolix_zc300 import ZolixZC300


def crc_wrap(body: bytes) -> bytes:
    return body + struct.pack("<H", crc16(body))


def decode(frame: bytes):
    """Decode a request frame to (fc, 1-based register, payload)."""
    fc = frame[1]
    if fc in (0x03, 0x04):
        addr, count = struct.unpack(">HH", frame[2:6])
        return (fc, addr + 1, count)
    if fc == 0x06:
        addr, val = struct.unpack(">HH", frame[2:6])
        return (fc, addr + 1, val)
    if fc == 0x10:
        addr, _count = struct.unpack(">HH", frame[2:6])
        bc = frame[6]
        vals = [
            struct.unpack(">H", frame[7 + i:9 + i])[0] for i in range(0, bc, 2)
        ]
        return (fc, addr + 1, vals)
    raise AssertionError(f"unexpected fc 0x{fc:02X}")


def float_regs(value: float):
    hi, lo = struct.unpack(">HH", struct.pack(">f", value))
    return hi, lo


class FakeZC300:
    """Scripted ZC300 double implementing the pyserial surface the driver uses."""

    def __init__(self):
        self.is_open = True
        self.frames = []          # every request frame written by the driver
        self.opcodes = []         # decoded value lists written to 30050
        self.registers = {}       # 1-based register -> unsigned 16-bit value
        self.motion_polls = {}    # axis idx -> remaining reads reporting moving
        self.default_move_polls = 2
        self.reject_next_opcode = None   # exception code for next opcode write
        self._pending = b""

        self.set_ascii(30001, "ZC300")
        for reg in (30022, 30023, 30024):
            self.registers[reg] = 0      # pulse mode

    # -- scripting helpers ------------------------------------------------

    def set_ascii(self, start_reg: int, text: str) -> None:
        data = text.encode("ascii")
        if len(data) % 2:
            data += b"\x00"
        for i in range(0, len(data), 2):
            self.registers[start_reg + i // 2] = (data[i] << 8) | data[i + 1]

    def set_float(self, start_reg: int, value: float) -> None:
        hi, lo = float_regs(value)
        self.registers[start_reg] = hi
        self.registers[start_reg + 1] = lo

    # -- pyserial surface --------------------------------------------------

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def flush(self):
        pass

    def close(self):
        self.is_open = False

    def write(self, frame):
        frame = bytes(frame)
        self.frames.append(frame)
        self._pending = self._respond(frame)

    def read(self, n):
        out, self._pending = self._pending, b""
        return out

    # -- device emulation --------------------------------------------------

    def _respond(self, frame: bytes) -> bytes:
        slave, fc = frame[0], frame[1]
        if fc in (0x03, 0x04):
            addr, count = struct.unpack(">HH", frame[2:6])
            values = [self._read_reg(addr + 1 + i) for i in range(count)]
            payload = bytes([count * 2]) + b"".join(
                struct.pack(">H", v & 0xFFFF) for v in values
            )
            return crc_wrap(bytes([slave, fc]) + payload)
        if fc == 0x06:
            return bytes(frame)  # device echoes the request
        if fc == 0x10:
            addr, count = struct.unpack(">HH", frame[2:6])
            reg = addr + 1
            bc = frame[6]
            values = [
                struct.unpack(">H", frame[7 + i:9 + i])[0]
                for i in range(0, bc, 2)
            ]
            if reg == 30050:
                if self.reject_next_opcode is not None:
                    code = self.reject_next_opcode
                    self.reject_next_opcode = None
                    return crc_wrap(bytes([slave, 0x90, code]))
                self.opcodes.append(values)
                self._on_opcode(values)
            else:
                for i, v in enumerate(values):
                    self.registers[reg + i] = v
            return crc_wrap(struct.pack(">BBHH", slave, 0x10, addr, count))
        raise AssertionError(f"unexpected fc 0x{fc:02X}")

    def _read_reg(self, reg: int) -> int:
        if reg in (30012, 30013, 30014):
            idx = reg - 30012
            remaining = self.motion_polls.get(idx, 0)
            if remaining > 0:
                self.motion_polls[idx] = remaining - 1
                return 1
            return 0
        return self.registers.get(reg, 0)

    def _on_opcode(self, values) -> None:
        opcode = values[0]
        if opcode in (0x0064, 0x0065, 0x0066, 0x0069) and len(values) >= 2:
            axis_sel = values[1]
            for idx, sel in ((0, 0x31), (1, 0x32), (2, 0x33)):
                if axis_sel in (sel, 0x30):
                    self.motion_polls[idx] = self.default_move_polls


def make_driver(**kwargs):
    fake = FakeZC300()
    driver = ZolixZC300(
        port="COM_TEST", serial_factory=lambda **_kw: fake, **kwargs
    )
    return driver, fake


class TestConnect:
    def test_connect_handshake(self):
        driver, fake = make_driver()

        driver.connect()

        decoded = [decode(f) for f in fake.frames]
        # liveness (motion states), model, units
        assert decoded[0] == (0x04, 30012, 3)
        assert decoded[1] == (0x04, 30001, 7)
        assert decoded[2] == (0x04, 30022, 3)
        # axis enables
        assert decoded[3:6] == [
            (0x06, 30066, 1), (0x06, 30067, 1), (0x06, 30068, 1),
        ]
        # acceleration floats
        acc_regs = [d[1] for d in decoded[6:9]]
        assert acc_regs == [30135, 30137, 30139]
        assert driver.is_connected
        assert driver.read_device_model() == "ZC300"

    def test_connect_unit_mismatch_warns(self, caplog):
        driver, fake = make_driver()
        fake.registers[30023] = 1  # Y axis configured in mm

        with caplog.at_level(logging.WARNING):
            driver.connect()

        assert any("pulse" in rec.message for rec in caplog.records)
        assert driver.is_connected

    def test_connect_unit_mismatch_strict_raises(self):
        driver, fake = make_driver(unit_check="strict")
        fake.registers[30023] = 1

        with pytest.raises(StageError, match="pulse"):
            driver.connect()
        assert not driver.is_connected


class TestMotionCommands:
    def test_move_relative_sequence(self):
        driver, fake = make_driver()
        driver.connect()
        fake.frames.clear()

        driver.move_relative("x", 500, wait=False)

        decoded = [decode(f) for f in fake.frames]
        # speed -> distance -> opcode, all fc 0x10
        assert decoded[0][0:2] == (0x10, 30129)
        assert decoded[1][0:2] == (0x10, 30114)
        assert decoded[1][2] == list(float_regs(500.0))
        assert decoded[2] == (0x10, 30050, [0x0065, 0x31, 0x50])

        # Same axis, same distance again: only the opcode is re-sent
        fake.frames.clear()
        driver.move_relative("x", 500, wait=False)
        assert [decode(f) for f in fake.frames] == [
            (0x10, 30050, [0x0065, 0x31, 0x50]),
        ]

    def test_move_relative_negative_direction(self):
        driver, fake = make_driver()
        driver.connect()

        driver.move_relative("y", -250, wait=False)

        assert fake.opcodes[-1] == [0x0065, 0x32, 0x4E]

    def test_move_absolute_sequence(self):
        driver, fake = make_driver()
        driver.connect()
        fake.frames.clear()

        driver.move_absolute("r", 12345.0, wait=False)

        decoded = [decode(f) for f in fake.frames]
        # target write to 30063 (Z/R pair) then 3-register opcode
        target = [d for d in decoded if d[1] == 30063]
        assert target and target[0][2] == list(float_regs(12345.0))
        assert decoded[-1] == (0x10, 30050, [0x0064, 0x33, 0x50])

    def test_stop_writes_two_registers(self):
        driver, fake = make_driver()
        driver.connect()

        driver.stop()

        assert fake.opcodes[-1] == [0x0067, 0x30]  # decel stop, all axes

    def test_home_sequence(self):
        driver, fake = make_driver()
        driver.connect()
        fake.frames.clear()

        driver.home("x", mode=0x02, wait=False)

        decoded = [decode(f) for f in fake.frames]
        assert decoded[0] == (0x06, 30105, 0x02)   # homing mode: neg limit
        assert decoded[-1] == (0x10, 30050, [0x0069, 0x31, 0x4E])

    def test_busy_rejection_maps_to_stage_busy_error(self):
        driver, fake = make_driver()
        driver.connect()
        fake.reject_next_opcode = 0x06

        with pytest.raises(StageBusyError):
            driver.move_relative("x", 100, wait=False)

    def test_move_on_unconfigured_axis_raises(self):
        driver, fake = make_driver(axes=("x", "y"))
        driver.connect()

        with pytest.raises(StageError, match="not configured"):
            driver.move_relative("r", 100, wait=False)

    def test_disconnected_move_raises(self):
        driver, _fake = make_driver()

        with pytest.raises(StageNotConnectedError):
            driver.move_relative("x", 100, wait=False)


class TestStatusAndWait:
    def test_get_position_decodes_floats(self):
        driver, fake = make_driver()
        driver.connect()
        fake.set_float(30016, 1234.5)
        fake.set_float(30018, -42.0)

        pos = driver.get_position()

        assert pos["x"] == 1234.5
        assert pos["y"] == -42.0
        assert pos["r"] == 0.0

    def test_wait_until_stopped_success(self):
        driver, fake = make_driver()
        driver.connect()
        fake.default_move_polls = 2

        driver.move_relative(
            "x", 100, wait=True, timeout=2.0,
        )

        # Motion countdown exhausted: the axis was polled to stop
        assert fake.motion_polls[0] == 0

    def test_wait_until_stopped_timeout(self):
        driver, fake = make_driver()
        driver.connect()
        fake.default_move_polls = 10 ** 9

        with pytest.raises(StageTimeoutError):
            driver.move_relative("x", 100, wait=True, timeout=0.1)

    def test_wait_until_stopped_limit_abort(self):
        driver, fake = make_driver()
        driver.connect()
        fake.default_move_polls = 10 ** 9
        fake.registers[30015] = 1 << 0   # X positive limit, still moving

        with pytest.raises(StageLimitError):
            driver.move_relative("x", 100, wait=True, timeout=2.0)

    def test_wait_until_stopped_estop_abort(self):
        driver, fake = make_driver()
        driver.connect()
        fake.default_move_polls = 10 ** 9
        fake.registers[30015] = 1 << 9   # emergency stop

        with pytest.raises(StageEstopError):
            driver.move_relative("x", 100, wait=True, timeout=2.0)

    def test_wait_until_stopped_alarm_abort(self):
        driver, fake = make_driver()
        driver.connect()
        fake.default_move_polls = 10 ** 9
        fake.registers[30015] = 1 << 10  # X axis alarm

        with pytest.raises(StageAlarmError):
            driver.move_relative("x", 100, wait=True, timeout=2.0)

    def test_get_status_shape(self):
        driver, fake = make_driver()
        driver.connect()

        status = driver.get_status()

        assert set(status.keys()) == {
            "position", "moving", "limits", "home_switch",
            "axis_alarms", "emergency_stop",
        }
        assert set(status["position"].keys()) == {"x", "y", "r"}
        assert set(status["limits"].keys()) == {"x+", "x-", "y+", "y-", "r+", "r-"}
