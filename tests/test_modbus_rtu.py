"""Golden-vector tests for the vendored Modbus RTU layer (stage/modbus_rtu.py).

Vectors come from the ZC300 protocol documentation:
- PDU register address = manual register number - 1 (30050 -> 0x7561)
- floats are IEEE 754 big-endian across 2 registers (15.0 -> 41 70 00 00)
- CRC-16 (poly 0xA001, init 0xFFFF) appended low byte first
"""

import struct

import pytest

from stage.modbus_rtu import (
    ModbusExceptionResponse,
    build_read_frame,
    build_write_multiple_floats,
    build_write_multiple_frame,
    build_write_single_frame,
    crc16,
    parse_float_pair,
    parse_multi_read_response,
    registers_to_ascii,
)


def valid_response(slave: int, fc: int, payload: bytes) -> bytes:
    """Assemble a CRC-valid response frame."""
    body = bytes([slave, fc]) + payload
    return body + struct.pack("<H", crc16(body))


class TestFrameBuilders:
    def test_register_offset_30050_is_0x7561(self):
        frame = build_read_frame(1, 0x04, 30050, count=1)

        # [slave][fc][addr_hi][addr_lo][count_hi][count_lo][crc_lo][crc_hi]
        assert frame[0] == 1
        assert frame[1] == 0x04
        assert frame[2:4] == bytes([0x75, 0x61])
        assert frame[4:6] == bytes([0x00, 0x01])
        assert len(frame) == 8

    def test_write_single_uses_offset_address(self):
        frame = build_write_single_frame(1, 30066, 0x01)

        assert frame[1] == 0x06
        assert frame[2:4] == struct.pack(">H", 30066 - 1)
        assert frame[4:6] == bytes([0x00, 0x01])

    def test_float_15_encodes_41_70_00_00(self):
        frame = build_write_multiple_floats(1, 30129, [15.0])

        # header: [slave][fc10][addr:2][count:2][byte_count] then data
        assert frame[1] == 0x10
        assert frame[5] == 2          # register count for one float
        assert frame[6] == 4          # byte count
        assert frame[7:11] == bytes([0x41, 0x70, 0x00, 0x00])

    def test_write_multiple_opcode_arity_encoding(self):
        # A 3-register opcode block (fixed-length move: opcode, axis, dir)
        frame = build_write_multiple_frame(1, 30050, [0x0065, 0x31, 0x50])

        assert frame[2:4] == bytes([0x75, 0x61])
        assert frame[4:6] == bytes([0x00, 0x03])   # 3 registers
        assert frame[6] == 6                        # 6 data bytes
        assert frame[7:13] == bytes([0x00, 0x65, 0x00, 0x31, 0x00, 0x50])

    def test_crc16_known_check_value(self):
        # Standard CRC-16/MODBUS check value for '123456789'
        assert crc16(b"123456789") == 0x4B37

    def test_crc_appended_low_byte_first(self):
        frame = build_read_frame(1, 0x04, 30012, count=3)

        crc = crc16(frame[:-2])
        assert frame[-2] == crc & 0xFF
        assert frame[-1] == (crc >> 8) & 0xFF


class TestParsers:
    def test_parse_float_pair_decodes_15(self):
        assert parse_float_pair(0x4170, 0x0000) == 15.0

    def test_parse_multi_read_response(self):
        # 3 registers: 0, 1, 0
        resp = valid_response(1, 0x04, bytes([6, 0, 0, 0, 1, 0, 0]))

        assert parse_multi_read_response(resp, 0x04) == [0, 1, 0]

    def test_exception_response_raises_with_code(self):
        resp = valid_response(1, 0x84, bytes([0x06]))

        with pytest.raises(ModbusExceptionResponse) as exc_info:
            parse_multi_read_response(resp, 0x04)
        assert exc_info.value.code == 0x06
        # Still a ValueError subclass (vendored-API compatibility)
        assert isinstance(exc_info.value, ValueError)

    def test_crc_mismatch_raises(self):
        resp = bytearray(valid_response(1, 0x04, bytes([2, 0, 5])))
        resp[-1] ^= 0xFF

        with pytest.raises(ValueError, match="CRC"):
            parse_multi_read_response(bytes(resp), 0x04)


class TestDeviceModel:
    def test_registers_to_ascii_device_model(self):
        # 'ZC300' packed 2 chars per register, big-endian, null-padded
        regs = [0x5A43, 0x3330, 0x3000, 0x0000]

        assert registers_to_ascii(regs) == "ZC300"
