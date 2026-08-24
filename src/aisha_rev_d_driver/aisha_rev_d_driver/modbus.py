"""Small, dependency-free Modbus RTU codec for the ZLAC8015D telemetry map.

The supplier V4 Series RS485 manual describes encoder positions as signed
32-bit values stored high-word first at 0x20A7--0x20AA.  Actual velocities are
signed 16-bit values in 0.1 RPM at 0x20AB and 0x20AC.

This module can build write frames for offline protocol conformance tests, but
the Phase 8B transport deliberately accepts function 0x03 reads only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


READ_HOLDING_REGISTERS = 0x03
WRITE_SINGLE_REGISTER = 0x06

REG_CONTROL_MODE = 0x200D
REG_CONTROL_WORD = 0x200E
REG_TARGET_VELOCITY_LEFT = 0x2088
REG_TARGET_VELOCITY_RIGHT = 0x2089
REG_STATUS_WORD = 0x20A2
REG_FAULT_LEFT = 0x20A5
REG_POSITION_LEFT_HIGH = 0x20A7
REG_POSITION_LEFT_LOW = 0x20A8
REG_POSITION_RIGHT_HIGH = 0x20A9
REG_POSITION_RIGHT_LOW = 0x20AA
REG_ACTUAL_VELOCITY_LEFT = 0x20AB
REG_ACTUAL_VELOCITY_RIGHT = 0x20AC


class ModbusProtocolError(ValueError):
    """Raised when a Modbus frame violates the expected contract."""


@dataclass(frozen=True)
class TelemetryRegisters:
    """Decoded contiguous 0x20A7--0x20AC telemetry registers."""

    left_count: int
    right_count: int
    left_rpm: float
    right_rpm: float


def crc16_modbus(payload: bytes) -> int:
    """Return the standard Modbus CRC-16 value for *payload*."""

    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def append_crc(payload: bytes) -> bytes:
    """Append Modbus wire-order CRC bytes (low byte, then high byte)."""

    crc = crc16_modbus(payload)
    return payload + bytes((crc & 0xFF, (crc >> 8) & 0xFF))


def verify_crc(frame: bytes) -> None:
    if len(frame) < 4:
        raise ModbusProtocolError("frame is too short")
    expected = crc16_modbus(frame[:-2])
    received = frame[-2] | (frame[-1] << 8)
    if received != expected:
        raise ModbusProtocolError(
            f"CRC mismatch: received 0x{received:04X}, expected 0x{expected:04X}"
        )


def _validate_unit(unit: int) -> None:
    if not 1 <= unit <= 127:
        raise ValueError("ZLAC8015D unit address must be in the supplier range 1..127")


def build_read_holding_registers(unit: int, start_register: int, count: int) -> bytes:
    _validate_unit(unit)
    if not 0 <= start_register <= 0xFFFF:
        raise ValueError("start register is outside the 16-bit Modbus address space")
    if not 1 <= count <= 125:
        raise ValueError("holding-register read count must be in 1..125")
    payload = bytes(
        (
            unit,
            READ_HOLDING_REGISTERS,
            (start_register >> 8) & 0xFF,
            start_register & 0xFF,
            (count >> 8) & 0xFF,
            count & 0xFF,
        )
    )
    return append_crc(payload)


def build_write_single_register(unit: int, register: int, value: int) -> bytes:
    """Build a function-0x06 frame for offline verification only."""

    _validate_unit(unit)
    if not 0 <= register <= 0xFFFF:
        raise ValueError("register is outside the 16-bit Modbus address space")
    if not -0x8000 <= value <= 0xFFFF:
        raise ValueError("register value is outside signed/unsigned 16-bit range")
    encoded = value & 0xFFFF
    payload = bytes(
        (
            unit,
            WRITE_SINGLE_REGISTER,
            (register >> 8) & 0xFF,
            register & 0xFF,
            (encoded >> 8) & 0xFF,
            encoded & 0xFF,
        )
    )
    return append_crc(payload)


def parse_read_holding_registers(
    frame: bytes,
    *,
    expected_unit: int,
    expected_count: int,
) -> list[int]:
    """Validate a function-0x03 response and return its unsigned registers."""

    verify_crc(frame)
    if frame[0] != expected_unit:
        raise ModbusProtocolError(
            f"unexpected unit {frame[0]}; expected {expected_unit}"
        )
    if frame[1] & 0x80:
        code = frame[2] if len(frame) > 2 else -1
        raise ModbusProtocolError(f"device returned Modbus exception 0x{code:02X}")
    if frame[1] != READ_HOLDING_REGISTERS:
        raise ModbusProtocolError(
            f"unexpected function 0x{frame[1]:02X}; read-only 0x03 required"
        )
    expected_bytes = expected_count * 2
    if frame[2] != expected_bytes:
        raise ModbusProtocolError(
            f"unexpected byte count {frame[2]}; expected {expected_bytes}"
        )
    if len(frame) != expected_bytes + 5:
        raise ModbusProtocolError(
            f"unexpected response length {len(frame)}; expected {expected_bytes + 5}"
        )
    return [
        (frame[index] << 8) | frame[index + 1]
        for index in range(3, 3 + expected_bytes, 2)
    ]


def signed_i16(value: int) -> int:
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def signed_i32_high_low(high_word: int, low_word: int) -> int:
    value = ((high_word & 0xFFFF) << 16) | (low_word & 0xFFFF)
    return value - 0x1_0000_0000 if value & 0x8000_0000 else value


def decode_telemetry_registers(registers: Sequence[int]) -> TelemetryRegisters:
    if len(registers) != 6:
        raise ModbusProtocolError("0x20A7 telemetry block must contain six registers")
    return TelemetryRegisters(
        left_count=signed_i32_high_low(registers[0], registers[1]),
        right_count=signed_i32_high_low(registers[2], registers[3]),
        left_rpm=signed_i16(registers[4]) * 0.1,
        right_rpm=signed_i16(registers[5]) * 0.1,
    )


def assert_read_only_request(frame: bytes) -> None:
    """Reject every outgoing RTU request except a valid function-0x03 read."""

    verify_crc(frame)
    if len(frame) != 8 or frame[1] != READ_HOLDING_REGISTERS:
        raise PermissionError("Phase 8B serial transport permits function 0x03 reads only")
