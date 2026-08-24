from __future__ import annotations

import pytest

from aisha_rev_d_driver.modbus import (
    REG_CONTROL_WORD,
    REG_POSITION_LEFT_HIGH,
    REG_TARGET_VELOCITY_LEFT,
    REG_TARGET_VELOCITY_RIGHT,
    append_crc,
    assert_read_only_request,
    build_read_holding_registers,
    build_write_single_register,
    decode_telemetry_registers,
    parse_read_holding_registers,
    signed_i32_high_low,
)


def test_supplier_example_frames_match_exactly() -> None:
    assert build_write_single_register(1, REG_CONTROL_WORD, 8).hex(" ").upper() == (
        "01 06 20 0E 00 08 E2 0F"
    )
    assert build_write_single_register(1, REG_TARGET_VELOCITY_LEFT, 100).hex(" ").upper() == (
        "01 06 20 88 00 64 03 CB"
    )
    assert build_write_single_register(1, REG_TARGET_VELOCITY_RIGHT, -100).hex(" ").upper() == (
        "01 06 20 89 FF 9C 12 79"
    )
    assert build_read_holding_registers(1, REG_POSITION_LEFT_HIGH, 2).hex(" ").upper() == (
        "01 03 20 A7 00 02 7E 28"
    )


def test_read_response_decodes_high_word_first_signed_values() -> None:
    # left=-2, right=0x00010002, speeds=-5.0/+5.0 RPM
    registers = [0xFFFF, 0xFFFE, 0x0001, 0x0002, 0xFFCE, 0x0032]
    payload = bytes([1, 3, 12]) + b"".join(value.to_bytes(2, "big") for value in registers)
    frame = append_crc(payload)
    decoded_words = parse_read_holding_registers(frame, expected_unit=1, expected_count=6)
    decoded = decode_telemetry_registers(decoded_words)
    assert decoded.left_count == -2
    assert decoded.right_count == 0x00010002
    assert decoded.left_rpm == pytest.approx(-5.0)
    assert decoded.right_rpm == pytest.approx(5.0)
    assert signed_i32_high_low(0x8000, 0x0000) == -0x80000000


def test_phase8b_transport_guard_rejects_every_write() -> None:
    read = build_read_holding_registers(1, REG_POSITION_LEFT_HIGH, 6)
    assert_read_only_request(read)
    write = build_write_single_register(1, REG_CONTROL_WORD, 8)
    with pytest.raises(PermissionError):
        assert_read_only_request(write)
