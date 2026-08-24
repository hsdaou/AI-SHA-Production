"""Replay and read-only RS485 transports for Phase 8B."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .modbus import (
    REG_ACTUAL_VELOCITY_RIGHT,
    REG_FAULT_LEFT,
    REG_POSITION_LEFT_HIGH,
    REG_STATUS_WORD,
    TelemetryRegisters,
    assert_read_only_request,
    build_read_holding_registers,
    decode_telemetry_registers,
    parse_read_holding_registers,
)


@dataclass(frozen=True)
class EncoderSample:
    stamp_s: float
    left_count: int
    right_count: int
    left_rpm: float
    right_rpm: float
    status_word: int = 0
    left_fault: int = 0
    right_fault: int = 0
    source: str = "replay"


class ReplayTransport:
    """Deterministic JSONL telemetry source; it cannot touch a serial device."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def samples(self) -> Iterator[EncoderSample]:
        for line_number, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            try:
                item = json.loads(raw)
                yield EncoderSample(
                    stamp_s=float(item["stamp_s"]),
                    left_count=int(item["left_count"]),
                    right_count=int(item["right_count"]),
                    left_rpm=float(item.get("left_rpm", 0.0)),
                    right_rpm=float(item.get("right_rpm", 0.0)),
                    status_word=int(item.get("status_word", 0)),
                    left_fault=int(item.get("left_fault", 0)),
                    right_fault=int(item.get("right_fault", 0)),
                    source="replay",
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid replay record at {self.path}:{line_number}") from exc


class ReadOnlyRs485Transport:
    """Poll encoder/status registers while making function-0x06 impossible.

    Construction imports pyserial lazily, so pure replay and tests do not need
    hardware dependencies.  Every outbound frame is checked immediately before
    serial transmission and must be a valid function-0x03 request.
    """

    def __init__(
        self,
        port: str,
        *,
        unit: int = 1,
        baud_rate: int = 115200,
        timeout_s: float = 0.10,
    ) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:
            raise RuntimeError("python3-serial is required for RS485 read-only mode") from exc
        self.unit = unit
        self.timeout_s = timeout_s
        self._serial = serial.Serial(
            port=port,
            baudrate=baud_rate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout_s,
            write_timeout=timeout_s,
        )

    def close(self) -> None:
        self._serial.close()

    def _read_exactly(self, size: int) -> bytes:
        data = self._serial.read(size)
        if len(data) != size:
            raise TimeoutError(f"RS485 response timed out after {len(data)}/{size} bytes")
        return data

    def _query(self, start_register: int, count: int) -> list[int]:
        request = build_read_holding_registers(self.unit, start_register, count)
        assert_read_only_request(request)
        self._serial.reset_input_buffer()
        self._serial.write(request)
        response = self._read_exactly(5 + 2 * count)
        return parse_read_holding_registers(
            response,
            expected_unit=self.unit,
            expected_count=count,
        )

    def sample(self) -> EncoderSample:
        telemetry_words = self._query(
            REG_POSITION_LEFT_HIGH,
            REG_ACTUAL_VELOCITY_RIGHT - REG_POSITION_LEFT_HIGH + 1,
        )
        telemetry: TelemetryRegisters = decode_telemetry_registers(telemetry_words)
        status_word = self._query(REG_STATUS_WORD, 1)[0]
        fault_words = self._query(REG_FAULT_LEFT, 2)
        return EncoderSample(
            stamp_s=time.monotonic(),
            left_count=telemetry.left_count,
            right_count=telemetry.right_count,
            left_rpm=telemetry.left_rpm,
            right_rpm=telemetry.right_rpm,
            status_word=status_word,
            left_fault=fault_words[0],
            right_fault=fault_words[1],
            source="rs485_read_only",
        )

    def __enter__(self) -> "ReadOnlyRs485Transport":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
