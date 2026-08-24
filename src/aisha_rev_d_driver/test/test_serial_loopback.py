"""Exercise the read-only serial transport against a pseudo-terminal device."""

from __future__ import annotations

import os
import pty
import threading
import tty

from aisha_rev_d_driver.modbus import append_crc, verify_crc
from aisha_rev_d_driver.transport import ReadOnlyRs485Transport


def _read_exactly(file_descriptor: int, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        data.extend(os.read(file_descriptor, size - len(data)))
    return bytes(data)


def test_read_only_transport_uses_only_function_03_and_decodes_sample() -> None:
    master_fd, slave_fd = pty.openpty()
    tty.setraw(master_fd)
    tty.setraw(slave_fd)
    slave_path = os.ttyname(slave_fd)
    observed_requests: list[bytes] = []
    responses = (
        [0x0000, 0x0064, 0xFFFF, 0xFF38, 0x0032, 0xFFCE],
        [0x0000],
        [0x0000, 0x0000],
    )

    def emulate_driver() -> None:
        for registers in responses:
            request = _read_exactly(master_fd, 8)
            verify_crc(request)
            observed_requests.append(request)
            assert request[0] == 1
            assert request[1] == 0x03
            assert int.from_bytes(request[4:6], "big") == len(registers)
            payload = bytes((1, 3, len(registers) * 2)) + b"".join(
                value.to_bytes(2, "big") for value in registers
            )
            os.write(master_fd, append_crc(payload))

    emulator = threading.Thread(target=emulate_driver, daemon=True)
    emulator.start()
    try:
        with ReadOnlyRs485Transport(slave_path, timeout_s=0.5) as transport:
            sample = transport.sample()
    finally:
        emulator.join(timeout=1.0)
        os.close(master_fd)
        os.close(slave_fd)

    assert not emulator.is_alive()
    assert len(observed_requests) == 3
    assert [int.from_bytes(request[2:4], "big") for request in observed_requests] == [
        0x20A7,
        0x20A2,
        0x20A5,
    ]
    assert sample.left_count == 100
    assert sample.right_count == -200
    assert sample.left_rpm == 5.0
    assert sample.right_rpm == -5.0
    assert sample.left_fault == sample.right_fault == 0
    assert sample.source == "rs485_read_only"
