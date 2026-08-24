from __future__ import annotations

from pathlib import Path

import pytest

from aisha_rev_d_driver.transport import ReplayTransport


FIXTURE = Path(__file__).parents[1] / "config/phase8b_encoder_replay.jsonl"


def test_replay_is_deterministic_and_hardware_free() -> None:
    samples = list(ReplayTransport(FIXTURE).samples())
    assert len(samples) == 21
    assert samples[0].left_count == samples[0].right_count == 0
    assert samples[-1].left_count == samples[-1].right_count == 1365
    assert {sample.source for sample in samples} == {"replay"}


def test_invalid_replay_fails_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text('{"stamp_s": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid replay record"):
        list(ReplayTransport(invalid).samples())
