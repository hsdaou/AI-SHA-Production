"""
Wall-clock parsing and room repair.

These are the two places where the old app silently compared incomparable
things: period labels across three different numbering schemes, and room names
that the report writer had truncated.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from etl.timegrid import hhmm, overlaps, parse_hhmm, parse_span, repair_rooms


@pytest.mark.parametrize("text,expected", [
    # As printed in the class timetable (explicit meridiem).
    ("8:00AM- 8:50AM", (8 * 60, 8 * 60 + 50)),
    ("10:00AM-10:50AM", (10 * 60, 10 * 60 + 50)),
    ("12:15PM- 1:05PM", (12 * 60 + 15, 13 * 60 + 5)),
    ("3:20PM- 4:10PM", (15 * 60 + 20, 16 * 60 + 10)),
    # As printed in the teacher schedule (no meridiem at all).
    (" 8:00- 8:50", (8 * 60, 8 * 60 + 50)),
    ("10:50-11:40", (10 * 60 + 50, 11 * 60 + 40)),
    ("1:05- 1:50", (13 * 60 + 5, 13 * 60 + 50)),
    ("4:10- 5:00", (16 * 60 + 10, 17 * 60)),
])
def test_spans(text, expected):
    assert parse_span(text) == expected


def test_afternoon_is_inferred_not_guessed():
    """A school has no lesson at 1:05 in the morning, so bare 1:05 is 13:05."""
    assert parse_span("1:05- 1:50")[0] == 13 * 60 + 5


def test_midday_crossing():
    """11:40-12:30 must not become 11:40-00:30."""
    assert parse_span("11:40-12:30") == (11 * 60 + 40, 12 * 60 + 30)


@pytest.mark.parametrize("junk", [None, "", "no times here", "8:00", "Lunch"])
def test_unreadable_spans_return_none(junk):
    """Returning None lets the caller drop the row. The alternative — a plausible
    default — is how a lesson ends up compared against the wrong hour."""
    assert parse_span(junk) is None


def test_the_period_label_collision_is_real():
    """Documents the mismatch this module exists to route around.

    The same three time slots, as labelled by three sources:
        13:05-13:50   grade 5 "Period 6" | grade 9 "SLO/Period 6" | sheet "7"
        13:50-14:35   grade 5 "SLO/P7"   | grade 9 "Period 7"     | sheet "8"
    Comparing labels across sources is therefore off by one whole period, and
    "Period 6" does not exist in the teacher sheet at all.
    """
    assert parse_span("1:05- 1:50") == parse_span("1:05PM- 1:50PM")
    assert parse_span("1:50- 2:35") != parse_span("1:05- 1:50")


def test_hhmm_roundtrip():
    assert hhmm(13 * 60 + 20) == "13:20"
    assert parse_hhmm("13:20") == 13 * 60 + 20
    assert parse_hhmm("09:05") == 9 * 60 + 5
    for junk in ("25:00", "1320", "", "aa:bb", "12:60"):
        assert parse_hhmm(junk) is None


def test_overlap_is_half_open():
    """Back-to-back periods must not both match a boundary minute."""
    assert not overlaps(480, 530, 530, 580)
    assert overlaps(480, 530, 529, 580)


@pytest.mark.parametrize("field,expected", [
    ("05A", ["05A"]),
    ("05A,05B,05C,05D", ["05A", "05B", "05C", "05D"]),
    # The first entry lost its grade prefix to the field width. Left unrepaired,
    # the group meeting in 12SB never matches the lesson that teaches it.
    ("SB,12SC,12SD", ["12SB", "12SC", "12SD"]),
    ("SC,09SD,09SE,09SF", ["09SC", "09SD", "09SE", "09SF"]),
    ("SD,11SE", ["11SD", "11SE"]),
    # Unpadded grade numbers.
    ("4E,04F", ["04E", "04F"]),
    ("1E", ["01E"]),
    ("K1A", ["K1A"]),
])
def test_room_repair(field, expected):
    rooms, unresolved = repair_rooms(field)
    assert rooms == expected
    assert not unresolved


def test_rooms_with_no_recoverable_grade_are_reported_not_invented():
    rooms, unresolved = repair_rooms("SB")
    assert rooms == []
    assert unresolved == ["SB"]


def test_empty_room_field():
    assert repair_rooms(None) == ([], [])
    assert repair_rooms("None") == ([], [])
