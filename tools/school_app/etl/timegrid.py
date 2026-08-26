"""
Wall-clock time as the one shared coordinate for the schedule.

THE BUG THIS MODULE EXISTS TO KILL
----------------------------------
The three schedule sources label the same school day three different ways:

    13:05-13:50   grade 5 calls it   "Period 6"
                  grade 9 calls it   "SLO/Period 6"
                  the teacher sheet  "Period 7"
    13:50-14:35   grade 5 calls it   "SLO/P7"
                  grade 9 calls it   "Period 7"
                  the teacher sheet  "Period 8"

and worse, the same label is a different hour in different grades:

    "Period 3"    grade 5:  10:00-10:50
                  grade 9:   9:40-10:30

The previous app joined these namespaces by label. Asking it for free teachers
in "Period 7" stripped the word and looked up teacher-sheet period 7, returning
everyone free at 13:05 to a caller who meant 13:50. Asking for "Period 6"
matched nothing at all — the teacher sheet has no period 6, it jumps 5 to 7 —
so every teacher in the school came back free. A downstream heuristic
(`busy_free_sane`) then refused the answer because it looked implausible, which
is a smoke alarm bolted over a gas leak.

Minutes since midnight have none of these problems. Labels become display
aliases resolved per grade; every join is an interval overlap.
"""

from __future__ import annotations

import re

# The school day. Used only to disambiguate clock times printed without AM/PM,
# as the teacher schedule does (" 1:05- 1:50" is quarter past one in the
# afternoon; there is no lesson at quarter past one in the morning).
DAY_START_HOUR = 7

_CLOCK_RE = re.compile(r"(\d{1,2}):(\d{2})\s*([AaPp][Mm])?")
SCHOOL_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
               "Saturday", "Sunday"]


def parse_clock(hour: int, minute: int, meridiem: str | None) -> int:
    """A single clock reading -> minutes since midnight."""
    if meridiem:
        pm = meridiem.lower() == "pm"
        hour = hour % 12 + (12 if pm else 0)
    elif hour < DAY_START_HOUR:
        hour += 12          # 1:05 means 13:05 in a school timetable
    return hour * 60 + minute


def parse_span(text: str | None) -> tuple[int, int] | None:
    """'8:00AM- 8:50AM' or ' 1:05- 1:50' -> (start_min, end_min).

    Returns None when the text does not hold two readable clock times, so the
    caller can record an unusable row rather than store a plausible guess.
    """
    if not text:
        return None
    found = _CLOCK_RE.findall(str(text))
    if len(found) < 2:
        return None
    start = parse_clock(int(found[0][0]), int(found[0][1]), found[0][2] or None)
    end = parse_clock(int(found[1][0]), int(found[1][1]), found[1][2] or None)
    if end <= start:
        # A span that ends before it starts is a misread meridiem, not a lesson
        # that runs backwards. One retry assuming the end is in the afternoon.
        end_pm = parse_clock(int(found[1][0]) % 12 + 12, int(found[1][1]), None)
        if end_pm > start:
            end = end_pm
        else:
            return None
    return start, end


def hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_hhmm(text: str) -> int | None:
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", text or "")
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h < 24 and 0 <= mi < 60):
        return None
    return h * 60 + mi


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Half-open interval overlap, so back-to-back periods do not both match."""
    return a_start < b_end and b_start < a_end


# ── Room identifiers ────────────────────────────────────────────────────────
# The teacher schedule prints a lesson's rooms as one fixed-width field, and
# when the list is too long the FIRST entry loses its grade prefix:
#
#     "SB,12SC,12SD"        should be 12SB, 12SC, 12SD
#     "SC,09SD,09SE,09SF"   should be 09SC, 09SD, 09SE, 09SF
#     "4E,04F"              should be 04E, 04F
#
# Left as-is, a group meeting in 12SB never matches the lesson that teaches it,
# and its students are reported free while they are sitting in that room.
_ROOM_RE = re.compile(r"^(\d{1,2}|K\d)?([A-Za-z]{0,2}\d?)$")


def repair_rooms(field: str | None) -> tuple[list[str], list[str]]:
    """Explode a Room/Class field into canonical rooms.

    Returns (rooms, unresolved) — `unresolved` holds entries whose grade could
    not be recovered, which the build report surfaces instead of guessing.
    """
    if not field or str(field).strip().lower() in ("", "none"):
        return [], []
    parts = [p.strip() for p in str(field).split(",") if p.strip()]

    grades = []
    for p in parts:
        m = _ROOM_RE.match(p)
        if m and m.group(1):
            g = m.group(1)
            grades.append(g if g.startswith("K") else f"{int(g):02d}")

    rooms, unresolved = [], []
    for p in parts:
        m = _ROOM_RE.match(p)
        if not m:
            unresolved.append(p)
            continue
        grade, rest = m.group(1), m.group(2)
        if grade:
            grade = grade if grade.startswith("K") else f"{int(grade):02d}"
        elif grades:
            # Borrow the grade from a sibling entry in the same field.
            grade = grades[0]
        else:
            unresolved.append(p)
            continue
        rooms.append(f"{grade}{rest.upper()}")
    return list(dict.fromkeys(rooms)), unresolved
