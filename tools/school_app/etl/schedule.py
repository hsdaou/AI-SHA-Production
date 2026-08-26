"""
Import the schedule sources and tie each teaching group to the lessons that are
actually it.

TWO SOURCES, DIFFERENT JOBS
---------------------------
`lessons`          teacher-centric, from the teacher schedule workbook. Says
                   WHEN something is taught and by whom. A teacher occupies one
                   slot at a time (verified: 3,060 rows, no teacher double-
                   booked), which makes this the reliable spine.
`section_periods`  class-centric, from the timetable database. Says what a
                   SECTION is nominally doing. Necessary for displaying a
                   timetable, but it cannot answer "is this particular student in
                   a lesson", because a section splits into language and ability
                   groups that the section grid renders as one cell.

MATCHING
--------
A group and a lesson are the same teaching event when the teacher matches and at
least one corroborating field agrees. Room alone is not enough and subject alone
is not enough, because:

  * Room vocabularies differ between sources. Grade 9 enrolment rooms are 09A,
    09B (physical rooms); the teacher schedule writes 09SA, 09SB (section
    names). Requiring a room match drops grade 9 from 83% resolvable to 4%.
  * Subject names are printed differently in each report, and a teacher usually
    teaches several groups of the same subject.

So both are scored, and either one plus teacher identity is accepted. This
replaces comparing normalised subject strings and hoping — the approach that
made every Mathematics lesson in the school look like a free period.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .subjects import compatible
from .timegrid import parse_span, repair_rooms

# Above this many matches a group is no longer identified — it is a teacher's
# whole week of one subject, every group of which looks alike once the room
# fails to distinguish them (grade 9 enrolment rooms are physical rooms, 09A;
# the teacher schedule writes section names, 09SA). Such matches are KEPT,
# because they still prove the teacher is teaching then, but marked 'ambiguous'
# so nothing claims a particular student sits in a particular one of them.
AMBIGUOUS_ABOVE = 14


@dataclass
class Lesson:
    teacher_code: str
    day: str
    start_min: int
    end_min: int
    sheet_period: str | None
    subject_text: str | None
    group_code: str | None
    room_text: str | None
    rooms: list[str] = field(default_factory=list)
    source: str = "teacher_schedule"


@dataclass
class ScheduleImport:
    lessons: list[Lesson] = field(default_factory=list)
    teacher_names: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    unresolved_rooms: dict[str, int] = field(default_factory=dict)


def load_lessons(xlsx_path: str) -> ScheduleImport:
    """Read the teacher schedule workbook into wall-clock lessons."""
    import openpyxl

    out = ScheduleImport()
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["Full Schedule"]
    headers = None
    for row in ws.iter_rows(values_only=True):
        if headers is None:
            headers = [str(h).strip() if h else "" for h in row]
            continue
        rec = dict(zip(headers, row))
        code = str(rec.get("Teacher Code") or "").strip()
        day = str(rec.get("Day") or "").strip()
        span = parse_span(rec.get("Time"))
        if not code or not day:
            continue
        if span is None:
            out.warnings.append(
                f"lesson for {code} on {day} has unreadable time "
                f"{rec.get('Time')!r} — dropped, since a lesson with no hour "
                f"cannot be compared to anything")
            continue
        rooms, unresolved = repair_rooms(rec.get("Room/Class"))
        for token in unresolved:
            out.unresolved_rooms[token] = out.unresolved_rooms.get(token, 0) + 1
        name = str(rec.get("Teacher Name") or "").strip()
        if name:
            out.teacher_names.setdefault(code, name)
        out.lessons.append(Lesson(
            teacher_code=code, day=day, start_min=span[0], end_min=span[1],
            sheet_period=str(rec.get("Period") or "") or None,
            subject_text=str(rec.get("Subject") or "") or None,
            group_code=str(rec.get("Group Code") or "") or None,
            room_text=str(rec.get("Room/Class") or "") or None,
            rooms=rooms,
        ))
    wb.close()
    return out


def room_keys(group) -> set[str]:
    """Every name the schedule might use for the place this group meets.

    The two sources do not mean the same thing by "room". The subject lists print
    the PHYSICAL room — grade 9's groups meet in 09A, 09B, 09C — while the teacher
    schedule's column is headed "Room/Class" and for the upper grades holds the
    SECTION: 09SA, 09SB, 09SC. Compared directly, 88 of grade 9's 156 groups match
    nothing, fall back to matching on subject alone, and become indistinguishable
    from the other groups that same teacher teaches of that same subject. That is
    what left `Mathematics L` claiming 29 lessons a week and 78 of the grade's
    groups marked ambiguous.

    The course group is the missing link: for these grades it IS the section name
    (SA, SB, LA), so `grade + course_group` reconstructs exactly the string the
    schedule wrote. It is added as a SECOND key rather than replacing the printed
    room, because in the lower grades the printed room already is the section
    (01A, 05C) and those matches must keep working.

    Only keys that actually occur in the schedule can match, so a derived key that
    corresponds to nothing simply has no effect.
    """
    keys = set()
    if group.room:
        keys.add(group.room)
    if group.course_group and group.grade:
        keys.add(f"{group.grade}{group.course_group.upper()}")
    return keys


def score_match(group, lesson: Lesson, keys: set[str] | None = None) -> int:
    """0 = not the same event. >=2 = the same event, higher is more corroborated.

    Deliberately additive rather than a chain of conditions, so that the reason a
    match was accepted survives into the database as a number the API can report.
    """
    if lesson.teacher_code != group.teacher_code:
        return 0
    score = 0
    if (keys if keys is not None else room_keys(group)) & set(lesson.rooms):
        score += 2
    if compatible(group.subject_name, lesson.subject_text):
        score += 2
    if (group.course_group and lesson.group_code
            and group.course_group.upper() in lesson.group_code.upper()):
        score += 1
    return score if score >= 2 else 0


def quality_of(best_score: int, room_hit: bool, subject_hit: bool) -> str:
    """Describe the evidence among the matches that actually survived.

    A score of four can only come from room + subject, so it is exact.  Below
    that, seeing a room hit in one lesson and a subject hit in a *different*,
    discarded lesson is not exact corroboration.  The old implementation kept
    those flags from every provisional hit and could therefore promote a
    room-only match to ``exact`` because a weaker subject-only coincidence
    existed elsewhere in the teacher's week.
    """
    if best_score >= 4:
        return "exact"
    if room_hit and not subject_hit:
        return "room"
    if subject_hit and not room_hit:
        return "subject"
    if room_hit and subject_hit:
        # Equal-scoring retained lessons disagree about *why* they match.  No
        # single event is corroborated by both facts, so an individual student
        # must not be placed through this group.
        return "ambiguous"
    return "none"


@dataclass
class MatchResult:
    # group -> list of (lesson_index, score)
    meetings: dict[object, list[tuple[int, int]]] = field(default_factory=dict)
    quality: dict[object, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    ambiguous: int = 0


def match_groups(groups, lessons: list[Lesson]) -> MatchResult:
    """Tie every teaching group to the lessons that are it."""
    by_teacher: dict[str, list[int]] = defaultdict(list)
    for i, lesson in enumerate(lessons):
        by_teacher[lesson.teacher_code].append(i)

    result = MatchResult()
    for group in groups:
        hits: list[tuple[int, int]] = []
        keys = room_keys(group)
        for i in by_teacher.get(group.teacher_code or "", ()):
            lesson = lessons[i]
            score = score_match(group, lesson, keys)
            if not score:
                continue
            hits.append((i, score))

        # Keep only the best-corroborated interpretation. If a group matched some
        # lessons on room AND subject, a room-only coincidence elsewhere in the
        # same teacher's week is noise, not a second meeting.
        if hits:
            best = max(s for _, s in hits)
            hits = [(i, s) for i, s in hits if s == best]
        else:
            best = 0

        # Compute quality from the retained interpretation only.  Evidence on
        # a lower-scoring lesson is evidence that it was correctly discarded,
        # not corroboration for the winner.
        room_hit = any(keys & set(lessons[i].rooms) for i, _ in hits)
        subject_hit = any(
            compatible(group.subject_name, lessons[i].subject_text)
            for i, _ in hits)

        result.meetings[group] = hits
        result.quality[group] = quality_of(best, room_hit, subject_hit)

        if (len(hits) > AMBIGUOUS_ABOVE
                or result.quality[group] == "ambiguous"):
            result.quality[group] = "ambiguous"
            result.ambiguous += 1

    return result


def lessons_from_section_grid(section_rows, existing: list[Lesson]) -> list[Lesson]:
    """Recover lessons for staff the teacher schedule does not cover.

    The teacher schedule holds 121 teachers; the section timetable names 137. The
    difference is not idle staff — it is 27 teachers whose lessons are recorded
    only class-first. OEA1 (Art) has 31 cells in the section grid and no row at
    all in the teacher schedule, so 911 of her students could never be placed and
    she herself was reported as "not in the schedule" rather than as teaching.

    A taught cell IS a lesson stated the other way round: it names the teacher,
    the day, the hour, the subject and the section. Turning it back into a lesson
    costs nothing and invents nothing.

    Cells for the same teacher at the same hour are merged into one lesson with
    several rooms — that is a combined class, not a double booking. A cell is
    skipped whenever the teacher schedule already places that teacher at that
    hour, so the two sources can never contradict or double-count each other.
    """
    busy: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for lesson in existing:
        busy[lesson.teacher_code].append((lesson.start_min, lesson.end_min, lesson.day))

    merged: dict[tuple, dict] = {}
    for r in section_rows:
        code = r.get("teacher_code")
        if not code or r.get("is_free") or r.get("start_min") is None:
            continue
        if any(d == r["day"] and s < r["end_min"] and r["start_min"] < e
               for s, e, d in busy.get(code, ())):
            continue
        key = (code, r["day"], r["start_min"], r["end_min"])
        room = f"{r['grade']}{r['letter']}"
        slot = merged.setdefault(key, {"rooms": [], "subjects": [],
                                       "label": r.get("label"),
                                       "group_code": r.get("group_code")})
        if room not in slot["rooms"]:
            slot["rooms"].append(room)
        if r.get("subject_text") and r["subject_text"] not in slot["subjects"]:
            slot["subjects"].append(r["subject_text"])

    return [
        Lesson(teacher_code=code, day=day, start_min=start, end_min=end,
               sheet_period=slot["label"],
               subject_text=slot["subjects"][0] if slot["subjects"] else None,
               group_code=slot["group_code"],
               room_text=",".join(slot["rooms"]),
               rooms=slot["rooms"],
               source="section_timetable")
        for (code, day, start, end), slot in sorted(merged.items())
    ]


def load_section_periods(db_path: str):
    """Read the printed per-section timetable, converting labels to wall clock.

    Yields dicts with grade, section letter, day, label, start/end minutes.
    """
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT cs.grade, cs.section_code, te.day, te.period_name, te.period_time,
               te.subject, te.teacher_code, te.group_code, te.is_free
        FROM timetable_entries te
        JOIN class_sections cs ON cs.id = te.class_section_id
    """).fetchall()
    conn.close()

    for r in rows:
        code = r["section_code"]
        grade, _, letter = code.partition("-")
        grade = grade if grade.startswith("K") else f"{int(grade):02d}"
        span = parse_span(r["period_time"])
        yield {
            "grade": grade,
            "letter": letter,
            "day": r["day"],
            "label": r["period_name"],
            "start_min": span[0] if span else None,
            "end_min": span[1] if span else None,
            "subject_text": r["subject"],
            "teacher_code": r["teacher_code"],
            "group_code": r["group_code"],
            "is_free": 1 if (r["is_free"] or not r["subject"]) else 0,
        }
