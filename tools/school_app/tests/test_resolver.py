"""
The free/busy resolver.

`test_never_defaults_to_free` is the reason this file exists. Everything else is
supporting detail.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tsapp.resolver import (FREE, IN_CLASS, UNKNOWN, Moment, current_period,
                            grade_snapshot, periods_for_grade, section_status,
                            student_day, student_now, teacher_status)

MON_0810 = Moment("Monday", 8 * 60 + 10)     # inside Period 1 (taught)
MON_0910 = Moment("Monday", 9 * 60 + 10)     # inside Period 2 (free)
MON_1200 = Moment("Monday", 12 * 60)         # outside every period


def test_taught_period_is_in_class(synthetic_db):
    snap = grade_snapshot("05", MON_0810)
    assert snap.counts() == {IN_CLASS: 3, FREE: 0, UNKNOWN: 0}


def test_free_period_is_free(synthetic_db):
    snap = grade_snapshot("05", MON_0910)
    assert snap.counts() == {IN_CLASS: 0, FREE: 3, UNKNOWN: 0}
    assert all("free for this section" in s["reason"]
               for s in snap.by_status(FREE))


def test_never_defaults_to_free(synthetic_db):
    """A student whose own group cannot be placed on the timetable.

    Student 1003 is in section 05-B, which is timetabled for "Periodic" at 08:10.
    Their only enrolment is a group with no matching lesson at all — exactly the
    situation the old app resolved as "not enrolled in what is being taught,
    therefore free". It is a gap in knowledge about WHICH lesson, not evidence of
    an absence of one.
    """
    snap = grade_snapshot("05", MON_0810)
    cara = next(s for s in snap.students if s["computer_number"] == "1003")

    assert cara["status"] == IN_CLASS
    assert cara["confidence"] == "low"
    assert "could not be identified" in cara["reason"]
    # And the count of such students is stated, not buried.
    assert any("could not be identified" in c for c in snap.caveats)


def test_identified_lesson_carries_subject_teacher_and_room(synthetic_db):
    snap = grade_snapshot("05", MON_0810)
    amal = next(s for s in snap.students if s["computer_number"] == "1001")
    assert amal["status"] == IN_CLASS
    assert amal["confidence"] == "high"
    assert amal["subject"] == "Mathematics N2"
    assert amal["teacher"] == "Ms. Maths Teacher"
    assert amal["room"] == "05A"


def test_grade_without_a_timetable_is_unknown_not_free(synthetic_db):
    """KG1 has a roster and enrolment but no timetable in any source.

    "0 free" would be read as "nobody is free", which is a claim about 270 real
    children that nothing in the data supports.
    """
    snap = grade_snapshot("K1", MON_0810)
    assert snap.counts() == {IN_CLASS: 0, FREE: 0, UNKNOWN: 1}
    assert snap.by_status(UNKNOWN)[0]["reason"] == "no timetable for this grade"
    assert any("not known" in c for c in snap.caveats)


def test_outside_timetabled_hours_is_free(synthetic_db):
    snap = grade_snapshot("05", MON_1200)
    assert snap.counts()[FREE] == 3
    assert all("no period is timetabled" in s["reason"]
               for s in snap.by_status(FREE))


def test_period_labels_are_resolved_within_a_grade(synthetic_db):
    assert [p["label"] for p in periods_for_grade("05")] == ["Period 1", "Period 2"]
    assert periods_for_grade("K1") == []
    assert current_period("05", MON_0810)["label"] == "Period 1"
    assert current_period("05", MON_1200) is None


def test_section_status_distinguishes_free_from_untimetabled(synthetic_db):
    rows = {r["section"]: r for r in section_status("05", MON_0910)}
    assert rows["A"]["status"] == FREE
    assert rows["B"]["status"] == FREE
    kg = section_status("K1", MON_0910)
    assert kg[0]["status"] == UNKNOWN     # no grid at all, not "free"


def test_overlapping_section_cells_are_not_resolved_by_row_order(synthetic_db):
    import sqlite3

    conn = sqlite3.connect(synthetic_db)
    conn.execute(
        "INSERT INTO section_periods (section_id,day,period_label,start_min,end_min,"
        "is_free) VALUES (1,'Monday','Bad duplicate',480,530,1)")
    conn.commit()
    conn.close()

    row = next(r for r in section_status("05", MON_0810) if r["section"] == "A")
    assert row["conflict"] is True
    assert row["status"] == IN_CLASS  # at least one overlapping cell is taught
    snap = grade_snapshot("05", MON_0810)
    assert any("overlapping timetable cells" in c for c in snap.caveats)


def test_teacher_availability_excludes_placeholders_and_the_unscheduled(synthetic_db):
    """Three ways a teacher is not teaching, and only one of them is 'free'."""
    result = teacher_status(MON_0810)
    busy = {t["code"] for t in result["busy"]}
    free = {t["code"] for t in result["free"]}

    assert busy == {"MTA1"}          # teaching at 08:10
    assert free == {"ARB1"}          # timetabled, but not at 08:10
    assert "SSTUDY" not in free | busy    # a placeholder, not a person
    assert "GHOST" not in free | busy     # absent from the schedule entirely
    assert result["not_in_schedule"] == 1


def test_teacher_availability_moves_with_the_clock(synthetic_db):
    at_0910 = teacher_status(MON_0910)
    assert {t["code"] for t in at_0910["busy"]} == {"ARB1"}
    assert {t["code"] for t in at_0910["free"]} == {"MTA1"}


def test_student_now(synthetic_db):
    rec = student_now("1001", MON_0810)
    assert rec["status"] == IN_CLASS
    assert rec["subject"] == "Mathematics N2"
    assert rec["grade"] == "05"
    assert student_now("nosuch", MON_0810) is None


def test_student_day_uses_the_student_not_the_section(synthetic_db):
    day = student_day("1001", "Monday")
    labels = [(s["period"], s["status"], s["subject"]) for s in day["slots"]]
    assert labels == [("Period 1", IN_CLASS, "Mathematics N2"),
                      ("Period 2", FREE, None)]

    # Same section, different enrolment: 1003's group is unplaceable, so their
    # Period 1 is in-class with no subject named rather than free.
    other = student_day("1003", "Monday")
    assert other["slots"][0]["status"] == IN_CLASS
    assert other["slots"][0]["subject"] is None
    assert other["slots"][0]["confidence"] == "low"


def test_individual_lesson_overrides_a_free_section_cell(synthetic_db):
    """A section-level free cell is not allowed to erase a student's own lesson.

    Real source data contains split groups in this shape. The old resolver
    checked ``is_free`` first and therefore reported those students available
    while their matched teacher schedule placed them in class.
    """
    import sqlite3

    conn = sqlite3.connect(synthetic_db)
    conn.execute(
        "INSERT INTO teaching_groups (id,grade,subject_code,course_group,room,"
        "teacher_code,match_quality,meeting_count) "
        "VALUES (3,'05','ARHL2','A','05C','ARB1','exact',1)")
    conn.execute("INSERT INTO enrolments (student_id,group_id) VALUES ('1001',3)")
    conn.execute(
        "INSERT INTO group_meetings (group_id,lesson_id,day,start_min,end_min,score) "
        "VALUES (3,2,'Monday',540,590,4)")
    conn.commit()
    conn.close()

    snap = grade_snapshot("05", MON_0910)
    amal = next(s for s in snap.students if s["computer_number"] == "1001")
    assert amal["status"] == IN_CLASS
    assert amal["subject"] == "Arabic Language- Arabs"
    assert amal["source_conflict"] is True
    assert "overrides" in amal["reason"]
    assert snap.counts() == {IN_CLASS: 1, FREE: 2, UNKNOWN: 0}
    assert any("individual lesson" in caveat for caveat in snap.caveats)

    day = student_day("1001", "Monday")
    assert day["slots"][1]["status"] == IN_CLASS
    assert day["slots"][1]["source_conflict"] is True


def test_kg_student_day_reports_no_timetable(synthetic_db):
    day = student_day("1004", "Monday")
    assert day["has_timetable"] is False
    assert day["slots"] == []


# ── choosing between concurrent matches ─────────────────────────────────────

from tsapp.resolver import _pick_lesson   # noqa: E402


def _cand(subject, teacher="MTA1", room="09B", score=4, quality="exact"):
    return {"subject": subject, "teacher_code": teacher, "room": room,
            "score": score, "match_quality": quality, "title": "Ms.",
            "teacher_name": "A Teacher"}


def test_two_records_of_one_lesson_collapse_to_high_confidence():
    """The commonest multi-match shape, and it is not really ambiguous.

    A student registered in both "Mathematics L1" and "Ace the IGCSE Exam - Math"
    with the same teacher in the same room is in ONE lesson recorded twice. Across
    a sample of the week 1,174 of 1,213 multi-match students looked like this.
    Where the child is is certain; only the label is not, so calling it low
    confidence understates what is known.
    """
    lesson, confidence = _pick_lesson(
        [_cand("Mathematics L1"), _cand("Ace the IGCSE Exam - Math")], None)
    assert confidence == "high"
    assert lesson["teacher_code"] == "MTA1"


def test_the_section_grid_breaks_the_tie_on_which_label_to_show():
    lesson, confidence = _pick_lesson(
        [_cand("Mathematics L1"), _cand("Ace the IGCSE Exam - Math")],
        "Mathematics")
    assert confidence == "high"
    assert lesson["subject"] == "Mathematics L1"


def test_two_teachers_at_once_stays_low_confidence():
    """39 of those 1,213 were genuine conflicts. A student cannot be in two rooms,
    so this is a matching failure and must not be dressed up as certainty."""
    lesson, confidence = _pick_lesson(
        [_cand("Mathematics L1", teacher="MTA1", room="09B"),
         _cand("Biology N2", teacher="SRC2", room="09A")], None)
    assert confidence == "low"
    assert lesson is None


def test_ambiguous_groups_are_not_used_to_place_a_student():
    """A group matched only by teacher+subject cannot say WHICH of that teacher's
    same-subject groups this is, so it may prove the teacher is busy but not where
    one student sits."""
    lesson, confidence = _pick_lesson(
        [_cand("Physical Education", score=2, quality="ambiguous")], None)
    assert confidence == "low"
    assert lesson is None


def test_a_single_strong_match_is_high_confidence():
    lesson, confidence = _pick_lesson([_cand("Chemistry L")], None)
    assert confidence == "high"
    assert lesson["subject"] == "Chemistry L"


def test_picking_is_deterministic():
    """Same inputs, same answer, whatever order they arrive in."""
    a, b = _cand("Zoology"), _cand("Algebra")
    assert _pick_lesson([a, b], None)[0] == _pick_lesson([b, a], None)[0]


def test_teacher_conflict_never_fabricates_a_subject_room_pair(synthetic_db):
    import sqlite3

    conn = sqlite3.connect(synthetic_db)
    conn.execute(
        "INSERT INTO lessons (id,teacher_code,day,start_min,end_min,subject_text,"
        "room_text) VALUES (3,'MTA1','Monday',480,530,'Physics','05B')")
    conn.commit()
    conn.close()

    result = teacher_status(MON_0810)
    teacher = next(r for r in result["busy"] if r["code"] == "MTA1")
    assert result["conflicts"] == 1
    assert teacher["conflict"] is True
    assert teacher["subject"] is None and teacher["room"] is None
    assert {row["subject"] for row in teacher["lessons"]} == {
        "Mathematics", "Physics"}
