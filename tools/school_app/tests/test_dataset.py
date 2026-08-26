"""
Assertions against the real built dataset.

These are not unit tests; they are the standing data-quality check. They fail if
a rebuild loses students, if the two independent reports stop agreeing about
which section a child is in, or if the historical "everybody is free" shape
reappears anywhere in the school week.

Skipped when data/school.db has not been built.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tests.conftest import needs_real_db

pytestmark = needs_real_db


@pytest.fixture(autouse=True)
def use_real_db(monkeypatch):
    monkeypatch.setenv("SCHOOL_DB",
                       str(Path(__file__).resolve().parents[1] / "data" / "school.db"))


@pytest.fixture
def conn():
    from tsapp import db
    return db.connect()


# ── coverage of the source reports ──────────────────────────────────────────

def test_every_grade_in_the_zips_is_present(conn):
    """KG1, KG2 and grades 1-12: fourteen class lists, fourteen grades.

    The previous dataset held grades 5-12 only — 1,878 of the school's students.
    """
    codes = {r["code"] for r in conn.execute("SELECT code FROM grades")}
    expected = {"K1", "K2"} | {f"{n:02d}" for n in range(1, 13)}
    assert expected <= codes


def test_student_count_is_plausible_and_complete(conn):
    n = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    assert n > 3400, f"only {n} students; a class list probably failed to parse"
    # Every student belongs to exactly one section, and every section to a grade.
    assert conn.execute("""SELECT COUNT(*) FROM students s
                           LEFT JOIN sections sec ON sec.id = s.section_id
                           WHERE sec.id IS NULL""").fetchone()[0] == 0


def test_no_grade_lost_its_roster(conn):
    """A grade with a class list but no students means a parse failure that the
    build reported as success. Grade 2's header omits its column captions, which
    is exactly how this happened once."""
    empty = conn.execute("""
        SELECT g.code FROM grades g
        LEFT JOIN sections sec ON sec.grade = g.code
        LEFT JOIN students s ON s.section_id = sec.id
        GROUP BY g.code HAVING COUNT(s.computer_number) = 0
    """).fetchall()
    assert [r["code"] for r in empty] == []


def test_enrolment_is_present_for_every_grade(conn):
    """The whole point of the subject lists.

    The previous builder wrote an EMPTY enrolment sheet, and because the app's
    rule was "free unless enrolled in what is being taught", an empty sheet made
    every student in the school report as free in every lesson.
    """
    rows = conn.execute("""
        SELECT g.code, COUNT(e.student_id) AS n
        FROM grades g
        LEFT JOIN sections sec ON sec.grade = g.code
        LEFT JOIN students s ON s.section_id = sec.id
        LEFT JOIN enrolments e ON e.student_id = s.computer_number
        GROUP BY g.code
    """).fetchall()
    for r in rows:
        assert r["n"] > 500, f"grade {r['code']} has only {r['n']} enrolment rows"


def test_almost_every_student_has_an_enrolment(conn):
    total = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    without = conn.execute("""
        SELECT COUNT(*) FROM students s
        WHERE NOT EXISTS (SELECT 1 FROM enrolments e
                          WHERE e.student_id = s.computer_number)
    """).fetchone()[0]
    assert without / total < 0.01, f"{without} of {total} students have no subjects"


def test_independent_source_sections_were_cross_checked(conn):
    meta = dict(conn.execute("SELECT key, value FROM build_meta"))
    assert int(meta["source_section_comparisons"]) > 50000
    assert meta["source_section_comparisons"] == meta["source_section_matches"]
    assert int(meta["source_section_mismatches"]) == 0
    assert int(meta["source_grade_mismatches"]) == 0


def test_no_cross_grade_enrolment_link_can_survive(conn):
    bad = conn.execute("""
        SELECT COUNT(*) FROM enrolments e
        JOIN students st ON st.computer_number = e.student_id
        JOIN sections sec ON sec.id = st.section_id
        JOIN teaching_groups tg ON tg.id = e.group_id
        WHERE sec.grade != tg.grade
    """).fetchone()[0]
    assert bad == 0


def test_every_teaching_group_has_a_teacher_and_a_room(conn):
    assert conn.execute("SELECT COUNT(*) FROM teaching_groups "
                        "WHERE teacher_code IS NULL OR room IS NULL"
                        ).fetchone()[0] == 0


def test_no_subject_name_normalises_to_nothing_by_accident(conn):
    """Subjects whose head is NULL are timetable activities with no subject in
    them ("Periodic"). A real subject losing its head would silently stop
    matching, which is the old Mathematics bug."""
    headless = [r["name"] for r in conn.execute(
        "SELECT name FROM subjects WHERE head IS NULL")]
    for name in headless:
        assert len(name.split()) <= 3, f"{name!r} lost its subject head"


# ── the time grid ───────────────────────────────────────────────────────────

def test_the_same_period_label_is_a_different_hour_in_different_grades(conn):
    """The mismatch the old app joined across. Asserted so that if the school
    ever aligns its bell schedules, this test tells us the workaround can go."""
    g5 = conn.execute("SELECT start_min FROM bell_slots "
                      "WHERE grade='05' AND label='Period 3'").fetchone()
    g9 = conn.execute("SELECT start_min FROM bell_slots "
                      "WHERE grade='09' AND label='Period 3'").fetchone()
    assert g5 and g9
    assert g5["start_min"] != g9["start_min"], (
        "grade 5 and grade 9 'Period 3' now start at the same time")


def test_every_lesson_and_bell_slot_has_a_sane_span(conn):
    for table in ("lessons", "bell_slots"):
        bad = conn.execute(f"SELECT COUNT(*) FROM {table} "
                           f"WHERE end_min <= start_min "
                           f"OR start_min < 6*60 OR end_min > 20*60").fetchone()[0]
        assert bad == 0, f"{bad} rows in {table} have an impossible time span"


def test_no_teacher_is_in_two_places_at_once(conn):
    """A teacher occupying two rooms in one minute would mean the schedule import
    merged two people, and every match through that teacher would be wrong."""
    clashes = conn.execute("""
        SELECT a.teacher_code, a.day, a.start_min, COUNT(*) AS n
        FROM lessons a JOIN lessons b
          ON b.teacher_code = a.teacher_code AND b.day = a.day
         AND b.id > a.id AND a.start_min < b.end_min AND b.start_min < a.end_min
        GROUP BY 1, 2, 3
    """).fetchall()
    assert clashes == [], f"{len(clashes)} teacher double-bookings"


# ── the resolver, on real data ──────────────────────────────────────────────

def test_no_grade_reports_everyone_free_during_a_taught_period(conn):
    """The historical failure, as a standing assertion.

    On the old code this fired constantly: grade 12 Monday "Period 1" (a
    "Periodic" lesson in all six sections) reported 163 of 163 students free;
    grade 5 Monday "Period 3" ("Math/English/2nd Lang" in all ten sections)
    reported 296 of 296; grade 5 Monday "Period 4" reported the 117 students
    sitting in its four Mathematics lessons as free.

    A period where every section is genuinely free — Lunch, After School — is a
    different thing, and is excluded by looking only at periods the timetable
    marks as taught.
    """
    from tsapp.resolver import FREE, Moment, grade_snapshot

    offenders = []
    slots = conn.execute("""
        SELECT DISTINCT g.code AS grade, sp.day, sp.period_label,
               sp.start_min, sp.end_min
        FROM section_periods sp
        JOIN sections sec ON sec.id = sp.section_id
        JOIN grades g ON g.code = sec.grade
        WHERE sp.is_free = 0 AND sp.start_min IS NOT NULL
    """).fetchall()
    assert len(slots) > 200, "too few taught periods to be a real check"

    for s in slots:
        # Only consider moments where NO section of the grade is free, i.e. the
        # whole grade is timetabled.
        any_free = conn.execute("""
            SELECT COUNT(*) FROM section_periods sp
            JOIN sections sec ON sec.id = sp.section_id
            WHERE sec.grade = ? AND sp.day = ? AND sp.is_free = 1
              AND sp.start_min < ? AND ? < sp.end_min
        """, (s["grade"], s["day"], s["end_min"], s["start_min"])).fetchone()[0]
        if any_free:
            continue
        moment = Moment(s["day"], (s["start_min"] + s["end_min"]) // 2)
        counts = grade_snapshot(s["grade"], moment).counts()
        if counts[FREE]:
            offenders.append((s["grade"], s["day"], s["period_label"],
                              counts[FREE]))
    assert offenders == [], (
        f"{len(offenders)} fully-timetabled periods reported free students: "
        f"{offenders[:5]}")


def test_free_students_do_exist_where_the_timetable_says_so(conn):
    """The mirror of the test above: the resolver must not simply never say free."""
    from tsapp.resolver import FREE, Moment, grade_snapshot

    row = conn.execute("""
        SELECT g.code AS grade, sp.day, sp.start_min, sp.end_min
        FROM section_periods sp
        JOIN sections sec ON sec.id = sp.section_id
        JOIN grades g ON g.code = sec.grade
        WHERE sp.is_free = 1 AND sp.start_min IS NOT NULL LIMIT 1
    """).fetchone()
    assert row is not None
    moment = Moment(row["day"], (row["start_min"] + row["end_min"]) // 2)
    assert grade_snapshot(row["grade"], moment).counts()[FREE] > 0


def test_grades_without_a_timetable_are_unknown_not_free(conn):
    from tsapp.resolver import FREE, IN_CLASS, Moment, grade_snapshot

    for r in conn.execute("SELECT grade FROM grade_coverage "
                          "WHERE has_section_timetable = 0"):
        counts = grade_snapshot(r["grade"], Moment("Monday", 9 * 60)).counts()
        assert counts[FREE] == 0 and counts[IN_CLASS] == 0, (
            f"grade {r['grade']} has no timetable but got a free/busy verdict")
        assert counts["unknown"] > 0


def test_no_timetable_always_means_an_insufficient_verdict(conn):
    rows = conn.execute(
        "SELECT grade, verdict FROM grade_coverage "
        "WHERE has_section_timetable = 0").fetchall()
    assert rows
    assert all(r["verdict"] == "insufficient" for r in rows)


def test_coverage_excludes_groups_that_cannot_place_an_individual(conn):
    for row in conn.execute("SELECT grade, resolved_enrolments FROM grade_coverage"):
        expected = conn.execute("""
            SELECT COUNT(*) FROM enrolments e
            JOIN teaching_groups tg ON tg.id = e.group_id
            WHERE tg.grade = ? AND tg.meeting_count > 0
              AND tg.match_quality != 'ambiguous'
        """, (row["grade"],)).fetchone()[0]
        assert row["resolved_enrolments"] == expected


def test_an_individual_lesson_is_never_erased_by_a_free_section_cell(conn):
    """Exercise real split-group conflicts from the supplied reports."""
    from tsapp.resolver import IN_CLASS, Moment, grade_snapshot

    examples = conn.execute("""
        SELECT DISTINCT sec.grade, sp.day, sp.start_min, sp.end_min, e.student_id
        FROM section_periods sp
        JOIN sections sec ON sec.id = sp.section_id
        JOIN students st ON st.section_id = sp.section_id
        JOIN enrolments e ON e.student_id = st.computer_number
        JOIN teaching_groups tg ON tg.id = e.group_id
                              AND tg.match_quality != 'ambiguous'
        JOIN group_meetings gm ON gm.group_id = tg.id AND gm.day = sp.day
             AND gm.start_min < sp.end_min AND sp.start_min < gm.end_min
        WHERE sp.is_free = 1
        LIMIT 10
    """).fetchall()
    assert examples, "expected at least one real split-group source conflict"
    for row in examples:
        minute = (row["start_min"] + row["end_min"]) // 2
        snap = grade_snapshot(
            row["grade"], Moment(row["day"], minute), row["student_id"])
        assert snap.students[0]["status"] == IN_CLASS
        assert snap.students[0].get("source_conflict") is True


def test_teacher_availability_is_never_the_whole_staff(conn):
    """"127 of 128 teachers are free" was the symptom of joining a period label
    against the teacher sheet's own numbering. Nothing joins on labels now."""
    from tsapp.resolver import Moment, teacher_status

    for day in ("Monday", "Wednesday"):
        for minute in (8 * 60 + 30, 10 * 60, 13 * 60 + 20, 15 * 60):
            result = teacher_status(Moment(day, minute))
            free, busy = len(result["free"]), len(result["busy"])
            assert busy > 20, (
                f"{day} {minute // 60}:{minute % 60:02d}: only {busy} teachers "
                f"teaching, {free} free — the period join has broken again")


def test_a_students_own_day_is_not_just_their_sections_day(conn):
    """Two students in one section diverge as soon as it splits by language.

    This is the capability the enrolment data adds, and it is what makes "where
    is this child now" answerable at all.
    """
    from tsapp.resolver import student_day

    row = conn.execute("""
        SELECT s.computer_number, s.section_id FROM students s
        JOIN sections sec ON sec.id = s.section_id
        WHERE sec.grade = '05' LIMIT 1
    """).fetchone()
    day = student_day(row["computer_number"], "Monday")
    assert day["slots"], "no timetable assembled for a grade-5 student"
    named = [s for s in day["slots"] if s["subject"]]
    assert named, "no lesson on this student's day could be identified"
    # At least one lesson names a subject more specific than the section grid's.
    assert any(s["subject"] != s["section_subject"] for s in named)


def test_grade_11_section_l_was_merged_into_la(conn):
    """The roster/timetable naming mismatch, resolved by a declared alias.

    Evidence for the merge: L held 31 students with no timetable while LA held a
    full 44-cell timetable and no students, and six of those students' own
    teachers teach in room 11LA while nobody in the school teaches in 11L.
    Grade 10 is the control — it really does have a section L, and must be
    untouched.
    """
    sections = {r["letter"]: r for r in conn.execute("""
        SELECT letter, student_count,
               (SELECT COUNT(*) FROM section_periods sp
                 WHERE sp.section_id = s.id) AS cells
        FROM sections s WHERE grade = '11'""")}
    assert "L" not in sections, "grade 11 section L should have been merged away"
    assert sections["LA"]["student_count"] == 31
    assert sections["LA"]["cells"] > 0

    # Grade 10's L is a real section and must survive.
    ten = conn.execute("""SELECT student_count,
                            (SELECT COUNT(*) FROM section_periods sp
                              WHERE sp.section_id = s.id) AS cells
                          FROM sections s WHERE grade='10' AND letter='L'""").fetchone()
    assert ten and ten["student_count"] > 0 and ten["cells"] > 0


def test_the_merge_is_recorded_not_silent(conn):
    """A rename that changes 31 children's answers must be visible in the audit."""
    aliases = [r["detail"] for r in conn.execute(
        "SELECT detail FROM build_issues WHERE category = 'section_alias'")]
    assert any("31 students" in d and "LA" in d for d in aliases)


@pytest.mark.parametrize("truncated,real", [("11L", "11LA"), ("09L", "09LA")])
def test_unambiguous_room_truncations_are_merged(conn, truncated, real):
    """A group whose room appears in no lesson can never match by room.

    `11L` and `09L` were such names: the schedule holds lessons in 11LA and 09LA
    and none in either short form, so those groups could only match on subject —
    which cannot tell one of a teacher's same-subject groups from another. On the
    real data this left `Mathematics L` claiming 29 weekly meetings instead of 5.
    """
    assert conn.execute("SELECT COUNT(*) FROM teaching_groups WHERE room = ?",
                        (truncated,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM lesson_rooms WHERE room = ?",
                        (real,)).fetchone()[0] > 0


def test_ambiguous_room_truncations_are_never_guessed(conn):
    """`12S` could be 12SA..12SE. Picking one would place children in a room they
    may not be in.

    The invariant is that no such name is ever REWRITTEN to one of its candidates.
    It may still be resolved — `12S` groups are placed by their course group,
    which is exact information rather than a guess — but the stored room keeps
    the name the report actually printed.
    """
    truncated = conn.execute("""
        SELECT tg.room,
               (SELECT COUNT(*) FROM (SELECT DISTINCT room FROM lesson_rooms
                                      WHERE room LIKE tg.room || '%')) AS candidates
        FROM teaching_groups tg
        WHERE tg.room IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr WHERE lr.room = tg.room)
        GROUP BY tg.room
    """).fetchall()
    ambiguous = [r["room"] for r in truncated if r["candidates"] > 1]
    assert ambiguous, "expected some genuinely ambiguous room names in this data"
    for room in ambiguous:
        # Still stored under its printed name, not silently expanded.
        assert conn.execute("SELECT COUNT(*) FROM teaching_groups WHERE room = ?",
                            (room,)).fetchone()[0] > 0

    # Whatever is left genuinely unplaced must be reported, not swallowed.
    unplaced = conn.execute("""
        SELECT DISTINCT tg.room FROM teaching_groups tg
        WHERE tg.room IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr WHERE lr.room = tg.room)
          AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr
                          WHERE lr.room = tg.grade || UPPER(tg.course_group))
    """).fetchall()
    reported = " ".join(r["detail"] for r in conn.execute(
        "SELECT detail FROM build_issues WHERE category IN "
        "('room_ambiguous', 'room_unknown', 'room_alias_available')"))
    for r in unplaced:
        assert r["room"] in reported, f"{r['room']} is unplaced but not reported"


def test_course_group_recovers_the_schedules_room_name(conn):
    """The grade-9 room mismatch.

    The subject lists print the physical room (09A); the teacher schedule's
    "Room/Class" column holds the section (09SA). Comparing them directly left 88
    of grade 9's 156 groups matchable only by subject, and therefore
    indistinguishable from the same teacher's other groups of that subject.
    `grade + course_group` reconstructs the schedule's own string.
    """
    placed = conn.execute("""
        SELECT COUNT(*) FROM teaching_groups tg
        WHERE tg.grade = '09'
          AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr WHERE lr.room = tg.room)
          AND EXISTS (SELECT 1 FROM lesson_rooms lr
                      WHERE lr.room = tg.grade || UPPER(tg.course_group))
    """).fetchone()[0]
    assert placed > 60, (
        f"only {placed} grade-9 groups recovered via course group; the room "
        f"reconciliation has regressed")

    # And the effect: grade 9 must no longer be dominated by ambiguous groups.
    quality = dict(conn.execute("""
        SELECT match_quality, COUNT(*) FROM teaching_groups
        WHERE grade = '09' GROUP BY match_quality"""))
    assert quality.get("exact", 0) > quality.get("ambiguous", 0), (
        f"grade 9 match quality regressed: {quality}")


def test_the_build_offers_the_next_alias_instead_of_hiding_it(conn):
    """Any remaining room with exactly one possible expansion is flagged with the
    command to fix it, so the next `09L` does not have to be hunted for."""
    rows = conn.execute("""
        SELECT tg.room FROM teaching_groups tg
        WHERE tg.room IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr WHERE lr.room = tg.room)
          AND (SELECT COUNT(*) FROM (SELECT DISTINCT room FROM lesson_rooms
                                     WHERE room LIKE tg.room || '%')) = 1
        GROUP BY tg.room""").fetchall()
    flagged = " ".join(r["detail"] for r in conn.execute(
        "SELECT detail FROM build_issues WHERE category = 'room_alias_available'"))
    for r in rows:
        assert r["room"] in flagged, (
            f"room {r['room']} has exactly one expansion but was not flagged")


def test_every_grade_11_student_now_gets_a_verdict(conn):
    """Before the merge, 31 of 173 were 'unknown' in every period of the week."""
    from tsapp.resolver import UNKNOWN, Moment, grade_snapshot

    for minute in (8 * 60 + 25, 14 * 60 + 10, 15 * 60 + 45):
        counts = grade_snapshot("11", Moment("Monday", minute)).counts()
        assert counts[UNKNOWN] == 0, (
            f"grade 11 still has {counts[UNKNOWN]} unplaceable students at "
            f"{minute // 60}:{minute % 60:02d}")
        assert counts["free"] + counts["in_class"] == 173


def test_build_recorded_what_it_could_not_do(conn):
    """A build with no issues at all would mean nobody is looking."""
    assert conn.execute("SELECT COUNT(*) FROM build_issues").fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM grade_coverage").fetchone()[0] >= 14
