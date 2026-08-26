"""
Fixtures.

`synthetic_db` builds a miniature but complete school in a temporary database:
one grade with a timetable, one without, sections that are taught and sections
that are free, and a student whose group cannot be tied to any lesson. That last
case is the one the old app got wrong — it is the shape that must come out as
"in class, group unidentified", never as "free".
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_DB = ROOT / "data" / "school.db"


@pytest.fixture
def synthetic_db(tmp_path, monkeypatch):
    path = tmp_path / "mini.db"
    conn = sqlite3.connect(path)
    conn.executescript((ROOT / "etl" / "schema.sql").read_text())

    conn.executemany("INSERT INTO grades (code, ordinal, label) VALUES (?,?,?)", [
        ("05", 5, "Grade 5"),      # has a timetable
        ("K1", -1, "KG1"),         # roster only, no timetable
    ])
    conn.executemany("INSERT INTO sections (id, grade, letter) VALUES (?,?,?)", [
        (1, "05", "A"),            # taught at 08:00, free at 09:00
        (2, "05", "B"),            # taught at 08:00 by an unmatchable group
        (3, "K1", "A"),            # no timetable at all
    ])
    conn.executemany("INSERT INTO teachers (code, title, name, is_placeholder) "
                     "VALUES (?,?,?,?)", [
        ("MTA1", "Ms.", "Maths Teacher", 0),
        ("ARB1", "Mr.", "Arabic Teacher", 0),
        ("SSTUDY", "Ms.", "Self Study", 1),   # placeholder, never counted
        ("GHOST", None, "Absent From Schedule", 0),
    ])
    conn.executemany("INSERT INTO subjects (code, name, head) VALUES (?,?,?)", [
        ("MTHR1", "Mathematics N2", "mathematics"),
        ("ARHL2", "Arabic Language- Arabs", "arabic"),
        ("XXXX1", "Unschedulable Studies", "unschedulable"),
    ])
    conn.executemany(
        "INSERT INTO students (computer_number, first_name, last_name, "
        "section_id, source) VALUES (?,?,?,?,?)", [
            ("1001", "Amal", "Alpha", 1, "t"),    # 05-A, matched group
            ("1002", "Bilal", "Beta", 1, "t"),    # 05-A, matched group
            ("1003", "Cara", "Gamma", 2, "t"),    # 05-B, UNMATCHABLE group
            ("1004", "Dana", "Delta", 3, "t"),    # KG1, no timetable
        ])
    conn.execute("""UPDATE sections SET student_count =
                    (SELECT COUNT(*) FROM students WHERE section_id = sections.id)""")

    # Group 1 is properly tied to a lesson; group 2 is not tied to anything.
    conn.executemany(
        "INSERT INTO teaching_groups (id, grade, subject_code, course_group, room, "
        "teacher_code, match_quality, meeting_count) VALUES (?,?,?,?,?,?,?,?)", [
            (1, "05", "MTHR1", "A", "05A", "MTA1", "exact", 1),
            (2, "05", "XXXX1", "B", "05B", "GHOST", "none", 0),
        ])
    conn.executemany("INSERT INTO enrolments (student_id, group_id) VALUES (?,?)",
                     [("1001", 1), ("1002", 1), ("1003", 2)])

    conn.execute("INSERT INTO lessons (id, teacher_code, day, start_min, end_min, "
                 "subject_text, room_text) VALUES (1,'MTA1','Monday',480,530,"
                 "'Mathematics','05A')")
    conn.execute("INSERT INTO lesson_rooms (lesson_id, room) VALUES (1,'05A')")
    conn.execute("INSERT INTO lessons (id, teacher_code, day, start_min, end_min, "
                 "subject_text, room_text) VALUES (2,'ARB1','Monday',540,590,"
                 "'Arabic Language','05C')")
    conn.execute("INSERT INTO lesson_rooms (lesson_id, room) VALUES (2,'05C')")
    conn.execute("INSERT INTO group_meetings (group_id, lesson_id, day, start_min, "
                 "end_min, score) VALUES (1,1,'Monday',480,530,4)")

    conn.executemany(
        "INSERT INTO section_periods (section_id, day, period_label, start_min, "
        "end_min, subject_text, teacher_code, is_free) VALUES (?,?,?,?,?,?,?,?)", [
            (1, "Monday", "Period 1", 480, 530, "Mathematics", "MTA1", 0),
            (1, "Monday", "Period 2", 540, 590, None, None, 1),      # free
            (2, "Monday", "Period 1", 480, 530, "Periodic", None, 0),  # taught
            (2, "Monday", "Period 2", 540, 590, None, None, 1),      # free
        ])
    conn.executemany("INSERT INTO bell_slots (grade, label, start_min, end_min) "
                     "VALUES (?,?,?,?)", [
        ("05", "Period 1", 480, 530),
        ("05", "Period 2", 540, 590),
    ])
    conn.executemany(
        "INSERT INTO grade_coverage (grade, students, enrolments, "
        "resolved_enrolments, coverage, has_section_timetable, verdict) "
        "VALUES (?,?,?,?,?,?,?)", [
            ("05", 3, 3, 2, 0.667, 1, "partial"),
            ("K1", 1, 0, 0, 0.0, 0, "insufficient"),
        ])
    conn.executemany("INSERT INTO build_meta (key, value) VALUES (?,?)",
                     [("students", "4"), ("lessons", "2"), ("enrolment_rows", "3")])
    conn.commit()
    conn.close()

    monkeypatch.setenv("SCHOOL_DB", str(path))
    return path


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("SCHOOL_STAFF_KEY", "staff-secret")
    monkeypatch.setenv("SCHOOL_ROBOT_KEY", "robot-secret")
    return {"staff": "staff-secret", "robot": "robot-secret"}


@pytest.fixture
def client(synthetic_db, keys):
    from tsapp import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


needs_real_db = pytest.mark.skipif(
    not REAL_DB.exists(),
    reason="data/school.db not built; run python3 -m etl.build")
