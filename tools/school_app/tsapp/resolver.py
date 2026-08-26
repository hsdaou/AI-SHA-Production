"""
Who is where, at a given moment.

THE INVERSION AT THE HEART OF THIS MODULE
-----------------------------------------
The previous app's rule was:

    a student is FREE if their section has no lesson
                     OR the student is not enrolled in the subject being taught

so anything it could not establish came out as "free". Every way of failing —
an empty enrolment sheet, a subject name that normalised to the empty string, a
combined cell like "Math/English/2nd Lang" that matches no single subject name —
produced the same confident answer: this child is available. That default is
backwards for a school. "I could not tell" and "this child is unsupervised and
free to be collected" are not the same sentence, and only one of them is safe to
say out loud in a corridor.

This module never defaults to free. Status is one of three values:

    IN_CLASS   the section is timetabled at this moment
    FREE       the section's own timetable says this period is free AND no
               matched individual group lesson overrides that coarse cell
    UNKNOWN    there is no schedule covering this student, so no claim is made

FREE comes only from a positive statement in the timetable. UNKNOWN is a real
answer that callers must handle, not an error.

WHAT THE ENROLMENT DATA ADDS
----------------------------
The section grid says whether a section is taught; it cannot say which of the
parallel groups a particular student sits in, because a single cell covers a
section that splits by language and ability. Enrolment plus the teacher schedule
supplies that: student -> group -> (teacher, room) -> lesson at this hour. That
is what makes "where is this student now" answerable at all, and it is reported
with the confidence it was actually established at, never upgraded silently.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from etl.subjects import compatible, subject_head

from . import db

IN_CLASS = "in_class"
FREE = "free"
UNKNOWN = "unknown"

# A lesson match corroborated by both room and subject.
STRONG_SCORE = 4


def school_tz() -> ZoneInfo:
    import os
    return ZoneInfo(os.environ.get("SCHOOL_TZ", "Asia/Dubai"))


@dataclass
class Moment:
    """A day and a wall-clock minute — the only schedule coordinate that is
    shared by all three source reports. See etl/timegrid.py for why period
    labels are not usable as one."""
    day: str
    minute: int
    source: str = "explicit"

    @property
    def hhmm(self) -> str:
        return f"{self.minute // 60:02d}:{self.minute % 60:02d}"

    def as_dict(self) -> dict:
        return {"day": self.day, "time": self.hhmm, "minute": self.minute,
                "source": self.source}


def now_moment() -> Moment:
    t = datetime.now(school_tz())
    return Moment(t.strftime("%A"), t.hour * 60 + t.minute, source="clock")


def resolve_label(grade: str, label: str) -> Moment | None:
    """A period label within a grade -> the middle of that period.

    Grade-scoped on purpose. "Period 3" is 10:00-10:50 in grade 5 and
    09:40-10:30 in grade 9; resolving it without knowing the grade is how the
    old app compared one grade's students against another grade's hour.
    """
    row = db.one("SELECT start_min, end_min FROM bell_slots "
                 "WHERE grade = ? AND label = ? COLLATE NOCASE",
                 (grade, label.strip()))
    if not row:
        return None
    return Moment("", (row["start_min"] + row["end_min"]) // 2, source="label")


def periods_for_grade(grade: str) -> list[dict]:
    return [{"label": r["label"], "start": f"{r['start_min']//60:02d}:{r['start_min']%60:02d}",
             "end": f"{r['end_min']//60:02d}:{r['end_min']%60:02d}",
             "start_min": r["start_min"], "end_min": r["end_min"]}
            for r in db.query("SELECT label, start_min, end_min FROM bell_slots "
                              "WHERE grade = ? ORDER BY start_min", (grade,))]


def current_period(grade: str, moment: Moment) -> dict | None:
    for p in periods_for_grade(grade):
        if p["start_min"] <= moment.minute < p["end_min"]:
            return p
    return None


# ── lesson identification ───────────────────────────────────────────────────

def _lessons_by_student(grade: str, moment: Moment,
                        computer_number: str | None = None
                        ) -> dict[str, list[dict]]:
    """For every student of a grade, the lessons of theirs that are running now.

    One indexed query for the whole grade. The old app issued a fresh SQLite
    connection per section and then compared strings in Python.
    """
    sql = """
        SELECT e.student_id,
               sub.name       AS subject,
               tg.room        AS room,
               tg.teacher_code,
               t.title, t.name AS teacher_name,
               tg.match_quality,
               gm.score
        FROM sections sec
        JOIN students  st ON st.section_id = sec.id
        JOIN enrolments e ON e.student_id  = st.computer_number
        JOIN teaching_groups tg ON tg.id = e.group_id
        JOIN group_meetings  gm ON gm.group_id = tg.id
        JOIN subjects sub ON sub.code = tg.subject_code
        LEFT JOIN teachers t ON t.code = tg.teacher_code
        WHERE sec.grade = ?
          AND gm.day = ? AND gm.start_min <= ? AND ? < gm.end_min
          AND tg.match_quality != 'ambiguous'
    """
    params: tuple = (grade, moment.day, moment.minute, moment.minute)
    if computer_number is not None:
        sql += " AND st.computer_number = ?"
        params += (computer_number,)
    rows = db.query(sql, params)

    out: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        out[r["student_id"]].append(dict(r))
    return out


def _one_place(candidates: list[dict]) -> bool:
    """Do these all describe a single teacher in a single room?

    A student is enrolled in more than one record of the same class more often
    than one might expect — "Mathematics L1" and "Ace the IGCSE Exam - Math" are
    the same period with the same teacher in the same room, registered twice.
    Across a sample of the week, 1,174 of 1,213 multi-match students had every
    strong candidate pointing at ONE teacher; only 39 pointed at two.

    When they agree on teacher and room, where the child is is not in doubt —
    only which label to print. Treating that as low confidence understates what
    is known, and "I am not sure where this student is" is a much more alarming
    sentence than "they are with Ms Yehia in 09B".
    """
    return (bool(candidates)
            and all(c.get("teacher_code") and c.get("room") for c in candidates)
            and len({c["teacher_code"] for c in candidates}) == 1
            and len({c["room"] for c in candidates}) == 1)


def _most_specific(candidates: list[dict], cell_subject: str | None) -> dict:
    """Of several records of one lesson, the label most worth printing.

    Prefers the record whose subject HEAD is the section timetable's subject.
    Mere compatibility is too weak to rank on: "Ace the IGCSE Exam - Math" is
    compatible with "Mathematics" (its token set contains it) but names the
    revision course rather than the lesson, and is a worse answer to "what is
    this student in right now".
    """
    if cell_subject:
        head = subject_head(cell_subject)
        same_head = [c for c in candidates if subject_head(c["subject"]) == head]
        if same_head:
            candidates = same_head
        else:
            agree = [c for c in candidates if compatible(c["subject"], cell_subject)]
            if agree:
                candidates = agree
    # Deterministic: strongest corroboration, then alphabetical.
    return sorted(candidates, key=lambda c: (-c["score"], c["subject"] or ""))[0]


def _pick_lesson(candidates: list[dict], cell_subject: str | None) -> tuple[dict | None, str]:
    """Choose which of a student's concurrent lessons this actually is.

    Returns (lesson, confidence). A student can only be in one place at a time,
    so more than one surviving candidate means either several records of the same
    lesson (resolvable — see _one_place) or a genuine matching failure (not
    resolvable, and reported as such rather than by taking the first).
    """
    if not candidates:
        return None, "none"

    strong = [c for c in candidates
              if c["score"] >= STRONG_SCORE and c["match_quality"] != "ambiguous"]
    if len(strong) == 1:
        return strong[0], "high"
    if strong:
        if _one_place(strong):
            return _most_specific(strong, cell_subject), "high"
        if cell_subject:
            agree = [c for c in strong if compatible(c["subject"], cell_subject)]
            if len(agree) == 1:
                return agree[0], "high"
            if agree and _one_place(agree):
                return _most_specific(agree, cell_subject), "high"
        # Two places claim this student at once.  Choosing one deterministically
        # would be repeatable, but still false precision; publish no teacher or
        # room for a conflict.
        return None, "low"

    usable = [c for c in candidates if c["match_quality"] != "ambiguous"]
    if len(usable) == 1:
        return usable[0], "medium"
    if usable and _one_place(usable):
        return _most_specific(usable, cell_subject), "medium"
    if cell_subject:
        agree = [c for c in (usable or candidates)
                 if compatible(c["subject"], cell_subject)]
        if len(agree) == 1:
            return agree[0], "medium"
        if agree and _one_place(agree):
            return _most_specific(agree, cell_subject), "medium"
    # Ambiguous groups deliberately remain in the database because they prove a
    # teacher is busy, but they cannot place an individual. Likewise, several
    # usable candidates in different places are a conflict, not a low-confidence
    # licence to pick one.
    return None, "low"


# ── the section grid ────────────────────────────────────────────────────────

def section_cells(grade: str, moment: Moment) -> dict[int, dict]:
    """section_id -> the timetable cell covering this minute (if any)."""
    rows = db.query("""
        SELECT sp.section_id, sp.period_label, sp.subject_text, sp.teacher_code,
               sp.is_free, sp.start_min, sp.end_min,
               t.title, t.name AS teacher_name
        FROM section_periods sp
        JOIN sections sec ON sec.id = sp.section_id
        LEFT JOIN teachers t ON t.code = sp.teacher_code
        WHERE sec.grade = ?
          AND sp.day = ? AND sp.start_min <= ? AND ? < sp.end_min
    """, (grade, moment.day, moment.minute, moment.minute))
    out: dict[int, dict] = {}
    for row in rows:
        current = dict(row)
        current["conflict"] = False
        section_id = row["section_id"]
        previous = out.get(section_id)
        if previous is None:
            out[section_id] = current
            continue
        # Do not let database row order choose one of two overlapping section
        # cells. Preserve only what is safe to conclude from all of them.
        out[section_id] = {
            "section_id": section_id,
            "period_label": "conflicting timetable cells",
            "subject_text": None,
            "teacher_code": None,
            "is_free": int(bool(previous["is_free"] and current["is_free"])),
            "start_min": min(previous["start_min"], current["start_min"]),
            "end_min": max(previous["end_min"], current["end_min"]),
            "title": None,
            "teacher_name": None,
            "conflict": True,
        }
    return out


@dataclass
class GradeSnapshot:
    grade: str
    moment: Moment
    verdict: str                    # good | partial | insufficient
    has_timetable: bool
    coverage: float
    students: list[dict] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def counts(self) -> dict:
        c = {IN_CLASS: 0, FREE: 0, UNKNOWN: 0}
        for s in self.students:
            c[s["status"]] += 1
        return c

    def by_status(self, status: str) -> list[dict]:
        return [s for s in self.students if s["status"] == status]


def grade_snapshot(grade: str, moment: Moment,
                   computer_number: str | None = None) -> GradeSnapshot:
    """Students of a grade, with a status and the reason for it.

    ``computer_number`` is the efficient single-student path used by
    :func:`student_now`; the public grade endpoints leave it unset.
    """
    cov = db.coverage(grade)
    snap = GradeSnapshot(
        grade=grade, moment=moment,
        verdict=cov["verdict"] if cov else "insufficient",
        has_timetable=bool(cov and cov["has_section_timetable"]),
        coverage=cov["coverage"] if cov else 0.0)

    student_sql = """
        SELECT st.computer_number, st.first_name, st.last_name,
               st.section_id, sec.letter AS section
        FROM students st JOIN sections sec ON sec.id = st.section_id
        WHERE sec.grade = ?
    """
    student_params: tuple = (grade,)
    if computer_number is not None:
        student_sql += " AND st.computer_number = ?"
        student_params += (computer_number,)
    student_sql += " ORDER BY st.last_name, st.first_name"
    students = db.query(student_sql, student_params)
    if not students:
        snap.caveats.append(f"No students are on file for grade {grade}.")
        return snap

    if not snap.has_timetable:
        # Grades KG1, KG2 and 1 have a roster and enrolment but no timetable in
        # any supplied source. Saying "0 free" here would be read as "nobody is
        # free", which is a claim; there is no basis for any claim.
        snap.caveats.append(
            f"No class timetable covers grade {grade}, so whether these "
            f"{len(students)} students are in a lesson is not known.")
        snap.students = [
            {"computer_number": s["computer_number"], "first_name": s["first_name"],
             "last_name": s["last_name"], "section": s["section"],
             "status": UNKNOWN, "confidence": "high",
             "reason": "no timetable for this grade",
             "subject": None, "teacher": None, "room": None}
            for s in students]
        return snap

    cells = section_cells(grade, moment)
    lessons = _lessons_by_student(grade, moment, computer_number)
    # Which of this grade's sections have a timetable AT ALL. A section with no
    # grid is not free — nothing is known about it. Grade 11's roster calls a
    # section "L" where the timetable calls it "LA", which left 31 real students
    # with no grid rows; without this distinction they read as free all week.
    timetabled = {r["section_id"] for r in db.query("""
        SELECT DISTINCT sp.section_id FROM section_periods sp
        JOIN sections sec ON sec.id = sp.section_id WHERE sec.grade = ?""", (grade,))}
    unplaced = 0
    source_conflicts = 0
    grid_conflicts = 0
    ungridded = set()

    for s in students:
        cell = cells.get(s["section_id"])
        rec = {"computer_number": s["computer_number"],
               "first_name": s["first_name"], "last_name": s["last_name"],
               "section": s["section"], "subject": None, "teacher": None,
               "room": None, "period": cell["period_label"] if cell else None}

        if s["section_id"] not in timetabled:
            ungridded.add(s["section"])
            rec.update(status=UNKNOWN, confidence="high",
                       reason=f"no timetable exists for section {s['section']}")
        else:
            if cell and cell.get("conflict"):
                grid_conflicts += 1
            lesson, conf = _pick_lesson(lessons.get(s["computer_number"], []),
                                        cell["subject_text"] if cell else None)
            if lesson:
                teacher = " ".join(x for x in (lesson["title"], lesson["teacher_name"])
                                   if x) or lesson["teacher_code"]
                conflict = cell is None or bool(cell["is_free"])
                if conflict:
                    source_conflicts += 1
                rec.update(status=IN_CLASS, confidence=conf,
                           reason=(
                               f"in {lesson['subject']} (the student's own "
                               f"schedule overrides the section-level free cell)"
                               if conflict else f"in {lesson['subject']}"),
                           subject=lesson["subject"], teacher=teacher,
                           room=lesson["room"], source_conflict=conflict)
            elif cell is None:
                rec.update(status=FREE, confidence="high",
                           reason="no period is timetabled at this time")
            elif cell["is_free"]:
                rec.update(status=FREE, confidence="high",
                           reason=f"{cell['period_label']} is free for this section")
            else:
                # The section is timetabled, so the student is in a lesson; we
                # just cannot say which one. Reporting that as "free" is exactly
                # the old bug, so it is reported as what it is.
                unplaced += 1
                teacher = " ".join(x for x in (cell["title"], cell["teacher_name"])
                                   if x) or cell["teacher_code"]
                rec.update(status=IN_CLASS, confidence="low",
                           reason="section is timetabled; this student's own "
                                  "group could not be identified",
                           subject=cell["subject_text"], teacher=teacher)
        snap.students.append(rec)

    if ungridded:
        many = len(ungridded) > 1
        snap.caveats.append(
            f"Section{'s' if many else ''} {', '.join(sorted(ungridded))} of grade "
            f"{grade} {'appear' if many else 'appears'} on the class list but in no "
            f"timetable, so {'those' if many else 'its'} students are reported as "
            f"unknown rather than free. The timetable may name "
            f"{'them' if many else 'it'} differently.")
    if unplaced:
        snap.caveats.append(
            f"{unplaced} of {len(students)} students are counted as in class "
            f"because their section is timetabled, but their specific group could "
            f"not be identified from the enrolment data.")
    if source_conflicts:
        snap.caveats.append(
            f"{source_conflicts} of {len(students)} students have a matched "
            f"individual lesson while the section grid is free or has no cell at "
            f"that time. The more specific individual schedule is used, so they "
            f"are counted in class rather than free.")
    if grid_conflicts:
        snap.caveats.append(
            f"{grid_conflicts} of {len(students)} students belong to a section "
            f"with overlapping timetable cells at this time. No arbitrary cell "
            f"was selected; individual lessons are used where available.")
    if snap.verdict != "good":
        snap.caveats.append(
            f"Only {snap.coverage:.0%} of grade {grade}'s enrolment could be tied "
            f"to a timetabled lesson, so which lesson each student is in is less "
            f"certain than usual.")
    return snap


# ── students ────────────────────────────────────────────────────────────────

def find_students(term: str, limit: int = 25) -> list[dict]:
    # '%' and '_' are user text, not an invitation to turn a two-character
    # search into an unbounded wildcard query.
    escaped = (term.strip().replace("!", "!!")
               .replace("%", "!%").replace("_", "!_"))
    like = f"%{escaped}%"
    limit = max(1, min(int(limit), 100))
    rows = db.query("""
        SELECT st.computer_number, st.first_name, st.last_name,
               sec.grade, sec.letter AS section
        FROM students st JOIN sections sec ON sec.id = st.section_id
        WHERE st.computer_number LIKE ? ESCAPE '!'
           OR st.last_name  LIKE ? ESCAPE '!'
           OR st.first_name LIKE ? ESCAPE '!'
        ORDER BY st.last_name, st.first_name
        LIMIT ?
    """, (like, like, like, limit))
    return [dict(r) for r in rows]


def student_now(computer_number: str, moment: Moment) -> dict | None:
    """Where one student is at one moment — a question the old app could not ask."""
    s = db.one("""SELECT st.computer_number, st.first_name, st.last_name,
                         st.section_id, sec.grade, sec.letter AS section
                  FROM students st JOIN sections sec ON sec.id = st.section_id
                  WHERE st.computer_number = ?""", (computer_number,))
    if not s:
        return None
    snap = grade_snapshot(s["grade"], moment, computer_number)
    rec = next((r for r in snap.students
                if r["computer_number"] == computer_number), None)
    if rec is None:
        return None
    return {**rec, "grade": s["grade"], "moment": moment.as_dict(),
            "caveats": snap.caveats}


def student_day(computer_number: str, day: str) -> dict | None:
    """A student's own timetable for a day, assembled from their enrolment.

    Not available from the section grid alone: two students in the same section
    have different days as soon as the section splits by language or ability.
    """
    s = db.one("""SELECT st.computer_number, st.first_name, st.last_name,
                         sec.grade, sec.letter AS section, st.section_id
                  FROM students st JOIN sections sec ON sec.id = st.section_id
                  WHERE st.computer_number = ?""", (computer_number,))
    if not s:
        return None

    own = db.query("""
        SELECT gm.start_min, gm.end_min, sub.name AS subject, tg.room,
               tg.teacher_code, t.title, t.name AS teacher_name,
               gm.score, tg.match_quality
        FROM enrolments e
        JOIN teaching_groups tg ON tg.id = e.group_id
        JOIN group_meetings  gm ON gm.group_id = tg.id
        JOIN subjects sub ON sub.code = tg.subject_code
        LEFT JOIN teachers t ON t.code = tg.teacher_code
        WHERE e.student_id = ? AND gm.day = ?
          AND tg.match_quality != 'ambiguous'
        ORDER BY gm.start_min
    """, (computer_number, day))

    grid = db.query("""
        SELECT period_label, start_min, end_min, subject_text, is_free
        FROM section_periods WHERE section_id = ? AND day = ?
        ORDER BY start_min
    """, (s["section_id"], day))

    slots = []
    for g in grid:
        mid = (g["start_min"] + g["end_min"]) // 2
        mine = [o for o in own if o["start_min"] <= mid < o["end_min"]]
        lesson, conf = _pick_lesson([dict(o) for o in mine], g["subject_text"])
        source_conflict = bool(lesson and g["is_free"])
        slots.append({
            "period": g["period_label"],
            "start": f"{g['start_min']//60:02d}:{g['start_min']%60:02d}",
            "end": f"{g['end_min']//60:02d}:{g['end_min']%60:02d}",
            "section_subject": g["subject_text"],
            # A student's matched group is more specific than the section grid.
            # A free section cell cannot make that individual lesson disappear.
            "status": IN_CLASS if lesson else (FREE if g["is_free"] else IN_CLASS),
            "subject": lesson["subject"] if lesson else None,
            "teacher": (" ".join(x for x in (lesson["title"], lesson["teacher_name"])
                                 if x) or lesson["teacher_code"]) if lesson else None,
            "room": lesson["room"] if lesson else None,
            "confidence": conf if lesson else ("high" if g["is_free"] else "low"),
            "source_conflict": source_conflict,
        })
    return {"student": dict(s), "day": day, "slots": slots,
            "has_timetable": bool(grid)}


# ── sections and teachers ───────────────────────────────────────────────────

def section_status(grade: str, moment: Moment) -> list[dict]:
    cells = section_cells(grade, moment)
    rows = db.query("""
        SELECT s.id, s.letter, s.student_count,
               (SELECT COUNT(*) FROM section_periods sp
                 WHERE sp.section_id = s.id) AS grid_cells
        FROM sections s WHERE s.grade = ? ORDER BY s.letter
    """, (grade,))
    out = []
    for r in rows:
        cell = cells.get(r["id"])
        teacher = None
        if cell:
            teacher = " ".join(x for x in (cell["title"], cell["teacher_name"])
                               if x) or cell["teacher_code"]
        if not r["grid_cells"]:
            status = UNKNOWN          # this section has no timetable at all
        elif cell is None or cell["is_free"]:
            status = FREE
        else:
            status = IN_CLASS
        out.append({
            "section": r["letter"], "student_count": r["student_count"],
            "status": status,
            "period": cell["period_label"] if cell else None,
            "subject": cell["subject_text"] if cell else None,
            "teacher": teacher,
            "conflict": bool(cell and cell.get("conflict")),
        })
    return out


def teacher_status(moment: Moment) -> dict:
    """Which teachers are teaching at this minute, and which are not.

    Joined on wall-clock minutes. The old implementation compared a period
    LABEL against the teacher sheet's own period numbering, which denotes
    different hours — and for "Period 6", which the teacher sheet does not have
    at all, it matched nothing and reported 127 of 128 teachers free.
    """
    # One query, one row per current lesson.  Keeping the current rows separate
    # matters: MAX(subject) and MAX(room) calculated independently can fabricate
    # a subject/room pair that appears in no source row when bad input double-
    # books a teacher. The build flags that defect, and the runtime still fails
    # safe if such a database reaches it.
    rows = db.query("""
        WITH weekly AS (
            SELECT teacher_code, COUNT(*) AS week_lessons
            FROM lessons GROUP BY teacher_code
        )
        SELECT t.code, t.title, t.name,
               COALESCE(w.week_lessons, 0) AS week_lessons,
               current_l.id AS lesson_id,
               current_l.subject_text AS now_subject,
               current_l.room_text AS now_room
        FROM teachers t
        LEFT JOIN weekly w ON w.teacher_code = t.code
        LEFT JOIN lessons current_l ON current_l.teacher_code = t.code
             AND current_l.day = ?
             AND current_l.start_min <= ? AND ? < current_l.end_min
        WHERE t.is_placeholder = 0
        ORDER BY t.name, t.code
    """, (moment.day, moment.minute, moment.minute))

    teachers: dict[str, dict] = {}
    for r in rows:
        rec = teachers.setdefault(r["code"], {
            "code": r["code"],
            "name": " ".join(x for x in (r["title"], r["name"]) if x)
                    or r["code"],
            "week_lessons": r["week_lessons"],
            "current": [],
        })
        if r["lesson_id"] is not None:
            rec["current"].append({"subject": r["now_subject"],
                                   "room": r["now_room"]})

    free_list, busy_list, unscheduled, conflicts = [], [], 0, 0
    for rec in teachers.values():
        current = rec.pop("current")
        if not rec["week_lessons"]:
            # Not free — absent from the schedule source. A different fact, and
            # counting them as free is how "127 of 128 teachers are free" happens.
            unscheduled += 1
        elif len(current) == 1:
            busy_list.append({**rec, **current[0], "conflict": False})
        elif current:
            conflicts += 1
            busy_list.append({**rec, "subject": None, "room": None,
                              "conflict": True, "lessons": current})
        else:
            free_list.append(rec)

    return {"free": free_list, "busy": busy_list,
            "not_in_schedule": unscheduled,
            "conflicts": conflicts,
            "moment": moment.as_dict()}
