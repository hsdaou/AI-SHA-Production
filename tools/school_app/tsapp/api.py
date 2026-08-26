"""
The staff-facing HTTP API.

Every endpoint that can be asked "when?" accepts either an explicit
`day` + `time` (HH:MM), or `day` + `period` resolved against THAT GRADE's bell
schedule, or nothing at all, meaning now. Whichever is used, the answer states
which moment it actually resolved to, because "Period 3" is a different hour in
grade 5 than in grade 9 and a caller comparing two grades needs to see that.

Responses carry `caveats` when the underlying data does not fully support the
answer. They are part of the answer, not decoration: a count with an unstated
30% margin is worse than no count.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from . import db
from .auth import SCOPE_ROBOT, SCOPE_STAFF, require, scope
from .resolver import (FREE, IN_CLASS, UNKNOWN, Moment, current_period,
                       find_students, grade_snapshot, now_moment,
                       periods_for_grade, section_status, student_day,
                       student_now, teacher_status)
from etl.timegrid import SCHOOL_DAYS, parse_hhmm

api = Blueprint("api", __name__, url_prefix="/api")

MAX_PAGE = 500


class BadRequest(Exception):
    def __init__(self, detail: str):
        self.detail = detail


@api.errorhandler(BadRequest)
def _bad_request(e):
    return jsonify({"error": "bad_request", "detail": e.detail}), 400


@api.errorhandler(db.DatabaseMissing)
def _no_db(e):
    return jsonify({"error": "no_data", "detail": str(e)}), 503


def resolve_moment(grade: str | None = None) -> Moment:
    """Turn the request's day/time/period arguments into one wall-clock moment."""
    day_arg = request.args.get("day")
    day = normalise_day(day_arg) if day_arg else None
    time_arg = request.args.get("time")
    period = request.args.get("period")

    if time_arg:
        minute = parse_hhmm(time_arg)
        if minute is None:
            raise BadRequest("time must be HH:MM in 24-hour form, e.g. 13:20")
        return Moment(day or now_moment().day, minute, source="explicit time")

    if period:
        if not grade:
            raise BadRequest(
                "a period can only be resolved within a grade, because the same "
                "period label is a different hour in different grades. Pass "
                "grade=, or use time=HH:MM.")
        row = db.one("SELECT label, start_min, end_min FROM bell_slots "
                     "WHERE grade = ? AND label = ? COLLATE NOCASE",
                     (grade, period.strip()))
        if not row:
            known = [p["label"] for p in periods_for_grade(grade)]
            raise BadRequest(
                f"grade {grade} has no period called {period!r}. It has: "
                f"{', '.join(known) if known else '(no bell schedule on file)'}")
        return Moment(day or now_moment().day,
                      (row["start_min"] + row["end_min"]) // 2,
                      source=f"period {row['label']} of grade {grade}")

    if day:
        return Moment(day, now_moment().minute, source="clock time on given day")
    return now_moment()


def normalise_day(day: str) -> str:
    canonical = {known.casefold(): known for known in SCHOOL_DAYS}
    value = canonical.get(day.strip().casefold())
    if value is None:
        raise BadRequest(f"day must be one of {', '.join(SCHOOL_DAYS)}")
    return value


def normalise_grade(grade: str) -> str:
    raw = grade.strip().upper().replace(" ", "")
    if raw.startswith("GRADE"):
        raw = raw[5:]
    if raw.startswith("KG"):
        raw = "K" + raw[2:]
    if raw.startswith("K"):
        grade = raw
    elif raw.isdigit():
        grade = raw.zfill(2)
    else:
        grade = raw
    if not grade or not db.one("SELECT 1 FROM grades WHERE code = ?", (grade,)):
        known = [r["code"] for r in db.grades()]
        raise BadRequest(f"unknown grade {grade!r}; known grades: {', '.join(known)}")
    return grade


def require_grade() -> str:
    grade = request.args.get("grade")
    if not grade:
        raise BadRequest("grade is required, e.g. grade=05 or grade=K1")
    return normalise_grade(grade)


def optional_grade() -> str | None:
    """A grade supplied only so a period LABEL can be resolved to a time.

    Used by school-wide endpoints: "which teachers are free in Grade 9's Period
    1" is a well-posed question — resolve the label against grade 9's bell
    schedule, then ask the whole school about that hour. The answer is not
    filtered by the grade, and says which moment it resolved to.
    """
    grade = request.args.get("grade")
    return normalise_grade(grade) if grade else None


def _paginate(rows: list[dict]) -> tuple[list[dict], dict]:
    try:
        requested_limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        raise BadRequest("limit and offset must be integers")
    if requested_limit < 1:
        raise BadRequest("limit must be at least 1")
    if offset < 0:
        raise BadRequest("offset must be zero or greater")
    limit = min(requested_limit, MAX_PAGE)
    page = rows[offset:offset + limit]
    return page, {"total": len(rows), "limit": limit, "offset": offset,
                  "returned": len(page)}


def _section_summary(grade: str, moment: Moment,
                     students: list[dict]) -> list[dict]:
    """Reconcile the coarse section grid with individual group schedules."""
    rows = section_status(grade, moment)
    counts = {row["section"]: {FREE: 0, IN_CLASS: 0, UNKNOWN: 0}
              for row in rows}
    for student in students:
        counts.setdefault(student["section"],
                          {FREE: 0, IN_CLASS: 0, UNKNOWN: 0})[
                              student["status"]] += 1
    for row in rows:
        row["grid_status"] = row["status"]
        row["student_status_counts"] = counts[row["section"]]
        present = [status for status in (IN_CLASS, FREE, UNKNOWN)
                   if counts[row["section"]][status]]
        if len(present) == 1:
            row["status"] = present[0]
        elif len(present) > 1:
            row["status"] = "mixed"
    return rows


# ── reference ───────────────────────────────────────────────────────────────

@api.route("/health")
def health():
    """Deliberately unauthenticated: liveness must not need a secret."""
    try:
        meta = db.meta()
        counts = db.one("""SELECT (SELECT COUNT(*) FROM students)  AS students,
                                  (SELECT COUNT(*) FROM lessons)   AS lessons,
                                  (SELECT COUNT(*) FROM enrolments) AS enrolments""")
        return jsonify({"ok": True, "students": counts["students"],
                        "lessons": counts["lessons"],
                        "enrolments": counts["enrolments"],
                        "schema_version": meta.get("schema_version"),
                        "built_at_utc": meta.get("built_at_utc"),
                        "source_sha256": {
                            k.removeprefix("built_from_").removesuffix("_sha256"): v
                            for k, v in meta.items() if k.endswith("_sha256")}})
    except db.DatabaseMissing as e:
        return jsonify({"ok": False, "error": str(e)}), 503


@api.route("/grades")
@require(SCOPE_ROBOT)
def grades():
    return jsonify([{
        "code": r["code"], "label": r["label"], "students": r["students"],
        "coverage": r["coverage"], "verdict": r["verdict"],
        "has_timetable": bool(r["has_section_timetable"]),
    } for r in db.grades()])


@api.route("/periods")
@require(SCOPE_ROBOT)
def periods():
    grade = require_grade()
    return jsonify({"grade": grade, "periods": periods_for_grade(grade)})


@api.route("/now")
@require(SCOPE_ROBOT)
def now():
    m = now_moment()
    out = {"moment": m.as_dict(), "periods_by_grade": {}}
    for r in db.grades():
        p = current_period(r["code"], m)
        out["periods_by_grade"][r["code"]] = p["label"] if p else None
    out["in_session"] = any(out["periods_by_grade"].values())
    return jsonify(out)


@api.route("/diagnostics")
@require(SCOPE_ROBOT)
def diagnostics():
    """What the build could not establish. Reading this is part of trusting the
    numbers, so it is a first-class endpoint rather than a log file."""
    return jsonify({
        "meta": db.meta(),
        "coverage": [dict(r) for r in db.query(
            "SELECT gc.* FROM grade_coverage gc JOIN grades g ON g.code=gc.grade "
            "ORDER BY g.ordinal")],
        "issues": [dict(r) for r in db.issues()],
    })


# ── the school right now ────────────────────────────────────────────────────

@api.route("/snapshot")
@require(SCOPE_ROBOT)
def snapshot():
    """Counts per status for a grade. Safe for a robot: no names."""
    grade = require_grade()
    m = resolve_moment(grade)
    snap = grade_snapshot(grade, m)
    counts = snap.counts()
    period = current_period(grade, m)
    return jsonify({
        "grade": grade, "moment": m.as_dict(),
        "period": period["label"] if period else None,
        "in_class": counts[IN_CLASS], "free": counts[FREE],
        "unknown": counts[UNKNOWN], "total": len(snap.students),
        "verdict": snap.verdict, "coverage": snap.coverage,
        "caveats": snap.caveats,
        "sections": _section_summary(grade, m, snap.students),
    })


@api.route("/students")
@require(SCOPE_STAFF)
def students():
    """Named students by status. Staff key only."""
    grade = require_grade()
    m = resolve_moment(grade)
    status = request.args.get("status")
    if status and status not in (IN_CLASS, FREE, UNKNOWN):
        raise BadRequest(f"status must be one of {IN_CLASS}, {FREE}, {UNKNOWN}")

    snap = grade_snapshot(grade, m)
    rows = snap.by_status(status) if status else snap.students
    page, meta = _paginate(rows)
    return jsonify({
        "grade": grade, "moment": m.as_dict(), "status": status,
        "counts": snap.counts(), "caveats": snap.caveats,
        "students": page, "page": meta,
    })


@api.route("/student/search")
@require(SCOPE_STAFF)
def student_search():
    term = (request.args.get("q") or "").strip()
    if len(term) < 2:
        raise BadRequest("q must be at least 2 characters")
    return jsonify({"query": term, "results": find_students(term)})


@api.route("/student/<computer_number>")
@require(SCOPE_STAFF)
def student(computer_number):
    # A period label is resolved against THIS student's grade — the only bell
    # schedule that means anything for them.
    own = db.one("""SELECT sec.grade FROM students st
                    JOIN sections sec ON sec.id = st.section_id
                    WHERE st.computer_number = ?""", (computer_number,))
    if own is None:
        return jsonify({"error": "not_found",
                        "detail": f"no student with computer number "
                                  f"{computer_number}"}), 404
    rec = student_now(computer_number, resolve_moment(own["grade"]))
    if rec is None:
        return jsonify({"error": "not_found",
                        "detail": f"no student with computer number "
                                  f"{computer_number}"}), 404
    return jsonify(rec)


@api.route("/student/<computer_number>/day")
@require(SCOPE_STAFF)
def student_timetable(computer_number):
    day_arg = request.args.get("day")
    day = normalise_day(day_arg) if day_arg else now_moment().day
    rec = student_day(computer_number, day)
    if rec is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(rec)


@api.route("/sections")
@require(SCOPE_ROBOT)
def sections():
    grade = require_grade()
    m = resolve_moment(grade)
    snap = grade_snapshot(grade, m)
    return jsonify({"grade": grade, "moment": m.as_dict(),
                    "sections": _section_summary(grade, m, snap.students),
                    "caveats": snap.caveats})


@api.route("/section/timetable")
@require(SCOPE_ROBOT)
def section_timetable():
    grade = require_grade()
    letter = request.args.get("section")
    if not letter:
        raise BadRequest("section is required, e.g. section=A")
    day_arg = request.args.get("day")
    day = normalise_day(day_arg) if day_arg else now_moment().day
    sec = db.one("SELECT id FROM sections WHERE grade = ? AND letter = ?",
                 (grade, letter.upper()))
    if not sec:
        raise BadRequest(f"grade {grade} has no section {letter!r}")
    rows = db.query("""
        SELECT sp.period_label, sp.start_min, sp.end_min, sp.subject_text,
               sp.is_free, sp.teacher_code, t.title, t.name AS teacher_name
        FROM section_periods sp
        LEFT JOIN teachers t ON t.code = sp.teacher_code
        WHERE sp.section_id = ? AND sp.day = ? ORDER BY sp.start_min
    """, (sec["id"], day))
    return jsonify({
        "grade": grade, "section": letter.upper(), "day": day,
        "periods": [{
            "period": r["period_label"],
            "start": f"{r['start_min']//60:02d}:{r['start_min']%60:02d}"
                     if r["start_min"] is not None else None,
            "end": f"{r['end_min']//60:02d}:{r['end_min']%60:02d}"
                   if r["end_min"] is not None else None,
            "status": FREE if r["is_free"] else IN_CLASS,
            "subject": r["subject_text"],
            "teacher": " ".join(x for x in (r["title"], r["teacher_name"]) if x)
                       or r["teacher_code"],
        } for r in rows],
    })


@api.route("/teachers")
@require(SCOPE_ROBOT)
def teachers():
    # Whole-school, but a grade may be passed to resolve a period label.
    m = resolve_moment(optional_grade())
    result = teacher_status(m)
    named = scope() == SCOPE_STAFF
    body = {
        "moment": result["moment"],
        "free_count": len(result["free"]),
        "busy_count": len(result["busy"]),
        "not_in_schedule": result["not_in_schedule"],
        "conflict_count": result["conflicts"],
    }
    if result["not_in_schedule"]:
        body["caveats"] = [
            f"{result['not_in_schedule']} members of staff appear in the subject "
            f"lists but in no schedule, so whether they are teaching now is not "
            f"known. They are excluded from both counts rather than counted free."]
    if result["conflicts"]:
        body.setdefault("caveats", []).append(
            f"{result['conflicts']} teachers have overlapping current lessons. "
            f"They are counted as teaching, but no subject or room is chosen "
            f"from the conflicting source rows.")
    if named:
        body["free"] = result["free"]
        body["busy"] = result["busy"]
    return jsonify(body)
