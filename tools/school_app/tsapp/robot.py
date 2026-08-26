"""
The robot-facing surface (AI-SHA).

The robot stands in a public corridor with a speaker. Whatever it receives, it
may say aloud to whoever is standing there. That single fact decides the whole
design of this module:

    SPEAKABLE   counts, section status, a class's timetable. Identifies nobody.
    EMAILED     any list of named people. Rendered and sent from this process;
                the robot receives {ok, count} and says "the list has been
                emailed". No child's name ever enters the robot's memory, disk,
                logs or speaker.

Each endpoint returns a `speakable` string written to be read out as-is,
including its caveats. If an answer is uncertain, the uncertainty is IN the
sentence — a robot cannot be relied on to add "approximately" by itself, and an
administrator who hears a bare number will act on a bare number.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from html import escape

from flask import Blueprint, jsonify, request

from . import db
from .api import (BadRequest, normalise_day, optional_grade, require_grade,
                  resolve_moment)
from .auth import SCOPE_ROBOT, require
from .resolver import (FREE, IN_CLASS, UNKNOWN, current_period, grade_snapshot,
                       now_moment, teacher_status)

robot = Blueprint("robot", __name__, url_prefix="/api/robot")


@robot.errorhandler(BadRequest)
def _bad_request(e):
    return jsonify({"error": "bad_request", "detail": e.detail,
                    "speakable": f"I could not answer that. {e.detail}"}), 400


def _recipients() -> list[str]:
    return [a.strip() for a in os.environ.get("SCHOOL_REPORT_TO", "").split(",")
            if a.strip()]


def send_report(subject: str, html: str) -> tuple[bool, str]:
    to = _recipients()
    if not to:
        return False, "SCHOOL_REPORT_TO is not set"
    user = os.environ.get("SCHOOL_SMTP_USER", "")
    password = os.environ.get("SCHOOL_SMTP_PASS", "")
    if not user or not password:
        return False, "SCHOOL_SMTP_USER / SCHOOL_SMTP_PASS are not set"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = ", ".join(to)
    msg.set_content("This report is formatted as HTML.")
    msg.add_alternative(html, subtype="html")
    try:
        with smtplib.SMTP(os.environ.get("SCHOOL_SMTP_HOST", "smtp.gmail.com"),
                          int(os.environ.get("SCHOOL_SMTP_PORT", "587")),
                          timeout=30) as s:
            s.starttls()
            s.login(user, password)
            s.send_message(msg)
        return True, f"sent to {len(to)} recipient(s)"
    except Exception as e:                                        # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


_CELL = "padding:8px;border:1px solid #e2e8f0;text-align:left;"


def render_people(title: str, subtitle: str, caveats: list[str],
                  rows: list[dict], columns: list[tuple[str, str]]) -> str:
    head = "".join(f'<th style="{_CELL}background:#f1f5f9;">{escape(label)}</th>'
                   for label, _ in columns)
    body = "".join(
        "<tr>" + "".join(
            f'<td style="{_CELL}">{escape(str(r.get(key) or ""))}</td>'
            for _, key in columns) + "</tr>"
        for r in rows)
    # The caveats travel with the list. A recipient acting on 400 names needs to
    # see, in the same document, that some of them are uncertain.
    warn = "".join(
        f'<p style="background:#fef3c7;border-left:4px solid #d97706;'
        f'padding:10px;margin:12px 0;">{escape(c)}</p>' for c in caveats)
    table = (f'<table style="border-collapse:collapse;width:100%;margin:16px 0;'
             f'font-size:14px;"><tr>{head}</tr>{body}</table>') if rows else \
            '<p style="color:#64748b;">Nobody matched.</p>'
    return f"""<div style="font-family:Arial,Helvetica,sans-serif;max-width:760px;margin:0 auto;">
  <h2 style="color:#1d4ed8;margin-bottom:4px;">{escape(title)}</h2>
  <p style="color:#475569;margin-top:0;">{escape(subtitle)}</p>
  {warn}{table}
  <hr style="border:none;border-top:1px solid #e2e8f0;margin:24px 0;" />
  <p style="color:#64748b;font-size:12px;">
    Requested via AI-SHA. Contains personal data about children. Do not forward.
  </p></div>"""


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


# ── speakable ───────────────────────────────────────────────────────────────

@robot.route("/now")
@require(SCOPE_ROBOT)
def r_now():
    m = now_moment()
    lines = []
    for r in db.grades():
        p = current_period(r["code"], m)
        if p:
            lines.append(f"{r['label']} is in {p['label']}")
    if lines:
        speak = f"It is {m.hhmm} on {m.day}. " + "; ".join(lines[:4]) + "."
    else:
        speak = (f"It is {m.hhmm} on {m.day}. No lesson is running at the moment.")
    return jsonify({"ok": True, **m.as_dict(), "in_session": bool(lines),
                    "speakable": speak})


@robot.route("/free-count")
@require(SCOPE_ROBOT)
def r_free_count():
    """SPEAKABLE. Counts only — and it says out loud what it cannot vouch for."""
    # Compatibility with AI-SHA's existing timetable client. It uses the shared
    # ``free_count`` intent for both populations and identifies the population
    # in ``subject``. Keep that wire contract while routing teacher questions to
    # the purpose-built whole-school resolver below.
    if (request.args.get("subject") or "").strip().lower() == "teachers":
        return r_teacher_count()
    grade = require_grade()
    m = resolve_moment(grade)
    snap = grade_snapshot(grade, m)
    counts = snap.counts()
    label = db.one("SELECT label FROM grades WHERE code = ?", (grade,))["label"]

    if not snap.students:
        return jsonify({"ok": False, "reason": "no_students",
                        "speakable": f"I have no student records for {label}."})

    if snap.verdict == "insufficient" or not snap.has_timetable:
        # The honest refusal. The old app answered this case with a number, and
        # the number was every student in the grade.
        return jsonify({
            "ok": False, "reason": "insufficient_schedule_data", "grade": grade,
            "unknown": counts[UNKNOWN], "total": len(snap.students),
            "speakable": (
                f"I cannot tell you that for {label}. I know its "
                f"{len(snap.students)} students and the subjects they take, but "
                f"not when those lessons are timetabled, so I would be guessing."),
        })

    free, busy = counts[FREE], counts[IN_CLASS]
    period = current_period(grade, m)
    where = f"{period['label']}" if period else f"at {m.hhmm}"
    speak = (f"In {label}, {free} {_plural(free, 'student')} "
             f"{'is' if free == 1 else 'are'} free in {where} on {m.day}, and "
             f"{busy} {'is' if busy == 1 else 'are'} in class.")
    for c in snap.caveats:
        speak += " " + c
    return jsonify({"ok": True, "grade": grade, "moment": m.as_dict(),
                    "period": period["label"] if period else None,
                    "free": free, "in_class": busy, "unknown": counts[UNKNOWN],
                    "total": len(snap.students), "verdict": snap.verdict,
                    "caveats": snap.caveats, "speakable": speak})


@robot.route("/free-sections")
@require(SCOPE_ROBOT)
def r_free_sections():
    """SPEAKABLE. Which sections are free — the actionable form of the question.

    A free period belongs to a section, not to scattered individuals, so this is
    both more useful to whoever asked and impossible to turn into a list of
    children.
    """
    grade = require_grade()
    m = resolve_moment(grade)
    from .resolver import section_status
    rows = section_status(grade, m)
    snap = grade_snapshot(grade, m)
    label = db.one("SELECT label FROM grades WHERE code = ?", (grade,))["label"]
    by_section = {
        row["section"]: {FREE: 0, IN_CLASS: 0, UNKNOWN: 0}
        for row in rows
    }
    for student in snap.students:
        by_section.setdefault(student["section"],
                              {FREE: 0, IN_CLASS: 0, UNKNOWN: 0})[
                                  student["status"]] += 1
    for row in rows:
        counts = by_section[row["section"]]
        row["grid_status"] = row["status"]
        row["student_status_counts"] = counts
        present = [status for status in (IN_CLASS, FREE, UNKNOWN)
                   if counts[status]]
        if len(present) == 1:
            row["status"] = present[0]
        elif len(present) > 1:
            row["status"] = "mixed"
        row["fully_free"] = bool(
            row["student_count"]
            and counts[FREE] == row["student_count"]
            and counts[IN_CLASS] == 0 and counts[UNKNOWN] == 0)
    free = [r for r in rows if r["fully_free"]]

    if not snap.has_timetable:
        speak = (f"I have no timetable for {label}, so I cannot say which "
                 f"sections are free.")
        return jsonify({"ok": False, "reason": "no_timetable", "speakable": speak})
    if not free:
        speak = (f"No entire section of {label} is free right now. Individual "
                 f"group schedules are included in that answer.")
    else:
        names = ", ".join(r["section"] for r in free)
        heads = sum(r["student_status_counts"][FREE] for r in free)
        speak = (f"In {label}, {_plural(len(free), 'section')} {names} "
                 f"{'is' if len(free) == 1 else 'are'} free — "
                 f"{heads} {_plural(heads, 'student')} in total.")
    return jsonify({"ok": True, "grade": grade, "moment": m.as_dict(),
                    "free_sections": free, "sections": rows, "speakable": speak})


@robot.route("/timetable")
@require(SCOPE_ROBOT)
def r_timetable():
    """SPEAKABLE. A section's day. Institutional, names nobody."""
    grade = require_grade()
    letter = (request.args.get("section") or "A").upper()
    day_arg = request.args.get("day")
    day = normalise_day(day_arg) if day_arg else now_moment().day
    sec = db.one("SELECT id FROM sections WHERE grade = ? AND letter = ?",
                 (grade, letter))
    if not sec:
        raise BadRequest(f"grade {grade} has no section {letter!r}")
    rows = db.query("""SELECT period_label, subject_text, is_free
                       FROM section_periods WHERE section_id = ? AND day = ?
                       ORDER BY start_min""", (sec["id"], day))
    if not rows:
        return jsonify({"ok": False, "reason": "no_timetable",
                        "speakable": f"I have no timetable for {grade} {letter}."})
    spoken = "; ".join(
        f"{r['period_label']}: {'free' if r['is_free'] else r['subject_text']}"
        for r in rows)
    return jsonify({"ok": True, "grade": grade, "section": letter, "day": day,
                    "periods": [dict(r) for r in rows],
                    "speakable": f"Grade {grade} section {letter} on {day}. {spoken}."})


@robot.route("/teacher-count")
@require(SCOPE_ROBOT)
def r_teacher_count():
    """SPEAKABLE. How many staff are free, joined on wall-clock time."""
    # Whole-school. A grade may be passed purely to resolve a period label,
    # since "Period 1" is a different hour in different grades.
    m = resolve_moment(optional_grade())
    result = teacher_status(m)
    free, busy = len(result["free"]), len(result["busy"])

    if busy == 0 and free > 5:
        # Kept as a guard, but it should now be unreachable for a real school
        # hour: the label-versus-clock mismatch that used to produce "127 of 128
        # free" cannot arise when both sides are minutes since midnight. If this
        # ever fires again, something new is wrong and silence is the right move.
        return jsonify({
            "ok": False, "reason": "implausible", "free": free, "busy": busy,
            "speakable": ("Something is wrong with my timetable data — it says "
                          "almost every teacher is free, which cannot be right "
                          "during a lesson. I would rather not answer than "
                          "mislead you."),
        })
    speak = (f"At {m.hhmm} on {m.day}, {free} {_plural(free, 'teacher')} "
             f"{'is' if free == 1 else 'are'} not teaching, and {busy} "
             f"{'is' if busy == 1 else 'are'}.")
    if result["not_in_schedule"]:
        speak += (f" {result['not_in_schedule']} more are on staff but absent from "
                  f"the timetable I hold, so I have not counted them either way.")
    if result["conflicts"]:
        speak += (f" {result['conflicts']} {_plural(result['conflicts'], 'teacher')} "
                  f"{'has' if result['conflicts'] == 1 else 'have'} overlapping "
                  f"timetable entries; I counted them as teaching "
                  f"but did not choose one room.")
    return jsonify({"ok": True, "moment": m.as_dict(), "free": free,
                    "in_class": busy,
                    "not_in_schedule": result["not_in_schedule"],
                    "conflicts": result["conflicts"],
                    "speakable": speak})


# ── emailed ─────────────────────────────────────────────────────────────────

@robot.route("/free-students")
@robot.route("/email/free-students", methods=["POST"])
@require(SCOPE_ROBOT)
def r_email_free_students():
    """EMAILED. The robot receives a count and a confirmation, never the names."""
    grade = require_grade()
    m = resolve_moment(grade)
    snap = grade_snapshot(grade, m)
    label = db.one("SELECT label FROM grades WHERE code = ?", (grade,))["label"]

    if snap.verdict == "insufficient" or not snap.has_timetable:
        return jsonify({
            "ok": False, "reason": "insufficient_schedule_data",
            "speakable": (f"I will not send that list. For {label} I do not know "
                          f"when lessons run, so the list would be wrong."),
        })

    rows = snap.by_status(FREE)
    period = current_period(grade, m)
    title = (f"Free students — {label}, "
             f"{period['label'] if period else m.hhmm}, {m.day}")
    html = render_people(
        title, f"{len(rows)} student(s) free at {m.hhmm} on {m.day}.",
        snap.caveats, rows,
        [("First name", "first_name"), ("Last name", "last_name"),
         ("Section", "section"), ("Computer no.", "computer_number"),
         ("Why free", "reason")])
    ok, info = send_report(title, html)
    if not ok:
        return jsonify({"ok": False, "count": len(rows), "emailed": False,
                        "error": info,
                        "speakable": "I could not send the e-mail. "
                                     "Please check the mail settings."}), 502
    speak = (f"I have e-mailed the list of {len(rows)} free "
             f"{_plural(len(rows), 'student')} in {label} to the administrator.")
    if snap.caveats:
        speak += " It carries a note about how certain the data is."
    return jsonify({"ok": True, "count": len(rows), "emailed": True,
                    "recipients": len(_recipients()), "speakable": speak})


@robot.route("/free-teachers")
@robot.route("/email/free-teachers", methods=["POST"])
@require(SCOPE_ROBOT)
def r_email_free_teachers():
    """EMAILED. Staff names are personal data too."""
    m = resolve_moment(optional_grade())
    result = teacher_status(m)
    rows = result["free"]
    if not rows and not result["busy"]:
        return jsonify({"ok": False, "reason": "no_schedule",
                        "speakable": "I hold no teacher timetable, so I cannot "
                                     "say who is free."})
    caveats = []
    if result["not_in_schedule"]:
        caveats.append(
            f"{result['not_in_schedule']} members of staff appear in the subject "
            f"lists but in no timetable. They are not on this list, because "
            f"whether they are teaching is unknown rather than known to be no.")
    if result["conflicts"]:
        caveats.append(
            f"{result['conflicts']} teachers have overlapping current timetable "
            f"entries. They are excluded from the free list and counted as "
            f"teaching, but no single room is asserted.")
    title = f"Teachers not teaching — {m.hhmm}, {m.day}"
    html = render_people(title, f"{len(rows)} of "
                         f"{len(rows) + len(result['busy'])} timetabled staff.",
                         caveats, rows,
                         [("Name", "name"), ("Code", "code"),
                          ("Lessons this week", "week_lessons")])
    ok, info = send_report(title, html)
    if not ok:
        return jsonify({"ok": False, "count": len(rows), "emailed": False,
                        "error": info,
                        "speakable": "I could not send the e-mail."}), 502
    return jsonify({"ok": True, "count": len(rows), "emailed": True,
                    "recipients": len(_recipients()),
                    "speakable": f"I have e-mailed the list of {len(rows)} "
                                 f"available {_plural(len(rows), 'teacher')}."})


@robot.route("/email/student", methods=["POST"])
@require(SCOPE_ROBOT)
def r_email_student():
    """EMAILED. "Where is this child?" is answered to an inbox, not to a corridor.

    The robot may be asked this by anyone standing in front of it. It gets a
    confirmation; the answer goes to staff.
    """
    number = (request.args.get("computer_number") or "").strip()
    if not number:
        raise BadRequest("computer_number is required")
    own = db.one("""SELECT sec.grade FROM students st
                    JOIN sections sec ON sec.id = st.section_id
                    WHERE st.computer_number = ?""", (number,))
    if own is None:
        return jsonify({"ok": False, "reason": "not_found",
                        "speakable": "I do not have a student with that number."})
    m = resolve_moment(own["grade"])
    from .resolver import student_now
    rec = student_now(number, m)
    if rec is None:
        return jsonify({"ok": False, "reason": "not_found",
                        "speakable": "I do not have a student with that number."})
    title = f"Student location — {m.hhmm}, {m.day}"
    html = render_people(
        title, f"Requested via AI-SHA at {m.hhmm} on {m.day}.",
        rec.get("caveats", []), [rec],
        [("First name", "first_name"), ("Last name", "last_name"),
         ("Grade", "grade"), ("Section", "section"), ("Status", "status"),
         ("Subject", "subject"), ("Teacher", "teacher"), ("Room", "room"),
         ("Confidence", "confidence")])
    ok, info = send_report(title, html)
    if not ok:
        return jsonify({"ok": False, "emailed": False, "error": info,
                        "speakable": "I could not send the e-mail."}), 502
    return jsonify({"ok": True, "emailed": True,
                    "recipients": len(_recipients()),
                    "speakable": "I have e-mailed that student's whereabouts to "
                                 "the school office. I will not read it out here."})
