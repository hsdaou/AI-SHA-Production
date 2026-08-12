"""Student Report Viewer web application.

The app reads a CSV report, indexes students by computer number, and exposes a
small JSON API used by the browser interface. The CSV is reloaded only when it
changes on disk, so updated reports are picked up without restarting the app.
"""

from __future__ import annotations

import csv
import hmac
import logging
import os
import re
import smtplib
import threading
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request


APP_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = Path(os.getenv("STUDENT_REPORT_DATA_FILE", APP_ROOT / "Houssam Report.csv"))
MAX_STUDENT_ID_LENGTH = 50
STUDENT_ID_PATTERN = re.compile(r"^[\w./-]+$", re.UNICODE)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "student_id": ("Student Number", "Student Computer Number", "Computer Number"),
    "student_name": ("Student Name", "Name"),
    "grade": ("Grade", "Class"),
    "section": ("Section",),
    "ams_avg": ("AMS Average T3", "AMS Average"),
    "periodic_avg": ("Periodic Average T3", "Periodic Average"),
    "behaviour": ("Behaviour Infraction", "Behavior Infraction"),
    "tardiness": ("Tardiness Infraction",),
    "academic": ("Academic Infraction",),
    "missed_periodic": ("Missed Periodic Exams",),
    # The source report historically contains the typo "Exasms".
    "missed_ams": ("Missed AMS Exams", "Missed AMS Exasms"),
}

SUBJECT_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Biology", ("Biology Average",)),
    ("Arabic", ("Arabic Average",)),
    ("Physics", ("Physics Average",)),
    ("English", ("English Average",)),
    ("French", ("French Average", "Frenh Average")),
    ("Computing", ("Computing Average",)),
    ("Chemistry", ("Chemistry Average",)),
    ("Mathematics", ("Math Average", "Mathematics Average")),
    ("Moral Education", ("Moral Average",)),
    ("Humanities", ("Humanities Average",)),
    ("Religion", ("Religion Average",)),
    ("Science", ("Science Average",)),
    ("Economics", ("Economics Average",)),
)

app = Flask(__name__)
app.config.update(JSON_SORT_KEYS=False, MAX_CONTENT_LENGTH=16 * 1024)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_cache_lock = threading.RLock()
_cache: dict[str, Any] = {"mtime_ns": None, "students": None, "warnings": []}


class DataFileError(RuntimeError):
    """Raised when the report file cannot be loaded safely."""


def _clean_header(value: str | None) -> str:
    return (value or "").strip()


def _clean_value(value: Any, default: str = "N/A") -> str:
    if value is None:
        return default
    cleaned = str(value).strip()
    if not cleaned or cleaned.casefold() in {"nan", "none", "null"}:
        return default
    return cleaned


def _counter_value(value: Any) -> str:
    cleaned = _clean_value(value, "0")
    return "0" if cleaned == "N/A" else cleaned


def _normalise_student_id(value: Any) -> str:
    cleaned = _clean_value(value, "")
    # Spreadsheet exports sometimes turn an integer identifier into 12345.0.
    if re.fullmatch(r"\d+\.0", cleaned):
        return cleaned[:-2]
    return cleaned


def _first_value(row: dict[str, str], aliases: tuple[str, ...], default: str = "N/A") -> str:
    for alias in aliases:
        if alias in row:
            value = _clean_value(row.get(alias), default)
            if value != default:
                return value
    return default


def _resolve_schema(headers: list[str]) -> list[str]:
    warnings: list[str] = []
    header_set = set(headers)
    for required in ("student_id", "student_name"):
        if not any(alias in header_set for alias in FIELD_ALIASES[required]):
            raise DataFileError(
                f"Required column is missing: one of {', '.join(FIELD_ALIASES[required])}"
            )
    for field, aliases in FIELD_ALIASES.items():
        if field not in {"student_id", "student_name"} and not any(
            alias in header_set for alias in aliases
        ):
            warnings.append(f"Optional field unavailable: {field}")
    return warnings


def _read_students() -> tuple[dict[str, dict[str, Any]], list[str]]:
    if not DATA_FILE.is_file():
        raise DataFileError(f"Report file not found: {DATA_FILE}")

    students: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()

    try:
        with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise DataFileError("The report file has no header row.")

            clean_headers = [_clean_header(header) for header in reader.fieldnames]
            warnings = _resolve_schema(clean_headers)

            for raw_row in reader:
                row = {
                    _clean_header(key): value
                    for key, value in raw_row.items()
                    if key is not None
                }
                student_id = _normalise_student_id(
                    _first_value(row, FIELD_ALIASES["student_id"], "")
                )
                if not student_id:
                    continue
                if student_id in students:
                    duplicate_ids.add(student_id)
                    continue

                subjects: list[dict[str, str]] = []
                for display_name, aliases in SUBJECT_FIELDS:
                    mark = _first_value(row, aliases)
                    if mark != "N/A":
                        subjects.append({"name": display_name, "mark": mark})

                students[student_id] = {
                    "student_id": student_id,
                    "student_name": _first_value(row, FIELD_ALIASES["student_name"]),
                    "grade": _first_value(row, FIELD_ALIASES["grade"]),
                    "section": _first_value(row, FIELD_ALIASES["section"]),
                    "ams_avg": _first_value(row, FIELD_ALIASES["ams_avg"]),
                    "periodic_avg": _first_value(row, FIELD_ALIASES["periodic_avg"]),
                    "behaviour": _counter_value(_first_value(row, FIELD_ALIASES["behaviour"], "0")),
                    "tardiness": _counter_value(_first_value(row, FIELD_ALIASES["tardiness"], "0")),
                    "academic": _counter_value(_first_value(row, FIELD_ALIASES["academic"], "0")),
                    "missed_periodic": _counter_value(
                        _first_value(row, FIELD_ALIASES["missed_periodic"], "0")
                    ),
                    "missed_ams": _counter_value(
                        _first_value(row, FIELD_ALIASES["missed_ams"], "0")
                    ),
                    "subjects": subjects,
                }
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataFileError(f"Unable to read the report file: {exc}") from exc

    if duplicate_ids:
        warnings.append(
            f"Ignored {len(duplicate_ids)} duplicate student number(s); the first row was used."
        )
    if not students:
        raise DataFileError("The report contains no usable student records.")
    return students, warnings


def load_students() -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return the cached student index, reloading after the CSV changes."""
    try:
        mtime_ns = DATA_FILE.stat().st_mtime_ns
    except OSError as exc:
        raise DataFileError(f"Report file is unavailable: {DATA_FILE}") from exc

    with _cache_lock:
        if _cache["students"] is None or _cache["mtime_ns"] != mtime_ns:
            students, warnings = _read_students()
            _cache.update(mtime_ns=mtime_ns, students=students, warnings=warnings)
            logger.info("Loaded %d student records from %s", len(students), DATA_FILE)
            for warning in warnings:
                logger.warning(warning)
        return _cache["students"], list(_cache["warnings"])


def _requested_student_id() -> str:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        return _normalise_student_id(payload.get("student_id"))
    return _normalise_student_id(request.form.get("student_id"))


def _robot_authorized() -> bool:
    expected = os.getenv("STUDENT_REPORT_ROBOT_KEY", "")
    provided = request.headers.get("x-robot-key", "")
    return bool(expected and provided and hmac.compare_digest(expected, provided))


def _report_recipients() -> list[str]:
    return [
        address.strip()
        for address in os.getenv("STUDENT_REPORT_TO", "").split(",")
        if address.strip()
    ]


def _email_student_report(student: dict[str, Any]) -> tuple[bool, str]:
    recipients = _report_recipients()
    smtp_user = os.getenv("STUDENT_REPORT_SMTP_USER", "")
    smtp_password = os.getenv("STUDENT_REPORT_SMTP_PASS", "")
    if not recipients or not smtp_user or not smtp_password:
        return False, "Report email is not configured."

    def safe(value: Any) -> str:
        return escape(str(value if value not in (None, "") else "N/A"))

    subjects = "".join(
        f"<tr><td>{safe(subject['name'])}</td><td>{safe(subject['mark'])}</td></tr>"
        for subject in student["subjects"]
    ) or "<tr><td colspan='2'>No subject marks recorded</td></tr>"
    html = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;max-width:720px;margin:auto">
    <h2>Student report: {safe(student['student_name'])}</h2>
    <p><b>Computer number:</b> {safe(student['student_id'])}<br>
    <b>Grade / section:</b> {safe(student['grade'])} / {safe(student['section'])}</p>
    <h3>Term 3 averages</h3><p>AMS: <b>{safe(student['ams_avg'])}</b> &nbsp; Periodic: <b>{safe(student['periodic_avg'])}</b></p>
    <h3>Attendance and conduct</h3><p>Missed periodic exams: {safe(student['missed_periodic'])}<br>
    Missed AMS exams: {safe(student['missed_ams'])}<br>Behaviour infractions: {safe(student['behaviour'])}<br>
    Tardiness infractions: {safe(student['tardiness'])}<br>Academic infractions: {safe(student['academic'])}</p>
    <h3>Subject marks</h3><table style="border-collapse:collapse;width:100%" border="1" cellpadding="8">
    <tr><th align="left">Subject</th><th align="left">Mark</th></tr>{subjects}</table>
    <p style="color:#667085;font-size:12px">ISC Sharjah — requested through AI-SHA. Contains personal data about a student; do not forward.</p>
    </body></html>"""
    message = EmailMessage()
    message["Subject"] = f"Student report — {student['student_name']} ({student['student_id']})"
    message["From"] = smtp_user
    message["To"] = ", ".join(recipients)
    message.set_content("This student report is available in the HTML version of this email.")
    message.add_alternative(html, subtype="html")
    try:
        with smtplib.SMTP(
            os.getenv("STUDENT_REPORT_SMTP_HOST", "smtp.gmail.com"),
            int(os.getenv("STUDENT_REPORT_SMTP_PORT", "587")),
            timeout=30,
        ) as smtp:
            smtp.starttls()
            smtp.login(smtp_user, smtp_password)
            smtp.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        logger.error("Unable to email student report: %s", type(exc).__name__)
        return False, "The report could not be emailed."
    return True, "sent"


@app.after_request
def add_security_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:"
    )
    return response


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/search")
def search():
    student_id = _requested_student_id()
    if not student_id:
        return jsonify(error="Please enter a student computer number."), 400
    if len(student_id) > MAX_STUDENT_ID_LENGTH or not STUDENT_ID_PATTERN.fullmatch(student_id):
        return jsonify(error="Enter a valid student computer number."), 400

    try:
        students, _warnings = load_students()
    except DataFileError:
        logger.exception("Student report data could not be loaded")
        return jsonify(error="Student data is temporarily unavailable."), 503

    student = students.get(student_id)
    if student is None:
        return jsonify(error=f"No student was found with number {student_id}."), 404
    return jsonify(student)


@app.get("/health")
def health():
    try:
        students, warnings = load_students()
    except DataFileError as exc:
        return jsonify(status="unhealthy", error=str(exc)), 503
    return jsonify(status="ok", student_count=len(students), warnings=warnings)


@app.get("/api/robot/student-report")
def robot_student_report():
    """Email one report without returning personal data to the robot client."""
    if not _robot_authorized():
        return jsonify(error="Unauthorized"), 401
    student_id = _normalise_student_id(request.args.get("student_id"))
    if not student_id or len(student_id) > MAX_STUDENT_ID_LENGTH or not STUDENT_ID_PATTERN.fullmatch(student_id):
        return jsonify(error="A valid student computer number is required."), 400
    try:
        students, _warnings = load_students()
    except DataFileError:
        logger.exception("Student report data could not be loaded")
        return jsonify(error="Student data is temporarily unavailable."), 503
    student = students.get(student_id)
    if student is None:
        return jsonify(ok=False, reason="not_found", speakable="I could not find a student with that computer number."), 404
    sent, detail = _email_student_report(student)
    if not sent:
        return jsonify(error=detail), 503
    return jsonify(
        ok=True,
        emailed=True,
        recipients=len(_report_recipients()),
        speakable="The student report has been emailed to the administrator.",
    )


if __name__ == "__main__":
    host = os.getenv("STUDENT_REPORT_HOST", "127.0.0.1")
    port = int(os.getenv("STUDENT_REPORT_PORT", "5000"))
    app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG") == "1")
