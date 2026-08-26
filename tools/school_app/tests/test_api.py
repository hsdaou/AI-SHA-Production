"""
HTTP behaviour: authentication, what a robot key may learn, and argument handling.

The previous app left every `/api/*` endpoint unauthenticated while it returned
full names, sections and computer numbers for 1,878 minors, and printed a warning
about it at start-up instead of refusing. These tests are the control that
warning was not.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

NAMED = ("/api/students?grade=05", "/api/student/1001",
         "/api/student/search?q=alpha", "/api/student/1001/day")
COUNTS = ("/api/snapshot?grade=05", "/api/sections?grade=05",
          "/api/grades", "/api/periods?grade=05", "/api/now")


@pytest.mark.parametrize("path", NAMED + COUNTS)
def test_no_key_is_refused(client, path):
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", NAMED)
def test_robot_key_cannot_reach_names(client, keys, path):
    r = client.get(path, headers={"x-api-key": keys["robot"]})
    assert r.status_code == 403
    assert "staff key" in r.get_json()["detail"]


@pytest.mark.parametrize("path", COUNTS)
def test_robot_key_may_read_counts(client, keys, path):
    assert client.get(path, headers={"x-api-key": keys["robot"]}).status_code == 200


@pytest.mark.parametrize("path", NAMED)
def test_staff_key_may_read_names(client, keys, path):
    assert client.get(path, headers={"x-api-key": keys["staff"]}).status_code == 200


def test_wrong_key_is_refused(client):
    assert client.get("/api/grades",
                      headers={"x-api-key": "wrong"}).status_code == 401


def test_key_in_query_string_is_not_accepted(client, keys):
    """URLs are logged and retained in history; authentication stays in headers."""
    assert client.get(f"/api/grades?key={keys['robot']}").status_code == 401


def test_unconfigured_service_refuses_everything(synthetic_db, monkeypatch):
    """An unconfigured deployment must not behave like an open one."""
    monkeypatch.delenv("SCHOOL_STAFF_KEY", raising=False)
    monkeypatch.delenv("SCHOOL_ROBOT_KEY", raising=False)
    from tsapp import create_app
    c = create_app().test_client()
    r = c.get("/api/grades")
    assert r.status_code == 503
    assert r.get_json()["error"] == "not_configured"
    # ...but liveness never needs a secret.
    assert c.get("/api/health").status_code == 200


def test_public_health_does_not_disclose_filesystem_paths(client):
    body = client.get("/api/health").get_json()
    assert body["ok"] is True
    assert "built_from" not in body


IDENTIFIERS = {"1001", "1002", "1003", "1004"}
NAMES = {"Alpha", "Beta", "Gamma", "Delta", "Amal", "Bilal", "Cara", "Dana"}


def _strings(value):
    """Every string that appears anywhere in a JSON response."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def assert_no_pii(body, where):
    """No child is identifiable in this response.

    Checked structurally rather than by searching the whole blob for a substring:
    the minute-of-day is a plain integer, and at 16:41 it is 1001, which is also
    a fixture computer number. A test that fails for one minute a day teaches
    people to ignore it.
    """
    for s in _strings(body):
        assert s not in IDENTIFIERS, f"computer number {s} leaked from {where}"
        for name in NAMES:
            assert name not in s, f"student name {name!r} leaked from {where}"


# Times are pinned so these never depend on when the suite runs.
def test_snapshot_never_carries_a_name(client, keys):
    """The robot's speaker is in a corridor. Counts may go there; names may not."""
    body = client.get("/api/snapshot?grade=05&day=Monday&time=08:10",
                      headers={"x-api-key": keys["robot"]}).get_json()
    assert body["in_class"] == 3          # it did answer, it is not empty
    assert_no_pii(body, "/api/snapshot")


def test_robot_endpoints_return_counts_only(client, keys):
    for path in ("/api/robot/free-count?grade=05&day=Monday&time=08:10",
                 "/api/robot/free-sections?grade=05&day=Monday&time=08:10",
                 "/api/robot/teacher-count?day=Monday&time=08:10",
                 "/api/robot/timetable?grade=05&section=A&day=Monday"):
        body = client.get(path, headers={"x-api-key": keys["robot"]}).get_json()
        assert body, f"{path} returned nothing"
        assert_no_pii(body, path)


def test_existing_ai_sha_email_routes_remain_compatible(client, keys, monkeypatch):
    """The deployed timetable client still calls the original GET route names.

    The rewritten app's canonical routes are POST /email/*, but replacing the
    backend must not strand the console client during a rolling deployment.
    """
    import importlib

    robot_module = importlib.import_module("tsapp.robot")
    monkeypatch.setattr(robot_module, "send_report",
                        lambda _subject, _html: (True, "sent"))
    headers = {"x-api-key": keys["robot"]}

    students = client.get(
        "/api/robot/free-students?grade=05&day=Monday&time=09:10",
        headers=headers)
    assert students.status_code == 200
    assert students.get_json()["emailed"] is True
    assert_no_pii(students.get_json(), "/api/robot/free-students")

    teachers = client.get(
        "/api/robot/free-teachers?day=Monday&time=08:10",
        headers=headers)
    assert teachers.status_code == 200
    assert teachers.get_json()["emailed"] is True


def test_existing_ai_sha_teacher_count_contract(client, keys):
    body = client.get(
        "/api/robot/free-count?subject=teachers&day=Monday&time=08:10",
        headers={"x-api-key": keys["robot"]}).get_json()
    assert body["ok"] is True
    assert "free" in body and "in_class" in body
    assert "teacher" in body["speakable"]


def test_robot_calls_only_an_entirely_free_section_free(client, keys, synthetic_db):
    """One student's split-group lesson means their section is not wholly free."""
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

    body = client.get(
        "/api/robot/free-sections?grade=05&day=Monday&time=09:10",
        headers={"x-api-key": keys["robot"]}).get_json()
    assert [row["section"] for row in body["free_sections"]] == ["B"]
    section_a = next(row for row in body["sections"] if row["section"] == "A")
    assert section_a["fully_free"] is False
    assert section_a["status"] == "mixed"
    assert section_a["grid_status"] == "free"
    assert section_a["student_status_counts"] == {
        "free": 1, "in_class": 1, "unknown": 0}


def test_a_period_label_resolves_school_wide_endpoints_too(client, keys):
    """"Which teachers are free during Grade 5's Period 1" is well-posed: resolve
    the label against grade 5's bell schedule, then ask the whole school.

    /api/teachers previously refused it, because it resolved the moment without a
    grade and a bare label cannot be resolved."""
    r = client.get("/api/teachers?grade=05&day=Monday&period=Period 1",
                   headers={"x-api-key": keys["staff"]})
    assert r.status_code == 200
    body = r.get_json()
    assert body["moment"]["time"] == "08:25"
    assert "grade 05" in body["moment"]["source"]
    # Still school-wide, not filtered to that grade.
    assert body["busy_count"] + body["free_count"] > 0


def test_teachers_endpoint_hides_names_from_the_robot(client, keys):
    robot = client.get("/api/teachers",
                       headers={"x-api-key": keys["robot"]}).get_json()
    assert "free_count" in robot and "free" not in robot
    staff = client.get("/api/teachers",
                       headers={"x-api-key": keys["staff"]}).get_json()
    assert "free" in staff


# ── argument handling ───────────────────────────────────────────────────────

def test_period_requires_a_grade(client, keys):
    """"Period 3" is a different hour in different grades, so it cannot be
    resolved without one. Refusing beats picking a grade's bell schedule."""
    r = client.get("/api/teachers?period=Period 1",
                   headers={"x-api-key": keys["staff"]})
    assert r.status_code == 400
    assert "within a grade" in r.get_json()["detail"]


def test_unknown_period_lists_the_real_ones(client, keys):
    r = client.get("/api/snapshot?grade=05&period=Period 9",
                   headers={"x-api-key": keys["staff"]})
    assert r.status_code == 400
    assert "Period 1" in r.get_json()["detail"]


def test_resolved_moment_is_reported_back(client, keys):
    body = client.get("/api/snapshot?grade=05&day=Monday&period=Period 1",
                      headers={"x-api-key": keys["staff"]}).get_json()
    assert body["moment"]["day"] == "Monday"
    assert body["moment"]["time"] == "08:25"        # the middle of 08:00-08:50
    assert "grade 05" in body["moment"]["source"]


def test_common_grade_day_and_period_spellings_are_normalised(client, keys):
    body = client.get(
        "/api/snapshot?grade=Grade%205&day=monday&period=period%201",
        headers={"x-api-key": keys["staff"]}).get_json()
    assert body["grade"] == "05"
    assert body["moment"]["day"] == "Monday"
    assert body["moment"]["time"] == "08:25"

    kg = client.get("/api/snapshot?grade=KG1&day=MONDAY&time=08:10",
                    headers={"x-api-key": keys["staff"]}).get_json()
    assert kg["grade"] == "K1"


def test_explicit_time_wins_over_period(client, keys):
    body = client.get("/api/snapshot?grade=05&day=Monday&time=09:10&period=Period 1",
                      headers={"x-api-key": keys["staff"]}).get_json()
    assert body["moment"]["time"] == "09:10"


@pytest.mark.parametrize("qs,fragment", [
    ("grade=99", "unknown grade"),
    ("grade=05&day=Funday", "day must be one of"),
    ("grade=05&time=99:99", "HH:MM"),
    ("", "grade is required"),
])
def test_bad_arguments_explain_themselves(client, keys, qs, fragment):
    r = client.get(f"/api/snapshot?{qs}", headers={"x-api-key": keys["staff"]})
    assert r.status_code == 400
    assert fragment in r.get_json()["detail"]


def test_pagination_is_bounded(client, keys):
    body = client.get("/api/students?grade=05&limit=9999",
                      headers={"x-api-key": keys["staff"]}).get_json()
    assert body["page"]["limit"] <= 500


@pytest.mark.parametrize("query,fragment", [
    ("limit=0", "at least 1"),
    ("limit=-5", "at least 1"),
    ("offset=-1", "zero or greater"),
])
def test_invalid_pagination_is_rejected(client, keys, query, fragment):
    response = client.get(f"/api/students?grade=05&{query}",
                          headers={"x-api-key": keys["staff"]})
    assert response.status_code == 400
    assert fragment in response.get_json()["detail"]


def test_robot_timetable_rejects_an_invalid_day(client, keys):
    response = client.get("/api/robot/timetable?grade=05&day=Funday",
                          headers={"x-api-key": keys["robot"]})
    assert response.status_code == 400
    assert "day must be" in response.get_json()["detail"]


def test_missing_student_is_404(client, keys):
    r = client.get("/api/student/999999", headers={"x-api-key": keys["staff"]})
    assert r.status_code == 404


def test_search_wildcards_are_literal_text(client, keys):
    body = client.get("/api/student/search?q=%25_",
                      headers={"x-api-key": keys["staff"]}).get_json()
    assert body["results"] == []


def test_diagnostics_are_a_first_class_endpoint(client, keys):
    body = client.get("/api/diagnostics",
                      headers={"x-api-key": keys["robot"]}).get_json()
    assert "coverage" in body and "issues" in body
    assert any(r["grade"] == "K1" and r["verdict"] == "insufficient"
               for r in body["coverage"])


def test_robot_refuses_a_grade_it_cannot_schedule(client, keys):
    body = client.get("/api/robot/free-count?grade=K1",
                      headers={"x-api-key": keys["robot"]}).get_json()
    assert body["ok"] is False
    assert body["reason"] == "insufficient_schedule_data"
    assert "guessing" in body["speakable"]


def test_robot_speakable_includes_its_own_caveats(client, keys):
    """A robot will not add "approximately" by itself, so uncertainty has to be
    inside the sentence it is handed."""
    body = client.get("/api/robot/free-count?grade=05&day=Monday&time=08:10",
                      headers={"x-api-key": keys["robot"]}).get_json()
    assert body["ok"] is True
    assert "could not be identified" in body["speakable"]
