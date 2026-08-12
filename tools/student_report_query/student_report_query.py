#!/usr/bin/env python3
"""AI-SHA client for securely emailing a student report by computer number."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HOME = os.path.expanduser("~/student_report_query")
CONFIG = os.path.join(HOME, "config.json")
SESSION = os.path.expanduser("~/face_auth/session.json")


def fail(message, code):
    print(f"[student-report] {message}", file=sys.stderr)
    raise SystemExit(code)


def student_number(text):
    matches = re.findall(r"\b\d{3,12}\b", text)
    return matches[-1] if matches else None


def session_valid():
    try:
        with open(SESSION, encoding="utf-8") as handle:
            return json.load(handle).get("expires_at", 0) > time.time()
    except (OSError, ValueError):
        return False


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "ask":
        fail('usage: student_report_query.py ask "student report for 12345"', 2)
    number = student_number(" ".join(sys.argv[2:]))
    if not number:
        fail("a student computer number is required", 2)
    if not session_valid():
        fail("DENIED - an administrator must authenticate before requesting a student report.", 3)
    try:
        with open(CONFIG, encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, ValueError) as exc:
        fail(f"invalid configuration: {exc}", 4)
    if not cfg.get("enabled"):
        fail("gate CLOSED", 3)
    url = cfg["base_url"].rstrip("/") + "/api/robot/student-report?" + urllib.parse.urlencode(
        {"student_id": number}
    )
    request = urllib.request.Request(url, headers={"x-robot-key": cfg["robot_key"]})
    try:
        with urllib.request.urlopen(request, timeout=cfg.get("timeout_s", 35)) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {"error": str(exc)}
        if exc.code == 404:
            print("SPEAK: I could not find a student with that computer number.")
            return
        fail(body.get("error", str(exc)), 5)
    except (OSError, urllib.error.URLError) as exc:
        fail(f"cannot reach the student report service: {exc}", 4)
    if not body.get("ok") or not body.get("emailed"):
        fail(body.get("error", "report was not sent"), 5)
    print("SPEAK: " + body.get("speakable", "The student report has been emailed to the administrator."))


if __name__ == "__main__":
    main()
