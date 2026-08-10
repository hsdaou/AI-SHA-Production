#!/usr/bin/env python3
"""
AI-SHA -> HRMS leave-report trigger.

AI-SHA is a THIN TRIGGER. It authorises the administrator locally (face + PIN),
maps an utterance to a fixed intent, and asks the HRMS to build and email the
report. Staff leave data is NEVER returned to the robot -- it does not reach
this device's memory, disk or logs, and is never spoken aloud.

Adds no new pip dependencies (stdlib urllib only).

    python3 hrms_query.py status
    python3 hrms_query.py report --intent on_leave --date 2026-08-09
    python3 hrms_query.py ask "who is on leave today"

Config: ~/hrms_query/config.json (0600), or $HRMS_QUERY_HOME.

    {
      "enabled": false,
      "base_url": "https://your-app.vercel.app",
      "robot_key": "<ROBOT_API_SECRET>",
      "timeout_s": 30
    }
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date as _date, timedelta

HOME = os.environ.get("HRMS_QUERY_HOME", os.path.expanduser("~/hrms_query"))
CONFIG = os.path.join(HOME, "config.json")

# Mirrors auth_gate.py's session contract (tools/face_auth/auth_gate.py).
# Read directly rather than importing auth_gate: that module imports
# face_auth -> onnxruntime/pyrealsense2, which would tie this trigger to
# camera hardware it does not need.
FACE_AUTH_HOME = os.environ.get("FACE_AUTH_HOME", os.path.expanduser("~/face_auth"))
SESSION = os.path.join(FACE_AUTH_HOME, "session.json")

# Fixed intents. The language model's only job is to pick one of these and
# extract a parameter -- it never composes a query.
INTENTS = {
    "on_leave": {
        "path": "/api/reports/on-leave",
        "keywords": ("on leave", "who is off", "absent", "away", "leave today",
                     "sick leave", "on sick", "off today", "who's off"),
        "takes_date": True,
    },
    "balance": {
        "path": "/api/reports/leave-balance",
        "keywords": ("days left", "leave balance", "balance for", "how many days",
                     "days remaining", "remaining leave", "annual leave for",
                     "vacation days"),
        "takes_employee": True,
    },
}

# Words that surround a name in a spoken request but are never part of it.
_NAME_STOP = {
    "how", "many", "days", "left", "does", "do", "has", "have", "is", "are",
    "the", "a", "an", "of", "his", "her", "their", "from", "in", "on", "for",
    "annual", "leave", "balance", "remaining", "vacation", "still", "got",
    "what", "s", "tell", "me", "please", "send", "report", "email", "aisha",
    "hey", "much", "and", "to", "employee", "staff", "member", "mr", "mrs",
    "ms", "dr", "this", "that", "year", "check",
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _fail(msg, code=1):
    print(f"[hrms] {msg}", file=sys.stderr)
    sys.exit(code)


def load_config():
    if not os.path.exists(CONFIG):
        _fail(f"no config at {CONFIG} -- run `status` for the expected shape.", 2)
    with open(CONFIG) as f:
        cfg = json.load(f)
    for key in ("base_url", "robot_key"):
        if not cfg.get(key):
            _fail(f"config.json missing `{key}`", 2)
    return cfg


def session_state():
    """(valid: bool, description: str) -- same rules as auth_gate check-session."""
    if not os.path.exists(SESSION):
        return False, "no active admin session"
    try:
        s = json.load(open(SESSION))
    except (OSError, ValueError) as e:
        return False, f"unreadable session file ({e})"
    left = s.get("expires_at", 0) - time.time()
    if left <= 0:
        return False, f"session EXPIRED for {s.get('user', '?')}"
    return True, f"session VALID for {s.get('user', '?')} ({int(left)}s left)"


def resolve_date(text):
    """Extract an explicit date, else today/tomorrow/yesterday, else today."""
    if not text:
        return None
    m = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if m:
        return m.group(0)
    low = text.lower()
    if "tomorrow" in low:
        return (_date.today() + timedelta(days=1)).isoformat()
    if "yesterday" in low:
        return (_date.today() - timedelta(days=1)).isoformat()
    return None


def route_intent(utterance):
    low = utterance.lower()
    for name, spec in INTENTS.items():
        if any(k in low for k in spec["keywords"]):
            return name
    return None


def resolve_employee(text):
    """Pull a person's name out of an utterance.

    Deliberately crude: strip the words that frame the question and keep what is
    left. The LLM is NOT asked to do this - a 1B model inventing a name would send
    another employee's leave record to the administrator, which is a privacy
    failure, not a formatting one. The server treats the result as a search term
    and refuses when it matches more than one person, so a bad guess fails loudly
    instead of answering about the wrong human.
    """
    if not text:
        return None
    cleaned = re.sub(r"[^a-zA-Z\s'-]", " ", text)
    words = [w for w in cleaned.split() if w.lower() not in _NAME_STOP]
    name = " ".join(words).strip()
    return name if len(name) >= 2 else None


def call_report(cfg, intent, date=None, employee=None):
    spec = INTENTS[intent]
    url = cfg["base_url"].rstrip("/") + spec["path"]
    params = {}
    if date and spec.get("takes_date"):
        params["date"] = date
    if employee and spec.get("takes_employee"):
        params["employee"] = employee
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, method="GET")
    req.add_header("x-robot-key", cfg["robot_key"])
    req.add_header("Accept", "application/json")

    timeout = cfg.get("timeout_s", 30)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": e.reason}
    except urllib.error.URLError as e:
        _fail(f"cannot reach HRMS at {cfg['base_url']}: {e.reason}", 4)


def speak(text):
    """Publish a confirmation on /robot_speech for the Pi TTS tier (optional)."""
    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError:
        print(f"[hrms] (ROS unavailable, not spoken) {text}")
        return
    rclpy.init()
    node = Node("hrms_query_speaker")
    pub = node.create_publisher(String, "/robot_speech", 10)
    time.sleep(0.5)  # let the publisher connect
    pub.publish(String(data=text))
    time.sleep(0.2)
    node.destroy_node()
    rclpy.shutdown()
    print(f"[hrms] spoke: {text}")


def cmd_status(a):
    print(f"config:  {CONFIG} {'(present)' if os.path.exists(CONFIG) else '(MISSING)'}")
    if os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))
        print(f"enabled: {cfg.get('enabled', False)}")
        print(f"base_url:{cfg.get('base_url', '<unset>')}")
        print(f"key:     {'<set>' if cfg.get('robot_key') else '<UNSET>'}")
    ok, desc = session_state()
    print(f"session: {desc}")
    print(f"intents: {', '.join(INTENTS)}")
    sys.exit(0 if os.path.exists(CONFIG) else 2)


def run_report(intent, date, skip_auth, do_speak, employee=None):
    cfg = load_config()

    if not cfg.get("enabled", False):
        _fail("gate CLOSED (`enabled: false` in config.json) -- refusing to query.", 3)

    if skip_auth:
        print("[hrms] *** WARNING: --skip-auth bypasses the admin gate. DEV ONLY. ***",
              file=sys.stderr)
    else:
        ok, desc = session_state()
        if not ok:
            _fail(f"DENIED - {desc}. Run: python3 auth_gate.py authenticate", 3)
        print(f"[hrms] {desc}")

    if INTENTS[intent].get("takes_employee") and not employee:
        _fail("this report needs an employee name -- say e.g. \"how many days "
              "does Sara have left\"", 2)

    status, body = call_report(cfg, intent, date, employee)

    if status == 200 and body.get("ok"):
        # Counts only -- the endpoint returns no personal data by design.
        if intent == "balance":
            print(f"[hrms] OK  balance emailed to "
                  f"{body.get('recipients')} recipient(s)")
        else:
            print(f"[hrms] OK  date={body.get('date')}  on_leave={body.get('count')}  "
                  f"emailed_to={body.get('recipientCount')} recipient(s)")
        if do_speak:
            # Deliberately data-free: never say who is on leave, or how many days
            # anyone has. The robot stands in a public corridor with a speaker.
            speak("The report has been sent to the administrator's email.")
        return 0

    # Name-matching failures deserve their own words: "no such employee" and
    # "several people match" are user errors the administrator can fix by
    # rephrasing, not faults. Still no names -- the count is all we may reveal.
    if status == 404 and body.get("matched") == 0:
        print("[hrms] NO MATCH - no employee by that name.", file=sys.stderr)
        if do_speak:
            speak("I could not find an employee by that name.")
        return 6
    if status == 409:
        print(f"[hrms] AMBIGUOUS - {body.get('matched')} employees match that name; "
              "be more specific.", file=sys.stderr)
        if do_speak:
            speak("Several staff match that name. Please be more specific.")
        return 7

    print(f"[hrms] FAILED (HTTP {status}): {body.get('error', body)}", file=sys.stderr)
    if do_speak:
        speak("I could not retrieve the leave report.")
    return 5


def cmd_report(a):
    if a.intent not in INTENTS:
        _fail(f"unknown intent `{a.intent}` -- known: {', '.join(INTENTS)}", 2)
    if a.date and not DATE_RE.match(a.date):
        _fail("--date must be YYYY-MM-DD", 2)
    sys.exit(run_report(a.intent, a.date, a.skip_auth, a.speak, a.employee))


def cmd_ask(a):
    intent = route_intent(a.utterance)
    if not intent:
        _fail(f"no intent matched {a.utterance!r} -- known: {', '.join(INTENTS)}", 2)
    date = resolve_date(a.utterance)
    employee = (resolve_employee(a.utterance)
                if INTENTS[intent].get("takes_employee") else None)
    print(f"[hrms] intent={intent} date={date or 'today'} employee={employee or '-'}")
    sys.exit(run_report(intent, date, a.skip_auth, a.speak, employee))


def main():
    p = argparse.ArgumentParser(description="AI-SHA -> HRMS leave report trigger")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="show config, session and known intents")
    s.set_defaults(func=cmd_status)

    r = sub.add_parser("report", help="run a specific intent")
    r.add_argument("--intent", required=True)
    r.add_argument("--date", help="YYYY-MM-DD (default: today, school timezone)")
    r.add_argument("--employee", help="name (required by the `balance` intent)")
    r.add_argument("--skip-auth", action="store_true", help="DEV ONLY")
    r.add_argument("--speak", action="store_true", help="publish confirmation on /robot_speech")
    r.set_defaults(func=cmd_report)

    k = sub.add_parser("ask", help="route a natural utterance to an intent")
    k.add_argument("utterance")
    k.add_argument("--skip-auth", action="store_true", help="DEV ONLY")
    k.add_argument("--speak", action="store_true", help="publish confirmation on /robot_speech")
    k.set_defaults(func=cmd_ask)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
