#!/usr/bin/env python3
"""AI-SHA skill: school timetable questions (free students, free teachers, class schedule).

Thin trigger over the school timetable app's /api/robot/* surface. Zero new pip
deps (stdlib urllib only), matching the HRMS and video-message skills.

TWO CLASSES OF ANSWER, and the difference is deliberate:

  SPEAKABLE  timetable for a section, and COUNTS of free students. These identify
             nobody, so the robot may say them out loud in the corridor. No admin
             session is required - a student may reasonably ask what Grade 7 A has
             on Monday.

  EMAILED    any list of NAMED students or teachers. These are minors and staff;
             the app renders and emails the list, the robot receives only
             {ok, count} and says it has been sent. Requires an authenticated
             admin session, exactly like the HRMS skill.

Config: ~/timetable_query/config.json (0600), or $TIMETABLE_QUERY_HOME.
    {"enabled": true, "base_url": "http://...", "robot_key": "...", "timeout_s": 20}

Exit codes: 0 ok · 2 usage/no intent · 3 gate closed or no admin session ·
            4 app unreachable · 5 app returned an error
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import date as _date_cls, timedelta as _timedelta
import urllib.error
import urllib.parse
import urllib.request

HOME = os.environ.get("TIMETABLE_QUERY_HOME", os.path.expanduser("~/timetable_query"))
CONFIG = os.path.join(HOME, "config.json")

FACE_AUTH_HOME = os.environ.get("FACE_AUTH_HOME", os.path.expanduser("~/face_auth"))
SESSION = os.path.join(FACE_AUTH_HOME, "session.json")

ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
            "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}

DAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# needs_admin marks the intents that cause a NAMED list to be emailed.
INTENTS = {
    "timetable": {
        "path": "/api/robot/timetable",
        "keywords": ("timetable", "time table", "schedule for", "what does grade",
                     "what do they have", "lessons for"),
        "needs_admin": False,
    },
    "free_count": {
        "path": "/api/robot/free-count",
        "keywords": ("how many students are free", "how many are free",
                     "how many free students", "count of free",
                     "how many students are available", "how many are available"),
        "needs_admin": False,
    },
    "free_students": {
        "path": "/api/robot/free-students",
        # "available" is how people actually ask. Its absence sent every such
        # question to the knowledge base, which invented teacher availability from
        # prospectus documents.
        "keywords": ("which students are free", "who is free", "who's free",
                     "free students", "students free", "list free students",
                     "which students are available", "students are available",
                     "which students available", "who is available"),
        "needs_admin": True,
    },
    "free_teachers": {
        "path": "/api/robot/free-teachers",
        "keywords": ("which teachers are free", "free teachers", "teachers free",
                     "which teacher is free", "which teachers are available",
                     "teachers are available", "which teacher is available",
                     "available teachers", "teachers available"),
        "needs_admin": True,
    },
}


def _fail(msg, code=1):
    print(f"[timetable] {msg}", file=sys.stderr)
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
    """(valid, description) -- same contract as auth_gate check-session."""
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


def route_intent(utterance):
    """Longest keyword wins, so "how many students are free" does not match the
    shorter "students free" and email a list nobody asked for."""
    low = utterance.lower()
    best, best_len = None, 0
    for name, spec in INTENTS.items():
        for kw in spec["keywords"]:
            if kw in low and len(kw) > best_len:
                best, best_len = name, len(kw)
    return best


def parse_params(utterance):
    """Grade, section and day out of plain speech. Anything absent is left to the
    server, which falls back to today and the current period."""
    low = utterance.lower()
    out = {}
    m = re.search(r"grade\s*(\d{1,2})", low) or re.search(r"\bg(\d{1,2})\b", low)
    if m:
        out["grade"] = m.group(1)
    m = re.search(r"section\s*([a-z])\b", low)
    if m:
        out["section"] = m.group(1).upper()
    elif "grade" in low:
        m = re.search(r"grade\s*\d{1,2}\s*([a-z])\b", low)
        if m:
            out["section"] = m.group(1).upper()
    for d in DAYS:
        if d in low:
            out["day"] = d.capitalize()
            break
    # "tomorrow" must be RESOLVED, not dropped. Dropping it let the server fall
    # back to today, so "who is free tomorrow" silently answered for today - the
    # worst kind of wrong, because it looks like an answer.
    if "tomorrow" in low:
        out["day"] = (_date_cls.today() + _timedelta(days=1)).strftime("%A")
    elif "today" in low:
        out["day"] = _date_cls.today().strftime("%A")

    m = re.search(r"period\s*(\d{1,2})", low) or re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+period", low)
    if m:
        out["period"] = f"Period {m.group(1)}"
    else:
        # People say "the third period", not "period 3".
        for word, n in ORDINALS.items():
            if re.search(rf"\b{word}\b\s+period|period\s+\b{word}\b", low):
                out["period"] = f"Period {n}"
                break
    return out


def call_api(cfg, intent, params):
    spec = INTENTS[intent]
    url = cfg["base_url"].rstrip("/") + spec["path"]
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET")
    req.add_header("x-robot-key", cfg["robot_key"])
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=cfg.get("timeout_s", 20)) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, {"error": e.reason}
    except urllib.error.URLError as e:
        _fail(f"cannot reach the timetable app at {cfg['base_url']}: {e.reason}", 4)


def run(intent, params, skip_auth=False):
    cfg = load_config()
    if not cfg.get("enabled", False):
        _fail("gate CLOSED (`enabled: false` in config.json) -- refusing to query.", 3)

    if INTENTS[intent]["needs_admin"]:
        if skip_auth:
            print("[timetable] *** WARNING: --skip-auth bypasses the admin gate. DEV ONLY. ***",
                  file=sys.stderr)
        else:
            ok, desc = session_state()
            if not ok:
                who = "teachers" if intent == "free_teachers" else "students"
                _fail(f"DENIED - {desc}. That answer names {who}, so it needs an "
                      "authenticated administrator.", 3)
            print(f"[timetable] {desc}")

    status, body = call_api(cfg, intent, params)

    if status == 200 and body.get("ok"):
        if body.get("emailed"):
            print(f"[timetable] OK  {body.get('count')} listed, emailed to "
                  f"{body.get('recipients')} recipient(s)")
            # SPEAK is data-free on purpose: never read out who is free.
            print("SPEAK: The list has been sent to the administrator's email.")
        else:
            print(f"[timetable] OK  {json.dumps({k: v for k, v in body.items() if k != 'periods'})[:200]}")
            print("SPEAK: " + (body.get("speakable") or "Done."))
        return 0

    # A 200 with ok=false is a REASON, not a fault: the school is between lessons,
    # or the enrolment data needed to answer is absent. Say the reason plainly
    # rather than reporting a failure the administrator cannot act on.
    if status == 200 and body.get("reason"):
        print("SPEAK: " + (body.get("speakable")
                           or "I cannot answer that at the moment."))
        return 0

    print(f"[timetable] FAILED (HTTP {status}): {body.get('error', body)}", file=sys.stderr)
    return 5


def cmd_ask(a):
    intent = route_intent(a.utterance)
    if not intent:
        _fail(f"no intent matched {a.utterance!r} -- known: {', '.join(INTENTS)}", 2)
    params = parse_params(a.utterance)
    if intent in ("timetable", "free_count", "free_students") and "grade" not in params:
        _fail("which grade? say e.g. \"what is the timetable for grade 7 section A on Monday\"", 2)
    print(f"[timetable] intent={intent} params={params}")
    sys.exit(run(intent, params, a.skip_auth))


def cmd_status(a):
    print(f"config:  {CONFIG} {'(present)' if os.path.exists(CONFIG) else '(MISSING)'}")
    if os.path.exists(CONFIG):
        cfg = json.load(open(CONFIG))
        print(f"enabled: {cfg.get('enabled')}")
        print(f"base_url:{cfg.get('base_url')}")
        print(f"key:     {'<set>' if cfg.get('robot_key') else '<MISSING>'}")
    ok, desc = session_state()
    print(f"session: {desc}")
    print("intents: " + ", ".join(
        f"{k}{'*' if v['needs_admin'] else ''}" for k, v in INTENTS.items()))
    print("         * = names people, so it is emailed and needs an admin session")


def main():
    p = argparse.ArgumentParser(description="AI-SHA -> school timetable trigger")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status"); s.set_defaults(func=cmd_status)
    k = sub.add_parser("ask"); k.add_argument("utterance")
    k.add_argument("--skip-auth", action="store_true", help="DEV ONLY")
    k.set_defaults(func=cmd_ask)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
