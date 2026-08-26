"""
Access control, and the rule about who may learn a child's name.

WHAT WAS WRONG BEFORE
---------------------
The previous app authenticated `/api/robot/*` with a shared secret and left
`/api/*` — the endpoints that return every student's full name, section and
computer number — completely open. Its own start-up banner said so:

    WARNING: listening beyond localhost with SCHOOL_ROBOT_KEY unset —
             /api/robot/* will refuse every request (fail closed),
             but /api/* has NO auth and exposes student names.

A warning printed to a terminal at boot is not a control. The service was
deployed to listen on a Jetson so a corridor robot could reach it, which means
anything else on that network could enumerate 1,878 minors by name over plain
HTTP.

THE MODEL HERE
--------------
Two scopes, and the default is the safe one:

    ROBOT   counts, timetables, section status. Never a name.
    STAFF   everything, including named lists.

Both need a key. With no key configured the app serves only its landing page and
health check — even diagnostics require authentication, because source paths and
data-quality details are operational information. An unconfigured deployment
must not behave like an open one. Requiring a STAFF key to see names is the
difference between a robot in a
public corridor being able to say "eleven students are free" and being able to
read out which eleven.
"""

from __future__ import annotations

import hmac
import os
from functools import wraps

from flask import g, jsonify, request

SCOPE_ROBOT = "robot"
SCOPE_STAFF = "staff"


def _key(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None


def configured() -> dict:
    return {"robot_key": bool(_key("SCHOOL_ROBOT_KEY")),
            "staff_key": bool(_key("SCHOOL_STAFF_KEY"))}


def identify() -> str | None:
    """The scope this request has proved, or None.

    A staff key grants the robot scope too — staff can see everything a robot
    can. The reverse is never true.
    """
    # Secrets in query strings are copied into browser history, reverse-proxy
    # access logs and referrer URLs. Accept headers only.
    presented = (request.headers.get("x-api-key")
                 or request.headers.get("x-robot-key"))
    if not presented:
        return None
    staff, robot = _key("SCHOOL_STAFF_KEY"), _key("SCHOOL_ROBOT_KEY")
    # compare_digest is constant-time; `==` leaks the key's prefix by timing.
    if staff and hmac.compare_digest(presented, staff):
        return SCOPE_STAFF
    if robot and hmac.compare_digest(presented, robot):
        return SCOPE_ROBOT
    return None


def require(scope: str):
    """Refuse anything that has not proved `scope`. Fails closed when unconfigured."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            keys = configured()
            if not keys["robot_key"] and not keys["staff_key"]:
                return jsonify({
                    "error": "not_configured",
                    "detail": "No SCHOOL_STAFF_KEY or SCHOOL_ROBOT_KEY is set. "
                              "This service refuses every data request until one "
                              "is, so that an unconfigured deployment is not an "
                              "open one.",
                }), 503
            have = identify()
            if have is None:
                return jsonify({"error": "unauthorized"}), 401
            if scope == SCOPE_STAFF and have != SCOPE_STAFF:
                return jsonify({
                    "error": "forbidden",
                    "detail": "This endpoint returns named students. It requires "
                              "the staff key, not the robot key.",
                }), 403
            g.scope = have
            return fn(*a, **kw)
        return wrapper
    return decorator


def scope() -> str:
    return getattr(g, "scope", SCOPE_ROBOT)


def redact(students: list[dict]) -> list[dict]:
    """Strip anything identifying. Applied whenever the caller is not staff."""
    return [{k: v for k, v in s.items()
             if k not in ("computer_number", "first_name", "last_name")}
            for s in students]
