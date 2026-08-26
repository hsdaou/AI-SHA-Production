"""
Teacher-Student App — application factory.

Run with:
    SCHOOL_STAFF_KEY=... python3 -m tsapp        (development)
    gunicorn 'tsapp:create_app()'                (production)
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

from . import db
from .api import BadRequest, api
from .auth import SCOPE_ROBOT, configured, identify, require
from .robot import robot

__version__ = "1.1.0"


def create_app() -> Flask:
    app = Flask(__name__, template_folder="../templates",
                static_folder="../static")
    app.config["JSON_SORT_KEYS"] = False
    app.register_blueprint(api)
    app.register_blueprint(robot)

    @app.errorhandler(BadRequest)
    def _bad_request(e):
        return jsonify({"error": "bad_request", "detail": e.detail}), 400

    @app.errorhandler(db.DatabaseMissing)
    def _no_db(e):
        return jsonify({"error": "no_data", "detail": str(e)}), 503

    @app.errorhandler(404)
    def _not_found(_e):
        return jsonify({"error": "not_found", "detail": request.path}), 404

    @app.errorhandler(500)
    def _server_error(_e):
        # Never echo an exception to the client: these tracebacks quote SQL that
        # contains student identifiers.
        app.logger.exception("unhandled error on %s", request.path)
        return jsonify({"error": "internal_error"}), 500

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    @app.route("/")
    def index():
        try:
            grades = [dict(r) for r in db.grades()]
            ready = True
            issues = len(db.issues("error"))
        except db.DatabaseMissing:
            grades, ready, issues = [], False, 0
        return render_template("index.html", grades=grades, ready=ready,
                               errors=issues, keys=configured(),
                               version=__version__)

    return app


def main() -> int:
    app = create_app()
    host = os.environ.get("SCHOOL_HOST", "127.0.0.1")
    port = int(os.environ.get("SCHOOL_PORT", "5000"))
    # Werkzeug's debugger is remote code execution for anyone who can reach the
    # port. Opt in explicitly, never by default, and never together with a
    # non-loopback bind.
    debug = os.environ.get("SCHOOL_DEBUG") == "1"
    keys = configured()

    print(f"\n  Teacher-Student App {__version__}")
    print(f"  database : {db.db_path()}")
    try:
        meta = db.meta()
        print(f"  contents : {meta.get('students', '?')} students, "
              f"{meta.get('enrolment_rows', '?')} enrolments, "
              f"{meta.get('lessons', '?')} lessons")
    except db.DatabaseMissing:
        print("  contents : NOT BUILT — run python3 -m etl.build")
    print(f"  auth     : staff key {'set' if keys['staff_key'] else 'NOT SET'}, "
          f"robot key {'set' if keys['robot_key'] else 'NOT SET'}")
    if not keys["staff_key"] and not keys["robot_key"]:
        print("             every data endpoint will return 503 until one is set")
    if debug and host != "127.0.0.1":
        print("  REFUSING to run the debugger on a non-loopback interface.")
        return 2
    print(f"  serving  : http://{host}:{port}  (debug={debug})\n")
    app.run(host=host, port=port, debug=debug)
    return 0
