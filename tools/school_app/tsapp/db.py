"""
Read-only access to the built database.

The previous app cached parsed spreadsheets in a module-level `_cache = {}` dict
with no size bound, no invalidation and no thread safety, and separately opened a
fresh `sqlite3.connect` for every section it examined. Answering "who is free in
grade 6" opened eleven connections and ran eleven queries; the first request to
touch students parsed a 1.5 MB and a 190 KB workbook inline, on the request
thread, holding ~23,000 rows in memory forever afterwards. Re-deploying the data
required restarting the process, because nothing ever noticed the file changed.

Here the database is opened read-only, one connection per thread, and the file's
mtime is checked so a rebuild is picked up without a restart. Nothing is cached
in Python: SQLite's own page cache is better at this, and it does not go stale.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

_local = threading.local()


class DatabaseMissing(RuntimeError):
    pass


def db_path() -> str:
    return os.environ.get("SCHOOL_DB") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "school.db")


def connect() -> sqlite3.Connection:
    """A read-only connection for this thread, reopened if the file changed."""
    path = os.path.realpath(db_path())
    if not os.path.exists(path):
        raise DatabaseMissing(
            f"{path} does not exist. Build it with:\n"
            f"  python3 -m etl.build --class-lists DIR --subject-lists DIR "
            f"--teacher-xlsx FILE --timetable-db FILE")
    # Keyed on the path as well as the timestamp: keying on mtime alone means
    # pointing SCHOOL_DB at a different file with a coincidentally equal mtime
    # keeps serving the old one.
    stat = os.stat(path)
    stamp = (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    conn = getattr(_local, "conn", None)
    if conn is not None and getattr(_local, "stamp", None) == stamp:
        return conn
    if conn is not None:
        conn.close()

    # mode=ro: the app has no business writing here, and saying so means a bug
    # cannot corrupt the dataset — it raises instead.
    # Path.as_uri quotes '?' and '#' in filenames instead of letting SQLite
    # mistake them for URI parameters.  The connection remains thread-bound;
    # each worker thread owns its own instance via ``_local``.
    conn = sqlite3.connect(f"{Path(path).as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA trusted_schema = OFF")
    _local.conn = conn
    _local.stamp = stamp
    return conn


def query(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    return connect().execute(sql, params).fetchall()


def one(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    return connect().execute(sql, params).fetchone()


def meta() -> dict[str, str]:
    return {r["key"]: r["value"] for r in query("SELECT key, value FROM build_meta")}


def grades() -> list[sqlite3.Row]:
    return query("""SELECT g.code, g.label, g.ordinal,
                           gc.students, gc.coverage, gc.verdict,
                           gc.has_section_timetable
                    FROM grades g
                    LEFT JOIN grade_coverage gc ON gc.grade = g.code
                    ORDER BY g.ordinal""")


def coverage(grade: str) -> sqlite3.Row | None:
    return one("SELECT * FROM grade_coverage WHERE grade = ?", (grade,))


def issues(severity: str | None = None) -> list[sqlite3.Row]:
    if severity:
        return query("SELECT severity, category, detail, n FROM build_issues "
                     "WHERE severity = ? ORDER BY category", (severity,))
    return query("SELECT severity, category, detail, n FROM build_issues "
                 "ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warning' "
                 "THEN 1 ELSE 2 END, category")
