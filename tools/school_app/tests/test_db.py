"""Read-only database connection lifecycle."""

import os
import sqlite3

from tsapp import db


def _database(path, value):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE build_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO build_meta VALUES ('version', ?)", (value,))
    conn.commit()
    conn.close()


def test_database_path_with_uri_characters_is_opened_literally(tmp_path, monkeypatch):
    path = tmp_path / "school?#copy.db"
    _database(path, "literal-path")
    monkeypatch.setenv("SCHOOL_DB", str(path))
    assert db.meta()["version"] == "literal-path"


def test_atomic_replacement_is_seen_even_with_same_timestamp(tmp_path, monkeypatch):
    live = tmp_path / "school.db"
    replacement = tmp_path / "replacement.db"
    _database(live, "one")
    original_stat = live.stat()
    monkeypatch.setenv("SCHOOL_DB", str(live))
    assert db.meta()["version"] == "one"

    # Same-length value and forced timestamp remove the two signals the old
    # cache key used. The inode still changes on atomic replacement.
    _database(replacement, "two")
    os.utime(replacement, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    os.replace(replacement, live)
    assert live.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert db.meta()["version"] == "two"
