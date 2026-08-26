#!/usr/bin/env python3
"""
Build data/school.db from the school's own reports.

    python3 -m etl.build --class-lists   <dir of "NN Class List.PDF"> \
                         --subject-lists <dir of "NN Subject List.PDF"> \
                         --teacher-xlsx  Teacher_Schedule_Database.xlsx \
                         --timetable-db  school_timetable.db \
                         --out data/school.db

Only the first two are required. The schedule sources are optional and the build
records exactly which grades they cover, so the app can distinguish "this
student is free" from "I have no schedule for this grade" — a distinction the
previous version could not make and therefore answered as "free".

The build is a full rebuild into a temporary file which is moved into place at
the end, so a failed build leaves the previous database serving.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone

from .class_lists import parse_class_list
from .schedule import (lessons_from_section_grid, load_lessons,
                       load_section_periods, match_groups)
from .subject_lists import parse_subject_list
from .subjects import subject_head
from .timegrid import hhmm

# A grade needs most of its enrolment tied to real lessons before a free/busy
# answer about it means anything. Below `PARTIAL` the app refuses to count.
GOOD_COVERAGE = 0.75
PARTIAL_COVERAGE = 0.40

PLACEHOLDER_TEACHERS = {"SSTUDY", "ENEW2"}

# ── Name reconciliation between the roster and the timetable ────────────────
# The class lists and the timetable occasionally spell the same section or room
# differently. Merging them is declared here explicitly rather than guessed,
# because merging two that are genuinely distinct would put children in the
# wrong room — a worse failure than leaving them unresolved.
#
# Every entry is checked against the data at build time (see _alias_sections and
# _alias_rooms) and skipped, with a warning, if its preconditions no longer hold.
# So an alias becomes inert by itself once the source reports are corrected.
#
# ("11", "L") -> "LA"
#   Grade 11's class list names a section L holding 31 students with no timetable
#   at all; the timetable names LA with 44 cells — the same as every other
#   grade-11 section — and no students. Six of those 31 students' own teachers
#   (MTA2, NMA1, ESB1, SRC2, AHI1, EJD2) teach lessons in room 11LA, and no
#   teacher in the school teaches in a room called 11L.
#   Grade 10 is the control: it really does have a section L, with its own 44
#   timetable cells and 41 lessons in room 10L. This is not a rule about "L".
SECTION_ALIASES: dict[tuple[str, str], str] = {("11", "L"): "LA"}

# Room names the subject lists print truncated, where exactly ONE real room
# extends them — so the expansion is forced, not chosen. Both are the same shape:
# the school's single "L" (Level) section, whose room the enrolment report cuts
# one character short.
#
# "11L" -> "11LA"   36 lessons in 11LA, none in 11L; 2 of the 3 affected groups
#                   carry course group "LA".
# "09L" -> "09LA"   grade 9's only L-ish section is LA; 12 of the 14 affected
#                   groups carry course group "LA" and 9 are taught by someone
#                   who teaches in 09LA.
#
# Deliberately NOT aliased, because more than one real room extends them and
# guessing would put children in a room they are not in — "12S" (12SA..12SE),
# "12L" (12LA/12LB), "11S", and bare "11". _report_unmatched_rooms lists these
# every build so they stay visible instead of being quietly wrong.
ROOM_ALIASES: dict[str, str] = {"11L": "11LA", "09L": "09LA"}
MAX_REPORT_ARCHIVE_BYTES = 512 * 1024 * 1024


@contextlib.contextmanager
def pdf_inputs(source: str, marker: str):
    """Yield the report PDFs from a directory, one PDF, or a ZIP archive.

    The school distributes these reports as ZIP files.  Making archives a
    first-class input avoids persistent plaintext copies of student reports and
    lets build provenance name the immutable file the user actually supplied.
    Archive members are copied into a private temporary directory by basename;
    paths inside the ZIP are never trusted or extracted directly.
    """
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, f"*{marker}*")))
        yield files
        return

    if os.path.isfile(source) and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive, tempfile.TemporaryDirectory(
                prefix="tsapp-reports-") as directory:
            members = [m for m in archive.infolist()
                       if not m.is_dir()
                       and marker.lower() in os.path.basename(m.filename).lower()
                       and m.filename.lower().endswith(".pdf")]
            names = [os.path.basename(m.filename) for m in members]
            total_size = sum(m.file_size for m in members)
            if total_size > MAX_REPORT_ARCHIVE_BYTES:
                raise SystemExit(
                    f"{source} expands to {total_size:,} bytes of reports; "
                    f"refusing an archive above {MAX_REPORT_ARCHIVE_BYTES:,} bytes")
            duplicates = sorted(name for name, n in Counter(names).items() if n > 1)
            if duplicates:
                raise SystemExit(
                    f"{source} contains duplicate report names: "
                    f"{', '.join(duplicates)}")
            files = []
            for member, name in sorted(zip(members, names), key=lambda pair: pair[1]):
                destination = os.path.join(directory, name)
                with archive.open(member) as src, open(destination, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                files.append(destination)
            yield files
        return

    if (os.path.isfile(source) and marker.lower() in os.path.basename(source).lower()
            and source.lower().endswith(".pdf")):
        yield [source]
        return

    yield []


def source_sha256(path: str, marker: str | None = None) -> str:
    """A stable digest for a source file or a directory of report PDFs."""
    digest = hashlib.sha256()

    def add_file(file_path: str, logical_name: str = ""):
        if logical_name:
            digest.update(logical_name.encode("utf-8"))
            digest.update(b"\0")
        with open(file_path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)

    if os.path.isfile(path):
        add_file(path)
        return digest.hexdigest()
    if os.path.isdir(path):
        pattern = f"*{marker}*" if marker else "*"
        for file_path in sorted(glob.glob(os.path.join(path, pattern))):
            if os.path.isfile(file_path):
                add_file(file_path, os.path.basename(file_path))
        return digest.hexdigest()
    return ""


def grade_label(code: str) -> str:
    if code.startswith("K"):
        return f"KG{code[1:]}"
    return f"Grade {int(code)}"


def grade_ordinal(code: str) -> int:
    return {"K1": -1, "K2": 0}.get(code, int(code) if code.isdigit() else 99)


def _split_title(full: str | None) -> tuple[str | None, str | None]:
    if not full:
        return None, None
    parts = full.split(None, 1)
    if parts and parts[0].rstrip(".") in ("Mr", "Ms", "Mrs", "Dr", "Miss"):
        return parts[0], (parts[1].strip() if len(parts) > 1 else None)
    return None, full.strip()


class Build:
    def __init__(self, out_path: str):
        self.out_path = out_path
        self.issues: list[tuple[str, str, str, int]] = []
        self.stats: dict[str, int] = {}
        fd, self.tmp_path = tempfile.mkstemp(
            suffix=".db", dir=os.path.dirname(os.path.abspath(out_path)) or ".")
        os.close(fd)
        os.unlink(self.tmp_path)
        self.conn = sqlite3.connect(self.tmp_path)
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "schema.sql")) as f:
            self.conn.executescript(f.read())

    def issue(self, severity: str, category: str, detail: str, n: int = 1):
        self.issues.append((severity, category, detail, n))

    def abort(self):
        """Close and remove only this build's unpublished temporary database."""
        try:
            self.conn.close()
        except sqlite3.ProgrammingError:
            pass
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    # ── sources ──────────────────────────────────────────────────────────
    def load_roster(self, source: str):
        rows, pages, skipped = [], 0, 0
        with pdf_inputs(source, "Class List") as files:
            if not files:
                raise SystemExit(f"no class lists found in {source}")
            for path in files:
                r = parse_class_list(path)
                rows += r.rows
                pages += r.pages_parsed
                skipped += r.pages_skipped
                for w in r.warnings:
                    self.issue("warning", "class_list", w)
        if skipped:
            self.issue("warning", "class_list",
                       f"{skipped} of {pages + skipped} class-list pages yielded "
                       f"no students", skipped)
        seen: dict[str, object] = {}
        for row in rows:
            if row.computer_number in seen:
                self.issue("warning", "duplicate_student",
                           f"computer number {row.computer_number} appears twice "
                           f"({seen[row.computer_number].source} and {row.source}); "
                           f"keeping the first")
                continue
            seen[row.computer_number] = row
        print(f"  roster        {len(seen)} students, {pages} pages")
        return list(seen.values())

    def load_enrolment(self, source: str):
        enrolments, groups, pages, skipped = [], set(), 0, 0
        with pdf_inputs(source, "Subject List") as files:
            if not files:
                raise SystemExit(f"no subject lists found in {source}")
            for path in files:
                r = parse_subject_list(path)
                enrolments += r.enrolments
                groups |= r.groups
                pages += r.pages_parsed
                skipped += r.pages_skipped
                for w in r.warnings:
                    self.issue("info", "subject_list", w)
        if skipped:
            self.issue("error", "subject_list",
                       f"{skipped} subject-list pages could not be parsed; the "
                       f"enrolment they carry is missing", skipped)
        print(f"  enrolment     {len(enrolments)} rows, {len(groups)} groups, "
              f"{pages} pages")
        return enrolments, sorted(
            groups, key=lambda g: (g.grade, g.subject_code, g.course_group,
                                   g.room or "", g.teacher_code or ""))

    # ── name reconciliation ──────────────────────────────────────────────
    def _alias_sections(self, roster, section_rows):
        """Merge class-list sections into the timetable's name for them.

        Applied only where the evidence still points the same way: the source
        section must have students and no timetable, and the target must have a
        timetable. If the reports are ever corrected upstream, the precondition
        fails, the alias is skipped, and the build says so.
        """
        timetabled = {(r["grade"], r["letter"]) for r in section_rows}
        on_roster = {(r.grade, r.section) for r in roster}

        applied: dict[tuple[str, str], str] = {}
        for (grade, src), dst in sorted(SECTION_ALIASES.items()):
            if (grade, src) not in on_roster:
                self.issue("info", "section_alias_unused",
                           f"the declared alias grade {grade} section {src} -> {dst} "
                           f"matched no section on any class list; it can be removed")
            elif (grade, src) in timetabled:
                self.issue("warning", "section_alias_skipped",
                           f"NOT merging grade {grade} section {src} into {dst}: "
                           f"{src} now has a timetable of its own, so they are two "
                           f"different sections. Remove the alias.")
            elif (grade, dst) not in timetabled:
                self.issue("warning", "section_alias_skipped",
                           f"NOT merging grade {grade} section {src} into {dst}: "
                           f"{dst} has no timetable either, so the merge would not "
                           f"resolve anything.")
            else:
                applied[(grade, src)] = dst

        if not applied:
            return roster

        moved: Counter = Counter()
        out = []
        for r in roster:
            dst = applied.get((r.grade, r.section))
            if dst is None:
                out.append(r)
            else:
                moved[(r.grade, r.section, dst)] += 1
                out.append(replace(r, section=dst))
        for (grade, src, dst), n in sorted(moved.items()):
            self.issue("info", "section_alias",
                       f"grade {grade}: {n} students listed under class-list section "
                       f"{src} were merged into timetable section {dst}, which had a "
                       f"timetable but no students. Without this they would be "
                       f"reported as unknown in every period.", n)
        return out

    @staticmethod
    def _name_key(name: str | None) -> str:
        if not name:
            return ""
        return re.sub(r"[^a-z]", "", re.sub(r"^(mr|ms|mrs|dr|miss)\.?\s*", "",
                                            name.strip().lower()))

    def _resolve_teacher_identities(self, groups, enrolments, sched):
        """Point each teaching group at the code the SCHEDULE uses for its teacher.

        The two reports do not agree on teacher codes, and — worse — they reuse
        them. `MKM2` is Mr Kevin Mercier teaching Music in the subject lists and
        Karen Merched teaching Mathematics in the teacher schedule. Keyed on the
        code alone, 628 of Mr Mercier's enrolments are filed under Ms Merched's
        name, and a Music group is one room-coincidence away from being matched to
        a Mathematics lesson.

        The name is the stable identity, so the code is resolved through it:

            MKM2  Kevin Mercier    -> OKM1   (his code in the schedule)
            EKM2  Karen Merched    -> MKM2   (hers)
            MRA1  Rafat Aly        -> ERA1
            ENIV1 Inneke Vermeulen -> EIV1

        A group is only re-pointed when the schedule holds exactly one code for
        that name, so an ambiguous name changes nothing.
        """
        if not sched:
            return groups, enrolments

        by_name: dict[str, set[str]] = defaultdict(set)
        for code, name in sched.teacher_names.items():
            key = self._name_key(name)
            if key:
                by_name[key].add(code)
        scheduled = {l.teacher_code for l in sched.lessons}

        # Key by BOTH source code and source name. Codes are known to be reused,
        # so a code-only remap can fix one person by incorrectly moving another
        # person who happens to share that code.
        remap: dict[tuple[str, str], str] = {}
        collisions: list[str] = []
        for g in groups:
            code, key = g.teacher_code, self._name_key(g.teacher_name)
            source_identity = (code or "", key)
            if not code or not key or source_identity in remap:
                continue
            candidates = by_name.get(key, set()) & scheduled
            if code in candidates:
                continue                       # code and name already agree
            if len(candidates) != 1:
                continue                       # unknown, or ambiguous — leave alone
            target = next(iter(candidates))
            remap[source_identity] = target
            if code in scheduled:
                collisions.append(
                    f"{code} is {g.teacher_name} in the subject lists but "
                    f"{sched.teacher_names.get(code)} in the teacher schedule")

        if not remap:
            return groups, enrolments

        for detail in collisions:
            self.issue("error", "teacher_code_collision",
                       f"{detail}. Enrolment recorded under it has been re-pointed "
                       f"by name; matching on the code alone would have attributed "
                       f"one teacher's students to another.")
        moved: Counter = Counter()
        for e in enrolments:
            identity = (e.group.teacher_code or "",
                        self._name_key(e.group.teacher_name))
            if identity in remap:
                moved[identity] += 1
        for (src, key), dst in sorted(remap.items()):
            self.issue("info", "teacher_identity",
                       f"teaching groups under code {src} were matched to schedule "
                       f"code {dst}, the same person under a different code "
                       f"({moved.get((src, key), 0)} enrolments)",
                       moved.get((src, key), 0))

        def canon(g):
            identity = (g.teacher_code or "", self._name_key(g.teacher_name))
            dst = remap.get(identity)
            return replace(g, teacher_code=dst) if dst else g

        new_groups = sorted({canon(g) for g in groups},
                            key=lambda g: (g.grade, g.subject_code, g.course_group,
                                           g.room or "", g.teacher_code or ""))
        return new_groups, [replace(e, group=canon(e.group)) for e in enrolments]

    def _audit_source_consistency(self, roster, enrolments):
        """Cross-check the two independent reports before transforming either.

        Subject reports repeat a student's grade and home section. They are not
        the roster authority, but agreement is a strong parser and source-data
        invariant. A cross-grade enrolment is never linked: it can place a child
        in another grade's lesson.
        """
        own = {r.computer_number: (r.grade, r.section) for r in roster}
        compared = matched = 0
        section_mismatches = []
        grade_mismatches = []
        for e in enrolments:
            roster_place = own.get(e.computer_number)
            if roster_place is None:
                continue
            if roster_place[0] != e.group.grade:
                grade_mismatches.append(
                    (e.computer_number, roster_place[0], e.group.grade))
            if e.section_hint:
                compared += 1
                if roster_place[1] == e.section_hint:
                    matched += 1
                else:
                    section_mismatches.append(
                        (e.computer_number, roster_place[1], e.section_hint))

        self.stats.update({
            "source_section_comparisons": compared,
            "source_section_matches": matched,
            "source_section_mismatches": len(section_mismatches),
            "source_grade_mismatches": len(grade_mismatches),
        })
        if section_mismatches:
            sample = ", ".join(
                f"{number} roster={roster_sec} subject-list={subject_sec}"
                for number, roster_sec, subject_sec in section_mismatches[:5])
            self.issue(
                "error", "student_section_mismatch",
                f"{len(section_mismatches)} enrolment rows disagree with the class "
                f"list about the student's home section ({sample}). The class "
                f"list remains authoritative; investigate the source or parser.",
                len(section_mismatches))
        if grade_mismatches:
            sample = ", ".join(
                f"{number} roster={roster_grade} subject-list={subject_grade}"
                for number, roster_grade, subject_grade in grade_mismatches[:5])
            self.issue(
                "error", "student_grade_mismatch",
                f"{len(grade_mismatches)} enrolment rows point a student at a "
                f"different grade ({sample}); those links were dropped rather "
                f"than placing a child into another grade's lessons.",
                len(grade_mismatches))

    def _alias_rooms(self, groups, enrolments, sched):
        """Canonicalise teaching-group room names against the schedule's."""
        if not sched:
            return groups, enrolments
        in_schedule = {room for lesson in sched.lessons for room in lesson.rooms}
        in_groups = {g.room for g in groups if g.room}

        applied: dict[str, str] = {}
        for src, dst in sorted(ROOM_ALIASES.items()):
            if src not in in_groups:
                self.issue("info", "room_alias_unused",
                           f"the declared room alias {src} -> {dst} matched no "
                           f"teaching group; it can be removed")
            elif src in in_schedule:
                self.issue("warning", "room_alias_skipped",
                           f"NOT renaming room {src} to {dst}: {src} is a real room "
                           f"in the schedule. Remove the alias.")
            elif dst not in in_schedule:
                self.issue("warning", "room_alias_skipped",
                           f"NOT renaming room {src} to {dst}: {dst} does not appear "
                           f"in the schedule either.")
            else:
                applied[src] = dst

        if not applied:
            return groups, enrolments

        def canon(g):
            dst = applied.get(g.room)
            return replace(g, room=dst) if dst else g

        n_groups = Counter(g.room for g in groups if g.room in applied)
        # Aliasing can collapse two group records into one; the set and the
        # enrolment pair-set below both de-duplicate, so a student enrolled
        # through either spelling ends up in the surviving group exactly once.
        new_groups = sorted({canon(g) for g in groups},
                            key=lambda g: (g.grade, g.subject_code, g.course_group,
                                           g.room or "", g.teacher_code or ""))
        new_enrolments = [replace(e, group=canon(e.group)) for e in enrolments]
        for src, n in sorted(n_groups.items()):
            self.issue("info", "room_alias",
                       f"{n} teaching group(s) recorded in room {src} were matched "
                       f"against the schedule's name for it, {applied[src]}", n)
        return new_groups, new_enrolments

    # ── writing ──────────────────────────────────────────────────────────
    def write(self, roster, enrolments, groups, sched, section_rows):
        c = self.conn
        self._audit_source_consistency(roster, enrolments)
        roster = self._alias_sections(roster, section_rows)
        groups, enrolments = self._resolve_teacher_identities(groups, enrolments, sched)
        groups, enrolments = self._alias_rooms(groups, enrolments, sched)

        grade_codes = sorted({r.grade for r in roster} | {g.grade for g in groups}
                             | {r["grade"] for r in section_rows},
                             key=grade_ordinal)
        c.executemany("INSERT INTO grades (code, ordinal, label) VALUES (?,?,?)",
                      [(g, grade_ordinal(g), grade_label(g)) for g in grade_codes])

        section_keys = sorted({(r.grade, r.section) for r in roster}
                              | {(r["grade"], r["letter"]) for r in section_rows},
                              key=lambda k: (grade_ordinal(k[0]), k[1]))
        c.executemany("INSERT INTO sections (grade, letter) VALUES (?,?)",
                      section_keys)
        section_id = {(g, l): i for i, (g, l) in enumerate(section_keys, start=1)}

        # Teachers: names come from whichever source has them.
        names: dict[str, str] = dict(sched.teacher_names if sched else {})
        for g in groups:
            if g.teacher_code and g.teacher_name:
                names.setdefault(g.teacher_code, g.teacher_name)
        codes = set(names) | {g.teacher_code for g in groups if g.teacher_code}
        if sched:
            codes |= {l.teacher_code for l in sched.lessons}
        # The section timetable names staff that neither the subject lists nor the
        # teacher schedule do. They are real teachers with real lessons; we simply
        # have no name for them, which is recorded below rather than used as a
        # reason to drop their lessons.
        codes |= {r["teacher_code"] for r in section_rows if r.get("teacher_code")}
        teacher_rows = []
        for code in sorted(codes):
            title, name = _split_title(names.get(code))
            teacher_rows.append((code, title, name,
                                 1 if code in PLACEHOLDER_TEACHERS else 0))
        c.executemany("INSERT INTO teachers (code, title, name, is_placeholder) "
                      "VALUES (?,?,?,?)", teacher_rows)
        for code in sorted(PLACEHOLDER_TEACHERS & codes):
            self.issue("info", "placeholder_teacher",
                       f"{code} ({names.get(code)}) is a staffing placeholder, not "
                       f"a person; excluded from teacher availability")

        subjects = {}
        for g in groups:
            subjects.setdefault(g.subject_code, g.subject_name)
        c.executemany("INSERT INTO subjects (code, name, head) VALUES (?,?,?)",
                      [(k, v, subject_head(v)) for k, v in sorted(subjects.items())])

        c.executemany(
            "INSERT INTO students (computer_number, first_name, last_name, "
            "section_id, family_number, siblings, source) VALUES (?,?,?,?,?,?,?)",
            [(r.computer_number, r.first_name, r.last_name,
              section_id[(r.grade, r.section)], r.family_number, r.siblings,
              r.source) for r in roster])
        c.execute("""UPDATE sections SET student_count =
                     (SELECT COUNT(*) FROM students WHERE section_id = sections.id)""")

        # Recover lessons for staff the teacher schedule omits, from the class
        # grid. Done before matching so those groups can be placed at all.
        if sched and section_rows:
            recovered = lessons_from_section_grid(section_rows, sched.lessons)
            if recovered:
                new_staff = {l.teacher_code for l in recovered} - {
                    l.teacher_code for l in sched.lessons}
                sched.lessons.extend(recovered)
                self.issue("info", "lessons_recovered",
                           f"{len(recovered)} lessons for {len(new_staff)} members "
                           f"of staff were recovered from the class timetable, "
                           f"which records them where the teacher schedule does "
                           f"not. Without this their students cannot be placed and "
                           f"they themselves read as absent rather than teaching.",
                           len(recovered))

        # Lessons and their repaired rooms.
        lesson_ids: dict[int, int] = {}
        if sched:
            for i, l in enumerate(sched.lessons):
                cur = c.execute(
                    "INSERT INTO lessons (teacher_code, day, start_min, end_min, "
                    "sheet_period, subject_text, group_code, room_text, source) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (l.teacher_code, l.day, l.start_min, l.end_min, l.sheet_period,
                     l.subject_text, l.group_code, l.room_text, l.source))
                lesson_ids[i] = cur.lastrowid
                c.executemany(
                    "INSERT OR IGNORE INTO lesson_rooms (lesson_id, room) VALUES (?,?)",
                    [(cur.lastrowid, room) for room in l.rooms])
            for token, n in sorted(sched.unresolved_rooms.items()):
                self.issue("warning", "room_unresolved",
                           f"room {token!r} appears with no grade prefix and no "
                           f"sibling to borrow one from; {n} lesson(s) affected "
                           f"cannot be matched by room", n)
            for w in sched.warnings:
                self.issue("warning", "lesson", w)

            teacher_clashes = c.execute("""
                SELECT COUNT(*) FROM lessons a
                JOIN lessons b ON b.teacher_code = a.teacher_code
                 AND b.day = a.day AND b.id > a.id
                 AND a.start_min < b.end_min AND b.start_min < a.end_min
            """).fetchone()[0]
            if teacher_clashes:
                self.issue(
                    "error", "teacher_schedule_overlap",
                    f"{teacher_clashes} pairs of lessons place one teacher in "
                    f"overlapping times. They are retained for audit, but the "
                    f"runtime will mark the exact lesson as conflicted instead "
                    f"of combining fields from different rows.", teacher_clashes)

        # Match groups to lessons.
        match = match_groups(groups, sched.lessons) if sched else None
        group_ids: dict[object, int] = {}
        for g in groups:
            hits = match.meetings.get(g, []) if match else []
            quality = match.quality.get(g, "none") if match else "none"
            cur = c.execute(
                "INSERT INTO teaching_groups (grade, subject_code, course_group, "
                "room, teacher_code, match_quality, meeting_count) "
                "VALUES (?,?,?,?,?,?,?)",
                (g.grade, g.subject_code, g.course_group, g.room, g.teacher_code,
                 quality, len(hits)))
            group_ids[g] = cur.lastrowid
            for idx, score in hits:
                l = sched.lessons[idx]
                c.execute(
                    "INSERT OR IGNORE INTO group_meetings (group_id, lesson_id, "
                    "day, start_min, end_min, score) VALUES (?,?,?,?,?,?)",
                    (cur.lastrowid, lesson_ids[idx], l.day, l.start_min,
                     l.end_min, score))
        if match:
            for w in match.warnings:
                self.issue("warning", "group_match", w)
            if match.ambiguous:
                self.issue(
                    "warning", "group_match_ambiguous",
                    f"{match.ambiguous} teaching groups matched too many events, "
                    f"or equal-scoring events with contradictory room/subject "
                    f"evidence. Those lessons prove the teacher is busy but do "
                    f"not identify one student's place, so they are excluded "
                    f"from individual placement and coverage",
                    match.ambiguous)

        known = {r.computer_number for r in roster}
        known_grade = {r.computer_number: r.grade for r in roster}
        orphans = Counter()
        pairs = set()
        for e in enrolments:
            if e.computer_number not in known:
                orphans[e.computer_number] += 1
                continue
            if e.group.grade != known_grade[e.computer_number]:
                continue
            pairs.add((e.computer_number, group_ids[e.group]))
        c.executemany("INSERT OR IGNORE INTO enrolments (student_id, group_id) "
                      "VALUES (?,?)", sorted(pairs))
        if orphans:
            self.issue("warning", "orphan_enrolment",
                       f"{len(orphans)} students appear in subject lists but on no "
                       f"class list, so their grade and section are unknown; their "
                       f"{sum(orphans.values())} enrolments were dropped",
                       len(orphans))
        no_enrol = known - {p[0] for p in pairs}
        if no_enrol:
            self.issue("info", "student_without_enrolment",
                       f"{len(no_enrol)} students have no subject enrolment at all "
                       f"({', '.join(sorted(no_enrol)[:5])}); they will read as "
                       f"unknown rather than free", len(no_enrol))

        # Section timetable + bell slots.
        bell: dict[tuple[str, str], tuple[int, int]] = {}
        bell_conflicts: dict[tuple[str, str], set[tuple[int, int]]] = defaultdict(set)
        missing_time = 0
        for r in section_rows:
            sid = section_id.get((r["grade"], r["letter"]))
            if sid is None:
                continue
            c.execute(
                "INSERT INTO section_periods (section_id, day, period_label, "
                "start_min, end_min, subject_text, teacher_code, group_code, "
                "is_free) VALUES (?,?,?,?,?,?,?,?,?)",
                (sid, r["day"], r["label"], r["start_min"], r["end_min"],
                 r["subject_text"], r["teacher_code"], r["group_code"], r["is_free"]))
            if r["start_min"] is None:
                missing_time += 1
            else:
                key = (r["grade"], r["label"])
                span = (r["start_min"], r["end_min"])
                existing = bell.setdefault(key, span)
                if existing != span:
                    bell_conflicts[key].update((existing, span))
        if missing_time:
            self.issue("warning", "period_time",
                       f"{missing_time} timetable cells carry no readable time, so "
                       f"they cannot take part in a 'right now' answer", missing_time)
        for (grade, label), spans in sorted(bell_conflicts.items()):
            printed = ", ".join(f"{hhmm(start)}-{hhmm(end)}"
                                for start, end in sorted(spans))
            self.issue(
                "error", "bell_slot_conflict",
                f"grade {grade} uses period label {label!r} for more than one "
                f"time ({printed}); the first span is retained for label lookup, "
                f"but explicit-time queries remain unambiguous",
                len(spans))
        c.executemany("INSERT INTO bell_slots (grade, label, start_min, end_min) "
                      "VALUES (?,?,?,?)",
                      [(g, lbl, s, e) for (g, lbl), (s, e) in sorted(bell.items())])

        section_clashes = c.execute("""
            SELECT COUNT(*) FROM section_periods a
            JOIN section_periods b ON b.section_id = a.section_id
             AND b.day = a.day AND b.id > a.id
             AND a.start_min < b.end_min AND b.start_min < a.end_min
        """).fetchone()[0]
        if section_clashes:
            self.issue(
                "error", "section_schedule_overlap",
                f"{section_clashes} pairs of timetable cells overlap for the "
                f"same section; a right-now lookup would otherwise choose one "
                f"arbitrarily", section_clashes)

        # Sections that exist on one side of the join only. A roster section with
        # no timetable is the dangerous direction: those are real children about
        # whom nothing can be said, and reporting them as free is how grade 11's
        # 31-student section "L" — which the timetable calls "LA" — became 31
        # students free in every period of the week.
        # Reported per grade, and only for grades that DO have a timetable for
        # their other sections — a grade with no timetable at all is already
        # covered by its coverage verdict, and repeating it per section buries the
        # genuine mismatches under thirty identical lines.
        for row in c.execute("""
            SELECT sec.grade,
                   GROUP_CONCAT(sec.letter) AS letters,
                   SUM(sec.student_count)   AS n
            FROM sections sec
            WHERE sec.student_count > 0
              AND NOT EXISTS (SELECT 1 FROM section_periods sp
                              WHERE sp.section_id = sec.id)
              AND EXISTS (SELECT 1 FROM sections s2
                          JOIN section_periods sp2 ON sp2.section_id = s2.id
                          WHERE s2.grade = sec.grade)
            GROUP BY sec.grade""").fetchall():
            grade, letters, count = row
            orphan_grid = [r[0] for r in c.execute(
                "SELECT letter FROM sections WHERE grade = ? AND student_count = 0",
                (grade,))]
            hint = (f" The timetable carries {', '.join(orphan_grid)} with no "
                    f"students, which may be the same section under another name."
                    if orphan_grid else "")
            self.issue("error", "section_without_timetable",
                       f"grade {grade} section(s) {letters} hold {count} students "
                       f"but appear in no timetable, so those students are reported "
                       f"as unknown, never free.{hint}", count)
        for row in c.execute("""
            SELECT sec.grade, sec.letter FROM sections sec
            WHERE sec.student_count = 0
              AND EXISTS (SELECT 1 FROM section_periods sp
                          WHERE sp.section_id = sec.id)""").fetchall():
            self.issue("info", "timetable_without_students",
                       f"grade {row[0]} section {row[1]} has a timetable but no "
                       f"students on any class list")

        self._report_unmatched_rooms(c)

        # Teacher codes in the timetable that no directory knows.
        nameless = [row[0] for row in c.execute(
            "SELECT code FROM teachers WHERE name IS NULL OR name = '' "
            "ORDER BY code")]
        if nameless:
            self.issue("warning", "teacher_without_name",
                       f"{len(nameless)} teacher codes appear in the timetable but "
                       f"in no staff list ({', '.join(nameless[:8])}"
                       f"{'...' if len(nameless) > 8 else ''}); their lessons are "
                       f"used, but reports will show the code instead of a name",
                       len(nameless))

        self._write_coverage(c)
        c.executemany("INSERT INTO build_issues (severity, category, detail, n) "
                      "VALUES (?,?,?,?)", self.issues)
        c.commit()

    def _report_unmatched_rooms(self, c):
        """List teaching-group rooms that match no lesson, and say which could be
        fixed by an alias and which genuinely cannot.

        A group whose room appears in no lesson can only ever be matched by
        subject, which is what leaves a teacher's same-subject groups
        indistinguishable. Surfacing the list is how the next `09L` gets found
        without anyone having to go looking for it.
        """
        # A group is only really unplaced if NEITHER the printed room nor the
        # section reconstructed from its course group appears in the schedule —
        # see etl.schedule.room_keys.
        rows = c.execute("""
            SELECT tg.room,
                   COUNT(*) AS groups,
                   (SELECT COUNT(*) FROM enrolments e
                     JOIN teaching_groups t2 ON t2.id = e.group_id
                    WHERE t2.room = tg.room) AS enrolments
            FROM teaching_groups tg
            WHERE tg.room IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr WHERE lr.room = tg.room)
              AND NOT EXISTS (SELECT 1 FROM lesson_rooms lr
                              WHERE lr.room = tg.grade || UPPER(tg.course_group))
            GROUP BY tg.room ORDER BY enrolments DESC
        """).fetchall()
        for room, n_groups, n_enrol in rows:
            candidates = [r[0] for r in c.execute(
                "SELECT DISTINCT room FROM lesson_rooms WHERE room LIKE ?||'%' "
                "ORDER BY room", (room,))]
            if len(candidates) == 1:
                self.issue("warning", "room_alias_available",
                           f"room {room} ({n_groups} groups, {n_enrol} enrolments) "
                           f"matches no lesson, and exactly one real room extends "
                           f"it: {candidates[0]}. Add ROOM_ALIASES[{room!r}] = "
                           f"{candidates[0]!r}, or --room-alias {room}={candidates[0]}",
                           n_enrol)
            elif candidates:
                self.issue("info", "room_ambiguous",
                           f"room {room} ({n_groups} groups, {n_enrol} enrolments) "
                           f"matches no lesson and could be any of "
                           f"{', '.join(candidates)}; left unresolved, because "
                           f"choosing one would place students in a room they may "
                           f"not be in. Those groups match by subject only.",
                           n_enrol)
            else:
                self.issue("info", "room_unknown",
                           f"room {room} ({n_groups} groups, {n_enrol} enrolments) "
                           f"appears in no lesson and no real room extends it. The "
                           f"enrolment reports name physical rooms where the "
                           f"schedule names sections, so these cannot be reconciled "
                           f"by name at all.", n_enrol)

    def _write_coverage(self, c):
        """How much of each grade can actually be placed on the timetable.

        The denominator excludes supervised self-study, which has no lesson by
        design. Including it made grade 6 read 81% when 89% of the enrolment that
        COULD be placed was placed, and the difference matters: the verdict
        decides whether the app answers a question or refuses it.
        """
        rows = c.execute("""
            SELECT g.code,
              (SELECT COUNT(*) FROM students s
                 JOIN sections sec ON sec.id = s.section_id
                WHERE sec.grade = g.code)                              AS students,
              (SELECT COUNT(*) FROM enrolments e
                 JOIN teaching_groups tg ON tg.id = e.group_id
                WHERE tg.grade = g.code)                               AS enrolments,
              (SELECT COUNT(*) FROM enrolments e
                 JOIN teaching_groups tg ON tg.id = e.group_id
                WHERE tg.grade = g.code AND tg.meeting_count > 0
                  AND tg.match_quality != 'ambiguous')                 AS resolved,
              (SELECT COUNT(*) FROM enrolments e
                 JOIN teaching_groups tg ON tg.id = e.group_id
                 JOIN teachers t ON t.code = tg.teacher_code
                WHERE tg.grade = g.code AND t.is_placeholder = 1)      AS self_study,
              (SELECT COUNT(*) FROM enrolments e
                 JOIN teaching_groups tg ON tg.id = e.group_id
                 LEFT JOIN teachers t ON t.code = tg.teacher_code
                WHERE tg.grade = g.code
                  AND COALESCE(t.is_placeholder, 0) = 0
                  AND NOT EXISTS (SELECT 1 FROM lessons l
                                  WHERE l.teacher_code = tg.teacher_code)) AS missing,
              (SELECT COUNT(*) FROM section_periods sp
                 JOIN sections sec2 ON sec2.id = sp.section_id
                WHERE sec2.grade = g.code)                             AS periods
            FROM grades g
        """).fetchall()

        for code, students, enrolments, resolved, self_study, missing, periods in rows:
            placeable = max(enrolments - self_study, 0)
            coverage = (resolved / placeable) if placeable else 0.0
            # Group coverage without a section timetable can help diagnose the
            # missing source, but it cannot support a free/busy verdict.  The
            # old report labelled Grade 1 and KG2 "partial" despite having no
            # class grid at all; callers then had to remember that the verdict
            # contradicted the has-timetable flag.
            if not periods:
                verdict = "insufficient"
            elif coverage >= GOOD_COVERAGE:
                verdict = "good"
            elif coverage >= PARTIAL_COVERAGE:
                verdict = "partial"
            else:
                verdict = "insufficient"
            c.execute(
                "INSERT INTO grade_coverage (grade, students, enrolments, "
                "resolved_enrolments, self_study, teacher_missing, coverage, "
                "has_section_timetable, verdict) VALUES (?,?,?,?,?,?,?,?,?)",
                (code, students, enrolments, resolved, self_study, missing,
                 round(coverage, 4), 1 if periods else 0, verdict))

            if missing:
                self.issue("warning", "teacher_missing_from_schedule",
                           f"grade {code}: {missing} enrolments are taught by staff "
                           f"who appear in neither the teacher schedule nor the "
                           f"class timetable, so their lessons cannot be placed. "
                           f"This is a gap in the source reports, not in the data "
                           f"we hold about the students.", missing)
            if not periods:
                self.issue(
                    "error", "coverage",
                    f"grade {code}: no section timetable was supplied, so "
                    f"free/busy answers are refused even though {coverage:.0%} "
                    f"of placeable enrolment matched the teacher schedule")
            elif verdict != "good":
                self.issue(
                    "warning" if verdict == "partial" else "error", "coverage",
                    f"grade {code}: only {coverage:.0%} of placeable enrolment can "
                    f"be tied to a timetabled lesson, so free/busy answers for it "
                    f"are {'caveated' if verdict == 'partial' else 'refused'}"
                    + (f" ({missing} enrolments belong to staff missing from every "
                       f"schedule)" if missing else ""))

    def finish(self, meta: dict):
        c = self.conn
        meta = {**meta, **self.stats}
        c.executemany("INSERT OR REPLACE INTO build_meta (key, value) VALUES (?,?)",
                      [(k, str(v)) for k, v in meta.items()])
        # Re-write issues that were appended during coverage analysis.
        c.execute("DELETE FROM build_issues")
        c.executemany("INSERT INTO build_issues (severity, category, detail, n) "
                      "VALUES (?,?,?,?)", self.issues)
        c.execute("PRAGMA optimize")
        c.commit()
        foreign_key_errors = c.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise RuntimeError(
                f"refusing to publish a database with {len(foreign_key_errors)} "
                f"foreign-key violation(s)")
        integrity = c.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(
                f"refusing to publish a database that failed integrity_check: "
                f"{integrity}")
        c.close()
        os.replace(self.tmp_path, self.out_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class-lists", required=True,
                    help="class-list ZIP, directory, or one class-list PDF")
    ap.add_argument("--subject-lists", required=True,
                    help="subject-list ZIP, directory, or one subject-list PDF")
    ap.add_argument("--teacher-xlsx")
    ap.add_argument("--timetable-db")
    ap.add_argument("--out", default=os.path.join("data", "school.db"))
    ap.add_argument("--report", default=None,
                    help="write the build report as JSON here")
    ap.add_argument("--section-alias", action="append", default=[], metavar="G:FROM=TO",
                    help="merge a class-list section into the timetable's name for "
                         "it, e.g. 11:L=LA. Repeatable. Checked against the data and "
                         "skipped with a warning if it does not hold.")
    ap.add_argument("--room-alias", action="append", default=[], metavar="FROM=TO",
                    help="canonicalise a teaching-group room name, e.g. 11L=11LA")
    ap.add_argument("--no-default-aliases", action="store_true",
                    help="ignore the aliases declared in etl/build.py")
    a = ap.parse_args(argv)

    if a.no_default_aliases:
        SECTION_ALIASES.clear()
        ROOM_ALIASES.clear()
    for spec in a.section_alias:
        try:
            grade, rest = spec.split(":", 1)
            src, dst = rest.split("=", 1)
        except ValueError:
            raise SystemExit(f"--section-alias must look like 11:L=LA, got {spec!r}")
        SECTION_ALIASES[(grade.strip(), src.strip())] = dst.strip()
    for spec in a.room_alias:
        if "=" not in spec:
            raise SystemExit(f"--room-alias must look like 11L=11LA, got {spec!r}")
        src, dst = spec.split("=", 1)
        ROOM_ALIASES[src.strip()] = dst.strip()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    build = Build(a.out)
    try:
        print("[1/4] roster")
        roster = build.load_roster(a.class_lists)
        print("[2/4] enrolment")
        enrolments, groups = build.load_enrolment(a.subject_lists)

        print("[3/4] schedule")
        sched = None
        if a.teacher_xlsx and os.path.exists(a.teacher_xlsx):
            sched = load_lessons(a.teacher_xlsx)
            print(f"  lessons       {len(sched.lessons)} "
                  f"({len({l.teacher_code for l in sched.lessons})} teachers)")
        else:
            build.issue("error", "no_schedule",
                        "no teacher schedule supplied, so nothing can be said "
                        "about when any group meets; every free/busy question "
                        "will be answered 'unknown'")
            print("  lessons       none supplied")

        section_rows = []
        if a.timetable_db and os.path.exists(a.timetable_db):
            section_rows = list(load_section_periods(a.timetable_db))
            print(f"  section grid  {len(section_rows)} cells")
        else:
            build.issue("warning", "no_section_timetable",
                        "no section timetable supplied; per-class timetables are "
                        "unavailable")

        print("[4/4] writing")
        build.write(roster, enrolments, groups, sched, section_rows)
        build.finish({
            "built_from_class_lists": os.path.abspath(a.class_lists),
            "built_from_class_lists_sha256": source_sha256(
                a.class_lists, "Class List"),
            "built_from_subject_lists": os.path.abspath(a.subject_lists),
            "built_from_subject_lists_sha256": source_sha256(
                a.subject_lists, "Subject List"),
            "built_from_teacher_xlsx": os.path.abspath(a.teacher_xlsx)
            if a.teacher_xlsx else "",
            "built_from_teacher_xlsx_sha256": source_sha256(a.teacher_xlsx)
            if a.teacher_xlsx else "",
            "built_from_timetable_db": os.path.abspath(a.timetable_db)
            if a.timetable_db else "",
            "built_from_timetable_db_sha256": source_sha256(a.timetable_db)
            if a.timetable_db else "",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "students": len(roster),
            "enrolment_rows": len(enrolments),
            "teaching_groups": len(groups),
            "lessons": len(sched.lessons) if sched else 0,
            "schema_version": "2",
        })
    except BaseException:
        build.abort()
        raise

    conn = sqlite3.connect(a.out)
    print(f"\nwrote {a.out}")
    print("\n  grade  students  enrolments  resolved  self-study  no-teacher  "
          "timetable  coverage  verdict")
    for row in conn.execute("""SELECT gc.grade, students, enrolments,
                               resolved_enrolments, self_study, teacher_missing,
                               has_section_timetable, coverage, verdict
                               FROM grade_coverage gc JOIN grades g ON g.code=gc.grade
                               ORDER BY g.ordinal"""):
        g, st, en, rs, ss, tm, tt, cov, verdict = row
        print(f"  {g:<5}  {st:8}  {en:10}  {rs:8}  {ss:10}  {tm:10}  "
              f"{'yes' if tt else 'no':9}  {cov:7.0%}  {verdict}")

    counts = Counter()
    for sev, cat, _detail, _n in build.issues:
        counts[sev] += 1
    print(f"\n  build issues: " + ", ".join(f"{n} {s}" for s, n in counts.items())
          or "  no issues")
    for sev in ("error", "warning"):
        for s, cat, detail, _n in build.issues:
            if s == sev:
                print(f"    [{sev}] {cat}: {detail[:150]}")

    if a.report:
        with open(a.report, "w") as f:
            json.dump({
                "meta": dict(conn.execute(
                    "SELECT key, value FROM build_meta ORDER BY key")),
                "coverage": [dict(zip(
                    ["grade", "students", "enrolments", "resolved", "coverage",
                     "has_timetable", "verdict"], r))
                    for r in conn.execute(
                        "SELECT grade, students, enrolments, resolved_enrolments,"
                        " coverage, has_section_timetable, verdict "
                        "FROM grade_coverage")],
                "issues": [{"severity": s, "category": c, "detail": d, "n": n}
                           for s, c, d, n in build.issues],
            }, f, indent=2)
        print(f"  report -> {a.report}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
