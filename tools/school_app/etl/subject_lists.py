"""
Parse the "<grade> Subject List.PDF" reports into subject enrolment.

This is the dataset the previous app did not have. Its builder wrote an empty
"Subject Enrollment" sheet, and because the free/busy rule was "free unless
enrolled in what is being taught", an empty sheet made every one of the school's
students report as free in every lesson. These 1,948 pages are what closes that
hole.

Each page is one teaching group:

    Subject List Class 05    ARHL1 Arabic Language- Non Arabs      AY :- 2526
    Couse Group A    RoomID :- 05A    AMB3 Ms. MANAHEL ABU AMER          Sibling
      1 AFRAH        ABDULLAHI     B   25443  AAFRAH26872@iscshj...      2

and therefore carries the four facts the resolver needs: which subject, which
group of it, WHERE the group meets, and WHO teaches it. Room and teacher are the
important ones — they let a student be matched to a lesson by identity instead
of by comparing subject names, which is what the old app did and what made
"Mathematics" match nothing at all.

A group larger than one page continues on the next page with the header
repeated, so blocks are accumulated by (subject, group, room, teacher) and
students de-duplicated.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .pdfgrid import Word, column_peaks, pdf_pages, text_of, to_lines

# The subject NAME is optional: a handful of codes (EnNO7, FrLO1) are printed
# with no name at all. Requiring one silently dropped 20 whole pages of grade
# 10-12 enrolment, which is exactly the kind of loss that looks like a quiet
# success.
SUBJECT_HDR_RE = re.compile(
    r"Subject List Class\s+(\d{1,2}|K\d)\s+([A-Za-z]{2,4}\d)(?:\s+(.+?))?\s+AY\s*:-")
# Teacher codes are mostly AMB3-shaped but not reliably: the school also uses
# EPK, R07, S77, E12 and the placeholders SSTUDY ("Ms. Self Study") and ENEW2
# ("Mr. New NEW"). Anything letter-initial in the code position is accepted, so
# an unusual code is carried through instead of being dropped.
TEACHER_CODE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,6}$")
# Rooms are digit- or K-initial: 05A, 12LA, K1A. Some print truncated — "09-",
# "K2-", bare "11" — which a stricter pattern rejects, and then the room token
# is misread as the teacher code and the real teacher vanishes into the name.
ROOM_RE = re.compile(r"^(?:\d{1,2}|K\d)[A-Za-z]{0,2}\d?-?$")
SEQ_RE = re.compile(r"^\d{1,3}$")
ID_RE = re.compile(r"^\d{4,6}$")


@dataclass(frozen=True)
class Group:
    grade: str
    subject_code: str
    subject_name: str
    course_group: str
    room: str | None
    teacher_code: str | None
    teacher_name: str | None


@dataclass
class Enrolment:
    computer_number: str
    group: Group
    section_hint: str | None   # the group sheet's own view of the home section
    source: str


@dataclass
class SubjectListResult:
    enrolments: list[Enrolment] = field(default_factory=list)
    groups: set[Group] = field(default_factory=set)
    warnings: list[str] = field(default_factory=list)
    pages_parsed: int = 0
    pages_skipped: int = 0


def _norm_grade(tok: str) -> str:
    return tok if tok.startswith("K") else f"{int(tok):02d}"


def _parse_group_line(text: str) -> tuple[str, str | None, str | None, str | None] | None:
    """'Couse Group A RoomID :- 05A AMB3 Ms. MANAHEL ABU AMER Sibling'
    -> (course_group, room, teacher_code, teacher_name)

    Written against tokens rather than one big regex because every field here is
    optional or misspelt in some report: KG pages wrap ':-' onto its own line, a
    few groups carry no room, and the caption is "CouseGroup" on some grades,
    "Couse Group" on others, and "Couse Goup:" — the report's own typo — for the
    whole of grade 7. Matching the caption exactly cost 162 pages, every
    enrolment record in that grade.

    So the caption is not matched at all. The group identifier is simply the
    last token before "RoomID", which holds however the caption is spelt.
    """
    toks = [t for t in text.split() if t != ":-"]
    try:
        room_i = toks.index("RoomID")
    except ValueError:
        return None
    if room_i == 0:
        return None
    course_group = toks[room_i - 1]
    if not re.fullmatch(r"[A-Za-z0-9]{1,4}", course_group):
        return None

    tail = toks[room_i + 1:]
    if tail and tail[-1] == "Sibling":
        tail = tail[:-1]

    room = None
    if tail and ROOM_RE.match(tail[0]):
        room, tail = tail[0], tail[1:]
    if not tail:
        return course_group, room, None, None

    # Usually a real code (AMB3). Sometimes a placeholder standing in for a
    # person: grade 11 Physics TN group OPT is staffed by "SSTUDY Ms. Self
    # Study", i.e. supervised self study. That is a real teaching arrangement
    # and its students are real, so the group is kept — it simply will not link
    # to any teacher's schedule, and the build report says so.
    return course_group, room, tail[0], " ".join(tail[1:]) or None


def parse_subject_list(path: str) -> SubjectListResult:
    out = SubjectListResult()
    source = os.path.basename(path)

    for page_no, words in pdf_pages(path):
        lines = to_lines(words)

        subject = next((SUBJECT_HDR_RE.search(ln.text) for ln in lines
                        if SUBJECT_HDR_RE.search(ln.text)), None)
        grp = next((g for ln in lines if (g := _parse_group_line(ln.text))), None)
        if not subject or not grp:
            out.pages_skipped += 1
            if subject or grp:
                out.warnings.append(
                    f"{source} p{page_no}: half a group header "
                    f"(subject={bool(subject)} group={bool(grp)}) — page skipped")
            continue

        course_group, room, teacher_code, teacher_name = grp
        name = (subject.group(3) or "").strip()
        if not name:
            # Printed with no name. Fall back to the code so the subject is
            # still identifiable, and say so rather than showing a blank.
            name = f"[{subject.group(2)}]"
            out.warnings.append(
                f"{source} p{page_no}: subject {subject.group(2)} has no printed "
                f"name; showing the code instead")
        group = Group(
            grade=_norm_grade(subject.group(1)),
            subject_code=subject.group(2),
            subject_name=name,
            course_group=course_group,
            room=room,
            teacher_code=teacher_code,
            teacher_name=teacher_name,
        )
        out.groups.add(group)

        data_lines = [
            ln for ln in lines
            if SEQ_RE.match(ln.words[0].text) and ln.words[0].x0 <= 55
            and any("@" in w.text for w in ln.words)
            and any(ID_RE.match(w.text) and w.x0 >= 55 for w in ln.words)
        ]
        if not data_lines:
            # A real possibility: an empty group. Recorded, not warned about.
            out.pages_parsed += 1
            continue

        def id_word(ln) -> Word | None:
            mail_x = min((w.x0 for w in ln.words if "@" in w.text), default=1e9)
            return next((w for w in ln.words
                         if ID_RE.match(w.text) and 55 <= w.x0 < mail_x), None)

        keep = [(ln, w) for ln in data_lines if (w := id_word(ln))]
        name_rows = [[w for w in ln.words[1:] if w.x0 < idw.x0] for ln, idw in keep]
        # Three left-aligned columns live here: surname, given name, section.
        peaks = column_peaks(name_rows)
        section_x = peaks[2] - 1.0 if len(peaks) >= 3 else None

        for (ln, idw), name_cells in zip(keep, name_rows):
            hint = None
            if section_x is not None:
                hint = text_of([w for w in name_cells if w.x0 >= section_x]) or None
                if hint and not re.fullmatch(r"[A-Z]{1,2}\d?", hint):
                    hint = None
            out.enrolments.append(Enrolment(
                computer_number=idw.text,
                group=group,
                section_hint=hint,
                source=f"{source}#p{page_no}",
            ))
        out.pages_parsed += 1

    return out
