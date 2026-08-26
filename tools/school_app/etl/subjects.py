"""
Subject-name reconciliation between the enrolment reports and the schedule.

WHAT WENT WRONG BEFORE
----------------------
The previous app normalised a subject name with this regex, meaning to strip
level suffixes like the "N2" in "Mathematics N2":

    re.sub(r"\\s*[-]?\\s*(n[0-9]*|l[0-9]*|m[0-9a-z]*|tn[0-9]*)$", "", name)

Nothing anchors the alternatives to a word boundary, and `m[0-9a-z]*$` will
match from any `m` to the end of the string. So:

    "Mathematics" -> ""          (matches from the leading m)
    "Music"       -> ""
    "Chemistry"   -> "che"
    "Economics"   -> "econo"
    "Drama"       -> "dra"

An empty normalised name made `student_is_enrolled` return False for every
student, and False there meant "not enrolled", which the app reported as FREE.
The result: during every Mathematics and every Music lesson in the school, all
students in the section were reported free. 353 Mathematics and 42 Music cells
in the timetable — the single largest source of wrong answers in the app, and
invisible because the failure looks like data, not like a crash.

WHAT THIS DOES INSTEAD
----------------------
Subject text is never the primary key here — lessons are matched to teaching
groups by teacher identity first (see etl.schedule). This module only breaks
ties between the lessons of one teacher, so it needs to be careful about false
matches rather than exhaustive.

`subject_head` reduces a name to its distinguishing token by removing level
markers and the generic nouns that several unrelated subjects share. Keeping
"education" would make "Islamic Education" match "Moral Education"; dropping it
leaves "islamic" against "moral", which do not.
"""

from __future__ import annotations

import re

# Level, track and delivery markers. Whole tokens only.
_LEVEL = re.compile(
    r"^(?:n|l|m|tn|s|r)-?\d*$|^(?:n|m|l)-\d+$|^\d+$", re.I)
_MARKERS = {
    "lab", "labs", "support", "sup", "extra", "advanced", "accelerated",
    "applied", "periodic", "ca", "opt", "option", "optional", "hl", "sl",
    "ap", "igcse", "as", "a2", "core", "elective", "revision", "remedial",
}
# Nouns shared by unrelated subjects. Removing them is what keeps
# "Islamic Education" from matching "Moral Education".
_GENERIC = {
    "education", "educ", "language", "lang", "studies", "study", "stud",
    "science", "sciences", "moe", "school", "middle", "high", "primary",
    "arabs", "arab", "non", "and", "of", "the", "for", "&", "second", "2nd",
    "first", "1st", "civics", "civic", "soc", "cultural", "cult", "general",
    "class", "classes", "group", "grade", "level",
}
_ALIAS = {
    "math": "mathematics", "maths": "mathematics", "maths.": "mathematics",
    "alg": "algebra", "geo": "geometry", "geom": "geometry",
    "eng": "english", "arab": "arabic", "isl": "islamic", "islamiat": "islamic",
    "phys": "physics", "chem": "chemistry", "bio": "biology",
    "comp": "computing", "computer": "computing", "cs": "computing",
    "pe": "sport", "sports": "sport", "physical": "sport",
    "econ": "economics", "bus": "business", "business": "business",
    "socstud": "social", "ss": "social",
}


def tokens(name: str | None) -> list[str]:
    """Significant, canonicalised tokens of a subject name, in order."""
    if not name:
        return []
    raw = re.split(r"[^A-Za-z0-9&]+", str(name).lower())
    out: list[str] = []
    for t in raw:
        t = t.strip(".")
        if not t or t in _GENERIC or t in _MARKERS or _LEVEL.match(t):
            continue
        out.append(_ALIAS.get(t, t))
    return out


def subject_head(name: str | None) -> str | None:
    """The token that distinguishes this subject from other subjects.

    "Mathematics N2"       -> "mathematics"
    "Math Support"         -> "mathematics"
    "Arabic Language- Arabs" -> "arabic"
    "Islamic Education"    -> "islamic"
    "Moral Education - Arabs" -> "moral"
    "Biology Lab"          -> "biology"
    """
    t = tokens(name)
    return t[0] if t else None


def compatible(a: str | None, b: str | None) -> bool:
    """Could these two names denote the same lesson?

    True when the heads agree, or when one name's token set is contained in the
    other's (so "Math Alg / Geo" still matches "Algebra"). Two names that
    normalise to nothing are NOT treated as compatible — the old code's habit of
    letting an empty normalisation match is precisely the bug above.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    if ta[0] == tb[0]:
        return True
    sa, sb = set(ta), set(tb)
    return bool(sa & sb) and (sa <= sb or sb <= sa)


# Regression fixtures for the exact strings that used to normalise to nothing.
# Asserted by tests/test_subjects.py.
NEVER_EMPTY = ["Mathematics", "Music", "Chemistry", "Economics", "Drama",
               "Computer Science", "Physics", "Biology", "Art", "Advising"]
