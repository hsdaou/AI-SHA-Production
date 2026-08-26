"""
Regressions for the subject normaliser.

The first test here is the one that matters. The previous implementation reduced
"Mathematics" to the empty string, and an empty string meant "this student is not
enrolled in what is being taught", which the app reported as free. On the real
dataset that put 117 grade-5 students who were sitting in a Mathematics lesson
into the free list, four times a week, per grade.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from etl.subjects import NEVER_EMPTY, compatible, subject_head, tokens


@pytest.mark.parametrize("name", NEVER_EMPTY)
def test_real_subjects_never_normalise_to_nothing(name):
    """The exact class of bug that made every Mathematics lesson look free."""
    assert subject_head(name), f"{name!r} normalised away to nothing"


def test_the_specific_historical_failures():
    # What the old regex produced, for the record:
    #   Mathematics -> ''    Music -> ''      Chemistry -> 'che'
    #   Economics   -> 'econo'  Drama -> 'dra'
    assert subject_head("Mathematics") == "mathematics"
    assert subject_head("Music") == "music"
    assert subject_head("Chemistry") == "chemistry"
    assert subject_head("Economics") == "economics"
    assert subject_head("Drama") == "drama"


@pytest.mark.parametrize("timetable,enrolment", [
    ("Mathematics", "Mathematics N2"),
    ("Mathematics", "Math Support"),
    ("Math Alg / Geo", "Mathematics N2"),
    ("Math N-2", "Mathematics"),
    ("Accelerated Math", "Mathematics N1"),
    ("Arabic Language", "Arabic Language- Arabs"),
    ("Arabic Language", "Arabic Language- Non Arabs"),
    ("Arabic Language", "Extra Arabic"),
    ("Islamic Education", "Islamic Education- Arabs"),
    ("Biology", "Biology Lab"),
    ("Biology", "Biology Support"),
    ("English", "English Language"),
    ("Computing", "Computer Science N2"),
    ("Economics", "Economics N2"),
    ("Physics", "Physics TN"),
    ("MOE Social Studies & Civics", "Social Studies"),
    ("Moral, Soc. & Cultural Stud.", "Moral Education - Arabs"),
])
def test_same_lesson_under_two_names(timetable, enrolment):
    assert compatible(timetable, enrolment), f"{timetable!r} !~ {enrolment!r}"


@pytest.mark.parametrize("a,b", [
    # These share a generic noun. Keeping "education" as significant would weld
    # Islamic and Moral studies into one subject.
    ("Islamic Education", "Moral Education - Arabs"),
    ("Arabic Language", "English Language"),
    ("Social Studies", "Islamic Education"),
    ("Biology", "Business Studies N"),
    ("Mathematics", "Music"),
    ("Chemistry", "Physics"),
    ("Economics", "English"),
    ("Art", "Arabic Language"),
])
def test_different_subjects_do_not_match(a, b):
    assert not compatible(a, b), f"{a!r} wrongly matched {b!r}"


def test_unmatchable_text_is_not_a_wildcard():
    """A name that normalises to nothing must match NOTHING, not everything.

    "Periodic" and "Support CA" are timetable activities with no subject in them.
    The old code let an empty normalisation short-circuit, and grade 12's Monday
    "Periodic" period therefore reported all 163 students free.
    """
    assert subject_head("Periodic") is None
    assert not compatible("Periodic", "Mathematics")
    assert not compatible("Periodic", "Periodic")
    assert not compatible(None, "Mathematics")
    assert not compatible("", "")


def test_level_markers_are_stripped_only_as_whole_tokens():
    assert tokens("Mathematics N2") == ["mathematics"]
    assert tokens("Chemistry N1") == ["chemistry"]
    # ...and not from the middle of a word, which is what went wrong before.
    assert "chemistry" in tokens("Chemistry")
    assert "music" in tokens("Music")
