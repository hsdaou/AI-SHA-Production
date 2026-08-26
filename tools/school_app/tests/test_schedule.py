"""Build-time lesson matching invariants."""

from etl.schedule import Lesson, match_groups
from etl.subject_lists import Group


def _group(**changes):
    values = {
        "grade": "05",
        "subject_code": "MTH1",
        "subject_name": "Mathematics",
        "course_group": "A",
        "room": "05A",
        "teacher_code": "T1",
        "teacher_name": "Ms. Teacher",
    }
    values.update(changes)
    return Group(**values)


def _lesson(subject, room, group_code=None):
    return Lesson(
        teacher_code="T1", day="Monday", start_min=480, end_min=530,
        sheet_period="1", subject_text=subject, group_code=group_code,
        room_text=room, rooms=[room])


def test_discarded_weaker_evidence_cannot_promote_match_to_exact():
    """The winning event is room-only; a weaker subject hit is discarded.

    Previously the flags from both provisional hits survived filtering, so the
    group was labelled exact even though no retained lesson matched both room
    and subject.
    """
    group = _group()
    result = match_groups([
        group,
    ], [
        _lesson("Music", "05A", group_code="A"),       # score 3, retained
        _lesson("Mathematics", "05B"),                 # score 2, discarded
    ])
    assert result.quality[group] == "room"
    assert result.meetings[group] == [(0, 3)]


def test_equal_matches_with_different_evidence_are_ambiguous():
    group = _group(course_group="Z")
    result = match_groups([group], [
        _lesson("Music", "05A"),                       # room only, score 2
        _lesson("Mathematics", "05B"),                 # subject only, score 2
    ])
    assert result.quality[group] == "ambiguous"
    assert result.ambiguous == 1


def test_one_event_with_room_and_subject_is_exact():
    group = _group()
    result = match_groups([group], [_lesson("Mathematics", "05A")])
    assert result.quality[group] == "exact"
    assert result.meetings[group] == [(0, 4)]
