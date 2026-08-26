"""Conversation regressions for AI-SHA's timetable router."""

from aisha_brain import skill_intents as si


def test_teacher_request_keeps_context_through_day_period_fragment():
    first = si.classify("send me the available teachers")
    follow = si.classify("on Thursday, the 6th.", first)

    assert follow["intent"] == si.INTENT_FREE_TEACHERS
    assert follow["day"] == "Thursday"
    assert follow["period"] == 6
    assert si.synthesize(follow) == (
        "send me the list of available teachers on Thursday period 6")


def test_student_request_retains_grade_when_fragment_adds_day_period():
    first = si.classify("I'll send the available students from grade 10")
    follow = si.classify("on Wednesday, the 6th.", first)

    assert follow["intent"] == si.INTENT_FREE_STUDENTS
    assert follow["grade"] == 10
    assert follow["day"] == "Wednesday"
    assert follow["period"] == 6
    assert si.synthesize(follow) == (
        "send me the list of available students in grade 10 "
        "on Wednesday period 6")


def test_slot_fragment_without_context_is_suppressed_from_rag():
    assert si.classify("on Wednesday, the 6th.")["intent"] == si.INTENT_FOLLOWUP
    assert si.is_skill("on Wednesday, the 6th.")


def test_explicit_exam_question_is_never_borrowed_from_context():
    context = si.classify("send me the available teachers")
    assert si.classify("Are there exams on Wednesday, the 6th?", context)[
        "intent"] == si.INTENT_NONE


def test_clock_question_is_not_mistaken_for_slot_fragment():
    assert si.classify("What period is it now?")["intent"] == si.INTENT_NONE


def test_count_followup_stays_count():
    first = si.classify("how many students are available in grade 8")
    follow = si.classify("Tuesday, period 4", first)
    assert follow["intent"] == si.INTENT_FREE_COUNT
    assert si.synthesize(follow) == (
        "how many students are available in grade 8 on Tuesday period 4")


def test_timetable_followup_reconstructs_timetable_question():
    first = si.classify("show me the timetable for grade 7")
    follow = si.classify("section A on Monday", first)
    assert follow["intent"] == si.INTENT_TIMETABLE
    assert si.synthesize(follow) == (
        "show me the timetable in grade 7 section A on Monday")
