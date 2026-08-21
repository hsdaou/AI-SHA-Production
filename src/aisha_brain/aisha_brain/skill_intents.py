"""Deterministic intent + slot extraction for AI-SHA's timetable skill.

ONE classifier, imported by BOTH brain_node (so the knowledge base never sees a
timetable question and invents an answer) and timetable_query.py (so the skill
knows what was asked). Two copies of this logic is what produced the failure it
replaces: the console recognised an utterance, the trigger did not, and
brain_node answered it with the LLM - all three disagreeing about one sentence.

WHY NOT KEYWORD SUBSTRINGS
The previous router held lists like "teachers are available". Real people say
"how many teachers do I HAVE available", "give me the teachers free tomorrow",
"the available students from grade 10". Each miss fell through to the LLM, which
answered confidently and wrongly. The space of phrasings is unbounded, so this
scores CONCEPTS - subject, availability, count-vs-list - instead of matching
sentences.

WHY NOT AN LLM
llama3.2:1b is the model that hallucinated in the first place, and a router that
sometimes emails a list of minors is worse than one that sometimes says "say that
again". Everything here is deterministic and unit-tested.
"""

from __future__ import annotations

import datetime
import re

INTENT_FREE_TEACHERS = "free_teachers"
INTENT_FREE_STUDENTS = "free_students"
INTENT_FREE_COUNT = "free_count"
INTENT_TIMETABLE = "timetable"
INTENT_NONE = "none"
# A follow-up that carries no subject of its own: "send me the report",
# "the name list please", "email it to me". Only resolvable against what was
# asked a moment ago.
INTENT_FOLLOWUP = "followup"

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday",
            "saturday", "sunday"]

_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12,
}
_NUM_WORD_RE = "|".join(sorted(_NUM_WORDS, key=len, reverse=True))

# ── Concept cues ────────────────────────────────────────────────────────────
_TEACHER = re.compile(r"\b(teacher|teachers|tutor|tutors|instructor|instructors"
                      r"|staff|faculty)\b")

# No noun, but the errand is unmistakably staff work: "who is free ... for
# assembly duty", "send me who is available to invigilate". Without this the
# subject-less fallback assumed students and would email the wrong list.
_STAFF_CONTEXT = re.compile(
    r"\b(cover|covering|substitut\w*|invigilat\w*|supervis\w*|duty|duties"
    r"|assembly|reception|gate|escort|chaperone|relief|stand\s*in)\b"
    r"|\bnot\s+teaching\b|\bno\s+(period|periods|lesson|lessons)\s+(today|tomorrow)\b"
    r"|\bsomeone\s+(for|to)\b|\bspare\s+(teacher|staff)\b"
    r"|\b(do\s+we|have\s+we|we)\s+have\s+(anyone|anybody|someone)\b")
_STUDENT = re.compile(r"\b(student|students|pupil|pupils|kid|kids|child|children)\b")

# "free" and "available" dominate, but people also describe the state.
_AVAILABLE = re.compile(
    r"\b(free|available|availability|unoccupied|idle|spare)\b"
    r"|\bnot\s+(busy|teaching|in\s+class|occupied)\b"
    r"|\b(do\s*n[o']?t|does\s*n[o']?t|dont|doesnt)\s+have\s+(a\s+)?(class|lesson|period)\b"
    r"|\bnot\s+having\s+(a\s+)?(class|lesson)\b"
    r"|\bhaving\s+no\s+(class|lesson|period)\b"
    r"|\bno\s+(class|lesson|lessons|classes)\b"
    r"|\bnothing\s+scheduled\b"
    r"|\bwithout\s+(a\s+)?(class|lesson)\b")

_COUNT = re.compile(r"\bhow\s*many\b|\bhowmany\b|\bnumber\s+of\b|\bcount\b"
                    r"|\bhow\s+much\b|\bow\s+many\b|\ba\s+lot\s+of\b|\btotal\b"
                    r"|\b(just\s+)?(tell\s+me\s+)?the\s+number\b|\bhow\s+man\b")

# An explicit request for identities. Beats a count cue when both appear:
# "how many teachers are free tomorrow AND WHO THEY ARE" wants the list.
_WANTS_NAMES = re.compile(
    r"\bnames?\b|\bwho\s+(they|these|those)\s+are\b|\bwho\s+is\s+free\b"
    r"|\bsend\s+me\s+who\b|\blist\s+(them|the)\b|\btell\s+me\s+his\s+name\b")

# A request for the LIST itself, even without "which/who".
_LIST = re.compile(r"\b(send|email|mail|report|list|give|show|share|names?|who|which|"
                   r"whos|who's|provide)\b")

# Deliberately broad. A timetable question that is NOT recognised falls through
# to the knowledge base, which invents a lesson - far worse than the skill asking
# "which grade?". Anything class-shaped should reach the skill.
_TIMETABLE = re.compile(
    r"\btime\s*table\b|\btimetable\b|\bschedule\b|\blesson\s*plan\b"
    r"|\b(subject|subjects|lesson|lessons|class|classes|period|periods)\b"
    r"|\b(studying|studies|study|doing|having|have|has|got|up\s+to|happening)\b"
    r"|\bwhat'?s?\s+(on|after)\b")

# ...but a timetable-shaped word attached to one of these is somebody else's
# document, not the class grid.
_TIMETABLE_NOT_OURS = re.compile(
    r"\b(exam|exams|football|training|club|clubs|trip|bus|assembly\s+schedule"
    r"|published|publish|duration|how\s+long)\b")

# ── Things that must NEVER be hijacked ──────────────────────────────────────
# Each of these belongs to another skill or to the knowledge base. A false
# positive here diverts a legitimate question and answers it wrongly.
_OTHER_SKILL = re.compile(
    r"\bvideo\s*message\b|\bvedio\s*message\b|\brecord\s+a\s+(video|message)\b"
    r"|\bsick\s+leave\b|\bannual\s+leave\b|\bon\s+leave\b|\bleave\s+balance\b"
    r"|\bdays?\s+(left|remaining)\b|\bwho.{0,6}\s+(off|absent|away)\b"
    r"|\bhrms\b|\bleave\s+(report|records?|request)\b|\b(casual|maternity)\s+leave\b"
    r"|\bauthenticate\b|\blog\s*in\b|\bsign\s*in\b|\bverify\s+me\b")

_KNOWLEDGE = re.compile(
    r"\b(tuition|fee|fees|admission|admissions|enrol|enroll|enrolment|enrollment)\b"
    r"|\bbus\b|\buniform\b|\bcanteen\b|\bholiday\b|\bvacation\b|\bterm\s+dates?\b"
    r"|\bexam\s+(date|dates|timetable\s+for\s+exams)\b"
    r"|\bwhat\s+time\s+does\b|\bopening\s+hours?\b|\bcontact\b|\bphone\s+number\b"
    r"|\baddress\b|\bprincipal\b|\bcurriculum\b|\bpolicy\b")

# "is the teacher available for a meeting" is about a PERSON'S diary, not the
# timetable grid; "which room is free" is about a facility. Both read as
# availability questions and must not be answered from the timetable.
# A FACILITY is never a person - always excluded.
_FACILITY = re.compile(
    r"\b(room|rooms|hall|lab|library|gym|field|court|space|seat|seats|desk"
    r"|classroom|office|bus)\b")
# A single person's diary. Only excludes when the ask is about ONE named adult
# ("is Mr Haddad available for a meeting") - NOT when counting or listing many
# ("how many teachers are available on Monday for the parent meeting").
_DIARY = re.compile(r"\b(meeting|appointment|interview|parent\s+conference)\b")
_SINGLE_PERSON = re.compile(r"\b(is|will)\s+(mr|mrs|ms|miss|dr|the)\b"
                            r"|\bis\s+[a-z]+\s+(available|free|in)\b")

# A request to be SENT something, with no subject of its own. On its own this is
# unanswerable; against the previous query it means "the named list of that".
# Without this it fell through to the knowledge base, which replied "I am AI-SHA,
# the administrative assistant" and then invented a paragraph about student life
# organisations.
_FOLLOWUP = re.compile(
    r"\b(send|email|mail|share|forward|give|get)\b[^.?]{0,24}"
    r"\b(it|them|those|that|report|list|names?|details|by\s*e?-?mail)\b"
    r"|\b(the\s+)?(name\s*list|report)\b"
    r"|\bemail\s+(it|them|me)\b")

# Speech-to-text renders "teachers" as "meters"/"metres" often enough to matter,
# and this robot is never asked about lengths. Only consulted inside the
# availability branch, so "how many metres of rope" could never reach it.
_STT_TEACHER = re.compile(r"\b(meters?|metres?|readers?|preachers?)\b")

# ── Slot extraction ─────────────────────────────────────────────────────────
_GRADE_RE = re.compile(
    r"\bgrade\s*(\d{1,2})\b"
    r"|\b(?:year|class)\s*(\d{1,2})\b"
    r"|\bg\s*(\d{1,2})\b(?!\s*[ap]m)"
    r"|\b(\d{1,2})\s*(?:st|nd|rd|th)\s+grade\b"
    rf"|\b(?:grade|class|year)\s+({_NUM_WORD_RE}|tan)\b"
    # compact "8D", "7B", "6C" - a grade glued to its section
    r"|\b(\d{1,2})\s*[a-j]\b(?!\s*m)")

# "section be"/"sec see" are how a speech-to-text engine renders "section B"/"C".
# Mapped ONLY directly after section/sec, never loose, or "be" and "see" would
# capture half the sentences in the corpus.
_STT_LETTER = {"be": "B", "bee": "B", "see": "C", "sea": "C", "dee": "D",
               "ay": "A", "eh": "A", "ee": "E", "gee": "G", "jay": "J"}
_SECTION_RE = re.compile(
    r"\b(?:section|sec)\s+([a-j])\b"
    r"|\b(?:section|sec)\s+(be|bee|see|sea|dee|ay|ee|gee|jay)\b"
    r"|\b(?:grade|class)\s*\d{1,2}\s*[-/ ]?\s*([a-j])\b(?!\w)"
    rf"|\b(?:grade|class)\s+(?:{_NUM_WORD_RE}|tan)\s+(be|bee|see|sea|dee|ay|ee)\b"
    r"|\b\d{1,2}\s*([a-j])\b(?!\s*m)")

# "period for"/"period too"/"period ate" are speech-to-text for 4/2/8. Accepted
# ONLY straight after the word "period" - "free teachers FOR Monday" must not
# become period 4.
_STT_NUM = {"for": 4, "fore": 4, "too": 2, "to": 2, "ate": 8, "won": 1,
            "tree": 3, "sex": 6, "tan": 10, "nan": 9}
_PERIOD_RE = re.compile(
    r"\bperiod\s*(\d{1,2})\b"
    r"|\b(\d{1,2})\s*(?:st|nd|rd|th)\s+period\b"
    rf"|\bperiod\s+({_NUM_WORD_RE})\b"
    rf"|\b({_NUM_WORD_RE})\s+period\b"
    r"|\bperiod\s+(for|fore|too|to|ate|won|tree|sex|tan|nan)\b"
    r"|\bp\s*(\d{1,2})\b(?!\w)"
    r"|\bthe\s+(\d{1,2})\s*(?:st|nd|rd|th)\b"
    rf"|\bthe\s+({_NUM_WORD_RE})\b(?=\s|$)")


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _first_int(m: re.Match) -> int | None:
    for g in m.groups():
        if not g:
            continue
        if g.isdigit():
            return int(g)
        if g in _NUM_WORDS:
            return _NUM_WORDS[g]
        if g in _STT_NUM:
            return _STT_NUM[g]
        if g == "tan":
            return 10
    return None


def extract_slots(text: str) -> dict:
    t = _norm(text)
    out: dict = {"grade": None, "section": None, "day": None, "period": None}

    m = _GRADE_RE.search(t)
    if m:
        g = _first_int(m)
        if g is not None and 1 <= g <= 12:
            out["grade"] = g

    m = _SECTION_RE.search(t)
    if m:
        for g in m.groups():
            if g:
                out["section"] = _STT_LETTER.get(g, g.upper() if len(g) == 1 else None)
                break

    if "tomorrow" in t:
        out["day"] = "tomorrow"
    elif "today" in t or re.search(r"\bright\s+now\b|\bat\s+the\s+moment\b|\bnow\b", t):
        out["day"] = "today"
    else:
        for d in WEEKDAYS:
            if re.search(rf"\b{d}\b", t):
                out["day"] = d.capitalize()
                break

    m = _PERIOD_RE.search(t)
    if m:
        p = _first_int(m)
        if p is not None and 1 <= p <= 12:
            out["period"] = p
    return out


def resolve_day(day: str | None, today: datetime.date | None = None) -> str | None:
    """'tomorrow' -> the actual weekday name. Resolving late, not early, is
    deliberate: dropping it and letting the server default to today silently
    answered the wrong day."""
    if not day:
        return None
    d = today or datetime.date.today()
    low = day.strip().lower()
    if low == "today":
        return d.strftime("%A")
    if low == "tomorrow":
        return (d + datetime.timedelta(days=1)).strftime("%A")
    if low in WEEKDAYS:
        return low.capitalize()
    return day


def is_followup(text: str) -> bool:
    """A 'send me the report' with no subject of its own."""
    t = _norm(text)
    if not _FOLLOWUP.search(t):
        return False
    return not (_TEACHER.search(t) or _STUDENT.search(t) or _AVAILABLE.search(t)
                or _TIMETABLE.search(t))


def synthesize(context: dict) -> str:
    """Rebuild a full question from a remembered query, so a follow-up can be
    answered by the ordinary path instead of a second, parallel code path."""
    if not context:
        return ""
    subject = context.get("subject") or (
        "teachers" if context.get("intent") == INTENT_FREE_TEACHERS else "students")
    parts = [f"send me the list of available {subject}"]
    if context.get("grade"):
        parts.append(f"in grade {context['grade']}")
    if context.get("section"):
        parts.append(f"section {context['section']}")
    if context.get("day"):
        parts.append(f"on {context['day']}")
    if context.get("period"):
        parts.append(f"period {context['period']}")
    return " ".join(parts)


def classify(text: str, context: dict | None = None) -> dict:
    """-> {intent, grade, section, day, period, why}. Never raises.

    `context` is the previously answered query. It is what lets "send me the
    report by email" mean anything at all."""
    t = _norm(text)
    slots = extract_slots(text)
    res = {"intent": INTENT_NONE, **slots, "why": ""}
    if not t:
        res["why"] = "empty"
        return res

    # Another skill or the knowledge base owns it. Checked FIRST: "send me the
    # video message" contains "send me", "sick leave" contains "leave".
    if _OTHER_SKILL.search(t):
        res["why"] = "belongs to another skill"
        return res
    if _KNOWLEDGE.search(t):
        res["why"] = "knowledge-base question"
        return res

    # "send me the report" - resolvable only against the last query.
    if is_followup(text):
        if context and context.get("intent") in (INTENT_FREE_TEACHERS,
                                                 INTENT_FREE_STUDENTS,
                                                 INTENT_FREE_COUNT):
            subject = context.get("subject") or (
                "teachers" if context["intent"] == INTENT_FREE_TEACHERS else "students")
            res.update({
                "intent": (INTENT_FREE_TEACHERS if subject == "teachers"
                           else INTENT_FREE_STUDENTS),
                "grade": context.get("grade"), "section": context.get("section"),
                "day": context.get("day"), "period": context.get("period"),
                "subject": subject,
                "why": "follow-up resolved against the previous question",
            })
            return res
        res["intent"] = INTENT_FOLLOWUP
        res["why"] = "asked to send something, but nothing was asked before it"
        return res

    has_teacher = bool(_TEACHER.search(t)) or bool(
        _STT_TEACHER.search(t) and _AVAILABLE.search(t))
    has_student = bool(_STUDENT.search(t))
    has_avail = bool(_AVAILABLE.search(t))
    has_count = bool(_COUNT.search(t))
    wants_names = bool(_WANTS_NAMES.search(t))
    has_tt = bool(_TIMETABLE.search(t)) and not _TIMETABLE_NOT_OURS.search(t)
    staff_ctx = bool(_STAFF_CONTEXT.search(t))
    named_class = slots["grade"] or slots["section"]

    # A facility is never a person.
    if _FACILITY.search(t) and not (has_teacher or has_student):
        res["why"] = "about a room or facility, not people"
        return res
    # One named adult's diary, not the class grid. Only when nobody is being
    # counted or listed - "how many teachers are available for the parent
    # meeting" is a legitimate availability count.
    if _DIARY.search(t) and _SINGLE_PERSON.search(t) and not (has_count or wants_names):
        res["why"] = "one person's diary, not the timetable"
        return res

    # An explicit timetable request for a NAMED class wins outright. Compound
    # asks ("give me the timetable for grade 9 D on Monday and also how many are
    # free") otherwise got hijacked by the trailing count.
    explicit_tt_word = re.search(
        r"\btime\s*table\b|\btimetable\b|\bschedule\b|\blesson\s*plan\b", t)
    if explicit_tt_word and named_class and not _TIMETABLE_NOT_OURS.search(t):
        res["intent"] = INTENT_TIMETABLE
        res["why"] = "explicit timetable request for a named class"
        return res

    # ── availability first when it is explicit ──────────────────────────────
    # "which students are free in grade 7 period 3" contains "period", which is
    # also a timetable cue; availability wins because it names the question.
    if has_avail:
        subject_teacher = has_teacher or (staff_ctx and not has_student)
        subject_student = has_student or bool(slots["grade"]) or bool(slots["section"])
        if has_count and not wants_names:
            res["intent"] = INTENT_FREE_COUNT
            res["subject"] = "teachers" if subject_teacher and not has_student else "students"
            res["why"] = "count of who is free -> spoken, no names"
            return res
        if subject_teacher and not has_student:
            res["intent"] = INTENT_FREE_TEACHERS
            res["why"] = "named teacher list -> emailed, admin only"
            return res
        if subject_student:
            res["intent"] = INTENT_FREE_STUDENTS
            res["why"] = "named student list -> emailed, admin only"
            return res
        if wants_names or slots["period"] or slots["day"]:
            # No noun at all: "who is free in period 3 on Tuesday?". Naming a
            # GRADE is what makes it a student question; without one it is the
            # director asking after staff, which is how it is actually used.
            res["intent"] = INTENT_FREE_STUDENTS if named_class else INTENT_FREE_TEACHERS
            res["why"] = ("availability, subject unstated; a class was named so "
                          "students" if named_class else
                          "availability, subject unstated and no class named; staff")
            return res

    # ── timetable ───────────────────────────────────────────────────────────
    # Route even WITHOUT a grade. An unrecognised timetable question falls to the
    # knowledge base and gets an invented lesson; routed here the skill simply
    # asks which class, which is the better failure.
    explicit_tt = re.search(r"\btime\s*table\b|\btimetable\b|\bschedule\b|\blesson\s*plan\b", t)
    if has_tt and (named_class or explicit_tt):
        res["intent"] = INTENT_TIMETABLE
        res["why"] = "timetable lookup"
        return res

    res["why"] = "no timetable intent detected"
    return res


def is_skill(text: str, context: dict | None = None) -> bool:
    """True when the timetable skill owns this utterance - brain_node uses this
    to stay silent instead of sending it to the knowledge base."""
    return classify(text, context)["intent"] != INTENT_NONE
