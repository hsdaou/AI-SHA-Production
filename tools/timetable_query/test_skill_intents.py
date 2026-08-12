#!/usr/bin/env python3
"""Regression test for the shared intent classifier.

    python3 test_skill_intents.py

240 labelled utterances covering how a director, a teacher and a speech-to-text
engine actually phrase these questions, plus 68 adversarial negatives that must
NOT be hijacked (tuition fees, sick leave, video messages, room bookings, "how
many teachers work here"). A false positive is the expensive class: it diverts a
question from the knowledge base and answers it wrongly, or emails a list of
minors nobody asked for.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("~/robot_ws/install/aisha_brain/lib/python3.10/site-packages/aisha_brain",
          "~/robot_ws/src/aisha_brain/aisha_brain"):
    sys.path.insert(0, os.path.expanduser(p))
import skill_intents  # noqa: E402

BASELINE_INTENT = 238   # of 240; the two known misses are documented below
BASELINE_SLOTS = 403    # of 406


def main():
    cases = json.load(open(os.path.join(HERE, "intent_corpus.json")))
    intent_ok, slot_ok, slot_tot, fails = 0, 0, 0, []
    for c in cases:
        r = skill_intents.classify(c["text"])
        if r["intent"] == c["intent"]:
            intent_ok += 1
        else:
            fails.append((c["text"], c["intent"], r["intent"]))
        if c["intent"] == "none" or r["intent"] != c["intent"]:
            continue
        for k in ("grade", "section", "period", "day"):
            exp = c.get(k)
            if exp in (None, ""):
                continue
            got = r.get(k)
            if k == "section":
                exp, got = str(exp).upper(), str(got or "")
            if k == "day":
                exp, got = str(exp).lower(), str(got or "").lower()
            slot_tot += 1
            if str(got) == str(exp):
                slot_ok += 1

    print(f"intent: {intent_ok}/{len(cases)}   slots: {slot_ok}/{slot_tot}")
    for t, want, got in fails:
        print(f"  want={want:<14} got={got:<14} | {t[:70]}")

    # Two accepted misses, both argued rather than ignored:
    #  - "Which teachers are free during periods on Saturday?" routes to the
    #    skill. The corpus wanted a refusal because school is closed; answering
    #    "no data for Saturday" from the timetable is the better failure.
    #  - "eh what class does seven be have tomorrow" - speech-to-text mangling
    #    past the point where "seven be" is recoverably grade 7 section B.
    bad = intent_ok < BASELINE_INTENT or slot_ok < BASELINE_SLOTS
    print("REGRESSION" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
