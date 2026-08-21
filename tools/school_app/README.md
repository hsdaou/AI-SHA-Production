# school_app — timetable backend for the AI-SHA timetable skill

Flask app on `127.0.0.1:5055`. `tools/timetable_query/` is the thin robot-side client that calls it;
this is the service that actually answers. Run by `school-app.service`.

## What is deliberately NOT in git

`data/` holds `Student_Database_1.xlsx` (**minors' personal data**), `Teacher_Schedule_Database.xlsx`
and the `school_timetable.db` built from them. They are gitignored and must be copied to the
machine out of band. Only `data/bell_schedule.json` is tracked — it is just period times.

`school_app.env` holds the robot key and a Gmail app password; only `school_app.env.example` is
tracked. Copy it to `~/.school_app.env`, fill it in, and `chmod 600` it.

## Two data sources that number time differently

This is the thing to know before touching period logic.

- The **class timetable** (`timetable_entries` in the sqlite db) *names* periods:
  `Period 1`..`Period 9`, plus `SLO/Period 6`, `SLO/P7`, `Lunch`, `After School`.
- The **teacher schedule** (`Teacher_Schedule_Database.xlsx`) *numbers* every slot in the day,
  **including lunch**. Lunch is slot 6 and nobody teaches in it, so that source has no slot `6`.

So the two agree up to period 5 and are **off by one after it**:

| Class timetable | Teacher slot |
|---|---|
| Period 1–5 | 1–5 |
| Period 6, SLO/Period 6 | 7 |
| Period 7, SLO/P7 | 8 |
| Period 8 | 9 |
| Period 9 | 10 |
| After School | 11 |

`robot_api.teacher_slot()` does this conversion and every teacher lookup must go through it.
Passing `6` straight through matches nothing, so no teacher is marked busy, the endpoint reports
every teacher free, and `busy_free_sane()` refuses the answer.

## Known data gap

`timetable_entries` contains **Monday, Tuesday, Wednesday and Thursday only — there is no Friday**.
Friday questions cannot be answered until the Friday timetable is loaded.

## Privacy split

`robot_api.py` enforces it: counts and class timetables are **spoken**; anything naming an
individual is **emailed** and never reaches the robot. AI-SHA stands in a public corridor with a
speaker, so it must never receive a list of named minors.
