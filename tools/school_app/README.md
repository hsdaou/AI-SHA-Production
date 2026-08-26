# Teacher-Student App

Timetable and availability service for the International School of Choueifat,
Sharjah, and the corridor robot (AI-SHA) that answers questions about it.

A rewrite of `school_app`. The defects it addresses are catalogued with
reproductions in [WEAKNESSES.md](WEAKNESSES.md); the two that shaped the design
were that the old app reported 117 students sitting in a Mathematics lesson as
free because a regex deleted the word "Mathematics", and that it treated *every*
failure to establish a fact as evidence that a child was available.

## The one rule

**Free is a positive finding, never a default.** Status is one of three values:

| | meaning |
|---|---|
| `in_class` | the student's matched group or their section is timetabled now |
| `free` | the section grid is free and no matched individual lesson overrides it |
| `unknown` | no schedule covers this student, so no claim is made |

`unknown` is an answer callers must handle, not an error. Grade 1, KG1 and KG2
have a roster and full enrolment but appear in no timetable, so the app says it
does not know rather than reporting 843 children as available.

## Current dataset

Built from 14 class-list PDFs and 14 subject-list PDFs (2,075 pages: 127 class
pages and 1,948 subject pages), plus the teacher schedule workbook and the
section timetable.

```
3,530 students      across KG1, KG2 and grades 1-12   (was 1,878, grades 5-12)
51,357 enrolments   student -> teaching group          (was 0 — sheet was empty)
1,794 groups        subject x cohort x room x teacher; 965 matched exactly
3,228 lessons       teacher-centric, in wall-clock minutes
                    (3,060 from the teacher schedule, 168 recovered from the
                     class grid for staff it omits)
  127 sections      across 14 grades
```

Cross-check: the class lists and subject lists carry each student's section
independently, and they agree on **51,328 of 51,328** comparisons. The build now
enforces and records this check; a cross-grade enrolment link is rejected rather
than allowed to place a child into another grade's lesson.

## Build

```bash
pip install -r requirements.txt

python3 -m etl.build \
  --class-lists   /path/to/fwclasslistupdated.zip \
  --subject-lists /path/to/fwsubjectlistupdated.zip \
  --teacher-xlsx  /path/to/Teacher_Schedule_Database.xlsx \
  --timetable-db  /path/to/school_timetable.db \
  --out data/school.db --report data/build_report.json
```

The report inputs can be ZIP archives, directories, or individual PDFs. ZIPs are
read through private temporary extraction, so student reports do not need a
persistent plaintext copy. Only the class and subject reports are required; the
schedule sources are optional and the build records which grades they cover. It
hashes every source, checks foreign keys and SQLite integrity, writes to a
temporary file, and atomically publishes only a successful build. Needs
`pdftotext` (`poppler-utils`).

The build prints a coverage table and every compromise it had to make. Both are
stored in the database (`grade_coverage`, `build_issues`) and served at
`/api/diagnostics`, because reading them is part of trusting the numbers:

```
  grade  students  enrolments  resolved  self-study  no-teacher  timetable  coverage  verdict
  K1          270        2184       609           0         450  no             28%  insufficient
  05          296        3976      3215          26         478  yes            81%  good
  07          283        4562      2924         118        1196  yes            66%  partial
  10          139        3089      2696         174          37  yes            92%  good
```

A grade whose verdict is `insufficient` gets refusals, not counts. The last two
gap columns make the shortfall attributable rather than mysterious:

Coverage excludes groups marked `ambiguous`, because those meetings prove that a
teacher is busy but cannot place an individual student. A grade with no section
timetable is always `insufficient`, even when part of its enrolment matches the
teacher schedule; the verdict and `has_timetable` flag can no longer contradict.

- **self-study** — enrolment in a supervised self-study group. There is no lesson
  to find, so this is excluded from the coverage denominator: it is a fact about
  the school, not a hole in the data.
- **no-teacher** — enrolment whose teacher appears in *neither* schedule source.
  This is the real remaining gap, and it is concentrated: one teacher missing
  from the staff reports (MGZ1, Georges Zakharia) accounts for 1,383 enrolments
  and is most of why grades 7 and 8 sit below the others.

### Name reconciliation

The class lists and the timetable sometimes spell the same section or room
differently. Merging them is **declared, not guessed** — merging two that are
genuinely distinct would put children in the wrong room, which is worse than
leaving them unresolved. The declarations live at the top of `etl/build.py` with
the evidence for each, and can be added to without editing code:

```bash
python3 -m etl.build ... --section-alias 11:L=LA --room-alias 11L=11LA
python3 -m etl.build ... --no-default-aliases          # ignore the built-ins
```

Every alias is re-checked against the data at build time and **skipped with a
warning** if its preconditions no longer hold — so it becomes inert by itself
once the source reports are corrected:

```
[warning] NOT merging grade 09 section SA into SB: SA now has a timetable of its
          own, so they are two different sections. Remove the alias.
```

Three are applied by default:

- **section `11:L` → `LA`.** Grade 11's class list names a section `L` holding 31
  students with no timetable; the timetable names `LA` with a full 44-cell week
  and no students. Six of those students' own teachers teach in room `11LA`, and
  nobody in the school teaches in `11L`. Grade 10 is the control — it really does
  have a section `L`, with its own 44 cells and 41 lessons in room `10L`, and is
  untouched.
- **room `11L` → `11LA`** and **room `09L` → `09LA`.** The enrolment reports print
  these one character short. The second matters: eight grade-9 groups could
  previously match only on subject, which cannot distinguish one of a teacher's
  same-subject groups from another, so `Mathematics L` claimed **29** weekly
  meetings instead of 5 and `English` claimed 25 instead of 4.

An alias is only ever added where **exactly one** real room extends the truncated
name. `12L` (could be 12LA or 12LB), `11S` and bare `11` are deliberately left
unresolved — choosing one would place children in a room they may not be in,
which is worse than matching by subject alone. Every build lists them, and flags
any remaining name that *does* have a single expansion together with the command
to fix it:

```
[warning] room 09L (14 groups, 265 enrolments) matches no lesson, and exactly one
          real room extends it: 09LA. Add ROOM_ALIASES['09L'] = '09LA', or
          --room-alias 09L=09LA
```

All of this is asserted in `tests/test_dataset.py`, including that the ambiguous
ones are never rewritten.

### "Room" means two different things

Aliases only fix names that were *truncated*. The larger mismatch is that the two
reports do not mean the same thing by room at all: the subject lists print the
**physical room** (grade 9's groups meet in `09A`, `09B`, `09C`) while the teacher
schedule's column is headed *Room/Class* and, for the upper grades, holds the
**section** (`09SA`, `09SB`). No rename reconciles those — they are different
facts about the same lesson.

The course group is the missing link: in those grades it *is* the section name,
so `grade + course_group` reconstructs exactly the string the schedule wrote. It
is used as a second room key alongside the printed one, so the lower grades —
where the printed room already is the section — keep working unchanged, and a
derived key that corresponds to no real room simply never matches.

The current conservative result is 965 exact groups school-wide. Grade 9 has 91
exact and 23 ambiguous groups, with 80% individually placeable enrolment. An
`exact` label now requires room and subject to corroborate the same retained
event; evidence on a weaker discarded event cannot promote a room-only match.

`Mathematics L` had claimed 29 meetings a week; it now correctly has 5.

### Teacher identity, and a code that means two people

Teacher codes are not reliable keys either — the two reports disagree, and they
**reuse** codes. `MKM2` is Mr Kevin Mercier teaching Music in the subject lists
and Karen Merched teaching Mathematics in the teacher schedule. Keyed on the code
alone, 628 of Mr Mercier's enrolments are filed under Ms Merched's name, and a
Music group is one room-coincidence away from being matched to a Maths lesson.

So identity is resolved through the *name*, and a group is only re-pointed when
the schedule holds exactly one code for that name. The remap key includes both
the source name and source code, so another person reusing the code is untouched:

```
MKM2  Kevin Mercier     -> OKM1      MRA1  Rafat Aly        -> ERA1
EKM2  Karen Merched     -> MKM2      ENIV1 Inneke Vermeulen -> EIV1
```

The collision is reported as an **error**, not fixed silently.

### Lessons recovered from the class grid

The teacher schedule holds 121 teachers; the section timetable names 137. The
difference is not idle staff — it is 27 teachers whose lessons exist only
class-first. Ms Abu Jaber (Art) has 31 cells in the class grid and no row at all
in the teacher schedule, so 911 of her students could not be placed and she read
as "not in the schedule" rather than as teaching.

A taught cell *is* a lesson stated the other way round, so 168 lessons are
recovered from it. Cells for one teacher at one hour merge into a single lesson
with several rooms (a combined class, not a double booking), and a cell is
skipped wherever the teacher schedule already places that teacher at that hour,
so the two sources can never double-count. `lessons.source` records which
supplied each one.

### Individual lessons override section-level free cells

The supplied schedules contain split-group cases where the section grid says
`free` while a student's own enrolment is tied to a teacher lesson at that same
minute. There are 1,158 such student/period overrides in the current data.
The resolver now checks the individual schedule first: a matched lesson produces
`in_class`, the response carries a source-conflict flag and caveat, and a robot
calls a section free only when every student in it is actually free. This closes
the remaining path by which a coarse section cell could erase a more specific
lesson.

## Run

```bash
export SCHOOL_STAFF_KEY=...      # named students
export SCHOOL_ROBOT_KEY=...      # counts and timetables only
python3 -m tsapp                 # development
gunicorn -w 4 -b 127.0.0.1:5000 'tsapp:create_app()'   # production
```

With neither key set every data endpoint returns 503. An unconfigured deployment
must not behave like an open one.

Send keys in the `X-API-Key` header. Query-string keys are rejected because URLs
are retained in browser history and routinely copied into proxy access logs. The
web UI keeps its key only in the current tab's session storage, not persistent
browser storage.

| variable | default | |
|---|---|---|
| `SCHOOL_DB` | `data/school.db` | |
| `SCHOOL_STAFF_KEY` | — | required for named data |
| `SCHOOL_ROBOT_KEY` | — | counts, timetables, section status |
| `SCHOOL_TZ` | `Asia/Dubai` | |
| `SCHOOL_HOST` / `SCHOOL_PORT` | `127.0.0.1` / `5000` | |
| `SCHOOL_DEBUG` | off | refuses to start on a non-loopback bind |
| `SCHOOL_REPORT_TO` | — | e-mail recipients for robot reports |
| `SCHOOL_SMTP_*` | `smtp.gmail.com:587` | `USER`, `PASS` (app password) |

## Asking about a moment in time

Every endpoint that can be asked "when?" accepts one of:

- nothing — now;
- `day=Monday&time=13:20` — an explicit wall-clock minute;
- `day=Monday&period=Period 3` — **resolved against that grade's own bell
  schedule**, so `grade` is required.

The response always reports the moment it resolved to. This matters because the
same label is a different hour in different grades — grade 5's `Period 3` is
10:00–10:50, grade 9's is 09:40–10:30 — and the old app silently summed the two.

## API

Staff key required for anything naming a person; robot key suffices for the rest.

| | |
|---|---|
| `GET /api/health` | no key; liveness and build provenance |
| `GET /api/grades` | grades with coverage verdicts |
| `GET /api/periods?grade=` | that grade's bell schedule |
| `GET /api/now` | current period per grade |
| `GET /api/diagnostics` | coverage and every build issue |
| `GET /api/snapshot?grade=` | counts per status + section breakdown + caveats |
| `GET /api/sections?grade=` | per-section status |
| `GET /api/section/timetable?grade=&section=&day=` | a section's day |
| `GET /api/teachers` | counts always; names with the staff key |
| **staff** `GET /api/students?grade=&status=` | named students, paginated |
| **staff** `GET /api/student/search?q=` | by name or computer number |
| **staff** `GET /api/student/<n>` | where one student is now |
| **staff** `GET /api/student/<n>/day?day=` | that student's own timetable |

### Robot surface

Counts and timetables are **speakable**; anything naming a person is **e-mailed**
from this process and the robot receives only `{ok, count}`. Each response carries
a `speakable` string written to be read aloud verbatim — including its caveats,
because a robot will not add "approximately" on its own.

| | |
|---|---|
| `GET /api/robot/now` | |
| `GET /api/robot/free-count?grade=` | refuses grades it cannot schedule |
| `GET /api/robot/free-sections?grade=` | the actionable form of the question |
| `GET /api/robot/timetable?grade=&section=` | |
| `GET /api/robot/teacher-count` | |
| `POST /api/robot/email/free-students?grade=` | |
| `POST /api/robot/email/free-teachers` | |
| `POST /api/robot/email/student?computer_number=` | never read out in a corridor |

## Layout

```
etl/            build the database from the school's reports
  pdfgrid.py      word-coordinate extraction; recovers columns from geometry
  class_lists.py  the roster
  subject_lists.py the enrolment — the dataset the old app lacked
  timegrid.py     wall-clock parsing and room-name repair
  subjects.py     subject-name reconciliation
  schedule.py     lessons, and tying groups to them by identity
  schema.sql      the normalised store
  build.py        orchestration, coverage analysis, issue log
tsapp/          the service
  db.py           read-only connections, reopened when the file changes
  resolver.py     the three-state availability engine
  auth.py         two scopes; names need the staff one
  api.py          staff API
  robot.py        speakable / e-mailed split
tests/          177 tests, including standing data-quality assertions
```

## Tests

```bash
python3 -m pytest tests/ -q      # 177 passed
```

`tests/test_dataset.py` runs against the real built database and is the standing
data-quality check. Its central assertion sweeps every fully-timetabled period in
the school week and fails if any of them reports a single free student — the shape
that the old code produced constantly. It found a real bug during development:
grade 11's `L`/`LA` mismatch was being reported as *free* rather than *unknown*,
which is the same defect this rewrite exists to remove. That is now fixed at two
levels — the resolver distinguishes "no timetable for this section" from "free",
and the alias above merges the two names.

## Known gaps

These are in `/api/diagnostics`, not hidden:

- **KG1, KG2, grade 1** — roster and enrolment, no section timetable. All
  availability questions answered `unknown`; all three coverage verdicts are
  `insufficient` regardless of partial teacher-schedule matches.
- **6,140 enrolments taught by staff in no schedule at all.** The largest
  remaining gap, and the reason grades 7 and 8 sit at 66% and 64% while their neighbours
  reach 90%+. It is concentrated: `MGZ1` (Georges Zakharia) alone accounts for
  1,383, `R07` for 628, `ECM6` for 555. These people teach real lessons; the
  staff reports we were given simply do not list them. Nothing in this codebase
  can recover that — it needs the source PDFs.
- **131 ambiguous groups** — a group whose room matches nothing (and whose course
  group reconstructs nothing) can be placed only by teacher and subject, which
  cannot tell one of a teacher's same-subject groups from another. Such lessons
  prove the teacher is busy but are not used to place an individual student.
- **253 of 3,294 taught timetable cells (8%) name no teacher**, so a student in
  one can be placed only through their own enrolment. Grade 5's Monday Period 3
  is the worst case: a combined `Math/English/2nd Lang` cell with no teacher,
  where the section splits into groups whose teachers are mostly among the
  missing above. The app reports those students as in class with low confidence
  and says why.
- **19 lessons with no usable room**, and 3 room tokens (`SB`, `SD`, `F`) printed
  with no grade prefix and no sibling to borrow one from.
- **`EHB1`** appears in the section timetable but in no staff list.
- The teacher schedule and section timetable come from a previous extraction of
  reports that are not in this repository. Re-parsing those PDFs directly with
  `etl/pdfgrid.py` would raise coverage further, particularly for KG–grade 2.
