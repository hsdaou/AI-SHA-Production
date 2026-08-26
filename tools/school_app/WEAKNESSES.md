# Audit of `school_app`

Findings from reading `app.py` (758 lines), `robot_api.py` (517) and
`build_data.py` (491), each reproduced by running the code against its own data.
Ordered by how wrong the answer a user receives is.

Test evidence throughout comes from executing `school_app/app.py`'s own functions
with `SCHOOL_DATA_DIR` pointed at `school_app/data/`.

---

## 1. `normalize_subject` deletes the subject name

`app.py:47`

```python
n = re.sub(r"\s*[-]?\s*(n[0-9]*|l[0-9]*|m[0-9a-z]*|tn[0-9]*)$", "", n)
```

The intent is to strip a level suffix, turning `"Mathematics N2"` into
`"Mathematics"`. But no alternative is anchored to a word boundary, and
`m[0-9a-z]*$` matches from *any* `m` to the end of the string:

```
normalize_subject("Mathematics")  ->  ''        (matches from the leading m)
normalize_subject("Music")        ->  ''
normalize_subject("Chemistry")    ->  'che'
normalize_subject("Economics")    ->  'econo'
normalize_subject("Drama")        ->  'dra'
```

`student_is_enrolled` then returns early on the empty string (`app.py:70-71`),
and its `False` means "not enrolled", which `find_free_students` reports as
**free**:

```
student_is_enrolled({"Mathematics N2"}, "Mathematics")  ->  False
```

**Effect.** Every student in every Mathematics and every Music lesson in the
school is reported free. Measured on the app's own data:

| Query | Old app says |
|---|---|
| Grade 5, Monday Period 4 | 117 free — reason `Not enrolled in: Mathematics` for all 117 |
| Grade 11, Monday Period 8 | 114 free — reason `Not enrolled in: Mathematics` |

Grade 5's Period 4 is Mathematics in sections A, B, C and G — 30+29+29+29 = 117
students, sitting in a Mathematics lesson, all reported available. The timetable
contains 353 Mathematics cells and 42 Music cells.

Chemistry and Economics escape only by accident: `'chemistry'.startswith('che')`
is true, so the truncation still matches.

---

## 2. Everything unknown defaults to "free"

`app.py:426-434`

```python
elif not student_is_enrolled(enrolled_subjects, tt_subject):
    free_students.append({... "reason": f"Not enrolled in: {tt_subject}"})
```

This is the structural defect, and finding 1 is only its most common trigger.
Every way of failing to establish a fact produces the same confident output:
*this child is free*.

Failure modes that all land here:

- a subject name that normalises to nothing (finding 1);
- a combined timetable cell — `"Math/English/2nd Lang"`, `"Math Alg / Geo"` —
  which cannot equal any single enrolment name. Grade 5 Monday Period 3 is
  `Math/English/2nd Lang` in all ten sections: **296 of 296 students free**;
- a timetable activity that is not a subject anyone enrols in. Grade 12 Monday
  Period 1 is `Periodic` in all six sections: **163 of 163 students free**;
- an empty enrolment sheet (finding 8): *every student in the school*, free,
  always.

For a school this default is the wrong way round. "I could not tell" and "this
child is unsupervised and free to be collected" are different sentences, and the
robot in the corridor says the second one.

`robot_api.py:169-198` adds `audit_free`, which detects the specific case where a
section is 100% free *and* every reason begins with `Not enrolled`, and attaches
"treat that as an over-estimate". It is a real improvement, but it is a smoke
alarm over a gas leak: it cannot fire when the section is only partly wrong (117
of 296 in grade 5 Period 4 — mixed reasons, `reliable: true`), and the answer is
still delivered.

---

## 3. Students whose section is not in the timetable are free forever

`app.py:388, 408-416`

`get_level_sections` collects timetable sections with no students, and
`find_free_students` skips those. The dangerous direction — a **roster** section
with no **timetable** — is not handled: `get_class_timetable` returns `[]`,
`period_entry` is `None`, and the student is appended as free with reason
`"No timetable entry"`.

Grade 11 is exactly this shape. The roster names the section `L`; the timetable
names it `LA`:

```
sections in the timetable: ['LA', 'SA', 'SB', 'SC', 'SD', 'SE']
sections in the roster   : ['L',  'SA', 'SB', 'SC', 'SD', 'SE']
get_level_sections(11)   = {'LA'}        <- guards the harmless direction
```

**Effect.** 31 real students are reported free in every period of every day,
permanently, with no caveat — `audit_free` cannot see it, because the reason is
`No timetable entry` rather than `Not enrolled`:

```
grade 11 Monday Period 1  free=174  busy=0    (31 of them "No timetable entry")
grade 11 Monday Period 4  free= 31  busy=143  (all 31 "No timetable entry")
grade 11 Monday Period 8  free=151  busy=23   (31 + 120 from finding 1)
```

---

## 4. Three period-numbering schemes, joined by label

`app.py:456`, `robot_api.py:490-491`

Period labels do not denote the same hour across sources, and the same label is a
different hour in different grades:

| Wall clock | Grade 5 calls it | Grade 9 calls it | Teacher sheet |
|---|---|---|---|
| 09:40–10:30 | — | Period 3 | 3 |
| 10:00–10:50 | Period 3 | — | 3 |
| 11:40–12:30 | — | Period 5 | 5 |
| 12:15–13:05 | Period 5 | — | 5 |
| 13:05–13:50 | Period 6 | SLO/Period 6 | **7** |
| 13:50–14:35 | SLO/P7 | Period 7 | **8** |
| 14:35–15:20 | Period 8 | Period 8 | **9** |

`find_free_teachers` compares `str(p) == str(period_num)` against the teacher
sheet's own numbering, and `robot_api` prepares the argument by deleting the word
"Period":

```python
period = str(period_raw).replace("Period", "").replace("period", "").strip()
```

So "who is free in Period 7" for a grade 9 caller (13:50) is answered with the
teachers free at 13:05 — off by a full period. And the teacher sheet has **no
period 6 at all** (it jumps 5 → 7), so "Period 6" matches nothing, no teacher is
ever marked busy, and the endpoint reports every teacher free. That is the bug
`robot_api.py:158-166` was written to suppress:

```python
def busy_free_sane(n_free, n_busy):
    """free=127 busy=0 is not a quiet day, it is a lookup that compared the
    wrong things"""
    return not (n_busy == 0 and n_free > 5)
```

Correctly diagnosed, and then handled by refusing to answer rather than by fixing
the comparison.

The same defect makes whole-school aggregation meaningless: `r_free_count`
(`robot_api.py:346-362`) loops over grades asking each for `period` **by name**,
summing grade 5's 10:00 students together with grade 9's 09:40 students and
calling the total "free in Period 3".

---

## 5. `/api/*` returns 1,878 children's names with no authentication

`app.py:503-733` vs `robot_api.py:210-219`

`/api/robot/*` requires a shared secret and fails closed. `/api/free-students`,
`/api/free-students-day` and `/api/section-status` require nothing, and return
full name, section, computer number and e-mail address for every student.

The app knows:

```
WARNING: listening beyond localhost with SCHOOL_ROBOT_KEY unset —
         /api/robot/* will refuse every request (fail closed),
         but /api/* has NO auth and exposes student names.
```

A message printed to a terminal at boot is not an access control. `deploy/` binds
the service to `0.0.0.0` as a systemd unit on a Jetson so the corridor robot can
reach it, which means anything else on that network can enumerate the school's
minors over plain HTTP.

`robot_api.py`'s SPEAKABLE/EMAILED split is a genuinely good design — personal
data is rendered to e-mail in-process and the robot only ever receives
`{ok, count}`. It is undermined completely by the unauthenticated endpoint next
door that returns the same names directly.

---

## 6. A connection per section, and a cache that never expires

`app.py:122-127`

```python
_cache = {}
def get_db():
    conn = sqlite3.connect(DB_PATH)
```

`get_class_timetable` opens a fresh connection, runs one query and closes it. It
is called once per section from `find_free_students` (`app.py:391`), from
`api_section_status` (`app.py:603`) and from `api_free_students_day`
(`app.py:687`) — so "who is free in grade 6" opens eleven connections and runs
eleven queries to assemble what one indexed join returns.

`_cache` is a module-level dict with no bound, no invalidation and no locking. The
first request that touches students parses a 1.5 MB and a 190 KB workbook inline
on the request thread and holds ~23,000 enrolment rows for the process lifetime;
rebuilding the data has no effect until someone restarts the service. Under
`app.run(threaded=True)` two concurrent first-requests race on the same key and
both do the work.

---

## 7. Debug mode reachable from the network

`app.py:751`

```python
debug = os.environ.get("SCHOOL_DEBUG") == "1"
app.run(debug=debug, host=host, port=port)
```

Opt-in is the right call and the comment above it is accurate. But nothing
prevents `SCHOOL_DEBUG=1` together with `SCHOOL_HOST=0.0.0.0`, which publishes
the Werkzeug console — arbitrary code execution — to the network the robot is on.

---

## 8. `build_data.py` cannot produce a file `app.py` can read

`build_data.py:405` writes the roster sheet as:

```python
m.title = "Students"
```

`app.py:136` reads:

```python
ws = wb["Student Master List"]
```

`openpyxl` raises `KeyError` on a missing sheet, so a workbook produced by the
builder crashes the app on first use. The workbook actually in `data/` contains
`Student Master List`, `Subject Enrollment`, `Class Summary`, `Teacher Directory`
and `Subject Catalog` — five sheets the builder never writes. It was produced by
some other tool that is not in the repository, so the documented build path has
never worked.

Separately, `build_student_xlsx` writes the enrolment sheet **empty by
construction** (`build_data.py:424-436`) and records `enrollment_present: false`.
The file's own docstring states the consequence plainly:

> with the sheet EMPTY the app treats every student as not-enrolled and reports
> ALL of them free

`robot_api.enrolment_present()` refuses student-level questions in that state,
which is the right response — but `/api/free-students`, unauthenticated and
un-guarded, answers them anyway.

---

## 9. Smaller things

- **`app.py:196-198`** — dead code that evaluates a generator and discards it:
  ```python
  raw = " ".join(w["text"] for w in group_lines(cell_words)[i]["words"]
                 for i in [0]) if False else " ".join(...)
  ```
- **`app.py:47`** — `"Islamic Education"` → `"islamic educatio"` and
  `"Physical Education"` → `"physical educatio"`: the trailing `n` is eaten by
  `n[0-9]*$`. Harmless only because both sides are normalised identically.
- **`robot_api.py:94-96`** — `resolve_now` picks a period with `start <= now <=
  end` inclusive on both ends, so at exactly 13:05 both Period 5 and Period 6
  match and the first wins by dict order.
- **`robot_api.py:59-68`** — the fallback bell schedule is invented data
  (`Period 1` at 07:45; the real one starts at 08:00). It is clearly labelled a
  placeholder, and `bell_schedule_is_real: false` is returned — but the times are
  still used to answer "right now" if `SCHOOL_BELL_JSON` is unset.
- **`app.py:290`** — `f"{int(grade):02d}-{section}"` raises `ValueError` on the
  KG grades in the source PDFs (`K1`, `K2`), so the schema cannot represent them.
- **No tests anywhere in the repository.** Every defect above is silent: none
  raises, none logs, each one returns a well-formed JSON answer that is wrong.

---

## What was already right

Worth preserving, and preserved in the rewrite:

- The **SPEAKABLE / EMAILED** split in `robot_api.py`. Counts may reach the
  robot's speaker; names are rendered to e-mail inside the process and the robot
  receives `{ok, count}`. This is the correct shape for a device standing in a
  public corridor.
- **`hmac.compare_digest`** for the shared secret, and failing closed when it is
  unset.
- **Geometry-based PDF parsing.** `build_data.py`'s header comment on why
  `pdftotext -bbox-layout` is used instead of `-layout` is correct and its
  reasoning carries over directly to the class and subject list reports.
- The instinct behind **`audit_free`** and **`busy_free_sane`**: notice when a
  result has the shape of a lookup failure and say so. In the rewrite the
  underlying joins are made exact instead, and these become assertions that
  should never fire.

---

## Summary

| # | Defect | Consequence |
|---|---|---|
| 1 | Regex deletes subject names | 117 students in a Maths lesson reported free |
| 2 | Unknown defaults to free | 296/296 and 163/163 grade-wide false positives |
| 3 | Roster section absent from timetable | 31 students free forever, no caveat |
| 4 | Period labels joined across 3 schemes | Teacher availability off by one period; 127/128 free |
| 5 | `/api/*` unauthenticated | 1,878 minors enumerable over the network |
| 6 | Connection per section, unbounded cache | 11 connections per query; stale data until restart |
| 7 | Debug mode can bind publicly | Remote code execution |
| 8 | Builder writes a schema the app cannot read | Documented build path has never worked |
