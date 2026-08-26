-- Teacher-Student App: one normalised store, built once, read many times.
--
-- The previous app kept no database of its own worth the name. It parsed a
-- 1.5 MB workbook and a 190 KB workbook on the first request that needed them,
-- held the result in a module-level dict with no invalidation, and opened a
-- fresh SQLite connection for every section it looked at — so answering "who is
-- free in grade 6" opened eleven connections and ran eleven queries to
-- reconstruct something a single indexed join returns.
--
-- Everything below is derived. Nothing here is edited by the app at runtime,
-- which is why there is no user or session table: the store is read-only in
-- production and can be rebuilt from the source PDFs at any time.

PRAGMA foreign_keys = ON;

CREATE TABLE build_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Every compromise the build had to make, kept as data rather than as a line on
-- someone's terminal that scrolled past. The API surfaces the ones that bear on
-- an answer it is about to give.
CREATE TABLE build_issues (
    id       INTEGER PRIMARY KEY,
    severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
    category TEXT NOT NULL,
    detail   TEXT NOT NULL,
    n        INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE grades (
    code    TEXT PRIMARY KEY,          -- 'K1','K2','01'..'12'
    ordinal INTEGER NOT NULL,          -- -1,0,1..12 for sorting
    label   TEXT NOT NULL              -- 'KG1','Grade 1'
);

CREATE TABLE sections (
    id            INTEGER PRIMARY KEY,
    grade         TEXT NOT NULL REFERENCES grades(code),
    letter        TEXT NOT NULL,       -- 'A','SA','LA','L'
    student_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (grade, letter)
);

CREATE TABLE students (
    computer_number TEXT PRIMARY KEY,
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    section_id      INTEGER NOT NULL REFERENCES sections(id),
    family_number   TEXT,
    siblings        INTEGER,
    source          TEXT NOT NULL
);
CREATE INDEX idx_students_section ON students(section_id);
CREATE INDEX idx_students_name    ON students(last_name, first_name);
CREATE INDEX idx_students_first   ON students(first_name, last_name);

CREATE TABLE teachers (
    code           TEXT PRIMARY KEY,
    title          TEXT,
    name           TEXT,
    -- 'Ms. Self Study' (SSTUDY) and 'Mr. New NEW' (ENEW2) are staffing
    -- placeholders, not people. Flagged so a report never addresses them and a
    -- free-teacher count never includes them.
    is_placeholder INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE subjects (
    code TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    head TEXT                          -- distinguishing token, see etl.subjects
);

-- One row per teaching group: a subject, a cohort of it, where it meets and who
-- teaches it. This is the unit a student is actually enrolled in — not a
-- subject, which is why "is this student in the lesson being taught" was never
-- answerable from a subject name alone.
CREATE TABLE teaching_groups (
    id            INTEGER PRIMARY KEY,
    grade         TEXT NOT NULL REFERENCES grades(code),
    subject_code  TEXT NOT NULL REFERENCES subjects(code),
    course_group  TEXT NOT NULL,
    room          TEXT,
    teacher_code  TEXT REFERENCES teachers(code),
    -- How securely this group was tied to the timetable: 'exact' (teacher, room
    -- and subject agree on one event), 'room', 'subject', 'ambiguous', or 'none'.
    match_quality TEXT NOT NULL DEFAULT 'none'
                  CHECK (match_quality IN
                         ('exact','room','subject','ambiguous','none')),
    meeting_count INTEGER NOT NULL DEFAULT 0 CHECK (meeting_count >= 0),
    UNIQUE (grade, subject_code, course_group, room, teacher_code)
);
CREATE INDEX idx_groups_teacher ON teaching_groups(teacher_code);

CREATE TABLE enrolments (
    student_id TEXT    NOT NULL REFERENCES students(computer_number),
    group_id   INTEGER NOT NULL REFERENCES teaching_groups(id),
    PRIMARY KEY (student_id, group_id)
);
CREATE INDEX idx_enrolments_group ON enrolments(group_id);

-- The teacher-centric schedule: the authoritative answer to "when".
-- start_min/end_min are minutes since midnight. Period NUMBERS are kept only
-- for provenance; nothing joins on them, because they mean different hours in
-- different sources. See etl/timegrid.py.
CREATE TABLE lessons (
    id           INTEGER PRIMARY KEY,
    teacher_code TEXT NOT NULL REFERENCES teachers(code),
    day          TEXT NOT NULL,
    start_min    INTEGER NOT NULL CHECK (start_min >= 0 AND start_min < 1440),
    end_min      INTEGER NOT NULL CHECK (end_min > start_min AND end_min <= 1440),
    sheet_period TEXT,
    subject_text TEXT,
    group_code   TEXT,
    room_text    TEXT,
    -- 'teacher_schedule' or 'section_timetable'. The second is recovered from
    -- the class grid for staff the teacher schedule omits — see
    -- etl.schedule.lessons_from_section_grid — and is kept distinguishable so a
    -- reader can tell which source placed a lesson.
    source       TEXT NOT NULL DEFAULT 'teacher_schedule'
);
CREATE INDEX idx_lessons_time    ON lessons(day, start_min, end_min);
CREATE INDEX idx_lessons_teacher ON lessons(teacher_code, day, start_min);

CREATE TABLE lesson_rooms (
    lesson_id INTEGER NOT NULL REFERENCES lessons(id),
    room      TEXT NOT NULL,
    PRIMARY KEY (lesson_id, room)
);

-- Materialised group-to-lesson join: when does each group actually meet.
-- Computed at build time so a request never has to fuzzy-match anything.
CREATE TABLE group_meetings (
    group_id  INTEGER NOT NULL REFERENCES teaching_groups(id),
    lesson_id INTEGER NOT NULL REFERENCES lessons(id),
    day       TEXT NOT NULL,
    start_min INTEGER NOT NULL CHECK (start_min >= 0 AND start_min < 1440),
    end_min   INTEGER NOT NULL CHECK (end_min > start_min AND end_min <= 1440),
    score     INTEGER NOT NULL,
    PRIMARY KEY (group_id, lesson_id)
);
CREATE INDEX idx_gm_time  ON group_meetings(day, start_min, end_min);
CREATE INDEX idx_gm_group ON group_meetings(group_id);

-- The printed per-section timetable. Secondary: it says what a SECTION is
-- doing, which is not the same as what each student in it is doing once the
-- section splits into ability or language groups.
CREATE TABLE section_periods (
    id           INTEGER PRIMARY KEY,
    section_id   INTEGER NOT NULL REFERENCES sections(id),
    day          TEXT NOT NULL,
    period_label TEXT NOT NULL,
    start_min    INTEGER,
    end_min      INTEGER,
    subject_text TEXT,
    teacher_code TEXT,
    group_code   TEXT,
    is_free      INTEGER NOT NULL DEFAULT 0 CHECK (is_free IN (0, 1)),
    CHECK ((start_min IS NULL AND end_min IS NULL)
           OR (start_min >= 0 AND start_min < 1440
               AND end_min > start_min AND end_min <= 1440))
);
CREATE INDEX idx_sp_section ON section_periods(section_id, day, start_min);
CREATE INDEX idx_sp_time    ON section_periods(day, start_min, end_min);

-- Bell times per grade, learnt from the timetable itself rather than typed in.
-- Grade-scoped because the same label is a different hour in different grades:
-- "Period 3" is 10:00 in grade 5 and 09:40 in grade 9.
CREATE TABLE bell_slots (
    grade     TEXT NOT NULL REFERENCES grades(code),
    label     TEXT NOT NULL,
    start_min INTEGER NOT NULL CHECK (start_min >= 0 AND start_min < 1440),
    end_min   INTEGER NOT NULL CHECK (end_min > start_min AND end_min <= 1440),
    PRIMARY KEY (grade, label)
);
CREATE INDEX idx_bell_time ON bell_slots(grade, start_min);

-- How much of each grade's schedule we can actually resolve. The API consults
-- this before answering: a grade whose enrolments mostly cannot be tied to a
-- lesson gets "I don't know", not a confident count.
CREATE TABLE grade_coverage (
    grade                 TEXT PRIMARY KEY REFERENCES grades(code),
    students              INTEGER NOT NULL,
    enrolments            INTEGER NOT NULL,
    resolved_enrolments   INTEGER NOT NULL,
    -- Enrolment in a supervised self-study group. There is no lesson to find, so
    -- this is excluded from the coverage denominator: it is a fact about the
    -- school, not a hole in the data, and counting it as unresolved would push a
    -- grade towards refusal for the wrong reason.
    self_study            INTEGER NOT NULL DEFAULT 0,
    -- Enrolment whose teacher appears in NEITHER schedule source. This is the
    -- real remaining gap and it is attributable: one teacher missing from the
    -- staff reports can account for a thousand enrolments.
    teacher_missing       INTEGER NOT NULL DEFAULT 0,
    coverage              REAL    NOT NULL CHECK (coverage >= 0 AND coverage <= 1),
    has_section_timetable INTEGER NOT NULL CHECK (has_section_timetable IN (0, 1)),
    verdict               TEXT    NOT NULL
                          CHECK (verdict IN ('good', 'partial', 'insufficient'))
);
