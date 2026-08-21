# AI-SHA Knowledge Base — Structure, Rules, and Change Register

This document governs `src/aisha_brain/aisha_raw_data/*.md`, the source of AI-SHA's RAG knowledge
base. It is **not** itself part of the knowledge base — it lives in `docs/` so the builder never
indexes it.

Last consolidation: **2026-08-21**.

---

## 1. How a document becomes an answer

`build_knowledge.py` splits each `.md` file into chunks, embeds them with `BAAI/bge-small-en-v1.5`,
and stores them in ChromaDB. `admin_node.py` retrieves the top 6 chunks (cosine distance ≤ 1.0) and
gives **only those chunks** to `llama3.2:1b`.

The chunker's behaviour dictates how these documents must be written:

| Markdown construct | Becomes |
|---|---|
| `# Title` / `## Section` / `###+ Subsection` | Not a chunk. Prepended to every chunk below it as `Title — Section — Subsection` |
| Each list item (`-`, `*`, `+`, `1.`) | **Its own chunk** |
| Each table data row | **Its own chunk**, each cell labelled by its column header |
| Consecutive non-list, non-table lines | Merged into **one** chunk per block |

**The consequence that causes hallucinations:** a list item or table row is retrieved *alone*. Any
qualifier sitting in a neighbouring paragraph — the academic year, the term, "UAE grades only",
"figures exclude transport" — is invisible to the model when that row is retrieved. The model then
fills the gap by inventing.

## 2. Writing rules

1. **Every chunk must stand alone.** A bullet or table row must be a complete, unambiguous statement
   on its own. Prefer "ISC-Sharjah Grade 7 classrooms GR7 A to GR7 J are all on the First Floor in
   Zone 6" over ten separate "GR7 A is on the First Floor" bullets.
2. **Put qualifiers in the heading.** Headings are prepended to every child chunk, so they are the
   cheapest way to attach a year or scope to a whole table — e.g.
   `## Annual Tuition Fees for the 2025-2026 Academic Year (AED)`.
3. **Facts that share one caveat go in a paragraph**, so they stay in one chunk with their caveat.
   **Facts that are independently askable go in a list or table**, with the qualifier in the heading.
4. **One fact, one wording.** Where a fact appears in more than one file (contacts, facility counts),
   the sentence must be **byte-identical**. Retrieval may surface either copy; identical wording means
   identical answers. Run the consistency lint in §5 after editing.
5. **Name the subject in the sentence.** Write "ISC-Sharjah has three science laboratories", not
   "There are three science laboratories" — a bare "there are" chunk is unanchored.
6. **State the academic year on anything dated.** Fees, term dates, holidays and timetables must all
   carry their year in the text or the heading.
7. **Never write in the first person.** Persona lines ("I am AI-SHA…", "I cannot access grades")
   belong in `admin_node.SYSTEM_PROMPT`, not in the knowledge base. As indexed chunks they get
   retrieved as if they were school facts and derail unrelated answers.
8. **Add an explicit "not held here" section** to every document. A retrievable statement of absence
   is what stops the model inventing an answer; silence does not.
9. **No robot-capability claims.** What AI-SHA can drive to or navigate is runtime state, not a
   school fact, and it goes stale silently.

## 3. Documents

| File | Authoritative for |
|---|---|
| `scope-and-limits.md` | Which academic year the KB covers; what it cannot answer; AI-SHA's role boundary |
| `school_facts.md` | Contacts, identity, facility counts, SLO departments, start of the school day |
| `homepage.md` | Short school overview and curriculum summary |
| `admissions.md` | Application process and required documents |
| `tuition-fees.md` | All fee figures |
| `school_calendar.md` | Term dates and holidays |
| `exam_schedule.md` | Term 2 school examination timetable, per grade |
| `cie_ap_exams_2026.md` | External Cambridge and AP examination timetable |
| `campus-map.md` | Room and grade locations by floor and zone |
| `campus-facilities.md` | Facility descriptions (counts must match `school_facts.md` exactly) |
| `sabis_system.md` | SABIS network, academic structure, assessment vocabulary, conduct code |

## 4. Change register — 2026-08-21 consolidation

### Corrected, with the source evidence
| # | Defect | Resolution |
|---|---|---|
| 1 | `school_calendar.md` said Term 2 finals were 9–17 March 2026; `exam_schedule.md` scheduled them 2–12 March. Both were retrieved together at near-identical distance (0.187 / 0.188), so the answer was a coin flip. | Reconciled using statements already present in both files: Grades 2–9 sat finals 2–12 March; Grades 10–12 had normal school days to 14 March; term ended 17 March. The calendar now defers to the timetable for per-grade dates. |
| 2 | Grade 8 and Grade 9 tuition: 3 × 11,000 = 33,000, but the annual fee was stated as 32,600. | Annual fee (32,600) kept as authoritative; those two rows removed from the installment table and replaced with an explicit "confirm the split with the Accounts Office" statement. **No figure was invented.** |
| 3 | Transport per-installment values were truncated, so 3 × 2,333 = 6,999 ≠ 7,000. | Relabelled "Approximate Amount per Installment" and rounded correctly. |
| 4 | Three different SLO department lists (`school_facts.md`, `sabis_system.md`, and `SYSTEM_PROMPT`). | `school_facts.md` is now the single ISC-Sharjah list; `sabis_system.md` describes the SLO structurally and points to it. |
| 5 | `cie_ap_exams_2026.md`: four weekday names did not match their 2026 dates. | "Tuesday 06 May" → "Wednesday 06 May" (×3; the surrounding 05/06/07 May sequence confirms the dates are right and the day names were wrong). See "needs confirmation" below for the fourth. |
| 6 | AQC offices were placed on both the Ground Floor and the First Floor with no distinction. | Stated explicitly: AQC Office and its two meeting rooms are on the Ground Floor; the AQC Coordinator's office is on the First Floor. Both statements now cross-reference each other. |
| 7 | "Two semi-Olympic-sized swimming pools" vs a campus-map note that the second is "Not In Use". | All three files now say: two pools, one currently in service. |
| 8 | Basketball courts described as indoor in one file and outdoor in another. | Stated as both: one indoor court in the gymnasium block, plus outdoor courts on the grounds. |
| 9 | `exam_schedule.md` was titled "Levels E–L (Grades 2–12 UAE / Grades 3–12 GULF)". Levels E–L are Grades 2–9, and the file's own notes said Grades 10–12 had normal school days. | Retitled and split: Grades 2–9 final examinations; Grades 10–12 Continuous Assessment and Periodic tests. |
| 10 | `school_calendar.md` mapped "Grades 7–12" to "Level A-B and Level J-N" — six grades to five levels. | Corrected to Levels J–O, and the full Level↔Grade mapping is now stated once, explicitly. |
| 11 | Section headings like `Grade 8 UAE / Grade 9 GULF` broke `admin_node`'s grade filter: a query for "Grade 9" matched the **Grade 8** section via its GULF label. | Section headings now carry the UAE grade only; the GULF and Level equivalences moved into the body. |
| 12 | Zone 5 index said Grade 6 classrooms "F–K"; the body and quick-lookup said "E–K". | Corrected to E–K throughout. |
| 13 | "Mix Play Areas surround the Kindergarten wing on the First Floor outdoor terrace", while the KG wing is Ground Floor Zone 1. | Rewritten as the terrace above the KG wing, reached from the First Floor. |
| 14 | `homepage.md` said the curriculum draws on "American, British, and Emirati" systems; `school_facts.md` said "American and British" only. | Unified: draws on the American and British systems and delivers the UAE Ministry of Education's required subjects (Arabic, Islamic Studies/Religion, Social Studies) — which the exam timetable confirms are taught. |
| 15 | `sabis_system.md` called the assessment system "CA, formerly AMS", while `exam_schedule.md` labelled everything "AMS". | Both terms defined together and explicitly equated, in both files. |
| 16 | `campus-map.md` claimed AI-SHA has a mecanum drive, a charging station, and a SLAM origin — none currently true, and none of them school facts. | Removed. Robot capability is runtime state, not knowledge-base content. |
| 17 | First-person persona chunks ("I am AI-SHA…", "I cannot access private academic records") were indexed and retrieved into unrelated answers — a failure already noted in `aisha_watch.py:580`. | Removed from the KB; this behaviour belongs to `SYSTEM_PROMPT`. |
| 18 | `school_facts.md` was titled "Verified Knowledge Base", an authority claim prepended to all its chunks. | Retitled "ISC-Sharjah School Facts". |
| 19 | `campus-map.md` produced 252 of 621 chunks, most of them near-identical one-line classroom statements, which crowded out real answers (an SLO question retrieved four campus-map office chunks). | Collapsed into per-grade range statements: 124 chunks, no information lost. |
| 20 | "KG Filling Room" (typo) alongside a separate First Floor "Filing Room". | Corrected to "KG Filing Room"; both are now distinguishable. |
| 21 | Every dated document was written in the present tense with no year marker, so expired 2025-2026 information was presented as current. | Every fee, date and timetable now carries its academic year, and `scope-and-limits.md` states the coverage year once, prominently. |
| 22 | No retrievable statement of what the KB does *not* know, so gap questions drifted to nearest-neighbour noise ("what time does school start" returned first-day-of-term and AP exam chunks). | Added `scope-and-limits.md` plus a "not held in this knowledge base" section to every document. |

### Found by end-to-end testing, after the first rewrite
Running real questions through retrieval + `llama3.2:1b` exposed two more defects, both fixed:

23. **The "cannot answer" bullets were not self-contained** — the rule this very document sets out.
    Each bullet chunked as, for example, "Bus routes, bus stops, bus timings…" while the instruction
    to give the office number sat in a sibling paragraph the model never saw. Asked about a bus
    route, AI-SHA reached for the *academic* refusal and replied "please ask your teacher for bus
    route information". Every bullet in `scope-and-limits.md` now names the topic, states that it is
    not held here, and gives +971 6 558 2211 — in one chunk. The role section now also says
    explicitly that "ask your teacher" is for academic requests only.
24. **The school start time did not retrieve** for the plain phrasing "What time does school
    start?". The 7:55 AM fact was buried mid-paragraph under the heading "Start of the School Day",
    which embeds poorly against that question. Rewritten as its own section, "What Time the School
    Day Starts at ISC-Sharjah", leading with a direct answer sentence.

### Needs confirmation from the school
These could not be resolved from the documents alone. Nothing was invented; each is flagged here.

1. **Term 2 revision week dates.** The old calendar put Revision Week on 3–6 March 2026, which
   collides head-on with examinations scheduled on 3, 4 and 5 March. Term 1 and Term 3 both place
   revision immediately *before* the examinations. The specific dates have been removed rather than
   guessed, and `school_calendar.md` lists them as not held here.
2. **Grade 8 / Grade 9 installment split.** Annual 32,600 vs installments summing to 33,000. The
   annual figure was kept; the split needs the Accounts Office.
3. **AP Computer Science A.** Listed as "Friday 16 May", but 16 May 2026 is a Saturday and falls
   outside the AP examination window, while 15 May is a Friday and its afternoon session is free.
   **Changed to Friday 15 May** as the single-error reading — verify before relying on it.
4. **Music rooms.** `campus-facilities.md` claimed two; the campus map locates only one (Grades 5–8,
   Zone 4). Standardised on one. If a second exists, add it to the map and update both files.
5. **AQC.** Ground-Floor office plus First-Floor coordinator is the reading that satisfies both
   original statements, but it was never stated that way. Confirm, and confirm which one the
   `aqc office` waypoint in `config/nav_locations.json` refers to.
6. **2026-2027 data.** Fees, calendar and timetables for the current year are absent. Until they are
   loaded, AI-SHA will correctly answer with 2025-2026 figures and say so — but that is a stopgap.

### Deliberately removed
- SABIS Vice-President and board-member names — outside AI-SHA's administrative scope, and the
  content most likely to go stale unnoticed.

## 5. Maintenance

Rebuild and redeploy after any edit to `aisha_raw_data/`:

```bash
cd ~/robot_ws/src/aisha_brain/aisha_brain && python3 build_knowledge.py
cd ~/robot_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select aisha_brain
sudo systemctl restart aisha-console.service
```

Note that `colcon build` copies `aisha_knowledge_db` into `install/aisha_brain/share/`. **The running
`admin_node` reads the `install/` copy, not the source tree** — editing markdown alone changes
nothing until both commands above have run.

Backups of the pre-consolidation KB are in `~/kb_backups/` on the Jetson.
