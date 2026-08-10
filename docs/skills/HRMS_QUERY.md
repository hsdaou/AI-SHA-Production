# Skill — HRMS leave query (AI-SHA → HRMS report by email)

An administrator asks AI-SHA who is on leave; the HRMS builds the report and emails it.
The robot is a **thin trigger**: it authorises the request locally, names an intent, and gets back
a confirmation. **Staff leave data never reaches the robot** and is never spoken aloud.

```
Admin ──face+PIN──► hrms_query.py ──HTTPS + x-robot-key──► HRMS /api/reports/on-leave
                          │                                        │
                    speaks only                            queries Supabase,
                 "report sent" ◄───{ok,count}───────      renders + emails report
```

## Why this shape

- **No database credential on the robot.** The Supabase service-role key bypasses all row-level
  security and grants full read/write on every staff record. The Jetson is physically accessible in
  a school corridor. It holds only a narrow, revocable read-only bearer token instead.
- **No leave data on the robot.** The endpoint responds with counts only — never names, emails or
  dates. Nothing sensitive lands in the robot's RAM, disk or logs.
- **Never spoken.** AI-SHA stands in a public area with a speaker. "Who is on sick leave" answered
  aloud is a personal-data disclosure to every passer-by. Spoken output is fixed and data-free.
- **No text-to-SQL.** `llama3.2:1b` is far too small to write reliable SQL, and a hallucinated query
  against HR data is both a correctness and a privacy failure. The model only picks an intent from a
  fixed table and extracts a date; every parameter is bound server-side.

## Files

| What | Where |
|---|---|
| Robot trigger | `tools/hrms_query/hrms_query.py` (repo) |
| Runtime config | `~/hrms_query/config.json` (0600, **git-ignored**) |
| HRMS endpoint | `src/app/api/reports/on-leave/route.ts` (HRMS repo) |
| Session contract | `~/face_auth/session.json`, written by `auth_gate.py` |

Adds **zero** new pip dependencies — stdlib `urllib` only.

## Config

```json
{
  "enabled": false,
  "base_url": "https://your-app.vercel.app",
  "robot_key": "<ROBOT_API_SECRET>",
  "timeout_s": 30
}
```

`enabled` is the gate and is **closed by default**, matching the video-message skill. For local
development the Jetson reaches the workstation over the USB link at `http://192.168.55.100:3000`.

HRMS side needs `ROBOT_API_SECRET` (generate with `openssl rand -hex 32`) and optionally
`ROBOT_REPORT_TO`. If `ROBOT_REPORT_TO` is unset, reports go to every user with the
`final_approver` role. **`ROBOT_API_SECRET` must differ from `SEED_SECRET`** — the seed key can
write; the robot must only ever read.

## Usage

```bash
python3 hrms_query.py status                                  # config, session, intents
python3 hrms_query.py report --intent on_leave [--date YYYY-MM-DD] [--speak]
python3 hrms_query.py ask "who is on leave tomorrow" [--speak]
```

`--skip-auth` bypasses the admin gate and is **development only** — it prints a loud warning.

Exit codes: `0` sent · `2` config/usage · `3` gate closed or no admin session · `4` HRMS
unreachable · `5` HRMS returned an error.

## Intents

| Intent | Keywords | Parameter |
|---|---|---|
| `on_leave` | "on leave", "who is off", "absent", "away", "leave today" | date (`today`/`tomorrow`/`yesterday`/`YYYY-MM-DD`) |

Add an intent by extending `INTENTS` in `hrms_query.py` and adding the matching read-only route.

## Validation status (2026-08-09)

**Verified live:**
- Endpoint rejects a missing key and a wrong key → `401` (constant-time comparison).
- Malformed and impossible dates → `400`.
- Robot refuses when the gate is closed → exit 3; unknown intent / bad date → exit 2.
- Robot refuses when there is no valid admin session → exit 3.
- Real transport Jetson → workstation → route, authenticating past the key check.
- Utterance routing, including "tomorrow" → next day's date.

**NOT yet verified — blocked:**
- The Supabase query, report rendering and email send. The project referenced by
  `NEXT_PUBLIC_SUPABASE_URL` returns **NXDOMAIN** (authoritative, confirmed via DNS-over-HTTPS), so
  the database is unreachable. Endpoint returns a truthful `500` with the error rather than a false
  success.
- Resend cannot currently deliver to `hsdaou@gmail.com`: the shared `onboarding@resend.dev` sender
  only permits the account owner's address (`houssam.daou@iscshj.sabis.net`) until a domain is
  verified. Confirmed with a live `403 validation_error`.
