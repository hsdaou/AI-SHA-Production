# Skill: Record & Deliver a Video Message (local-first, gated internet)

**Board:** Jetson Orin Nano · **Sensors:** RealSense D435 (color) + ReSpeaker 4-Mic Array
**New system dep:** `ffmpeg` (apt). **No new pip packages.**

The robot (stationed in the administration) offers a visitor to record a short
video message for the admin (Sam). Recording is **100 % local** — no internet
needed, messages queue in a local outbox. Delivery to the outside world is a
**separate, explicitly gated step** that is **CLOSED by default**: the robot
only sends when an admin has (a) filled in credentials and (b) flipped
`"enabled": true` in the config. This preserves the design rule that the robot
operates totally locally, while leaving a deliberate gate for this one service.

---

## Files

| File | Role |
|---|---|
| `tools/video_messenger/video_message.py` | `record` (camera+mic → H.264/AAC mp4 in outbox), `list`, `offer` (prints + publishes the offer on `/robot_speech` for the future TTS tier). |
| `tools/video_messenger/deliver.py` | The **gate**: `init` (writes a *disabled* config template), `status`, `send [--dry-run]`. Transports: WhatsApp Cloud API (official) or SMTP email. |

**Runtime data lives OUTSIDE the repo** at `~/video_messages/` (override
`VIDEO_MSG_HOME`): `outbox/` (queued), `sent/` (delivered), `config.json`
(credentials, 0600, git-ignored). Recorded messages are personal data — never
commit them.

## One-time setup

```bash
sudo apt-get install -y ffmpeg
mkdir -p ~/video_messages && cd ~/video_messages
cp ~/robot_ws/tools/video_messenger/{video_message.py,deliver.py} .
python3 deliver.py init          # writes a DISABLED config template
```

Camera up (color only is enough):
```bash
source /opt/ros/humble/setup.bash && export ROS_DOMAIN_ID=99
ros2 launch realsense2_camera rs_launch.py &
```

## Record a message (fully local)

```bash
cd ~/video_messages
source /opt/ros/humble/setup.bash && source ~/robot_ws/install/setup.bash
export ROS_DOMAIN_ID=99

python3 video_message.py offer                      # the robot's spoken invitation
python3 video_message.py record --seconds 15        # 3-2-1 countdown, then record
python3 video_message.py list                       # see the queue
```

Implementation notes:
- Frames are JPEG-buffered with the **measured** fps (not nominal 30) so audio
  and video stay in sync; ffmpeg muxes to H.264 + AAC + `+faststart`
  (plays on WhatsApp/phones directly). 10 s ≈ 0.6 MB at 640×480.
- The ReSpeaker (UAC1.0) has a device-release race (ALSA `-9985`); the recorder
  retries ×4 like `stt_node` does. Mic matcher: `VIDEO_MSG_MIC` (default
  "ReSpeaker" — the *sounddevice* name, not the ALSA card name `ArrayUAC10`).
- Hard cap 60 s per message; `max_mb` guard before sending.

## The delivery gate

```bash
python3 deliver.py status     # gate CLOSED/OPEN, network, queue
python3 deliver.py send       # refuses while "enabled": false  ← default
```

To open the gate, the ADMIN edits `~/video_messages/config.json`:

**WhatsApp (recommended, official route)** — requires a (free) Meta developer
app with the WhatsApp Cloud API product: fill `access_token`,
`phone_number_id`, `to` (international format), set `"transport": "whatsapp"`,
`"enabled": true`. We deliberately do **not** use unofficial WhatsApp bridges —
they violate WhatsApp's ToS and risk a permanent number ban.

**Email fallback** — fill `user`, `app_password` (an app-specific password,
e.g. Gmail App Password — never the real account password), `to`; set
`"transport": "email"`.

Then:
```bash
python3 deliver.py send --dry-run   # shows what would go, sends nothing
python3 deliver.py send             # uploads + sends; moves files to sent/
```

Failures keep the file in the outbox (nothing is lost when WiFi drops); the
robot keeps recording regardless of gate state.

## Validated (2026-07-22, Jetson)
- 10 s live message recorded: h264+aac, 24.5 fps measured, audio peak −8 dB,
  0.6 MB, queued. Thumbnail + playback verified.
- Gate refusal verified: `send` with `enabled:false` → exit 1, nothing sent.
- `offer` published on `/robot_speech` (TTS tier will speak it at CP4).

## Security & privacy notes
- Credentials only in `config.json` (0600, outside repo, git-ignored).
- Recording requires a deliberate command — tie it to the face-auth session
  (`docs/skills/FACE_AUTH.md`) if you want only admins to open the gate:
  `python3 ~/face_auth/auth_gate.py check-session && python3 deliver.py send`.
- Consider announcing "recording now" (the countdown does this) and deleting
  `sent/` periodically — these are visitors' personal videos.

## TODO / next
- Voice-triggered flow: wake word → "yes" intent → auto-run `record` (hook
  into `stt_node`'s `/speech/text`).
- WhatsApp credentials not yet configured (needs the admin's Meta app).
- Auto-`send` timer (systemd/cron) once the school-WiFi deployment happens at CP4.
