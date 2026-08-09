# Skill: Face + PIN Admin Authentication (with anti-spoofing)

**Board:** Jetson Orin Nano (JetPack 6.2.2) · **Sensor:** Intel RealSense D435 (color + aligned depth)
**Adds no new pip packages** — reuses the CP5/CP6 stack (onnxruntime, mediapipe Tasks API,
cv_bridge, rclpy). The pinned ML stack is untouched.

This skill lets AI-SHA recognise an enrolled administrator by face and grant a short-lived,
authenticated session that gates access to protected data/actions. It is deliberately
**multi-factor** because face recognition alone is spoofable:

1. **Identity** — ArcFace embedding match against an enrolled admin's face gallery.
2. **Passive liveness** — the RealSense *depth* frame must look like a real 3-D face,
   not a flat photo (planarity + depth-spread test).
3. **Active liveness** — a random head-turn challenge the user must perform live.
4. **PIN** — a salted PBKDF2-SHA256 secret (never stored in plaintext).

Only when **all four** pass is a 15-minute session token issued.

> ⚠️ **Security honesty:** this is a solid *defence-in-depth demo*, not a certified
> biometric access-control product. Depth liveness defeats printed/screen photos; it does
> **not** claim to defeat 3-D masks or video-on-a-curved-surface attacks. The PIN is the
> factor that must not be bypassed — treat face+liveness as convenience + spoof-resistance,
> and the PIN as the real secret.

---

## Files

| File | Role |
|---|---|
| `tools/face_auth/face_auth.py` | Core: `enroll` a face gallery, `verify` identity + liveness. Classes `Embedder` (ArcFace/onnxruntime), `Detector` (mediapipe FaceDetector + 3-pt alignment), `depth_liveness()`. |
| `tools/face_auth/auth_gate.py` | The gate: `set-pin`, `authenticate` (3-phase), `check-session`, `logout`. Admin registry + session token. |
| `tools/face_auth/fetch_face_models.sh` | Downloads the ArcFace recogniser (InsightFace `buffalo_s` → `w600k_mbf.onnx`). |
| `tools/face_auth/liveness_diag.py` | Calibration: prints the passive-liveness distribution (range/residual) for the current face. |
| `tools/face_auth/yaw_diag.py` | Calibration: prints the head-turn (yaw-proxy) swing range. |

**Biometric data + models live OUTSIDE the repo** (default `~/face_auth/`, override with
`FACE_AUTH_HOME`) and are git-ignored. Never commit a gallery, `admins.json`, `session.json`,
or the `.onnx`/`.zip` models.

---

## One-time setup

```bash
# 1. Face DETECTOR model (shared with CP5/CP6 vision) — if not already present:
bash ~/robot_ws/scripts/fetch_mediapipe_task_models.sh

# 2. Face RECOGNISER model (ArcFace):
bash ~/robot_ws/tools/face_auth/fetch_face_models.sh   # -> ~/face_auth/models/w600k_mbf.onnx
```

Bring up the camera (color + **aligned depth** is required for liveness):

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=99
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true &
```

---

## Enrol an administrator ("Sam")

```bash
cd ~/robot_ws/tools/face_auth
source /opt/ros/humble/setup.bash && source ~/robot_ws/install/setup.bash
export ROS_DOMAIN_ID=99

# capture ~25 good frames; turn head slightly between shots, stay well-lit
python3 face_auth.py enroll --name Sam --count 25 --timeout 120
# -> writes ~/face_auth/gallery/sam.npz  (512-d embeddings; git-ignored)

# set the admin's PIN (stored as salted PBKDF2, never plaintext)
python3 auth_gate.py set-pin --name Sam --pin 4729
```

## Authenticate (the gate)

```bash
python3 auth_gate.py authenticate
#   Phase 1/3 IDENTITY  : look at the camera, hold still (~6 s)
#   Phase 2/3 CHALLENGE : turn your head fully LEFT<->RIGHT when prompted
#   Phase 3/3 PIN       : type the PIN (or pass --pin for automation/testing)
# -> "ACCESS GRANTED to admin 'Sam'"  + a 15-minute session token
```

Downstream tools check the session instead of re-authenticating every call:

```bash
python3 auth_gate.py check-session   # exit 0 = valid, 1 = none/expired
python3 auth_gate.py logout          # clears the session
```

---

## How the anti-spoofing works

**Passive (depth) liveness** — `depth_liveness()` samples the central face ROI of the
aligned-depth frame and computes two things:
- **depth spread** (90th–10th percentile) — a real face spans ~20–50 mm; a flat photo ~0.
- **planarity residual** — RMS deviation from a best-fit plane `z = ax+by+c`. A convex face
  gives a large residual (~6–11 mm here); a photo is planar even when tilted, so its
  residual collapses. This is what defeats the *tilted-photo* bypass.

A frame is classed **live**, **flat** (valid depth but planar = spoof evidence), or
**nodata** (missing depth from motion/edge — carries *no* spoof signal). The gate rejects
the attempt only when the **flat fraction among depth-valid frames** exceeds 30 %. Motion
dropout is skipped, so a moving *live* user isn't falsely rejected while a photo (consistently
flat) is.

**Active liveness** — `active_challenge()` prompts a random LEFT/RIGHT turn and measures a
**yaw proxy**: signed nose offset relative to the eye-midpoint, normalised by inter-eye
distance. Only a real 3-D head rotation shifts this (a rigid photo — even slid or rotated —
stays ~0). Requires a peak-to-peak swing ≥ 0.20.

---

## Validated behaviour (Jetson, RealSense D435, 2026-07-22)

| Scenario | Result |
|---|---|
| Live admin + real head-turn + correct PIN | ✅ ACCESS GRANTED + 15-min session |
| Live admin, no/small head-turn | ❌ DENIED — active liveness (swing < 0.20) |
| Live admin, wrong PIN | ❌ DENIED — PIN mismatch |
| **Photo of admin + correct PIN** | ❌ DENIED — passive liveness (45 % frames flat) |
| Session after grant / after 15 min / after logout | ✅ VALID / EXPIRED / cleared |

Measured margins (hold still, frontal): real face **0 %** flat frames, depth residual
6.4–11.4 mm, spread 22–50 mm. Photo: **45–54 %** flat frames. Clean separation around the
30 % threshold.

## Tuning (use the diag tools before changing thresholds)

- `python3 liveness_diag.py` → set `LIVENESS_MIN_RESIDMM` / `LIVENESS_MIN_RANGE_MM` in
  `face_auth.py` **below** your real-face p05, **above** a photo's values.
- `python3 yaw_diag.py` → set `CHALLENGE_YAW_RANGE` in `auth_gate.py` between a rigid
  photo (~0) and a comfortable real turn (~0.3+).
- `authenticate --threshold` (default 0.45) is the cosine match floor. **Calibrate against a
  real impostor before deployment**: an enrolled admin scores ~0.55–0.79 here; run a
  different person through `face_auth.py verify` and set the threshold above their best score.

## Known limitations / TODO
- Match threshold not yet calibrated against a live different person (needs a 2nd subject).
- No protection against 3-D masks or a phone *video* on a curved surface.
- PIN entry is terminal-based; a deployed kiosk should use a shielded keypad.
