# ADR 0001 — Time-multiplex the GPU between NAVIGATING and CONVERSING states

- **Status:** Implemented — `gpu_arbiter` (YOLO kill/respawn) + `pause_inference` + `admin_node` `/aisha/mode` reaction + **"Hey Aisha stop" wake word**, all wired into `cerebro_aisha.launch.py` (`enable_gpu_arbiter`, default on) and verified (2026-06-02). Accelerated answer works (CONVERSING ~6.5 s first / ~2 s warm vs ~16–22 s CPU). Remaining polish: GPU-load fallback; optional sleep-phrase; ~10 s respawn time. See Validation.
- **Date:** 2026-06-02
- **Deciders:** AI-SHA team
- **Hardware:** Jetson Orin Nano 8GB (unified CPU+GPU memory), JetPack 6 / L4T R36.4.7, ROS 2 Humble
- **Related:** `cerebro_aisha.launch.py`, `jetson_launch.py`, `admin_node.py`, `yolov8_ros`

## Context

AI-SHA answers administrative questions via a local RAG stack (`admin_node` → Ollama
`llama3.2:1b`). On the 8GB Orin Nano, GPU and CPU **share one physical memory pool**.
During normal operation the GPU is occupied by:

- the **YOLOv8m TensorRT engine** (vision, ~85% GPU while inferring), and
- the **gemma3:270m** router (~0.84 GB resident in Ollama).

Because that VRAM is taken, `llama3.2:1b` is forced to run **CPU-only** (`num_gpu=0`)
to avoid `cudaMalloc` OOM → `llama-server` segfaults. The user-facing cost is **~16 s
per answer**, which feels broken to someone standing in front of the robot.

### Measured data (2026-06-02, direct Ollama API sweeps on the live board)

| Condition | `num_gpu` | Result | Latency |
|---|---|---|---|
| Full stack up (YOLO@85% + gemma3 resident) | 0 (CPU) | OK | **16.2 s** |
| Full stack up, steady state | ≤12 | OK (fragile) | 12.3 s |
| Full stack up, steady state | ≥16 | **OOM → segfault** | — |
| Full stack up, **memory spike** (startup / KV growth / router co-load) | even 4 | **OOM** | — |
| **Clean GPU** (YOLO unloaded, nothing resident) | 99 (full) | **OK** | gen 1.1–1.4 s @ 35–38 tok/s |
| Clean GPU, multi-turn (model kept resident) | 99 | OK | turn 1 **6.5 s** (incl 4.4 s load), warm turns **~5–6 s** |

**Key finding:** under live vision, *no nonzero offload is crash-safe* — but on a **clean
GPU, llama3.2:1b runs fully on GPU at ~5× the generation speed.** The vision VRAM is the
only thing in the way. (An earlier note claiming "num_gpu=99 OOMs even vision-off" was a
transient-residency artifact; a truly clean GPU offloads fully.)

## Validation (2026-06-02, `scripts/gpu_release_probe.py` + follow-up)

A standalone probe loaded the real `yolov8m.engine` and measured system
`MemAvailable` (the cross-process number that governs `cudaMalloc`):

| Step | Observation |
|---|---|
| Idle engine static footprint | **~700–820 MB** |
| llama `num_gpu=99` **with idle engine resident** | **OK, ~28 tok/s** — the idle engine is NOT a blocker |
| **In-process release** (`del model` + `gc` + `torch.cuda.empty_cache()`) | reclaimed **~0 MB** (−20 of 702); `torch.cuda.memory_allocated` 8→0 MB but system RAM unchanged |
| Engine resident (paused) + **3-turn** llama `num_gpu=99` | turn 1 **5.7 s** (incl 4.5 s load), warm turns **1.7–2.9 s @ 36–38 tok/s** |

**Probe conclusions:**

1. **In-process "release" does NOT free VRAM.** `del`+`empty_cache` returns only
   torch-tracked memory; the CUDA primary context + TensorRT runtime retain the rest
   until the **process exits**. A passive lifecycle deactivate / `del` is therefore
   *not* a way to reclaim the GPU. (Process kill does work: full-stack teardown dropped
   RAM 3.4 GB → 1.7 GB.)

2. *(Initially believed)* the idle engine coexists with GPU llama, so pausing alone would
   suffice. **The full-stack integration test below disproved this** — see correction.

### Full-stack integration test (2026-06-02, `gpu_arbiter` + live camera)

Ran the real `cerebro_aisha` stack (RealSense + YOLO @ 22 Hz + brain + admin) and drove
the new `gpu_arbiter`:

| Action | Result |
|---|---|
| NAVIGATING baseline | detection 22 Hz; llama `num_gpu=99` → **OOM** (expected) |
| `set_conversing true` → arbiter pauses YOLO + unloads gemma3 | mode=CONVERSING; **detection stops ✅**; gemma3 unloaded ✅ |
| llama `num_gpu=99` **while paused** | **STILL OOM ❌** — only ~1.25 GB free |
| Then **kill `yolov8_node`** (frees engine + CUDA ctx, ~920 MB) | llama `num_gpu=99` → **OK, 5.6 s @ 35 tok/s ✅** |
| `set_conversing false` | detection resumes at 22 Hz ✅ |

**CORRECTION — pausing is NOT sufficient under the full stack.** The isolated probe had
~4.8 GB free; the real stack does not. With YOLO paused, the resident processes still hold:
`yolov8_node` ~1.2 GB (engine + CUDA context, **not** reclaimed by pausing — see conclusion 1),
`admin_node` ~0.8 GB (ChromaDB + bge embeddings), plus RealSense/brain/OS — leaving only
~1.25 GB, too little for llama's GPU buffers. **You must actually free the YOLO engine's
GPU reservation, which means killing/respawning `yolov8_node`** (pausing only drains
*transient* inference buffers). Killing it confirmed the fix (920 MB freed → llama on GPU).

### Respawn-cycle test (2026-06-02, `gpu_arbiter` supervising `yolov8_node`)

The arbiter now spawns/kills `yolov8_node` as a managed child (admin_node up for
realistic ~0.8 GB pressure):

| Phase | YOLO proc | mode | RAM free | llama `num_gpu=99` |
|---|---|---|---|---|
| NAVIGATING (arbiter spawned YOLO) | pid A | NAVIGATING | 1.34 GB | OOM |
| `set_conversing true` → pause + **kill** YOLO | gone | CONVERSING | **2.09 GB** | **OK** ✅ |
| `set_conversing false` → **respawn** YOLO | pid B (new) | NAVIGATING | 1.35 GB | — |

Respawn-to-ready ≈ **10.6 s** with `ros2 run` defaults (also loads plant-disease +
mediapipe); expect faster with cerebro's params (faces/gestures off) via the `yolo_cmd`
param. Motion stays locked (`_vision_ready=False`) for the entire window vision is down,
cleared only once the respawned node's pause service reappears. Graceful arbiter stop
reaps the child (no orphan).

### admin_node `/aisha/mode` reaction test (2026-06-02)

`admin_node` now subscribes to the latched `/aisha/mode` and rebuilds its Ollama client
(num_gpu + keep_alive) **between queries** (worker thread, never mid-stream), so an
in-flight answer is never disrupted. Verified with `llm_model:=llama3.2:1b`, GPU free:

| Mode published | LLM rebuilt | Query latency | Answer |
|---|---|---|---|
| CONVERSING | num_gpu=99, keep_alive=120s | **~6.5 s** (incl GPU load) | valid (43 ch) ✅ |
| NAVIGATING | num_gpu=0, keep_alive=30s | ~21.8 s (CPU) | valid (96 ch) ✅ |

Params: `conversing_num_gpu`/`conversing_keep_alive` (99/120s),
`navigating_num_gpu`/`navigating_keep_alive` (0/30s). Defaults preserve the env-based
behaviour when no arbiter publishes. **Note:** the GPU path requires `llm_model=llama3.2:1b`
— the 3B `llama3.2` OOMs at num_gpu=99 even on a free GPU on this 8 GB board.

## Decision

Introduce a **GPU arbiter** that time-multiplexes the GPU between two mutually exclusive
states, since the robot does not need to navigate and converse at the same time:

### State 1 — NAVIGATING
- **GPU owner:** RealSense + YOLOv8 inference (active), Nav/Action node.
- **LLM:** CPU-only or unloaded (`keep_alive=0`).
- **Behavior:** roams, avoids obstacles, listens only for a wake word / button.

### State 2 — CONVERSING
- **Trigger:** wake word or physical button.
- **Transition (enter):**
  1. Publish **zero-velocity** and latch a "motion-inhibited" flag (hard safety invariant).
  2. **Pause YOLO inference** (`pause_inference` service) to drain in-flight GPU work
     cleanly before teardown.
  3. **Kill `yolov8_node`** to actually release its ~1.2 GB (engine + CUDA context) —
     pausing alone does NOT free this (integration test). **Keep the RealSense node alive.**
  4. **Unload gemma3** (`keep_alive=0`) — the heuristic router covers routing meanwhile.
  5. Route the admin query with `num_gpu=99`, `keep_alive=<conversation window>`.
- **Behavior:** stationary, answers admin queries at **~1.7–2.9 s warm** (first turn ~5.7 s
  incl. one-time llama load) instead of ~16 s.
- **Transition (exit):** llama `keep_alive=0`, **respawn `yolov8_node`** (engine reload from
  cache ~3–5 s; camera was never torn down), clear motion-inhibit, resume navigation.

**Scope decision: build a tiny GPU arbiter, NOT a lifecycle-node rebuild.** The arbiter
pauses inference, **respawns `yolov8_node`** to reclaim its GPU, manages ollama
`num_gpu`/`keep_alive`, and gates `/cmd_vel`. The camera node is never touched.

## Design constraints (validated on the live board)

Ignoring these yields a scheme that stalls vision without reclaiming memory.

1. **You must KILL/RESPAWN `yolov8_node` to free its GPU — pausing is not enough under the
   full stack.** Two proven facts combine: (a) in-process release (`del`+`empty_cache`)
   reclaims ~0 MB (CUDA/TRT context persists until the process exits); (b) with YOLO merely
   paused, the full stack leaves only ~1.25 GB free — llama `num_gpu=99` still OOMs.
   Killing `yolov8_node` freed ~920 MB and llama then ran on GPU (5.6 s). Pause first (drain
   in-flight work) **then** kill. The isolated probe's "pause is enough" was an artifact of
   having ~4.8 GB free with nothing else loaded.

2. **Llama is not a ROS node.** Its VRAM is owned by `ollama serve` and controlled by two
   request fields only: `num_gpu` and `keep_alive`. "Load Llama onto GPU" = send a request
   with `num_gpu=99, keep_alive=<window>`; "unload" = `keep_alive=0`. **Do not** wrap Llama
   in a lifecycle node — the arbiter logic belongs on the **vision/router side only**.

3. **Keep RealSense and YOLO as separate processes regardless.** RealSense re-init cost
   10–30 s in tests (USB re-enumeration). The pause-not-kill design never touches the
   camera; but if you ever hit the Constraint-1 fallback and must respawn YOLO, respawn
   only that node so the camera keeps streaming.

4. **Drain inference before promoting llama to GPU.** "Pause" must stop *new* `predict`
   calls and let in-flight GPU work finish, else a transient YOLO buffer can still collide
   with llama's load. An `inference_enabled` flag at the top of the image callback plus a
   short settle delay suffices.

## Implementation sketch

```
                         ┌─────────────────────────┐
   wake word / button ─► │   gpu_arbiter            │  [BUILT]
                         │   state machine          │
                         └───────────┬─────────────┘
            zero-vel /cmd_vel  ◄──────┤ enter CONVERSING
            pause_inference(srv)◄──────┤   1. drain in-flight YOLO GPU work   [BUILT]
            kill yolov8_node    ◄──────┤   2. free ~1.2GB (camera stays up)   [BUILT]
              gemma3 keep_alive=0      │   3. unload router                   [BUILT]
              admin num_gpu=99         │   4. llama on GPU                     [partial]
            respawn yolov8_node ◄──────┤ exit CONVERSING (engine reload ~10s)  [BUILT]
            clear motion-inhibit ◄─────┘   (only after vision ready)          [BUILT]
```

- **`gpu_arbiter` node [BUILT]:** owns NAVIGATING/CONVERSING; publishes `/aisha/mode`;
  gates `/cmd_vel` (drops motion + republishes zero at 10 Hz whenever vision isn't ready);
  **supervises `yolov8_node` as a managed child** (`manage_yolo=True`): pauses then kills it
  on CONVERSING, respawns + waits-for-ready on NAVIGATING; unloads gemma3/llama; auto-returns
  after `conversation_timeout_s`. Slow ops run in a worker thread so the executor stays live.
- **`yolov8_node` `pause_inference` service [BUILT]:** `std_srvs/SetBool` flips an
  `inference_enabled` flag gating `process_loop`; publishes `/detection/inference_active`.
  Used as the clean-drain step before the kill.
- **Launch [BUILT]:** `cerebro_aisha.launch.py` has `enable_gpu_arbiter` (default true): an
  `OpaqueFunction` launches `gpu_arbiter` with a `yolo_cmd` reproducing the cerebro YOLO
  params (and does NOT start the standalone `yolov8_node`); set false to fall back to
  always-on vision. Mutually exclusive — never both, to avoid two `detection_node`s.
- **`admin_node` change:** make `num_gpu` and `keep_alive` settable at request time
  (driven by `/aisha/mode`): CONVERSING → `num_gpu=99, keep_alive=120s`; NAVIGATING →
  `num_gpu=0, keep_alive=0`. The `num_gpu` plumbing already exists via the `llm_num_gpu`
  launch arg / `OLLAMA_NUM_GPU` env.
- **Router:** on entering CONVERSING, `keep_alive=0` gemma3 so it isn't co-resident; the
  smoke test showed the heuristic (`?`-detection) router carries routing fine without it.
- **Safety:** `gpu_arbiter` holds the motion-inhibit latch for the *entire* CONVERSING
  window — the robot is blind (inference paused), so it must be provably stationary.

## Consequences

**Positive**
- ~16 s → **~1.7–2.9 s warm** per answer (first turn ~5.7 s incl. load). Biggest UX win available on this hardware.
- No new models, no quantization, no extra hardware. RealSense never torn down (camera stays warm).
- Clear safety story (stationary + motion-inhibited while blind).

**Negative / risks**
- Requires **killing + respawning `yolov8_node`** each transition (pausing alone doesn't free
  the GPU under the full stack). Resume costs a ~3–5 s engine reload — robot is blind/stalled
  that long after each chat (acceptable: already stationary). Adds respawn failure modes
  (engine reload timeout → robot blind; needs a watchdog).
- llama may fail to load on GPU → fall back to `num_gpu=0` (CPU) mid-conversation (needs timeout).
- Robot is blind for the whole conversation — must be provably stationary.
- Wake-word detector must run cheaply in NAVIGATING without the GPU.
- *Mitigation worth a spike:* shrink the ~0.8 GB `admin_node` footprint (lighter embedding
  model / external Chroma). If the paused-stack baseline dropped ~0.8–1 GB, **pause-only might
  then suffice** — eliminating the respawn entirely. Re-measure if admin is trimmed.

## Alternatives considered

1. **Status quo (CPU-only, always-on vision).** Simplest, crash-safe, but ~16 s answers. Rejected: UX.
2. **Persistent partial offload (`num_gpu=12`) with vision running.** Gives ~24% (16→12 s)
   but is **not crash-safe** — OOM-segfaults on memory spikes. Rejected: unreliable for live demos.
3. **Smaller / more-quantized LLM (e.g. qwen2.5:0.5b, or llama3.2:1b q3).** Could cut CPU
   latency without GPU juggling. Worth a separate spike; orthogonal to this ADR and could
   stack with it.
4. **Stream the first sentence to TTS as it generates.** Cuts *perceived* latency cheaply,
   no architecture change. Recommended regardless of this ADR.

## Open questions / next steps

- [x] **Prototype the make-or-break VRAM question** (`scripts/gpu_release_probe.py`, 2026-06-02):
      in-process release reclaims ~0 MB; CUDA/TRT context persists until process exit.
- [x] **Build `gpu_arbiter` + `pause_inference` service** (2026-06-02): both work — pause/resume,
      gemma3 unload, `/cmd_vel` gating, `/aisha/mode` all verified live.
- [x] **Verify under real camera load:** done — revealed **pause alone is insufficient**
      (only ~1.25 GB free); killing `yolov8_node` freed 920 MB → llama on GPU 5.6 s.
- [x] **Extend `gpu_arbiter` to respawn `yolov8_node`** (2026-06-02): kill on CONVERSING,
      respawn + wait-for-ready on NAVIGATING, motion locked until vision back, child reaped on
      exit, ~10 s respawn. Full cycle verified (table above). `manage_yolo`/`yolo_cmd` params.
- [x] **Wire `admin_node` to `/aisha/mode`** (2026-06-02): rebuilds Ollama client between
      queries — CONVERSING→num_gpu=99/keep_alive=120s (~6.5 s), NAVIGATING→num_gpu=0 (~22 s
      CPU). Both verified. End-to-end accelerated answer now works. Requires `llm_model=llama3.2:1b`.
- [x] **Update `cerebro_aisha.launch.py`** (2026-06-02): added `enable_gpu_arbiter` arg
      (default true) — launches `gpu_arbiter` (owning `yolov8_node` via `yolo_cmd` with the
      cerebro params) instead of the standalone YOLO; set false to fall back. Verified end to
      end from one `ros2 launch`: 5 nodes incl. exactly ONE `detection_node`; CONVERSING kills
      YOLO + admin→num_gpu=99; NAVIGATING respawns YOLO + admin→num_gpu=0.
- [ ] Shrink respawn time: it is **~10 s even with faces/gestures/ocr off** — the TensorRT
      engine deserialize + warmup dominates, not the detectors. To cut it, investigate keeping
      a pre-warmed engine or a faster-loading model; disabling detectors alone does NOT help.
- [ ] Spike: trim `admin_node` ~0.8 GB footprint — if baseline drops enough, pause-only may
      suffice and the respawn can be dropped.
- [x] **Wake-word trigger** (2026-06-02): `gpu_arbiter` watches `/speech/text` (Whisper STT,
      already running) for **"Hey Aisha stop"** (tolerant regex: an Aisha-name variant + "stop";
      a bare "stop" stays brain_node's emergency halt) → enters CONVERSING. Verified: non-wake
      speech ignored, wake phrase switches the mode. Params `wake_enabled`/`speech_topic`. Reuses
      STT, so no new wake-word engine/dependency; upgrade to openWakeWord/Porcupine later if a
      lower-latency, always-on (STT-independent) detector is wanted.
- [ ] Define fallback: llama GPU-load failure → `num_gpu=0` (CPU) mid-conversation, with timeout.
- [ ] Optional: a "sleep phrase" (e.g. "Hey Aisha go") to end CONVERSING early instead of waiting
      for `conversation_timeout_s`.
