#!/usr/bin/env python3
"""AI-SHA web console — watch the camera and talk to the robot from a browser.

Runs ON THE JETSON. Serves one page (default http://192.168.55.1:8088) that shows
the live camera and lets you type a question and read AI-SHA's answer. Needs NO ROS
on the machine with the browser, and NO new Python packages on the Jetson.

  Camera  : /detection/image_annotated  (YOLO boxes) -> falls back to raw colour
  Ask     : publishes your text on /speech/text  (same topic the mic would feed)
  Answer  : shows whatever the robot publishes on /robot_speech
"""
import json, os, queue, re, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request

VOICE = os.environ.get("AISHA_VOICE") == "1"     # set by the voice-mode launcher

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

PORT        = 8088
# The LiDAR is on the Pi 5, not here. It is read over plain HTTP rather than ROS
# because Jazzy->Humble DDS does not deliver to real subscriptions (Iron+ type
# hashes), so a /scan subscriber on this board would silently receive nothing.
LIDAR_URL   = os.environ.get("AISHA_LIDAR_URL", "http://192.168.0.117:8090/scan.json")
# slam_toolbox's occupancy map, bridged off the Pi the same way and for the same
# reason. Base URL: /map.png is the picture, /map.json the metadata.
MAP_URL     = os.environ.get("AISHA_MAP_URL", "http://192.168.0.117:8091")
# Speaker volume also lives on the Pi, and for the same HTTP-not-ROS reason. The
# control it drives is a softvol plugin, not a hardware mixer: the hifiberry-dac
# driver the MAX98357A runs under exposes no hardware volume control at all.
VOLUME_URL  = os.environ.get("AISHA_VOLUME_URL", "http://192.168.0.117:8092")
# The robot's VOICE is on the Pi. Answers go there as an HTTP POST rather than on
# a ROS topic for the usual reason: this board is Humble, the Pi is Jazzy, and
# cross-distro DDS discovers endpoints but never delivers to a real subscription,
# so a /tts_text publisher here would be silently ignored. speech_bridge.py on the
# Pi receives the POST and republishes it on /tts_text locally.
SPEECH_URL  = os.environ.get("AISHA_SPEECH_URL", "http://192.168.0.117:8093/say")
# Same bridge, reporting whether the speaker is currently talking. Polled so the
# mic can be held muted for exactly the length of an utterance.
SPEAKING_URL = os.environ.get("AISHA_SPEAKING_URL", "http://192.168.0.117:8093/speaking")
ANNOTATED   = "/detection/image_annotated"          # camera + detections
RAW_COLOR   = "/camera/camera/color/image_raw"       # fallback if vision is off
ASK_TOPIC   = "/speech/text"                          # question in
VIDEO_HOME  = os.path.expanduser("~/video_messages")  # video-message skill lives here
HRMS_HOME   = os.path.expanduser("~/hrms_query")       # HRMS skill config lives here
HRMS_TOOL   = os.path.expanduser("~/robot_ws/tools/hrms_query/hrms_query.py")
SR_HOME     = os.path.expanduser("~/student_report_query")
SR_TOOL     = os.path.expanduser("~/robot_ws/tools/student_report_query/student_report_query.py")
TT_HOME     = os.path.expanduser("~/timetable_query")
TT_TOOL     = os.path.expanduser("~/robot_ws/tools/timetable_query/timetable_query.py")
FACE_HOME   = os.path.expanduser("~/face_auth")        # face-auth gate lives here
# Camera window for a scan. 8 s was too short: auth_gate loads the ArcFace model
# BEFORE creating its ROS node, so model load + DDS discovery ate most of the
# window and it collected <3 usable frames -> "I couldn't see a face" while the
# user sat perfectly framed. 20 s leaves a real sampling window.
AUTH_SECS   = 20
# Camera mounting rotation for the face gate: "0" now that the camera has been
# re-aimed upright. It was 180 while the D435 was mounted inverted. Keep this in
# step with the PHYSICAL mounting — a stale 180 flips an already-upright face and
# the gate fails as "not recognized", which reads like a recognition fault.
CAM_ROTATE  = os.environ.get("AISHA_CAM_ROTATE", "0")
# Experimental/demo default: face + passive anti-spoof only — no head-turn, no
# PIN. Set AISHA_AUTH_RELAXED=0 to restore the full four-factor gate.
AUTH_RELAXED = os.environ.get("AISHA_AUTH_RELAXED", "1") == "1"
VIDEO_SECS  = 15                                       # length of a recorded message

# Spoken/typed phrases that trigger the video-message skill instead of the LLM.
VIDEO_INTENT = re.compile(
    r"(record|leave|send|take).{0,20}(video|vedio)\s*message"
    r"|video\s*message.{0,20}(to|for)\s+(sam|admin|hsdaou)", re.I)

# Student reports name a minor and carry academic and behaviour records, so the
# report is EMAILED to an authenticated administrator and never spoken or shown.
#
# This pattern is a verbatim copy of the one in brain_node._classify. brain_node
# routes the utterance to SKILL_STUDENT_REPORT and then deliberately stays
# silent, expecting the skill layer to answer. Until now the console had no
# student-report handler at all, so nothing ran and the robot simply never
# replied - the request vanished between the two. Keep the two patterns
# identical: if brain_node routes something the console cannot match, that
# request goes silent again.
STUDENT_REPORT_INTENT = re.compile(
    r"\b(student|pupil)\s+(report|marks?|grades?|results?)\b"
    r"|\b(report|marks?|grades?|results?)\s+(for|of)\s+(student|pupil)\b", re.I)

# HRMS staff-leave questions. Broad on purpose: hrms_query.py does the precise
# routing and refuses anything it cannot map, so a false positive here costs a
# clear refusal rather than a wrong answer. The ANSWER IS NEVER SPOKEN OR SHOWN -
# the HRMS emails it. AI-SHA stands in a public corridor; reading out who is on
# sick leave would disclose it to every passer-by.
HRMS_INTENT = re.compile(
    r"sick leave|on leave|leave balance|annual leave"
    r"|who('?s| is)\s+(off|absent|away|out)"
    r"|days\s+(left|remaining)|how many days", re.I)

# School timetable questions. Unlike the HRMS skill these are NOT all private:
# a class timetable and a COUNT of free students identify nobody, so the robot
# says those out loud. A list of NAMED students is emailed instead - they are
# minors. timetable_query.py enforces that split and requires an admin session
# only for the naming answers.
# The console uses the SAME classifier as brain_node and the trigger. The regex
# below survives only as a fallback if the import fails, so a broken deploy
# degrades to the old behaviour rather than routing nothing at all.
try:
    import sys as _sys
    _sys.path.insert(0, os.path.expanduser(
        "~/robot_ws/install/aisha_brain/lib/python3.10/site-packages/aisha_brain"))
    import skill_intents as _si
except Exception:                                            # pragma: no cover
    _si = None

TIMETABLE_INTENT = re.compile(
    r"time\s*table|timetable"
    r"|(which|who|how many).{0,24}(students?|teachers?).{0,16}free"
    r"|free\s+(students?|teachers?)"
    # "available" is how the question is actually asked; without it the request
    # went to the knowledge base and came back invented.
    r"|(which|who|how many).{0,24}(students?|teachers?).{0,16}available"
    r"|available\s+(students?|teachers?)"
    r"|(students?|teachers?)\s+(are\s+)?available"
    r"|lessons? for grade|schedule for grade", re.I)

# "Authenticate me." Without this the request fell through to the knowledge base,
# which answered "I am an administrative assistant, please call the office" — the
# robot could ASK for authentication but had no way to START it. This runs the
# same face + liveness + head-turn + PIN gate that was previously CLI-only.
AUTH_INTENT = re.compile(
    r"authenticate|log ?me ?in|sign ?me ?in|verify me|verify my|scan my face"
    r"|i('?m| am)\s+(an?\s+)?admin|log in as|it'?s me", re.I)
REPLY_TOPIC = "/robot_speech"                         # answer out


def _is_timetable(text: str, context=None) -> bool:
    if _si is not None:
        return _si.is_skill(text, context)
    return bool(TIMETABLE_INTENT.search(text or ""))


class Hub(Node):
    def __init__(self):
        super().__init__("aisha_watch")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self._jpeg = None
        self._jpeg_src = "waiting"
        self._annot_t = 0.0
        self._vm_busy = False
        self._hrms_busy = False
        self._tt_busy = False
        self._last_skill = None   # last answered timetable query, for follow-ups
        self._auth_busy = False
        self._awaiting_pin = False      # next TYPED line is a PIN, not a question
        self._pin_deadline = 0.0
        self._pending_admin = None      # (kind, utterance) to replay after auth
        self.log = []            # list of dicts: {t, who, text}
        img_q = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, ANNOTATED, self._annot, img_q)
        self.create_subscription(Image, RAW_COLOR, self._raw, img_q)
        self.create_subscription(String, REPLY_TOPIC, self._reply, 10)
        self.create_subscription(String, ASK_TOPIC, self._heard, 10)
        self.pub = self.create_publisher(String, ASK_TOPIC, 10)

        # ── Give the LLM the GPU ────────────────────────────────────────────
        # admin_node defaults to mode NAVIGATING, which means num_gpu=0 — the
        # LLM then runs entirely on the CPU (30-80 s for a 1B model). Normally
        # gpu_arbiter publishes the mode, but console mode runs with the
        # arbiter (and YOLO) disabled, so nobody ever does. Since no vision node
        # is competing for the GPU here, we claim CONVERSING ourselves — this is
        # the single biggest answer-latency win.
        mode_q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL,
                            history=HistoryPolicy.KEEP_LAST)
        self.mode_pub = self.create_publisher(String, "/aisha/mode", mode_q)
        self._claim_gpu()
        self.create_timer(30.0, self._claim_gpu)   # re-assert for late joiners

        # ── Mic mute ────────────────────────────────────────────────────────
        # stt_node already mutes itself while the speaker is playing, via
        # /speaker/playing. We reuse that: it is the only mute path that exists,
        # and it avoids killing stt_node (which would reload Whisper on every
        # toggle, ~10 s). Caveat: stt_node has a 90 s stuck-mute WATCHDOG that
        # force-unmutes, so a one-shot True would silently wear off — we
        # re-assert every 20 s for as long as the mute is held.
        from std_msgs.msg import Bool
        self._Bool = Bool
        self.mic_muted = False
        self._speaking = False        # held while the Pi's speaker is talking
        self.mic_pub = self.create_publisher(Bool, "/speaker/playing", 10)
        self.create_timer(20.0, self._reassert_mic)

    def set_mic_mute(self, muted: bool):
        muted = bool(muted)
        changed = muted != self.mic_muted
        self.mic_muted = muted
        m = self._Bool(); m.data = muted
        self.mic_pub.publish(m)
        # Only announce a real change. Echoing every request filled the transcript
        # with "muted"/"on" pairs when the page re-sent the state.
        if changed:
            self._push("AI-SHA", "Microphone muted — I am not listening."
                       if muted else "Microphone on — I am listening again.")

    def speaker_gate(self, playing: bool):
        """Mute stt_node for as long as the Pi's speaker is talking.

        The Pi publishes /speaker/playing itself, but on Jazzy -- the Jetson's
        Humble subscription never receives it, so the mute has to be asserted
        here instead. Without this the robot hears its own answer and re-triggers.

        Touches ONLY the topic, never mic_muted or the console's Mic button:
        those belong to the operator. On release it republishes the operator's own
        mute state rather than a flat False, so someone who muted by hand stays
        muted once the robot stops talking.
        """
        self._speaking = bool(playing)
        m = self._Bool()
        m.data = self._speaking or self.mic_muted
        self.mic_pub.publish(m)

    def _reassert_mic(self):
        # Beat stt_node's 90 s stuck-mute watchdog, for a hand mute OR a long answer.
        if self.mic_muted or self._speaking:
            m = self._Bool(); m.data = True
            self.mic_pub.publish(m)

    def _claim_gpu(self):
        m = String(); m.data = "CONVERSING"
        self.mode_pub.publish(m)

    # ── Power off the whole robot ───────────────────────────────────────────
    def shutdown_all(self):
        threading.Thread(target=self._shutdown_worker, daemon=True).start()

    def _shutdown_worker(self):
        """Pi first (best-effort), then the Jetson last — the Jetson kills this
        console, so it must go after everything the console still needs to do.

        The Pi's SSH password lives ONLY in ~/robot_console/shutdown.json (0600,
        git-ignored), never in this source. The Jetson powers itself off through a
        NOPASSWD sudoers entry (see the deploy note), so no password is embedded
        here either.

        ⚠️ This DOES power off the Pi now. The note that used to sit here said the
        Jetson could not route to the Pi so the Pi step would harmlessly report
        'unreachable' — that stopped being true when CP4 put both boards on the
        house WiFi. On 2026-08-13 a press took the Pi down at the same instant as
        the Jetson, and the Pi's graceful poweroff was misread for days as the
        board crashing on a weak power supply. Both boards go down, and neither
        can be switched back on remotely."""
        import shlex
        cfg_path = os.path.expanduser("~/robot_console/shutdown.json")
        pi_msg = "Pi: not configured (skipped)"
        try:
            if os.path.exists(cfg_path):
                pi = json.load(open(cfg_path)).get("pi", {})
                host, user, pw = pi.get("host"), pi.get("user"), pi.get("password")
                if host and user and pw:
                    self._push("AI-SHA", f"Powering off the Pi ({user}@{host})…")
                    cmd = ("sshpass -p {pw} ssh -o StrictHostKeyChecking=no "
                           "-o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "
                           "-o PreferredAuthentications=password -o PubkeyAuthentication=no "
                           "{user}@{host} 'echo {pw} | sudo -S poweroff'").format(
                               pw=shlex.quote(pw), user=shlex.quote(user),
                               host=shlex.quote(host))
                    r = subprocess.run(cmd, shell=True, capture_output=True,
                                       text=True, timeout=30)
                    pi_msg = ("Pi: shutdown sent" if r.returncode == 0
                              else "Pi: unreachable — power it off manually")
        except Exception as e:                                    # never block the Jetson
            pi_msg = f"Pi: error ({type(e).__name__}) — power it off manually"

        self._push("AI-SHA", f"{pi_msg}. Powering off the Jetson now. Goodbye.")
        time.sleep(2)                        # let the message reach the browser
        # A successful poweroff never returns (the system goes down mid-call). So
        # if this call DOES return, the shutdown FAILED — almost always a missing
        # NOPASSWD sudoers rule (see SHUTDOWN_BUTTON.md). Report it instead of
        # leaving the browser on "Shutting down…" forever, which is what happened
        # when the sudoers file was silently empty.
        try:
            r = subprocess.run("sudo -n /usr/bin/systemctl poweroff", shell=True,
                               capture_output=True, text=True, timeout=20)
            self._push("AI-SHA", "SHUTDOWN FAILED — the Jetson could not power "
                       "itself off (sudo was denied). It is still running; power "
                       "it off physically. " + (r.stderr or "").strip()[:100])
        except subprocess.TimeoutExpired:
            pass                             # poweroff is in progress; expected

    def _encode(self, msg, src):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self.lock:
                self._jpeg = buf.tobytes(); self._jpeg_src = src

    def _annot(self, m):
        self._annot_t = time.time()
        self._encode(m, "annotated")

    def _raw(self, m):
        # only use raw if the annotated stream is stale (vision off / arbiter flipped)
        if time.time() - self._annot_t > 1.5:
            self._encode(m, "raw colour")

    def _reply(self, m):
        self._push("AI-SHA", m.data)

    def _heard(self, m):
        if "__selftest__" in m.data:      # boot pipeline probe; not a real question
            # NB: the probe routes to SKILL_VIDEO and produces NO /robot_speech
            # reply, so there is nothing to suppress. The old 120 s reply-swallow
            # window here was pure harm — it ate every real answer for two minutes
            # after any restart, which looked like "the console stopped responding".
            return
        if (m.data or "").strip() == "wake_word_triggered":
            # Internal STT control event. brain_node owns the spoken greeting;
            # never display this token as something the visitor said.
            return
        self._push("you", m.data)
        if VIDEO_INTENT.search(m.data or ""):
            self._start_video_message()
        elif AUTH_INTENT.search(m.data or ""):
            self._start_auth_prompt()      # PIN is then typed, never spoken
        elif HRMS_INTENT.search(m.data or ""):
            self._start_hrms_query(m.data)
        elif STUDENT_REPORT_INTENT.search(m.data or ""):
            self._start_student_report_query(m.data)
        elif _is_timetable(m.data or "", self._last_skill):
            self._start_timetable_query(m.data)

    # ── video-message skill ─────────────────────────────────────────────────
    def _start_video_message(self):
        """Record a video message and deliver it. Runs in a thread so the ROS
        executor and the HTTP server keep serving (the camera must stay live —
        the person needs to see themselves while recording)."""
        with self.lock:
            if self._vm_busy:
                return
            self._vm_busy = True
        threading.Thread(target=self._video_message_worker, daemon=True).start()

    # ── student report skill ────────────────────────────────────────────────
    def _start_student_report_query(self, utterance):
        with self.lock:
            if getattr(self, "_sr_busy", False):
                return
            self._sr_busy = True
        threading.Thread(target=self._student_report_worker,
                         args=(utterance,), daemon=True).start()

    def _student_report_worker(self, utterance):
        """Ask student_report_query.py to email one student's report.

        Nothing about the student comes back here - the app renders and sends it,
        and the robot receives only a confirmation. AI-SHA stands in a public
        corridor; a minor's marks must never reach its screen or its speaker.
        """
        try:
            r = subprocess.run(
                f'python3 -u "{SR_TOOL}" ask {json.dumps(utterance)}',
                shell=True, cwd=SR_HOME, capture_output=True, text=True, timeout=90)
            out = (r.stdout or "") + (r.stderr or "")
            spoken = [l[len("SPEAK: "):] for l in out.splitlines()
                      if l.startswith("SPEAK: ")]
            if r.returncode == 0 and spoken:
                self._push("AI-SHA", spoken[-1])
                return
            msg = {
                2: "Tell me the student's computer number - for example "
                   "\"send the student report for 18140\".",
                3: None,
                4: "I could not reach the student report service.",
            }.get(r.returncode)
            if msg is None:
                denied = [l for l in out.splitlines() if "DENIED" in l]
                if denied:
                    msg = denied[-1].split("- ", 1)[-1].strip()
                    if "authenticate" in msg.lower() or "administrator" in msg.lower():
                        # Queue it so the report is sent the moment the gate opens,
                        # instead of making the administrator re-type the number.
                        self._pending_admin = ("student_report", utterance)
                        msg += " Say \"authenticate me\" and I will then send it."
                else:
                    detail = [l for l in out.splitlines() if l.strip()]
                    msg = ("The student report could not be sent. "
                           + (detail[-1][:140] if detail else ""))
            self._push("AI-SHA", msg)
        except subprocess.TimeoutExpired:
            self._push("AI-SHA", "The student report service did not respond in time.")
        except Exception as e:
            self._push("AI-SHA",
                       f"Sorry - the student report request failed ({type(e).__name__}).")
        finally:
            with self.lock:
                self._sr_busy = False

    # ── HRMS staff-leave skill ──────────────────────────────────────────────
    def _start_hrms_query(self, utterance):
        with self.lock:
            if self._hrms_busy:
                return
            self._hrms_busy = True
        threading.Thread(target=self._hrms_worker, args=(utterance,), daemon=True).start()

    def _hrms_worker(self, utterance):
        """Hand the question to hrms_query.py, which authorises the admin, calls the
        HRMS and lets the HRMS render and email the report. Nothing about any
        employee comes back here - by design the robot only learns whether it
        worked, so staff leave data never reaches this device."""
        try:
            self._push("AI-SHA", "Checking with the HR system. The report will be "
                                 "emailed to the administrator — I will not read it out.")
            r = subprocess.run(
                f'python3 -u "{HRMS_TOOL}" ask {json.dumps(utterance)}',
                shell=True, cwd=HRMS_HOME, capture_output=True, text=True, timeout=120)
            code = r.returncode
            if code == 3:
                # Preserve the private question and replay it once the face/PIN
                # gate succeeds, so the administrator need not ask twice.
                self._pending_admin = ("hrms", utterance)
            msg = {
                0: "The report has been sent to the administrator's email.",
                2: "I could not tell which report you meant. Try: \"who is on sick "
                   "leave today\", or \"how many days does <name> have left\".",
                3: "An administrator must authenticate first — that report is "
                   "restricted.",
                4: "I could not reach the HR system.",
                6: "I could not find an employee by that name.",
                7: "Several staff match that name. Please be more specific.",
            }.get(code)
            if msg is None:
                detail = (r.stderr or r.stdout or "").strip().splitlines()
                msg = ("The HR system could not complete that request. "
                       + (detail[-1][:140] if detail else ""))
            self._push("AI-SHA", msg)
        except subprocess.TimeoutExpired:
            self._push("AI-SHA", "The HR system did not respond in time.")
        except Exception as e:
            self._push("AI-SHA", f"Sorry — the HR request failed ({type(e).__name__}).")
        finally:
            with self.lock:
                self._hrms_busy = False

    # ── School timetable skill ──────────────────────────────────────────────
    def _start_timetable_query(self, utterance):
        with self.lock:
            if self._tt_busy:
                return
            self._tt_busy = True
        threading.Thread(target=self._timetable_worker, args=(utterance,), daemon=True).start()

    def _timetable_worker(self, utterance):
        # A follow-up ("send me the report") carries no subject. Rebuild the full
        # question from the last one and run it down the ordinary path, so there
        # is no second code path that can drift from the first.
        if _si is not None:
            c = _si.classify(utterance, self._last_skill)
            if c["intent"] == "followup":
                self._push("AI-SHA", "Send you which report? Ask me about free "
                                     "teachers or students first, then say "
                                     "\"send me the list\".")
                with self.lock:
                    self._tt_busy = False
                return
            if _si.is_followup(utterance) and self._last_skill:
                utterance = _si.synthesize(c)
        """Ask the timetable app. Speakable answers (a class timetable, a count of
        free students) come back as text and are shown; anything that would name a
        student is emailed by the app and only a confirmation comes back here."""
        try:
            if _si is not None:
                c2 = _si.classify(utterance, self._last_skill)
                if c2["intent"] != "none":
                    with self.lock:
                        self._last_skill = c2      # remember for the next follow-up
            r = subprocess.run(
                f'python3 -u "{TT_TOOL}" ask {json.dumps(utterance)}',
                shell=True, cwd=TT_HOME, capture_output=True, text=True, timeout=90)
            out = (r.stdout or "") + (r.stderr or "")
            spoken = [l[len("SPEAK: "):] for l in out.splitlines() if l.startswith("SPEAK: ")]
            if r.returncode == 0 and spoken:
                self._push("AI-SHA", spoken[-1])
                return
            msg = {
                2: "I need a bit more detail — say e.g. \"what is the timetable for "
                   "grade 7 section A on Monday\", or \"how many students are free "
                   "in grade 10 period 3\".",
                # Wording comes from the tool, which knows whether the question
                # was about students or teachers; saying "students" for a teacher
                # query made the robot look like it had misheard.
                3: None,
                4: "I could not reach the timetable system.",
            }.get(r.returncode)
            if msg is None:
                denied = [l for l in out.splitlines() if "DENIED" in l]
                if denied:
                    # e.g. "... That answer names teachers, so it needs an
                    # authenticated administrator." Use the tool's own wording.
                    msg = denied[-1].split(". ", 1)[-1].strip()
                    # Remember it, so after the user authenticates we answer it
                    # automatically instead of making them re-type it.
                    if "authenticate" in msg.lower() or "administrator" in msg.lower():
                        self._pending_admin = ("timetable", utterance)
                        msg += " Say \"authenticate me\" and I will then answer this."
                else:
                    detail = [l for l in out.splitlines() if l.strip()]
                    msg = ("The timetable system could not answer that. "
                           + (detail[-1][:140] if detail else ""))
            self._push("AI-SHA", msg)
        except subprocess.TimeoutExpired:
            self._push("AI-SHA", "The timetable system did not respond in time.")
        except Exception as e:
            self._push("AI-SHA", f"Sorry — the timetable request failed ({type(e).__name__}).")
        finally:
            with self.lock:
                self._tt_busy = False
        self._last_skill = None   # last answered timetable query, for follow-ups

    def _sh(self, cmd, timeout=180):
        return subprocess.run(cmd, shell=True, cwd=VIDEO_HOME, capture_output=True,
                              text=True, timeout=timeout)

    def _video_message_worker(self):
        try:
            self._push("AI-SHA", f"Sure — a {VIDEO_SECS}-second video message for the "
                                 "administrator. Look at my camera; I will count down.")
            # stt_node holds the ReSpeaker exclusively (ALSA capture is not shareable),
            # so the recorder cannot open the mic until STT lets go. The launch only
            # LOGS when a node exits, so stopping stt_node here is survivable; we
            # start it again ourselves afterwards.
            stt_was_running = subprocess.run("pgrep -f 'stt_nod[e]'", shell=True,
                                             capture_output=True).returncode == 0
            if stt_was_running:
                subprocess.run("pkill -f 'stt_nod[e]'", shell=True)
                time.sleep(4)          # let ALSA actually release the device

            # Stream the recorder's own output instead of waiting for it to finish,
            # so the countdown the visitor reads is the REAL one - a countdown faked
            # here would drift from the process that is actually recording.
            proc = subprocess.Popen(
                # -u is REQUIRED: writing to a pipe makes Python block-buffer stdout,
                # so without it the whole countdown arrives at once AFTER the
                # recording has already finished - useless to the person in front.
                f'python3 -u video_message.py record --seconds {VIDEO_SECS} '
                f'--note "requested from the web console"',
                shell=True, cwd=VIDEO_HOME, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            out = ""
            for line in proc.stdout:
                out += line
                t = line.strip()
                if re.search(r"^\[record\]\s+([321])\s*\.\.\.", t):
                    n = re.search(r"([321])", t).group(1)
                    # Countdown ticks arrive ~1 s apart but each takes longer than
                    # that to speak, so speaking them would run into the recording.
                    self._push("AI-SHA", f"{n}...", speak=False)
                elif "RECORDING" in t:
                    self._push("AI-SHA", f"● RECORDING NOW — speak your message "
                                         f"({VIDEO_SECS} seconds).")
                elif t.startswith("[record] ■ done"):
                    self._push("AI-SHA", "Recording finished. Sending it now...")
            proc.wait(timeout=60)
            if "saved:" not in out:
                tail = [l for l in out.splitlines() if "ERROR" in l or "error" in l]
                self._push("AI-SHA", "Sorry — the recording failed. "
                                     + (tail[-1] if tail else "No file was produced."))
                return

            d = self._sh("python3 deliver.py send", timeout=300)
            dout = (d.stdout or "") + (d.stderr or "")
            if "SENT" in dout:
                self._push("AI-SHA", "Your video message has been sent to the "
                                     "administrator by email. Thank you!")
            elif "gate CLOSED" in dout:
                self._push("AI-SHA", "The message is recorded and queued, but sending is "
                                     "switched off. An administrator must open the "
                                     "delivery gate.")
            else:
                detail = dout.strip().splitlines()[-1][:120] if dout.strip() else ""
                self._push("AI-SHA", "The message is recorded and queued, but sending did "
                                     "not succeed — it stays in the outbox and can be "
                                     f"retried. {detail}")
        except subprocess.TimeoutExpired:
            self._push("AI-SHA", "Sorry — the video message timed out.")
        except Exception as e:                                    # never kill the console
            self._push("AI-SHA", f"Sorry — the video message failed ({type(e).__name__}).")
        finally:
            # bring speech recognition back however we exited
            try:
                subprocess.Popen(
                        "source /opt/ros/humble/setup.bash && "
                        "source ~/robot_ws/install/setup.bash && "
                        "ros2 run aisha_brain stt_node --ros-args "
                        "-p whisper_model:=small -p whisper_device:=cpu "
                        "-p audio_device:=ReSpeaker -p wake_word_enabled:=true "
                    "-p wake_word_timeout:=6.0 >> /tmp/aisha_stt_restart.log 2>&1",
                    shell=True, executable="/bin/bash", start_new_session=True)
            except Exception:
                pass
            with self.lock:
                self._vm_busy = False

    def _push(self, who, text, speak=True):
        with self.lock:
            self.log.append({"t": time.strftime("%H:%M:%S"), "who": who, "text": text})
            self.log = self.log[-60:]
        # Everything shown as AI-SHA's turn is also said out loud. The privacy
        # filtering already happened upstream -- the HRMS skill emails its answer
        # instead of returning one, and the timetable skill emails NAMED lists --
        # so whatever reaches here was already cleared to appear on a screen in a
        # public corridor, and speaking it exposes nothing more. speak=False is for
        # UI echoes that are not the robot talking.
        if speak and who == "AI-SHA":
            SPEAKER.say(text)

    def jpeg(self):
        with self.lock:
            return self._jpeg, self._jpeg_src

    def events(self):
        with self.lock:
            return list(self.log)

    def ask(self, text):

        # PIN entry: if we asked for a PIN, the next typed line IS the PIN. Consume
        # it here, mask it in the transcript, and never publish it anywhere.
        if self._awaiting_pin and time.time() < self._pin_deadline:
            self._awaiting_pin = False
            if re.fullmatch(r"\d{3,8}", text.strip()):
                self._push("you", "•••• (PIN entered)")
                self._start_auth(text.strip())
                return
            self._push("you", text + "  (typed)")
            self._push("AI-SHA", "That did not look like a PIN. Say "
                       "\"authenticate me\" to try again.")
            return
        self._awaiting_pin = False

        self._push("you", text + "  (typed)")
        # A video-message request is handled HERE, not by the LLM. Publishing it on
        # /speech/text as well would also reach brain_node, which routes it to the
        # knowledge base and answers something irrelevant ("I cannot access private
        # information...") right next to the recording prompt - confusing for the
        # visitor. Spoken requests still pass through brain_node; only the typed
        # path can be kept clean.
        if VIDEO_INTENT.search(text or ""):
            self._start_video_message()
            return
        if AUTH_INTENT.search(text or ""):
            # "authenticate me and send me the report by email" is TWO commands.
            # Queue the second so it runs the moment the gate opens, instead of
            # letting the leftover clause fall through to the knowledge base.
            rest = re.sub(AUTH_INTENT, " ", text or "", count=1)
            rest = re.sub(r"^[\s,.]*\b(and|then|also|please)\b", " ", rest.strip(),
                          flags=re.I).strip(" ,.")
            if rest:
                if HRMS_INTENT.search(rest):
                    self._pending_admin = ("hrms", rest)
                elif STUDENT_REPORT_INTENT.search(rest):
                    self._pending_admin = ("student_report", rest)
                elif _is_timetable(rest, self._last_skill):
                    self._pending_admin = ("timetable", rest)
            self._start_auth_prompt()
            return
        if HRMS_INTENT.search(text or ""):
            self._start_hrms_query(text)
            return
        if _is_timetable(text or "", self._last_skill):
            self._start_timetable_query(text)
            return
        msg = String(); msg.data = text
        self.pub.publish(msg)

    # ── admin face authentication ───────────────────────────────────────────
    def _start_auth_prompt(self):
        if self._auth_busy:
            return
        if AUTH_RELAXED:
            # Demo mode: just look at the camera. No PIN, no head turn.
            self._push("AI-SHA", "Sure — look at my camera and hold still for a "
                       "few seconds while I recognise you.")
            self._start_auth(None)
            return
        # Full mode: the PIN is collected first, by TYPING, so it is never spoken
        # aloud in the corridor and never leaves as a /speech/text message.
        self._awaiting_pin = True
        self._pin_deadline = time.time() + 90
        self._push("AI-SHA", "Let's authenticate you. Type your PIN in the box "
                   "below and press Ask — then look at my camera.")

    def _start_auth(self, pin):
        with self.lock:
            if self._auth_busy:
                return
            self._auth_busy = True
        threading.Thread(target=self._auth_worker, args=(pin,), daemon=True).start()

    def _auth_worker(self, pin):
        try:
            import shlex
            # INHERIT the console's own domain — never hardcode it. This was
            # pinned to "99"; when the stack moved to domain 42 for the two-board
            # mesh, the auth subprocess kept looking on 99, saw no camera at all,
            # and reported "I couldn't see a face" while the user sat perfectly
            # framed. Same class of bug as the stale camera rotation.
            env = dict(os.environ, AISHA_CAM_ROTATE=CAM_ROTATE)
            env.setdefault("ROS_DOMAIN_ID", os.environ.get("ROS_DOMAIN_ID", "42"))
            if AUTH_RELAXED:
                cmd = f"python3 -u auth_gate.py authenticate --relaxed --seconds {AUTH_SECS}"
            else:
                self._push("AI-SHA", "Look at my camera now, and when I ask, slowly "
                           "turn your head left and right. Keep within arm's length.")
                cmd = (f"python3 -u auth_gate.py authenticate --pin {shlex.quote(pin)} "
                       f"--seconds {AUTH_SECS}")
            r = subprocess.run(cmd, shell=True, cwd=FACE_HOME, capture_output=True,
                               text=True, timeout=90, env=env)
            out = (r.stdout or "") + (r.stderr or "")
            if "ACCESS GRANTED" in out:
                m = re.search(r"admin '([^']+)'", out)
                who = m.group(1) if m else "administrator"
                self._push("AI-SHA", f"Access granted — welcome, {who}. You are "
                           "authenticated for 15 minutes. I can now send named lists.")
                self._replay_pending()
            elif "not recognized" in out:
                self._push("AI-SHA", "I did not recognise you as an enrolled "
                           "administrator. Access denied.")
            elif "SPOOF" in out or "too far" in out.lower():
                self._push("AI-SHA", "I couldn't verify a live face — please sit "
                           "within arm's length, facing the camera, and say "
                           "\"authenticate me\" to try again.")
            elif "active liveness" in out:
                self._push("AI-SHA", "I didn't see a clear head turn. Say "
                           "\"authenticate me\" and turn left then right when asked.")
            elif "wrong PIN" in out:
                self._push("AI-SHA", "Your face was recognised, but that PIN was "
                           "incorrect. Access denied.")
            elif "no face seen" in out or "no admin" in out:
                self._push("AI-SHA", "I couldn't see a face — my camera may be "
                           "pointed away from you. Check the live preview on this "
                           "page shows your face, then say \"authenticate me\".")
            else:
                tail = [l for l in out.splitlines() if "DENIED" in l or "ERROR" in l]
                self._push("AI-SHA", "Authentication did not complete. "
                           + (tail[-1] if tail else ""))
        except subprocess.TimeoutExpired:
            self._push("AI-SHA", "The scan timed out. Say \"authenticate me\" to retry.")
        except Exception as e:                                    # never kill the console
            self._push("AI-SHA", f"Sorry — authentication failed ({type(e).__name__}).")
        finally:
            with self.lock:
                self._auth_busy = False

    def _replay_pending(self):
        """Re-run the admin question that was blocked, so the user does not have to
        re-ask it after authenticating."""
        pend = self._pending_admin
        self._pending_admin = None
        if not pend:
            return
        kind, utt = pend
        # A status echo of what is about to be answered, not speech.
        self._push("AI-SHA", f"Now answering: \"{utt}\"", speak=False)
        if kind == "timetable":
            self._start_timetable_query(utt)
        elif kind == "hrms":
            self._start_hrms_query(utt)
        elif kind == "student_report":
            self._start_student_report_query(utt)


PAGE = """<!DOCTYPE html><html><head><meta charset=utf-8>
<title>AI-SHA console</title><style>
body{margin:0;background:#11151c;color:#e6ecf3;font-family:Arial,Helvetica,sans-serif}
header{background:#4136b8;padding:10px 18px;font-weight:bold;font-size:18px}
header small{font-weight:normal;opacity:.8;font-size:12px;margin-left:10px}
.wrap{display:flex;gap:14px;padding:14px;flex-wrap:wrap}
.cam{flex:1;min-width:480px}.cam img{width:100%;border-radius:8px;background:#000;display:block}
.src{font-size:11px;color:#9aa4b2;margin-top:4px}
.side{flex:1;min-width:340px;display:flex;flex-direction:column}
#log{flex:1;height:420px;overflow-y:auto;background:#1a2029;border-radius:8px;padding:10px}
.msg{margin:6px 0;padding:7px 10px;border-radius:7px;line-height:1.4;font-size:14px}
.you{background:#26324a}.aisha{background:#203a2c}
.who{font-size:11px;opacity:.7;margin-bottom:2px}
form{display:flex;gap:8px;margin-top:10px}
input{flex:1;padding:11px;border-radius:7px;border:1px solid #333;background:#0d1117;color:#fff;font-size:15px}
button{padding:11px 18px;border:0;border-radius:7px;background:#4136b8;color:#fff;font-size:15px;cursor:pointer}
button:hover{background:#5346d8}
.bar{display:flex;align-items:center;gap:10px;margin-top:8px}
.mic{padding:9px 14px;border:0;border-radius:7px;font-size:14px;cursor:pointer;color:#fff}
.vol{display:inline-flex;align-items:center;gap:7px;margin-left:10px;color:#cfe0d6;font-size:13px}
.vol input{width:104px;accent-color:#2f6f4a;cursor:pointer;vertical-align:middle}
.vol b{min-width:40px;display:inline-block;text-align:right}
.vol.bad b{color:#e08a8a}
.mic.on{background:#2f6f4a}.mic.off{background:#8a3030}
.stat{font-size:12px;color:#9aa4b2}
.stat b.ok{color:#57d98b}.stat b.bad{color:#e08a8a}
.voicebar{background:#203a2c;color:#bfe6cd;padding:9px 18px;font-size:14px;border-bottom:1px solid #2c5a3e}
.voicebar b{color:#8fe0a8}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#57d98b;margin-right:7px;
     animation:pulse 1.3s infinite}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}
.lidar{flex:1;min-width:360px}
.lidar h3{margin:0 0 8px;font-size:15px;font-weight:bold}
.lidar h3 small{font-weight:normal;font-size:11px;color:#9aa4b2;margin-left:8px}
#radar{width:100%;max-width:460px;aspect-ratio:1;background:#0d1117;border-radius:8px;display:block}
.lbar{display:flex;align-items:center;gap:12px;margin-top:8px;flex-wrap:wrap}
.lbar select{background:#0d1117;color:#e6ecf3;border:1px solid #333;border-radius:6px;padding:6px 8px;
     font-size:13px}
.near{font-size:13px;color:#e6ecf3}
.near b{color:#ffd479}
.tabs{display:flex;gap:6px;margin-bottom:8px}
.tabs button{padding:6px 14px;font-size:13px;background:#1a2029;color:#9aa4b2;border-radius:6px}
.tabs button.sel{background:#4136b8;color:#fff}
#mapwrap{display:none}
#mapimg{width:100%;max-width:460px;aspect-ratio:1;object-fit:contain;background:#0d1117;
        border-radius:8px;display:block;image-rendering:pixelated}
</style></head><body>
<header>AI-SHA console
  <button onclick="shutdownAll()" title="Power off the robot (Jetson, and the Pi if reachable)"
    style="float:right;background:#8a3030;color:#fff;border:0;border-radius:7px;padding:7px 13px;
           font-size:13px;cursor:pointer;font-weight:bold">&#9211; Shut down</button>
  <small>camera + ask &amp; answer</small></header>
__VOICEBAR__
<div class=wrap>
  <div class=cam><img id=cam src="/stream">
    <div class=bar>
      <button id=micbtn class="mic on" onclick="togglemic()">&#127908; Mic: ON</button>
      <span class=vol title="Speaker volume on the Pi 5">
        <span>&#128266;</span>
        <input id=volslider type=range min=0 max=100 step=5 value=70
               oninput="volPreview(this.value)" onchange="volSet(this.value)">
        <b id=volval>&hellip;</b>
      </span>
      <span class=stat id=src></span>
      <span class=stat>camera <b id=camstat class=ok>live</b></span>
    </div>
  </div>
  <div class=lidar>
    <h3>LiDAR <small>360&deg; scan from the Pi 5 &middot; front of robot is up</small></h3>
    <div class=tabs>
      <button id=tabradar class=sel onclick="showLidarView('radar')">Live radar</button>
      <button id=tabmap onclick="showLidarView('map')">SLAM map</button>
    </div>
    <canvas id=radar width=460 height=460></canvas>
    <div id=mapwrap>
      <img id=mapimg alt="SLAM map">
      <div class=stat id=mapstat style="margin-top:8px">map: waiting&hellip;</div>
    </div>
    <div class=lbar>
      <select id=lrange onchange="drawRadar()">
        <option value=0>range: auto</option>
        <option value=1>1 m</option><option value=2>2 m</option>
        <option value=4 selected>4 m</option><option value=8>8 m</option>
        <option value=12>12 m</option>
      </select>
      <span class=stat>lidar <b id=lstat class=bad>offline</b></span>
      <span class=stat id=lrpm></span>
    </div>
    <div class=near id=lnear></div>
  </div>
  <div class=side>
    <div id=log></div>
    <form onsubmit="ask(event)">
      <input id=q autocomplete=off placeholder="Ask a question, or say: record a video message">
      <button>Ask</button>
    </form>
  </div>
</div>
<script>
// ── keep the camera alive ───────────────────────────────────────────────────
// The MJPEG response is one long-lived HTTP connection. When it drops - Wi-Fi
// blip, the console being restarted, a proxy timing it out - the <img> simply
// stops updating and keeps showing the LAST frame. Nothing errors, so the page
// looks fine while the picture is minutes old; that is what forced a manual
// refresh every 20-30 s. Watch for staleness and rebuild the connection.
var lastFrame = Date.now();
var cam = document.getElementById('cam');
cam.addEventListener('load', function(){ lastFrame = Date.now(); });
cam.addEventListener('error', function(){ reconnect(); });
function reconnect(){
  document.getElementById('camstat').textContent = 'reconnecting';
  document.getElementById('camstat').className = 'bad';
  cam.src = '/stream?t=' + Date.now();     // cache-buster forces a NEW connection
}
setInterval(function(){
  // A live MJPEG stream fires no 'load' events in some browsers, so also poll a
  // single frame: if THAT fails the server is genuinely gone.
  fetch('/frame?t=' + Date.now(), {cache:'no-store'}).then(function(r){
    if(!r.ok) throw 0;
    lastFrame = Date.now();
    document.getElementById('camstat').textContent = 'live';
    document.getElementById('camstat').className = 'ok';
  }).catch(function(){
    if(Date.now() - lastFrame > 6000) reconnect();
  });
}, 4000);

var micBusy = false;
async function shutdownAll(){
  // Two-step confirm: this powers the robot OFF, and it cannot be turned back on
  // remotely (a powered-off board has no network). A single misclick should not
  // strand the robot.
  if(!confirm('Power OFF the robot? This shuts down the Jetson (and the Pi 5 if '
    +'reachable). Neither can be switched back on remotely — someone has to press '
    +'power physically. Continue?')) return;
  if(!confirm('Are you sure? Final confirmation — the console will go offline.')) return;
  try{
    await fetch('/shutdown', {method:'POST'});
  }catch(e){}   // the connection drops as the Jetson goes down; that is expected
  document.body.innerHTML =
    '<div style="font-family:Arial;color:#e6ecf3;padding:40px;font-size:18px">'
    +'⏻ Shutting the robot down…<br><br>'
    +'<span style="font-size:14px;color:#9aa4b2">The Pi is powered off first (if reachable), '
    +'then the Jetson. This page will stop responding. To use the robot again, press the '
    +'power button on the board(s).</span></div>';
}

async function togglemic(){
  if(micBusy) return;                 // a double-click used to fire mute then unmute
  micBusy = true;
  try{
    var cur = await (await fetch('/mic')).json();   // server is the source of truth
    var d = await (await fetch('/mic?mute=' + (cur.muted ? '0' : '1'))).json();
    setmic(d.muted);
  } finally { micBusy = false; }
}
function setmic(muted){
  var b = document.getElementById('micbtn');
  b.className = 'mic ' + (muted ? 'off' : 'on');
  b.innerHTML = muted ? '&#128263; Mic: MUTED' : '&#127908; Mic: ON';
}

async function ask(e){e.preventDefault();var q=document.getElementById('q');
  if(!q.value.trim())return; await fetch('/ask?q='+encodeURIComponent(q.value)); q.value=''; poll();}
async function poll(){try{var r=await fetch('/events');var d=await r.json();
  var el=document.getElementById('log');el.innerHTML='';
  d.forEach(function(m){var c=m.who=='you'?'you':'aisha';
    el.innerHTML+='<div class="msg '+c+'"><div class=who>'+m.who+' &middot; '+m.t+'</div>'+
      m.text.replace(/</g,'&lt;')+'</div>';});
  el.scrollTop=el.scrollHeight;
  var s=await fetch('/src');document.getElementById('src').textContent='source: '+await s.text();
  if(!micBusy){ var m=await (await fetch('/mic')).json(); setmic(m.muted); }
}catch(e){}}
setInterval(poll,1000);poll();

// ── LiDAR radar ─────────────────────────────────────────────────────────────
// Points arrive as [angle_deg, distance_mm, intensity], 0 deg = straight ahead,
// angle increasing clockwise. Screen keeps that convention: front of the robot is
// up, so what you see on the radar matches what the camera above is looking at.
var lastScan = null;
function drawRadar(){
  var c = document.getElementById('radar'), g = c.getContext('2d');
  var W = c.width, H = c.height, cx = W/2, cy = H/2, R = Math.min(cx,cy) - 26;
  g.clearRect(0,0,W,H);

  var pts = (lastScan && lastScan.points) ? lastScan.points : [];
  var sel = parseFloat(document.getElementById('lrange').value);
  var maxm = sel;
  if(!maxm){                                  // auto: fit the farthest return
    maxm = 0.5;
    pts.forEach(function(p){ var m = p[1]/1000; if(m > maxm) maxm = m; });
    maxm = Math.ceil(maxm*2)/2;
  }

  // range rings
  g.strokeStyle = '#243044'; g.fillStyle = '#5c6b80'; g.font = '10px Arial';
  g.lineWidth = 1;
  for(var i=1;i<=4;i++){
    var rr = R*i/4;
    g.beginPath(); g.arc(cx,cy,rr,0,Math.PI*2); g.stroke();
    g.fillText((maxm*i/4).toFixed(maxm<2?2:1)+' m', cx+4, cy-rr-3);
  }
  g.beginPath(); g.moveTo(cx-R,cy); g.lineTo(cx+R,cy);
  g.moveTo(cx,cy-R); g.lineTo(cx,cy+R); g.stroke();
  g.fillStyle = '#7b8aa0'; g.font = '11px Arial'; g.textAlign = 'center';
  g.fillText('FRONT 0°', cx, cy-R-12);
  g.fillText('180°', cx, cy+R+18);
  g.fillText('90°', cx+R+16, cy+4);
  g.fillText('270°', cx-R-16, cy+4);
  g.textAlign = 'left';

  // returns
  var nearD = 1e9, nearA = 0;
  pts.forEach(function(p){
    var m = p[1]/1000;
    if(m <= 0) return;
    if(m < nearD){ nearD = m; nearA = p[0]; }
    if(m > maxm) return;                      // outside the selected range
    var a = p[0]*Math.PI/180, rr = R*m/maxm;
    var x = cx + rr*Math.sin(a), y = cy - rr*Math.cos(a);
    // colour by proximity: close things are the ones you care about
    var f = m/maxm;
    g.fillStyle = f < 0.25 ? '#ff6b6b' : (f < 0.5 ? '#ffd479' : '#57d98b');
    g.fillRect(x-1.5, y-1.5, 3, 3);
  });

  // the robot
  g.fillStyle = '#4136b8';
  g.beginPath(); g.moveTo(cx,cy-9); g.lineTo(cx-7,cy+7); g.lineTo(cx+7,cy+7);
  g.closePath(); g.fill();

  var near = document.getElementById('lnear');
  if(pts.length && nearD < 1e9){
    near.innerHTML = 'nearest obstacle <b>' + nearD.toFixed(2) + ' m</b> at <b>'
                   + nearA.toFixed(0) + '°</b> &middot; ' + pts.length + ' points';
  } else {
    near.textContent = lastScan ? 'no returns' : '';
  }
}

async function pollLidar(){
  try{
    var d = await (await fetch('/lidar.json', {cache:'no-store'})).json();
    var st = document.getElementById('lstat');
    if(d.ok && d.points){
      lastScan = d;
      st.textContent = 'live'; st.className = 'ok';
      document.getElementById('lrpm').textContent =
        d.rpm + ' rpm · ' + (d.crc_errors||0) + ' crc err';
    } else if(d.warming){
      st.textContent = 'connecting'; st.className = '';
    } else {
      // d.age is the Pi's own scanner age; if THAT is stale the scanner stopped,
      // which is a different fault from the link to the Pi being down.
      st.textContent = (d.age > 3) ? 'scanner stopped' : (d.stale ? 'stale ' + d.stale + 's' : 'offline');
      st.className = 'bad';
      document.getElementById('lrpm').textContent = d.error ? String(d.error).slice(0,60) : '';
    }
  }catch(e){
    document.getElementById('lstat').textContent = 'offline';
    document.getElementById('lstat').className = 'bad';
  }
  drawRadar();
}
setInterval(pollLidar, 200); pollLidar();

// ── SLAM map ────────────────────────────────────────────────────────────────
// Only polled while its tab is showing: the radar is the default view and the
// map costs a round trip to the Pi plus a PNG decode for a picture that changes
// about once a second.
var lidarView = 'radar';
function showLidarView(v){
  lidarView = v;
  document.getElementById('radar').style.display   = (v=='radar') ? 'block' : 'none';
  document.getElementById('lrange').style.display  = (v=='radar') ? '' : 'none';
  document.getElementById('mapwrap').style.display = (v=='map')   ? 'block' : 'none';
  document.getElementById('tabradar').className = (v=='radar') ? 'sel' : '';
  document.getElementById('tabmap').className   = (v=='map')   ? 'sel' : '';
  if(v=='map') pollMap();
}
async function pollMap(){
  if(lidarView != 'map') return;
  try{
    var d = await (await fetch('/map.json', {cache:'no-store'})).json();
    var el = document.getElementById('mapstat');
    if(d.ok){
      document.getElementById('mapimg').src = '/map.png?t=' + Date.now();
      var m = (d.width*d.resolution).toFixed(1) + ' x ' + (d.height*d.resolution).toFixed(1) + ' m';
      el.innerHTML = 'map ' + m + ' @ ' + d.resolution + ' m/cell &middot; '
                   + d.occupied + ' occupied, ' + d.free + ' free cells';
    } else {
      el.textContent = 'map unavailable — is SLAM running on the Pi?';
    }
  }catch(e){
    document.getElementById('mapstat').textContent = 'map unavailable';
  }
}
setInterval(pollMap, 1500);

// ── Speaker volume (lives on the Pi, proxied through /volume*) ──────────────
var volBusy = false;
var volTouched = 0;              // last time the user moved the slider themselves
function volPreview(v){
  volTouched = Date.now();
  document.getElementById('volval').textContent = v + '%';
}
async function volSet(v){
  if(volBusy) return;              // dragging fires change repeatedly; one write at a time
  volBusy = true; volTouched = Date.now();
  try{
    volShow(await (await fetch('/volume/set?percent=' + encodeURIComponent(v),
                               {cache:'no-store'})).json());
  }catch(e){ volShow({ok:false}); }
  finally{ volBusy = false; }
}
function volShow(d){
  var box = document.querySelector('.vol'), val = document.getElementById('volval');
  if(d && d.ok){
    box.classList.remove('bad');
    val.textContent = d.percent + '%';
    document.getElementById('volslider').value = d.percent;
  }else{
    // Pi unreachable or aisha-volume down. Say so rather than leaving the slider
    // sitting at a number that is not what the speaker is actually set to.
    box.classList.add('bad');
    val.textContent = 'n/a';
  }
}
async function volLoad(){
  // Skip while the user is working the slider, so a poll cannot yank it out from
  // under their finger and fight them mid-drag.
  if(volBusy || Date.now() - volTouched < 3000) return;
  try{ volShow(await (await fetch('/volume.json', {cache:'no-store'})).json()); }
  catch(e){ volShow({ok:false}); }
}
// Re-sync from the Pi, which is the source of truth. Without this the slider was
// fetched ONCE at page load and then silently lied: a failed write (Pi briefly
// unreachable) or a volume_service restart restoring a different level left the
// console showing 100% while the speaker sat at 53% -- i.e. -23.8 dB down.
setInterval(volLoad, 5000); volLoad();
</script></body></html>"""


class Speaker:
    """Sends AI-SHA's spoken turns to the Pi so they come out of the speaker.

    Owns a thread and a queue because _push() runs inside ROS callbacks: a
    blocking HTTP POST there would stall the answer pipeline behind the network.
    The queue also keeps utterances in order, and being bounded means a Pi that
    has gone away cannot grow an unbounded backlog here.
    """
    MAXQ = 8
    # brain_node streams an answer a LINE at a time, so a single reply arrives as
    # several _push calls. Speaking each one separately sounded chopped up -- every
    # fragment paid piper's synthesis time and the node's 1 s pre-roll. Gather lines
    # that arrive close together and speak them as one utterance instead.
    DEBOUNCE   = 0.6      # quiet gap that ends an answer. Every 0.1 s here is 0.1 s
                          # of silence the visitor sits through, so it is kept just
                          # long enough to catch the next streamed line of one answer.
    MAX_BUFFER = 600      # ...but never hold a long answer back waiting for more

    def __init__(self):
        self.q = queue.Queue(maxsize=self.MAXQ)
        self.error = None
        self.sent = 0
        self._pending = []
        self._pending_at = 0.0
        self._plock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._flusher, daemon=True).start()

    @staticmethod
    def clean(text):
        """Strip decoration that belongs to the transcript, not to speech."""
        t = (text or "").strip()
        # Markdown the LLM emits: a leading bullet, and ** emphasis **. Left in,
        # piper reads "star" out loud.
        t = re.sub(r"^\s*(?:[\*\-\u2022\u25cf]|\d+[.)])\s+", "", t)
        t = t.replace("**", "").replace("__", "")
        t = re.sub(r"[\u25cf\u2022\u2192\u2026]+", " ", t)   # bullets/arrows/ellipsis
        t = re.sub(r"\s*[\u2014\u2013]\s*", ", ", t)          # em/en dash -> a pause.
                                                              # ASCII "-" is left alone
                                                              # so "AI-SHA" survives.
        return re.sub(r"\s+", " ", t).strip()

    def say(self, text):
        text = self.clean(text)
        if not text:
            return
        with self._plock:
            self._pending.append(text)
            self._pending_at = time.time()
            total = sum(len(x) + 2 for x in self._pending)
        if total >= self.MAX_BUFFER:
            self._flush()

    def _flush(self):
        with self._plock:
            if not self._pending:
                return
            parts, self._pending = self._pending, []
        # Give every fragment terminal punctuation so piper puts a pause between
        # what were separate lines instead of running them into one breath.
        self._enqueue(" ".join(p if p[-1] in ".!?:;," else p + "." for p in parts))

    def _flusher(self):
        while True:
            time.sleep(0.3)
            with self._plock:
                due = bool(self._pending) and (time.time() - self._pending_at) >= self.DEBOUNCE
            if due:
                self._flush()

    def _enqueue(self, text):
        try:
            self.q.put_nowait(text)
        except queue.Full:
            # Speech is far slower than text. If we are behind, drop the OLDEST
            # line: a robot that is current is better than one reciting history.
            try:
                self.q.get_nowait()
                self.q.put_nowait(text)
            except (queue.Empty, queue.Full):
                pass

    START_WAIT = 25.0     # give piper time to synthesise before deciding it never spoke
    MAX_HOLD   = 150.0    # hard cap: never leave the microphone muted for ever

    def _worker(self):
        while True:
            text = self.q.get()
            hub = globals().get("HUB")
            try:
                if hub:
                    hub.speaker_gate(True)
                req = Request(SPEECH_URL,
                              data=json.dumps({"text": text}).encode(),
                              headers={"Content-Type": "application/json"})
                with urlopen(req, timeout=5.0) as r:
                    r.read()
                self.error = None
                self.sent += 1
                self._wait_until_quiet()
            except Exception as e:
                # Never raise: the console must keep working with the Pi down.
                self.error = f"{type(e).__name__}: {e}"
            finally:
                # Always release, including on error -- a failed POST must not
                # leave the microphone muted.
                if hub:
                    hub.speaker_gate(False)

    def _wait_until_quiet(self):
        """Block until the Pi reports the speaker has finished.

        Doing this in the worker also serialises the queue: one utterance is
        fully spoken before the next is sent, so answers do not overlap. Every
        exit path is bounded, because the cost of getting this wrong is a
        microphone that stays muted.
        """
        deadline = time.time() + self.MAX_HOLD
        start_by = time.time() + self.START_WAIT
        started = False
        while time.time() < deadline:
            try:
                with urlopen(SPEAKING_URL, timeout=3.0) as r:
                    speaking = bool(json.loads(r.read().decode()).get("speaking"))
            except Exception:
                return                    # cannot tell; do not hold the mute blind
            if speaking:
                started = True
            elif started:
                return                    # spoke, and has now finished
            elif time.time() > start_by:
                return                    # never started; release rather than hang
            time.sleep(0.4)

    def status(self):
        return {"ok": self.error is None, "sent": self.sent, "error": self.error}


def volume_call(path):
    """Proxy one call through to the Pi's volume service.

    Deliberately NOT a background poller like Lidar: volume is read once when a
    page loads and written only when someone moves the slider, so there is no
    steady stream of polls to shield the console from. The short timeout is what
    keeps an unreachable Pi from tying up a request thread.
    """
    try:
        with urlopen(VOLUME_URL + path, timeout=2.0) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


class Lidar:
    """Mirrors the Pi's LiDAR feed so the browser never talks to the Pi directly.

    A background thread owns the only connection to the Pi and the HTTP handler
    serves whatever it last got. That matters because the Pi drops off regularly
    (power) — if each browser poll did its own fetch, every one of them would block
    on the dead socket and the whole console would stall, camera included. Polling
    is lazy: it stops when nobody is looking at the LiDAR panel.
    """
    IDLE_AFTER = 10.0                 # stop polling this long after the last view

    def __init__(self):
        self.lock = threading.Lock()
        self.data = None
        self.error = "starting"
        self.fetched = 0.0
        self.wanted = 0.0
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            if time.time() - self.wanted > self.IDLE_AFTER:
                time.sleep(0.5)
                continue
            try:
                with urlopen(LIDAR_URL, timeout=2.0) as r:
                    d = json.loads(r.read().decode())
                with self.lock:
                    self.data, self.error, self.fetched = d, None, time.time()
            except Exception as e:
                with self.lock:
                    self.error = f"{type(e).__name__}: {e}"
                time.sleep(1.0)       # back off; the Pi is down or rebooting
            time.sleep(0.15)

    def snapshot(self):
        # Was the poller asleep? If so the cached sweep is old only because nobody
        # was watching, which is not a fault -- report it as warming up rather than
        # flashing "stale 15 s" at someone who just opened the page.
        warming = (time.time() - self.wanted) > self.IDLE_AFTER
        self.wanted = time.time()
        with self.lock:
            if self.data is None:
                return {"ok": False, "error": self.error or "no data yet"}
            if warming:
                return {"ok": False, "warming": True}
            # Staleness is reported rather than hidden: a frozen radar picture that
            # looks live is the same trap as the frozen camera frame handled above.
            stale = time.time() - self.fetched
            out = dict(self.data)
            # Two independent ways this can be stale, and BOTH must be clean:
            # our fetch may be old, or the Pi may still be serving happily while its
            # scanner has stopped (a yanked/re-enumerated USB port reads as silence,
            # not as an error). Trusting only our own fetch time would paint a frozen
            # sweep as "live" -- the exact trap the camera code warns about.
            age = self.data.get("age")
            out["ok"] = stale < 3.0 and (age is None or age < 3.0)
            out["stale"] = round(stale, 2)
            if self.error:
                out["error"] = self.error
            return out


class MapView:
    """Mirrors the SLAM map PNG off the Pi. Same lazy-poll shape as Lidar.

    The map changes about once a second and is only a couple of kB, so this polls
    far more slowly than the radar and holds the last good picture across the Pi's
    frequent reboots instead of blanking the panel.
    """
    IDLE_AFTER = 15.0

    def __init__(self):
        self.lock = threading.Lock()
        self.png = None
        self.meta = {}
        self.fetched = 0.0
        self.wanted = 0.0
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            if time.time() - self.wanted > self.IDLE_AFTER:
                time.sleep(1.0)
                continue
            try:
                with urlopen(MAP_URL + "/map.png", timeout=3.0) as r:
                    png = r.read()
                with urlopen(MAP_URL + "/map.json", timeout=3.0) as r:
                    meta = json.loads(r.read().decode())
                with self.lock:
                    self.png, self.meta, self.fetched = png, meta, time.time()
            except Exception:
                time.sleep(2.0)
            time.sleep(1.0)

    def image(self):
        self.wanted = time.time()
        with self.lock:
            return self.png

    def info(self):
        self.wanted = time.time()
        with self.lock:
            out = dict(self.meta)
            out["ok"] = self.png is not None and (time.time() - self.fetched) < 20
            return out


SPEAKER = Speaker()
LIDAR = Lidar()
MAPVIEW = MapView()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # silence access log
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/shutdown":
            # Reply BEFORE anything powers off, so the browser gets a clean 200
            # rather than a dropped connection it might retry.
            self._send(200, "application/json", b'{"ok":true}')
            HUB.shutdown_all()
        else:
            self._send(404, "text/plain", b"not found")

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            if VOICE:
                bar = ('<div class=voicebar><span class=dot></span>'
                       '<b>Robot mic is live.</b> &nbsp;Say &nbsp;<b>&ldquo;Hey Aisha&rdquo;</b>&nbsp; then your '
                       'question to the robot&rsquo;s microphone &mdash; e.g. '
                       '<i>&ldquo;Hey Aisha, what are the tuition fees?&rdquo;</i>. '
                       'You can also still type below.</div>')
            else:
                bar = ('<div class=voicebar style="background:#3a2c20;color:#e6cdbf;border-color:#5a412c">'
                       'Voice input is off &mdash; type your question below. '
                       '(Launch with <code>aisha_console_voice.sh</code> to use the robot&rsquo;s mic.)</div>')
            self._send(200, "text/html; charset=utf-8", PAGE.replace("__VOICEBAR__", bar).encode())
        elif p.path == "/events":
            self._send(200, "application/json", json.dumps(HUB.events()).encode())
        elif p.path == "/frame":
            frame, _ = HUB.jpeg()
            if not frame:
                self._send(503, "text/plain", b"no frame")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
        elif p.path == "/mic":
            want = parse_qs(p.query).get("mute", ["?"])[0]
            if want in ("1", "0"):
                HUB.set_mic_mute(want == "1")
            self._send(200, "application/json",
                       json.dumps({"muted": HUB.mic_muted}).encode())
        elif p.path == "/lidar.json":
            self._send(200, "application/json", json.dumps(LIDAR.snapshot()).encode())
        elif p.path == "/speech.json":
            self._send(200, "application/json", json.dumps(SPEAKER.status()).encode())
        elif p.path == "/volume.json":
            self._send(200, "application/json",
                       json.dumps(volume_call("/volume.json")).encode())
        elif p.path == "/volume/set":
            # Clamp here rather than trusting the query string: the value is
            # interpolated into the URL sent on to the Pi.
            try:
                want = max(0, min(100, int(float(parse_qs(p.query).get("percent", [""])[0]))))
            except ValueError:
                self._send(400, "application/json",
                           b'{"ok": false, "error": "percent must be a number"}')
                return
            self._send(200, "application/json",
                       json.dumps(volume_call(f"/set?percent={want}")).encode())
        elif p.path == "/map.json":
            self._send(200, "application/json", json.dumps(MAPVIEW.info()).encode())
        elif p.path == "/map.png":
            png = MAPVIEW.image()
            if not png:
                self._send(503, "text/plain", b"no map")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
        elif p.path == "/src":
            self._send(200, "text/plain", HUB.jpeg()[1].encode())
        elif p.path == "/ask":
            q = parse_qs(p.query).get("q", [""])[0].strip()
            if q:
                HUB.ask(q)
            self._send(200, "text/plain", b"ok")
        elif p.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    frame, _ = HUB.jpeg()
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(frame)).encode()
                                         + b"\r\n\r\n" + frame + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self._send(404, "text/plain", b"not found")


def main():
    global HUB
    rclpy.init()
    HUB = Hub()

    def _spin():
        # If this thread dies the node stops receiving camera frames, but the HTTP
        # server keeps serving the LAST one — a console that looks alive and shows a
        # frozen picture, which is worse than an obvious failure. Take the whole
        # process down so the supervisor restarts it.
        try:
            rclpy.spin(HUB)
        except Exception as e:                                # incl. ExternalShutdown
            print(f"[watch] ROS spin ended ({type(e).__name__}); exiting so the "
                  f"supervisor can restart a working console", flush=True)
        finally:
            os._exit(1)

    threading.Thread(target=_spin, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[aisha_watch] console at http://192.168.55.1:{PORT}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
