#!/usr/bin/env python3
"""AI-SHA web console — watch the camera and talk to the robot from a browser.

Runs ON THE JETSON. Serves one page (default http://192.168.55.1:8088) that shows
the live camera and lets you type a question and read AI-SHA's answer. Needs NO ROS
on the machine with the browser, and NO new Python packages on the Jetson.

  Camera  : /detection/image_annotated  (YOLO boxes) -> falls back to raw colour
  Ask     : publishes your text on /speech/text  (same topic the mic would feed)
  Answer  : shows whatever the robot publishes on /robot_speech
"""
import json, os, re, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

VOICE = os.environ.get("AISHA_VOICE") == "1"     # set by the voice-mode launcher

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

PORT        = 8088
ANNOTATED   = "/detection/image_annotated"          # camera + detections
RAW_COLOR   = "/camera/camera/color/image_raw"       # fallback if vision is off
ASK_TOPIC   = "/speech/text"                          # question in
VIDEO_HOME  = os.path.expanduser("~/video_messages")  # video-message skill lives here
VIDEO_SECS  = 15                                       # length of a recorded message

# Spoken/typed phrases that trigger the video-message skill instead of the LLM.
VIDEO_INTENT = re.compile(
    r"(record|leave|send|take).{0,20}(video|vedio)\s*message"
    r"|video\s*message.{0,20}(to|for)\s+(sam|admin|hsdaou)", re.I)
REPLY_TOPIC = "/robot_speech"                         # answer out


class Hub(Node):
    def __init__(self):
        super().__init__("aisha_watch")
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self._jpeg = None
        self._jpeg_src = "waiting"
        self._annot_t = 0.0
        self._vm_busy = False
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

    def _claim_gpu(self):
        m = String(); m.data = "CONVERSING"
        self.mode_pub.publish(m)

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
            return
        self._push("you", m.data)
        if VIDEO_INTENT.search(m.data or ""):
            self._start_video_message()

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

    def _sh(self, cmd, timeout=180):
        return subprocess.run(cmd, shell=True, cwd=VIDEO_HOME, capture_output=True,
                              text=True, timeout=timeout)

    def _video_message_worker(self):
        try:
            self._push("AI-SHA", f"Sure. I will record a {VIDEO_SECS}-second video message "
                                 "and send it to the administrator. Look at my camera — "
                                 "recording starts after the countdown.")
            # stt_node holds the ReSpeaker exclusively (ALSA capture is not shareable),
            # so the recorder cannot open the mic until STT lets go. The launch only
            # LOGS when a node exits, so stopping stt_node here is survivable; we
            # start it again ourselves afterwards.
            stt_was_running = subprocess.run("pgrep -f 'stt_nod[e]'", shell=True,
                                             capture_output=True).returncode == 0
            if stt_was_running:
                subprocess.run("pkill -f 'stt_nod[e]'", shell=True)
                time.sleep(4)          # let ALSA actually release the device

            r = self._sh(f"python3 video_message.py record --seconds {VIDEO_SECS} "
                         f'--note "requested from the web console"', timeout=180)
            out = (r.stdout or "") + (r.stderr or "")
            if "saved:" not in out:
                tail = [l for l in out.splitlines() if "ERROR" in l or "error" in l]
                self._push("AI-SHA", "Sorry — the recording failed. "
                                     + (tail[-1] if tail else "No file was produced."))
                return
            self._push("AI-SHA", "Recording finished. Sending it now...")

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

    def _push(self, who, text):
        with self.lock:
            self.log.append({"t": time.strftime("%H:%M:%S"), "who": who, "text": text})
            self.log = self.log[-60:]

    def jpeg(self):
        with self.lock:
            return self._jpeg, self._jpeg_src

    def events(self):
        with self.lock:
            return list(self.log)

    def ask(self, text):
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
        msg = String(); msg.data = text
        self.pub.publish(msg)


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
.voicebar{background:#203a2c;color:#bfe6cd;padding:9px 18px;font-size:14px;border-bottom:1px solid #2c5a3e}
.voicebar b{color:#8fe0a8}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#57d98b;margin-right:7px;
     animation:pulse 1.3s infinite}
@keyframes pulse{0%,100%{opacity:.35}50%{opacity:1}}
</style></head><body>
<header>AI-SHA console <small>camera + ask &amp; answer &nbsp;|&nbsp; nothing is spoken until the Pi 5 (TTS) exists</small></header>
__VOICEBAR__
<div class=wrap>
  <div class=cam><img src="/stream"><div class=src id=src></div></div>
  <div class=side>
    <div id=log></div>
    <form onsubmit="ask(event)">
      <input id=q autocomplete=off placeholder="Ask a question, or say: record a video message">
      <button>Ask</button>
    </form>
  </div>
</div>
<script>
async function ask(e){e.preventDefault();var q=document.getElementById('q');
  if(!q.value.trim())return; await fetch('/ask?q='+encodeURIComponent(q.value)); q.value=''; poll();}
async function poll(){try{var r=await fetch('/events');var d=await r.json();
  var el=document.getElementById('log');el.innerHTML='';
  d.forEach(function(m){var c=m.who=='you'?'you':'aisha';
    el.innerHTML+='<div class="msg '+c+'"><div class=who>'+m.who+' &middot; '+m.t+'</div>'+
      m.text.replace(/</g,'&lt;')+'</div>';});
  el.scrollTop=el.scrollHeight;
  var s=await fetch('/src');document.getElementById('src').textContent='camera source: '+await s.text();
}catch(e){}}
setInterval(poll,1000);poll();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):        # silence access log
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
    threading.Thread(target=rclpy.spin, args=(HUB,), daemon=True).start()
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
