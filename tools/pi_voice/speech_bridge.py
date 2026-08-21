#!/usr/bin/env python3
"""AI-SHA speech bridge (Raspberry Pi 5).

Accepts answer text over plain HTTP from the Jetson console and republishes it
locally on /tts_text, where tts_speaker_node picks it up and speaks it.

This exists because the two boards cannot talk over ROS. The Jetson runs Humble
and this board runs Jazzy; cross-distro DDS discovers the endpoints but does not
deliver to real subscriptions (Iron+ added type hashes that Humble does not
supply), so a /tts_text publisher on the Jetson is silently ignored. Publishing
the topic HERE, same-distro and same-process-tree, sidesteps DDS entirely --
the only thing crossing the network is an HTTP POST.

    POST /say   {"text": "..."}   -> {"ok": true, "queued": "..."}
    GET  /say?text=...            -> same, for quick curl testing
    GET  /speaking                -> {"ok": true, "speaking": bool}
    GET  /health                  -> {"ok": true, "published": N}

/speaking mirrors tts_speaker_node's /robot/speaking so the Jetson can hold its
microphone muted for exactly as long as the speaker is talking. The Pi publishes
that topic itself, but on Jazzy, so the Jetson's Humble subscription never sees
it -- the console polls this instead and asserts the mute locally. Without it the
robot hears its own answer through the mic and re-triggers on it.

Deliberately does NOT run piper itself: tts_speaker_node already owns the
speaker, the /robot/speaking + /speaker/playing mic-mute signalling, and the
playback queue. Duplicating that here would give two things fighting for one
sound card.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote_plus

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

PORT = 8093
TOPIC = "/tts_text"
SPEAKING_TOPIC = "/robot/speaking"
MAX_CHARS = 1200          # a runaway answer should not lock the speaker for minutes

_node = None
_pub = None
_lock = threading.Lock()
_count = 0
_speaking = False
_speaking_at = 0.0
# If the TTS node dies mid-utterance its last message is "speaking", which would
# pin the Jetson's mic muted for ever. Treat a stale flag as quiet.
SPEAKING_TTL = 180.0


class Bridge(Node):
    def __init__(self):
        super().__init__("speech_bridge")
        self.pub = self.create_publisher(String, TOPIC, 10)
        self.create_subscription(Bool, SPEAKING_TOPIC, self._on_speaking, 10)

    @staticmethod
    def _on_speaking(msg):
        global _speaking, _speaking_at
        _speaking = bool(msg.data)
        _speaking_at = time.time()


def publish(text):
    global _count
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > MAX_CHARS:
        # Cut on a sentence boundary where possible rather than mid-word.
        cut = text[:MAX_CHARS]
        dot = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: dot + 1] if dot > MAX_CHARS // 2 else cut
    m = String()
    m.data = text
    with _lock:
        _pub.publish(m)
        _count += 1
    print(f"speech_bridge: -> {TOPIC}: {text[:90]!r}", flush=True)
    return text


def _is_speaking():
    if not _speaking:
        return False
    return (time.time() - _speaking_at) < SPEAKING_TTL


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        p = urlparse(self.path)
        if p.path != "/say":
            return self._send({"ok": False, "error": "not found"}, 404)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                text = json.loads(raw.decode()).get("text", "")
            except (ValueError, UnicodeDecodeError):
                text = raw.decode("utf-8", "replace")   # tolerate a bare body
            said = publish(text)
            if said is None:
                return self._send({"ok": False, "error": "empty text"}, 400)
            return self._send({"ok": True, "queued": said})
        except Exception as e:
            return self._send({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health":
            return self._send({"ok": True, "published": _count, "speaking": _is_speaking()})
        if p.path == "/speaking":
            return self._send({"ok": True, "speaking": _is_speaking()})
        if p.path == "/say":
            text = unquote_plus(parse_qs(p.query).get("text", [""])[0])
            said = publish(text)
            if said is None:
                return self._send({"ok": False, "error": "empty text"}, 400)
            return self._send({"ok": True, "queued": said})
        return self._send({"ok": False, "error": "not found"}, 404)

    def log_message(self, *a):
        pass


def main():
    global _node, _pub
    rclpy.init()
    _node = Bridge()
    _pub = _node.pub
    threading.Thread(
        target=lambda: ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever(),
        daemon=True).start()
    print(f"speech_bridge: HTTP :{PORT} -> {TOPIC}", flush=True)
    try:
        rclpy.spin(_node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
