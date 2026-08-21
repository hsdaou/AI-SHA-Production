#!/usr/bin/env python3
"""AI-SHA speaker volume service (Raspberry Pi 5).

Serves the speaker's ALSA volume over plain HTTP so the console -- which runs on
the JETSON, not here -- can read and set it. HTTP rather than ROS for the same
reason the LiDAR feed uses HTTP: Jazzy->Humble DDS does not deliver to real
subscriptions, so a topic published here would silently never arrive there.

    GET /volume.json      -> {"ok": true, "percent": 70}
    GET /set?percent=NN   -> clamps to 0..100, applies, returns the same shape

The control being driven is the *softvol* "Master" in ~/.asoundrc, not a hardware
mixer: the hifiberry-dac driver this speaker runs under exposes no hardware
volume control at all.
"""

import json
import os
import re
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

PORT = 8092
CARD = "0"
CONTROL = "Master"
# Last applied level, so a reboot comes back at the volume the user chose rather
# than at softvol's default of full scale.
STATE_FILE = os.path.expanduser("~/.aisha_volume")
DEFAULT_PERCENT = 70

_lock = threading.Lock()


def _amixer(*args):
    return subprocess.run(
        ["amixer", "-c", CARD] + list(args),
        capture_output=True, text=True, timeout=5,
    )


def _parse_percent(text):
    m = re.search(r"\[(\d+)%\]", text)
    return int(m.group(1)) if m else None


def prime_softvol():
    """Instantiate the softvol control by playing a moment of silence.

    softvol is an ALSA *plugin*: its "Master" control does not exist until
    something has opened the plugin at least once. Straight after a reboot the
    control is therefore missing and amixer fails, so the service creates it
    itself instead of waiting for the first bit of speech.
    """
    path = "/tmp/.aisha_silence.wav"
    try:
        if not os.path.exists(path):
            w = wave.open(path, "wb")
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(b"\0" * (4 * 4800))      # 0.1 s of stereo silence
            w.close()
        subprocess.run(["aplay", "-D", "default", path],
                       capture_output=True, timeout=15)
    except Exception as e:
        print(f"prime_softvol: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def get_percent():
    r = _amixer("sget", CONTROL)
    if r.returncode != 0:
        return None
    return _parse_percent(r.stdout)


def set_percent(pct):
    pct = max(0, min(100, int(pct)))
    r = _amixer("sset", CONTROL, f"{pct}%")
    if r.returncode != 0:
        return None
    try:
        with open(STATE_FILE, "w") as f:
            f.write(str(pct))
    except OSError:
        pass                                        # volume still applied; only the memo failed
    return _parse_percent(r.stdout) if _parse_percent(r.stdout) is not None else pct


def restore():
    prime_softvol()
    want = DEFAULT_PERCENT
    try:
        with open(STATE_FILE) as f:
            want = int(f.read().strip())
    except (OSError, ValueError):
        pass
    got = set_percent(want)
    print(f"volume_service: restored to {got}%", flush=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path)
        try:
            if p.path in ("/volume.json", "/"):
                with _lock:
                    pct = get_percent()
                if pct is None:
                    # Most likely the softvol control vanished (card re-created).
                    with _lock:
                        prime_softvol()
                        pct = get_percent()
                if pct is None:
                    return self._send({"ok": False, "error": "no Master control"}, 503)
                return self._send({"ok": True, "percent": pct})

            if p.path == "/set":
                q = parse_qs(p.query)
                if "percent" not in q:
                    return self._send({"ok": False, "error": "percent required"}, 400)
                try:
                    want = int(float(q["percent"][0]))
                except ValueError:
                    return self._send({"ok": False, "error": "percent must be a number"}, 400)
                with _lock:
                    pct = set_percent(want)
                    if pct is None:
                        prime_softvol()
                        pct = set_percent(want)
                if pct is None:
                    return self._send({"ok": False, "error": "amixer failed"}, 503)
                return self._send({"ok": True, "percent": pct})

            self._send({"ok": False, "error": "not found"}, 404)
        except Exception as e:
            self._send({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def log_message(self, *a):
        pass                                        # do not spam the journal per poll


if __name__ == "__main__":
    restore()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
