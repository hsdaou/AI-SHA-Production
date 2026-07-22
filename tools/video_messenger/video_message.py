#!/usr/bin/env python3
"""AI-SHA skill: record a video message for the administration (Sam).

Runs fully LOCALLY: RealSense color + ReSpeaker audio -> H.264/AAC .mp4 in a
local outbox. Delivery to the outside world (WhatsApp/email) is a SEPARATE,
gated step — see deliver.py. The robot never needs internet to record.

Commands:
  record  --seconds 15 [--note "for Sam"]   capture a message into the outbox
  list                                       show outbox + sent messages
  offer                                      speak/print the offer prompt (TTS hook)

Data lives OUTSIDE the git repo (default ~/video_messages, override
VIDEO_MSG_HOME). Nothing here is ever committed.
"""
import argparse, json, os, subprocess, sys, tempfile, time
import numpy as np
import cv2

HOME      = os.path.expanduser("~")
DATA_HOME = os.environ.get("VIDEO_MSG_HOME", f"{HOME}/video_messages")
OUTBOX    = f"{DATA_HOME}/outbox"
SENT      = f"{DATA_HOME}/sent"

AUDIO_DEV_MATCH = os.environ.get("VIDEO_MSG_MIC", "ReSpeaker")  # ReSpeaker
MAX_SECONDS     = 60

COLOR_TOPIC = "/camera/camera/color/image_raw"

OFFER_TEXT = ("Would you like to record a short video message for the "
              "administration? I can pass it on to Sam. Just say yes, look at "
              "my camera, and speak after the countdown.")


# ── camera grabber (same pattern as the face-auth skill) ─────────────────────
def make_grabber():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    class Grab(Node):
        def __init__(self):
            super().__init__('video_msg_grab')
            self.bridge = CvBridge(); self.color = None; self.stamp = 0.0
            q = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(Image, COLOR_TOPIC, self._c, q)
        def _c(self, m):
            self.color = self.bridge.imgmsg_to_cv2(m, 'bgr8'); self.stamp = time.time()
    return rclpy, Grab


def _find_mic():
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if AUDIO_DEV_MATCH in d["name"] and d["max_input_channels"] > 0:
            return i, int(d["default_samplerate"]), int(d["max_input_channels"])
    return None, None, None


def cmd_record(a):
    import sounddevice as sd
    os.makedirs(OUTBOX, exist_ok=True)
    seconds = min(float(a.seconds), MAX_SECONDS)

    mic_idx, sr, in_ch = _find_mic()
    if mic_idx is None:
        print(f"[record] ERROR: microphone matching '{AUDIO_DEV_MATCH}' not found"); sys.exit(2)

    rclpy, Grab = make_grabber(); rclpy.init(); node = Grab()
    # wait for the camera
    t0 = time.time()
    while node.color is None and time.time() - t0 < 8.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.color is None:
        print("[record] ERROR: no camera frames — is the RealSense launched?"); sys.exit(2)

    print("[record] Recording starts in:")
    for n in (3, 2, 1):
        print(f"[record]   {n} ...")
        t = time.time() + 1.0
        while time.time() < t:
            rclpy.spin_once(node, timeout_sec=0.05)

    audio_chunks = []
    def _cb(indata, frames, t_, status):
        audio_chunks.append(indata[:, 0].copy())   # ch0 of the mic array

    frames_jpg, last_stamp = [], 0.0
    print(f"[record] ● RECORDING {seconds:.0f}s — speak now!")
    # ReSpeaker UAC1.0 release race (ALSA -9985): retry like stt_node does
    stream = None
    for attempt in range(4):
        try:
            stream = sd.InputStream(device=mic_idx, samplerate=sr, channels=in_ch,
                                    dtype="int16", callback=_cb)
            break
        except sd.PortAudioError as e:
            if attempt == 3:
                print(f"[record] ERROR: mic busy after 4 tries: {e}"); sys.exit(2)
            print(f"[record] mic busy, retrying ({attempt+1}/4)..."); time.sleep(3.0)
    t_start = time.time()
    with stream:
        t_end = t_start + seconds
        while time.time() < t_end and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.color is None or node.stamp == last_stamp:
                continue
            last_stamp = node.stamp
            ok, jpg = cv2.imencode(".jpg", node.color,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                frames_jpg.append(jpg)
    elapsed = time.time() - t_start
    node.destroy_node(); rclpy.shutdown()
    print(f"[record] ■ done: {len(frames_jpg)} frames / {elapsed:.1f}s, "
          f"audio {sum(len(c) for c in audio_chunks)/max(1,sr):.1f}s")

    if len(frames_jpg) < 10:
        print("[record] ERROR: too few frames captured"); sys.exit(2)

    fps = len(frames_jpg) / elapsed
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_mp4 = f"{OUTBOX}/VM_{stamp}.mp4"

    with tempfile.TemporaryDirectory() as td:
        # raw video at the MEASURED fps (keeps A/V in sync)
        h, w = cv2.imdecode(frames_jpg[0], cv2.IMREAD_COLOR).shape[:2]
        avi = f"{td}/v.avi"
        vw = cv2.VideoWriter(avi, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
        for j in frames_jpg:
            vw.write(cv2.imdecode(j, cv2.IMREAD_COLOR))
        vw.release()
        # audio wav (int16 mono)
        import wave
        wav = f"{td}/a.wav"
        pcm = np.concatenate(audio_chunks) if audio_chunks else np.zeros(1, np.int16)
        with wave.open(wav, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        # mux -> H.264 + AAC (plays everywhere incl. WhatsApp)
        r = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", avi, "-i", wav,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
             "-shortest", "-movflags", "+faststart", out_mp4])
        if r.returncode != 0:
            print("[record] ERROR: ffmpeg mux failed"); sys.exit(2)

    meta = {"file": os.path.basename(out_mp4), "created": time.time(),
            "created_str": stamp, "duration_s": round(elapsed, 1),
            "frames": len(frames_jpg), "fps": round(fps, 2),
            "audio_sr": sr, "note": a.note, "status": "queued"}
    with open(out_mp4.replace(".mp4", ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    sz = os.path.getsize(out_mp4) / 1e6
    print(f"[record] saved: {out_mp4}  ({sz:.1f} MB, {fps:.1f} fps)")
    print("[record] queued in the outbox — deliver.py sends it when the gate is enabled.")


def cmd_list(_a):
    for label, d in (("OUTBOX (queued)", OUTBOX), ("SENT", SENT)):
        print(f"── {label}: {d}")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".mp4"):
                sz = os.path.getsize(os.path.join(d, f)) / 1e6
                print(f"   {f}  ({sz:.1f} MB)")


def cmd_offer(_a):
    """Print the offer + publish it on /robot_speech so the TTS tier (Pi 5,
    CP2/CP4) will speak it once that half of the robot exists."""
    print(f"[offer] {OFFER_TEXT}")
    try:
        import rclpy
        from std_msgs.msg import String
        rclpy.init()
        n = rclpy.create_node("video_msg_offer")
        pub = n.create_publisher(String, "/robot_speech", 1)
        time.sleep(0.5)                       # let the pub connect
        m = String(); m.data = OFFER_TEXT
        pub.publish(m)
        time.sleep(0.5)
        n.destroy_node(); rclpy.shutdown()
        print("[offer] published on /robot_speech (spoken once the TTS tier is up)")
    except Exception as e:
        print(f"[offer] ROS publish skipped ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record"); r.add_argument("--seconds", type=float, default=15)
    r.add_argument("--note", default="video message for Sam"); r.set_defaults(fn=cmd_record)
    l = sub.add_parser("list"); l.set_defaults(fn=cmd_list)
    o = sub.add_parser("offer"); o.set_defaults(fn=cmd_offer)
    args = ap.parse_args(); args.fn(args)
