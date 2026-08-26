#!/usr/bin/env python3
"""AI-SHA face authentication gate — face + active-liveness challenge + PIN.

Layered access control for administrator data (defense in depth):
  Factor 1  IDENTITY   : ArcFace face match vs an enrolled admin gallery.
  Factor 2  LIVENESS   : passive depth planarity (real convex face, not a photo)
                         + ACTIVE challenge (random head-turn on command).
  Factor 3  KNOWLEDGE  : a per-admin PIN (stored as a salted pbkdf2 hash).
Only when ALL pass is an authenticated SESSION issued (time-limited token).

SECURITY NOTE: face recognition is spoofable; this is why the PIN and the
active challenge exist. Never rely on face alone. See docs/skills/FACE_AUTH.md.

Commands:
  set-pin --name Sam --pin 4729         # register/replace an admin's PIN (hashed)
  authenticate --pin 4729               # full gate -> session on success
  check-session                          # is there a valid unexpired session?
  logout                                 # clear the session
Reuses primitives from face_auth.py (Detector, Embedder, depth_liveness, grabber).
"""
import argparse, os, json, time, hashlib, hmac, secrets, sys, glob
from collections import Counter, deque
import numpy as np

import face_auth as fa   # same directory

ADMINS   = os.path.join(os.path.dirname(fa.GALLERY_DIR), "admins.json")
SESSION  = os.path.join(os.path.dirname(fa.GALLERY_DIR), "session.json")
SESSION_TTL_S = 15 * 60          # a granted session lasts 15 minutes
CHALLENGE_YAW_RANGE = 0.20       # normalized nose-offset swing required (look L<->R)
CHALLENGE_SECONDS   = 8


# ── PIN storage (salted PBKDF2, never plaintext) ─────────────────────────────
def _load_admins():
    return json.load(open(ADMINS)) if os.path.exists(ADMINS) else {}

def _save_admins(d):
    with open(ADMINS, "w") as f:
        json.dump(d, f, indent=2)
    os.chmod(ADMINS, 0o600)

def _hash_pin(pin, salt=None, iters=200_000):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), iters).hex()
    return salt, iters, h

def _verify_pin(name, pin):
    a = _load_admins().get(name)
    if not a:
        return False
    _, _, h = _hash_pin(pin, a["salt"], a["iters"])
    return hmac.compare_digest(h, a["hash"])


# ── Active liveness: random head-turn challenge ──────────────────────────────
def _yaw_proxy(bgr, det):
    """Signed horizontal nose offset relative to eye-midpoint, normalized by
    inter-eye distance. Swings negative/positive as the head turns L/R."""
    r = det.detect_align(bgr)
    if r is None:
        return None
    # recompute keypoints (detect_align returns aligned crop; get raw kps here)
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = det.mp.Image(image_format=det.mp.ImageFormat.SRGB, data=rgb)
    res = det.fd.detect(img)
    if not res.detections:
        return None
    d = max(res.detections, key=lambda x: x.bounding_box.width * x.bounding_box.height)
    kp = d.keypoints
    h, w = bgr.shape[:2]
    eyes = [(kp[0].x * w, kp[0].y * h), (kp[1].x * w, kp[1].y * h)]
    nose = (kp[2].x * w, kp[2].y * h)
    eye_mid = ((eyes[0][0] + eyes[1][0]) / 2, (eyes[0][1] + eyes[1][1]) / 2)
    inter = max(1.0, abs(eyes[0][0] - eyes[1][0]))
    return (nose[0] - eye_mid[0]) / inter


def active_challenge(node, det, rclpy):
    """Prompt a live head-turn and require a real yaw swing to both sides.
    A static photo cannot produce this; a random side adds unpredictability."""
    import random
    side = random.choice(["LEFT", "RIGHT"])
    print(f"[auth] LIVENESS CHALLENGE: slowly look to your {side}, then back to center.")
    print("[auth]   (then look the other way too — I need to see real head motion)")
    t_end = time.time() + CHALLENGE_SECONDS
    ys = []
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.color is None:
            continue
        y = _yaw_proxy(node.color.copy(), det)
        if y is not None:
            ys.append(y)
    if len(ys) < 8:
        return False, {"reason": "face not tracked during challenge", "n": len(ys)}
    swing = float(max(ys) - min(ys))
    return swing >= CHALLENGE_YAW_RANGE, {"swing": round(swing, 3),
                                          "need": CHALLENGE_YAW_RANGE, "n": len(ys)}


# ── Session ──────────────────────────────────────────────────────────────────
def _write_session(name):
    tok = secrets.token_hex(24)
    now = time.time()
    s = {"user": name, "token": tok, "granted_at": now, "expires_at": now + SESSION_TTL_S}
    with open(SESSION, "w") as f:
        json.dump(s, f)
    os.chmod(SESSION, 0o600)
    return s

def cmd_check_session(_a):
    if not os.path.exists(SESSION):
        print("[session] none"); sys.exit(1)
    s = json.load(open(SESSION))
    left = s["expires_at"] - time.time()
    if left <= 0:
        print(f"[session] EXPIRED for {s['user']}"); sys.exit(1)
    print(f"[session] VALID for {s['user']} ({int(left)}s left)"); sys.exit(0)

def cmd_logout(_a):
    if os.path.exists(SESSION):
        os.remove(SESSION)
    print("[session] cleared")


def cmd_set_pin(a):
    admins = _load_admins()
    gal = os.path.join(fa.GALLERY_DIR, f"{a.name.lower()}.npz")
    if not os.path.exists(gal):
        print(f"[set-pin] WARNING: no enrolled face gallery for {a.name} "
              f"(run: face_auth.py enroll --name {a.name}). PIN saved anyway.")
    salt, iters, h = _hash_pin(a.pin)
    admins[a.name] = {"salt": salt, "iters": iters, "hash": h,
                      "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save_admins(admins)
    print(f"[set-pin] PIN set for admin '{a.name}' (stored as salted pbkdf2 hash).")


# ── The gate ─────────────────────────────────────────────────────────────────
def cmd_authenticate(a):
    admins = _load_admins()
    gals = {}
    for f in glob.glob(os.path.join(fa.GALLERY_DIR, "*.npz")):
        d = np.load(f, allow_pickle=True)
        name = str(d["name"])
        if name in admins:                 # only enrolled ADMINS are candidates
            gals[name] = d["embeddings"]
    if not gals:
        print("[auth] DENIED - no admin has both a face gallery and a PIN set."); sys.exit(2)

    # Create the subscriptions before loading the models. DDS discovery then
    # overlaps the ~4 s detector/embedder startup instead of beginning after it.
    rclpy, Grab = fa.make_grabber(); rclpy.init(); node = Grab()
    try:
        det, emb = fa.Detector(), fa.Embedder()
    except Exception:
        node.destroy_node(); rclpy.shutdown()
        raise

    # Phase 1 — identity + passive depth liveness. The time limit remains a
    # fallback for poor positioning; a clear, live, consistently matching face
    # finishes as soon as enough independent fresh frames agree.
    print("[auth] Phase 1/3 IDENTITY: look at the camera...")
    votes, flat, nodata = [], 0, 0
    fast_matches = max(3, a.fast_matches)
    recent = deque(maxlen=fast_matches)
    last_color_seq = -1
    sample_started = time.monotonic()
    t_end = sample_started + a.seconds
    while rclpy.ok() and time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.color is None:
            continue
        # spin_once also wakes for depth callbacks. Only process a newly arrived
        # colour frame so the fast path represents independent camera samples.
        color_seq = getattr(node, "color_seq", last_color_seq + 1)
        if color_seq == last_color_seq:
            continue
        last_color_seq = color_seq
        r = det.detect_align(node.color.copy())
        if r is None:
            continue
        aligned, bbox, score = r
        if score < 0.7 or bbox[2] < 90:
            continue
        live, linfo = fa.depth_liveness(node.depth, bbox)
        if not live:
            # planar-with-valid-depth = photo-spoof evidence; missing depth
            # (motion/edge dropout) carries no spoof signal, so skip it.
            if linfo.get("kind") == "flat":
                flat += 1
                recent.clear()
            else:
                nodata += 1
            continue
        v = emb.embed(aligned)
        best_name, best = "UNKNOWN", -1.0
        for name, E in gals.items():
            s = float(np.max(E @ v))
            if s > best:
                best, best_name = s, name
        decided = best_name if best >= a.threshold else "UNKNOWN"
        votes.append((decided, best))
        recent.append(decided)
        # Five consecutive depth-live matches is stricter than the normal final
        # three-vote minimum. Never fast-pass after planar/photo evidence; such
        # attempts continue through the full window for the spoof-fraction test.
        if (flat == 0 and len(recent) == fast_matches and
                recent[0] != "UNKNOWN" and len(set(recent)) == 1):
            elapsed = time.monotonic() - sample_started
            print(f"[auth]   fast match: {fast_matches} fresh live frames agreed "
                  f"in {elapsed:.1f}s.")
            break
    depth_valid = len(votes) + flat
    spoof_frac = flat / max(1, depth_valid)
    if depth_valid >= 3 and spoof_frac > 0.30:
        node.destroy_node(); rclpy.shutdown()
        print(f"[auth] DENIED - possible SPOOF ({spoof_frac*100:.0f}% of depth-valid frames "
              f"were planar/flat: {flat}/{depth_valid}). nodata_skipped={nodata}"); sys.exit(3)
    tally = Counter(n for n, _ in votes)
    top = tally.most_common(1)[0] if tally else ("UNKNOWN", 0)
    # Distinguish "saw nobody" from "saw someone I don't know". The camera on this
    # robot wanders off-target, and reporting "not recognized" when it is actually
    # pointed at a wall sends the user hunting for a recognition fault that isn't
    # there. Too few qualifying frames = no face in view.
    if len(votes) + flat < 3:
        node.destroy_node(); rclpy.shutdown()
        print("[auth] DENIED - no face seen. Aim the camera at your face "
              "(use the console preview) and try again."); sys.exit(6)
    if top[0] == "UNKNOWN" or top[1] < max(3, len(votes) // 2):
        node.destroy_node(); rclpy.shutdown()
        print("[auth] DENIED - face not recognized as an enrolled admin."); sys.exit(3)
    identity = top[0]
    best_overall = max((s for _, s in votes), default=0.0)
    print(f"[auth]   identity candidate: {identity} (best={best_overall:.3f}, {top[1]}/{len(votes)} frames)")

    # Phases 2 & 3 (head-turn challenge + PIN) are skipped in --relaxed mode.
    # Passive depth liveness in Phase 1 still runs, so a printed photo is still
    # rejected — relaxed lowers the friction, not the spoof resistance to a flat
    # image. Intended for the experimental/demo phase; drop --relaxed to restore
    # the full four-factor gate.
    if getattr(a, "relaxed", False):
        node.destroy_node(); rclpy.shutdown()
        print("[auth]   relaxed mode: head-turn challenge and PIN skipped.")
    else:
        # Phase 2 — ACTIVE liveness challenge
        print("[auth] Phase 2/3 LIVENESS CHALLENGE:")
        ok, info = active_challenge(node, det, rclpy)
        node.destroy_node(); rclpy.shutdown()
        if not ok:
            print(f"[auth] DENIED - active liveness failed ({info})."); sys.exit(4)
        print(f"[auth]   challenge passed (head swing={info['swing']} >= {info['need']}).")

        # Phase 3 — PIN
        print("[auth] Phase 3/3 PIN:")
        pin = a.pin if a.pin is not None else _read_pin()
        if not _verify_pin(identity, pin):
            print("[auth] DENIED - wrong PIN."); sys.exit(5)

    s = _write_session(identity)
    print(f"[auth] ==================================================")
    print(f"[auth] ACCESS GRANTED to admin '{identity}'.")
    print(f"[auth]   session token: {s['token'][:16]}...  expires in {SESSION_TTL_S//60} min")
    print(f"[auth] ==================================================")


def _read_pin():
    # Headless robots would use a keypad or spoken PIN; stdin for POC/testing.
    try:
        import getpass
        return getpass.getpass("[auth] enter PIN: ")
    except Exception:
        return sys.stdin.readline().strip()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("set-pin"); sp.add_argument("--name", required=True)
    sp.add_argument("--pin", required=True); sp.set_defaults(fn=cmd_set_pin)
    au = sub.add_parser("authenticate"); au.add_argument("--pin", default=None)
    au.add_argument("--relaxed", action="store_true",
                    help="demo mode: face + passive liveness only, no challenge/PIN")
    au.add_argument("--threshold", type=float, default=0.45)
    au.add_argument("--seconds", type=float, default=5)
    au.add_argument("--fast-matches", type=int, default=5,
                    help="finish early after this many consecutive fresh live matches")
    au.set_defaults(fn=cmd_authenticate)
    cs = sub.add_parser("check-session"); cs.set_defaults(fn=cmd_check_session)
    lo = sub.add_parser("logout"); lo.set_defaults(fn=cmd_logout)
    args = ap.parse_args(); args.fn(args)
