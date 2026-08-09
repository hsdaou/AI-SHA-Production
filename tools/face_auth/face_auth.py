#!/usr/bin/env python3
"""AI-SHA face enrollment + verification (proof-of-concept).

Pipeline (no new pip deps — reuses onnxruntime, cv2, mediapipe from CP5/CP6):
  RealSense color+aligned_depth (ROS topics)
    -> mediapipe FaceDetector (bbox + 6 keypoints)
    -> 3-point similarity alignment (eyes+nose, order-by-x) to 112x112
    -> ArcFace MobileFaceNet ONNX -> 512-d L2-normalized embedding
  Depth liveness: a real 3D face has cm-scale depth variation across it;
  a flat phone/photo does not. Rejects the easy 2D-photo spoof.

Commands:
  enroll --name Sam --count 25       # build a gallery from live frames
  verify --threshold 0.45            # match a live face vs the gallery
NOTE: face recognition is ONE factor. For real access control, pair with a PIN.
"""
import argparse, os, time, sys
import numpy as np
import cv2

HOME = os.path.expanduser("~")
# Biometric data + models live OUTSIDE the git repo (never committed).
# Override the location with FACE_AUTH_HOME; default ~/face_auth.
DATA_HOME   = os.environ.get("FACE_AUTH_HOME", f"{HOME}/face_auth")
REC_MODEL   = f"{DATA_HOME}/models/w600k_mbf.onnx"                       # ArcFace (buffalo_s)
FACE_MODEL  = os.environ.get("FACE_DETECTOR_MODEL",
                             f"{HOME}/robot_ws/models/blaze_face_short_range.tflite")  # shared w/ CP5/6
GALLERY_DIR = f"{DATA_HOME}/gallery"

# ArcFace canonical 5-point template (112x112); we use the first 3 (l-eye,r-eye,nose).
_ARCFACE = np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366]], dtype=np.float32)

# Liveness thresholds (mm) — a real face spans several cm front-to-back.
LIVENESS_MIN_RANGE_MM = 18.0   # nose-to-cheek/ear depth spread
LIVENESS_MIN_VALID    = 0.35   # fraction of face pixels with valid depth
LIVENESS_MIN_RESIDMM  = 4.5    # RMS deviation from best-fit PLANE: a real face
                               # is convex (nose protrudes) -> high; a flat photo
                               # is planar even when tilted -> low.
LIVENESS_NEAR_MM, LIVENESS_FAR_MM = 200, 1200  # plausible face distance


class Embedder:
    def __init__(self):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.s = ort.InferenceSession(REC_MODEL, sess_options=so,
                                      providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def embed(self, aligned_bgr):
        b = ((aligned_bgr.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        v = self.s.run(None, {self.inp: b})[0][0]
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class Detector:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision as mpv
        self.mp = mp
        self.fd = mpv.FaceDetector.create_from_options(mpv.FaceDetectorOptions(
            base_options=mpp.BaseOptions(model_asset_path=FACE_MODEL),
            min_detection_confidence=0.5))

    def detect_align(self, bgr):
        """Return (aligned112_bgr, bbox(x,y,w,h), score) for the largest face, or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.fd.detect(img)
        if not res.detections:
            return None
        det = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        kp = det.keypoints  # 6 normalized keypoints; [0],[1] eyes, [2] nose
        h, w = bgr.shape[:2]
        eyes = sorted([(kp[0].x*w, kp[0].y*h), (kp[1].x*w, kp[1].y*h)], key=lambda p: p[0])
        nose = (kp[2].x*w, kp[2].y*h)
        src = np.array([eyes[0], eyes[1], nose], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, _ARCFACE, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(bgr, M, (112, 112), borderValue=0)
        bb = det.bounding_box
        score = det.categories[0].score if det.categories else 0.0
        return aligned, (int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height)), float(score)


def depth_liveness(depth_u16, bbox):
    """Real 3D face -> cm-scale depth spread. Flat photo -> ~uniform. Returns (bool, info)."""
    if depth_u16 is None:
        return False, {"reason": "no depth", "kind": "nodata"}
    x, y, w, h = bbox
    H, W = depth_u16.shape[:2]
    # sample the central face region (avoid bbox edges/background)
    x0 = max(0, x + w//4); x1 = min(W, x + 3*w//4)
    y0 = max(0, y + h//4); y1 = min(H, y + 3*h//4)
    roi = depth_u16[y0:y1, x0:x1].astype(np.float32)
    if roi.size == 0:
        return False, {"reason": "empty roi", "kind": "nodata"}
    valid = roi[(roi > LIVENESS_NEAR_MM) & (roi < LIVENESS_FAR_MM)]
    frac = valid.size / roi.size
    if frac < LIVENESS_MIN_VALID:
        return False, {"reason": f"valid depth {frac:.2f}<{LIVENESS_MIN_VALID}", "valid": frac, "kind": "nodata"}
    rng = float(np.percentile(valid, 90) - np.percentile(valid, 10))
    med = float(np.median(valid))
    # Planarity residual: fit z = a*x+b*y+c to valid face-depth points; RMS
    # deviation is large for a convex face, tiny for a flat photo (even tilted).
    mask = (roi > LIVENESS_NEAR_MM) & (roi < LIVENESS_FAR_MM)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    resid = 0.0
    if mask.sum() >= 30:
        X = xs[mask].astype(np.float32); Y = ys[mask].astype(np.float32); Z = roi[mask]
        A = np.stack([X, Y, np.ones_like(X)], 1)
        coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
        resid = float(np.sqrt(np.mean((Z - A @ coef) ** 2)))
    live = (rng >= LIVENESS_MIN_RANGE_MM) and (resid >= LIVENESS_MIN_RESIDMM)
    kind = "live" if live else "flat"   # valid depth but planar => photo-spoof evidence
    return live, {"range_mm": round(rng, 1), "resid_mm": round(resid, 1),
                  "median_mm": round(med, 1), "valid": round(frac, 2), "kind": kind}


# ── ROS camera grabber ────────────────────────────────────────────────────────
def make_grabber():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    class Grab(Node):
        def __init__(self):
            super().__init__('face_auth_grab')
            self.bridge = CvBridge(); self.color = None; self.depth = None
            q = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                           history=HistoryPolicy.KEEP_LAST)
            self.create_subscription(Image, '/camera/camera/color/image_raw', self._c, q)
            self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self._d, q)
        def _c(self, m): self.color = self.bridge.imgmsg_to_cv2(m, 'bgr8')
        def _d(self, m): self.depth = self.bridge.imgmsg_to_cv2(m, '16UC1')
    return rclpy, Grab


def cmd_enroll(a):
    import rclpy
    os.makedirs(GALLERY_DIR, exist_ok=True)
    det, emb = Detector(), Embedder()
    rclpy, Grab = make_grabber(); rclpy.init(); node = Grab()
    print(f"[enroll] Look at the camera. Capturing {a.count} good shots as '{a.name}'.")
    print("[enroll] Turn your head slightly L/R/up/down between captures; keep well lit.")
    embs, thumbs, last = [], [], 0.0
    t_end = time.time() + a.timeout
    while rclpy.ok() and len(embs) < a.count and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.color is None:
            continue
        bgr = node.color.copy()
        r = det.detect_align(bgr)
        if r is None:
            continue
        aligned, bbox, score = r
        if score < 0.7 or bbox[2] < 90:            # confident + face reasonably close
            continue
        live, info = depth_liveness(node.depth, bbox)
        now = time.time()
        if not live:
            if now - last > 1.5:
                print(f"  … face seen but liveness low ({info}) — sit ~40cm, no photo"); last = now
            continue
        if now - last < 0.4:                        # space out captures for variety
            continue
        v = emb.embed(aligned)
        # avoid near-duplicate frames (encourage pose variety)
        if embs and max(float(v @ e) for e in embs) > 0.985:
            continue
        embs.append(v); thumbs.append(aligned); last = now
        print(f"  ✓ captured {len(embs)}/{a.count}  (score={score:.2f}, depth_range={info['range_mm']}mm)")
    node.destroy_node(); rclpy.shutdown()
    if len(embs) < max(5, a.count // 3):
        print(f"[enroll] FAILED — only {len(embs)} good shots. Better light / closer / no glasses glare.")
        sys.exit(2)
    E = np.stack(embs)
    out = os.path.join(GALLERY_DIR, f"{a.name.lower()}.npz")
    np.savez(out, name=a.name, embeddings=E.astype(np.float32),
             created=time.strftime("%Y-%m-%d %H:%M:%S"))
    # intra-gallery cohesion (sanity: should be high, same person)
    sims = [float(E[i] @ E[j]) for i in range(len(E)) for j in range(i+1, len(E))]
    cv2.imwrite(os.path.join(GALLERY_DIR, f"{a.name.lower()}_montage.png"),
                np.hstack(thumbs[:min(8, len(thumbs))]))
    print(f"[enroll] SAVED {len(embs)} embeddings -> {out}")
    print(f"[enroll] intra-person similarity: mean={np.mean(sims):.3f} min={np.min(sims):.3f} "
          f"(high+tight = clean capture)")


def cmd_verify(a):
    import rclpy
    import glob
    gals = {}
    for f in glob.glob(os.path.join(GALLERY_DIR, "*.npz")):
        d = np.load(f, allow_pickle=True); gals[str(d["name"])] = d["embeddings"]
    if not gals:
        print("[verify] no gallery — run enroll first"); sys.exit(2)
    det, emb = Detector(), Embedder()
    rclpy, Grab = make_grabber(); rclpy.init(); node = Grab()
    print(f"[verify] galleries: {list(gals)} ; threshold={a.threshold}")
    votes, t_end = [], time.time() + a.seconds
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.color is None: continue
        r = det.detect_align(node.color.copy())
        if r is None: continue
        aligned, bbox, score = r
        if score < 0.7 or bbox[2] < 90: continue
        live, info = depth_liveness(node.depth, bbox)
        if not live:
            votes.append((("FLAT" if info.get("kind") == "flat" else "NODATA"), 0.0, info)); continue
        v = emb.embed(aligned)
        best_name, best = "UNKNOWN", -1.0
        for name, E in gals.items():
            s = float(np.max(E @ v))   # best match against that person's gallery
            if s > best: best, best_name = s, name
        decided = best_name if best >= a.threshold else "UNKNOWN"
        votes.append((decided, best, info))
    node.destroy_node(); rclpy.shutdown()
    if not votes:
        print("[verify] no face seen"); sys.exit(2)
    live_votes = [x for x in votes if x[0] not in ("FLAT", "NODATA")]
    names = [x[0] for x in live_votes]
    from collections import Counter
    tally = Counter(names)
    flat   = sum(1 for x in votes if x[0] == "FLAT")     # planar w/ valid depth = spoof evidence
    nodata = sum(1 for x in votes if x[0] == "NODATA")   # motion/edge dropout = skip, not spoof
    best_overall = max((x[1] for x in live_votes), default=0.0)
    print(f"[verify] frames: live={len(live_votes)} flat(spoof)={flat} nodata(skipped)={nodata}")
    print(f"[verify] tally: {dict(tally)} ; best score={best_overall:.3f}")
    # SECURITY: a photo is CONSISTENTLY planar -> high flat-fraction among
    # depth-valid frames. A live person (still or moving) yields ~0% flat.
    # nodata frames carry no spoof signal, so they are excluded from the ratio.
    depth_valid = len(live_votes) + flat
    spoof_frac = flat / max(1, depth_valid)
    top = tally.most_common(1)[0] if tally else ("UNKNOWN", 0)
    if depth_valid >= 3 and spoof_frac > 0.30:
        print(f"[verify] REJECTED - possible SPOOF: {spoof_frac*100:.0f}% of depth-valid frames "
              f"were planar/flat ({flat}/{depth_valid}). A live face gives ~0%.")
    elif top[0] != "UNKNOWN" and top[1] >= max(3, len(live_votes)//2):
        print(f"[verify] RECOGNIZED: {top[0]}  ({top[1]}/{len(live_votes)} live frames, "
              f"spoof={spoof_frac*100:.0f}%, best={best_overall:.3f})")
    else:
        print(f"[verify] NOT recognized. "
              f"{'Liveness failed on all frames.' if not live_votes else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("enroll"); e.add_argument("--name", required=True)
    e.add_argument("--count", type=int, default=25); e.add_argument("--timeout", type=float, default=120)
    e.set_defaults(fn=cmd_enroll)
    v = sub.add_parser("verify"); v.add_argument("--threshold", type=float, default=0.45)
    v.add_argument("--seconds", type=float, default=6); v.set_defaults(fn=cmd_verify)
    args = ap.parse_args(); args.fn(args)
