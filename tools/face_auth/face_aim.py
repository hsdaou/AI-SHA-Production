#!/usr/bin/env python3
"""Live aiming helper: reports whether a face is visible, how big, how far."""
import time, numpy as np, face_auth as fa
det = fa.Detector()
rclpy, Grab = fa.make_grabber(); rclpy.init(); node = Grab()
t0 = time.time(); last = 0
while time.time() - t0 < 25 and rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.05)
    if node.color is None or node.depth is None:
        continue
    if time.time() - last < 2.5:
        continue
    last = time.time()
    r = det.detect_align(node.color.copy())
    if r is None:
        print("  NO FACE detected — aim the camera at your face", flush=True)
        continue
    _, bbox, score = r
    x, y, w, h = bbox
    roi = node.depth[max(0, y):y+h, max(0, x):x+w]
    valid = roi[(roi > 200) & (roi < 1200)]
    dist = float(np.median(valid)) if valid.size else -1
    hint = "GOOD" if (w >= 120 and 430 <= dist <= 650) else (
        "too close" if 0 < dist < 350 else "too far / small" if dist > 700 else "adjust")
    print(f"  FACE: width={w}px conf={score:.2f} dist={dist:.0f}mm -> {hint}", flush=True)
node.destroy_node(); rclpy.shutdown()
