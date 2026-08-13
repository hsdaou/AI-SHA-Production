import time, numpy as np, face_auth as fa
from collections import Counter
det = fa.Detector()
rclpy, Grab = fa.make_grabber(); rclpy.init(); node = Grab()
t0=time.time(); kinds=Counter(); rows=[]
while time.time()-t0 < 25 and rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.05)
    if node.color is None or node.depth is None: continue
    r = det.detect_align(node.color.copy())
    if r is None: continue
    _, bbox, score = r
    if score < 0.7 or bbox[2] < 90: continue
    live, info = fa.depth_liveness(node.depth, bbox)
    kinds[("LIVE" if live else info.get("kind","?"))] += 1
    x,y,w,h = bbox
    roi = node.depth[max(0,y):y+h, max(0,x):x+w]
    valid = roi[(roi>200)&(roi<1200)]
    frac = valid.size/max(1,roi.size)
    rows.append((frac, float(np.median(valid)) if valid.size else -1, w))
node.destroy_node(); rclpy.shutdown()
print("RESULT kinds:", dict(kinds))
if rows:
    import statistics as st
    print(f"RESULT valid_frac med={st.median([r[0] for r in rows]):.2f} (need >= {fa.LIVENESS_MIN_VALID})")
    print(f"RESULT resid med={st.median([r[3] for r in rows]):.1f} thr={fa.LIVENESS_MIN_RESIDMM}") if False else None; print(f"RESULT dist med={st.median([r[1] for r in rows if r[1]>0]):.0f}mm  facewidth={st.median([r[2] for r in rows]):.0f}px")
