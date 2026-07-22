import time, numpy as np, face_auth as fa
from collections import Counter
def main(seconds=8):
    det = fa.Detector()
    rclpy, Grab = fa.make_grabber(); rclpy.init(); node = Grab()
    t0=time.time(); rows=[]
    while time.time()-t0 < seconds and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.color is None or node.depth is None: continue
        r = det.detect_align(node.color.copy())
        if r is None: continue
        _, bbox, score = r
        if score < 0.7 or bbox[2] < 90: continue
        live, info = fa.depth_liveness(node.depth, bbox)
        rows.append((live, info.get("range_mm"), info.get("resid_mm"), info.get("valid"), info.get("kind")))
    node.destroy_node(); rclpy.shutdown()
    if not rows: print("DIAG no qualifying frames"); return
    n=len(rows)
    kinds=Counter(x[4] for x in rows)
    rngs=[x[1] for x in rows if x[1] is not None]
    res =[x[2] for x in rows if x[2] is not None]
    print(f"DIAG frames={n} kinds={dict(kinds)}")
    if rngs: print(f"  range_mm p05/p50/p95 = {np.percentile(rngs,5):.1f}/{np.percentile(rngs,50):.1f}/{np.percentile(rngs,95):.1f} (min={min(rngs):.1f})")
    if res:  print(f"  resid_mm p05/p50/p95 = {np.percentile(res,5):.1f}/{np.percentile(res,50):.1f}/{np.percentile(res,95):.1f} (min={min(res):.1f})")
if __name__=="__main__": main()
