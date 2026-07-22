import time, numpy as np, face_auth as fa, auth_gate as ag
def main(seconds=12):
    det = fa.Detector()
    rclpy, Grab = fa.make_grabber(); rclpy.init(); node = Grab()
    t0=time.time(); ys=[]; none=0; noframe=0
    print("YAW DIAG: turn your head FULLY left, center, FULLY right, repeat.")
    while time.time()-t0 < seconds and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.03)
        if node.color is None: noframe+=1; continue
        y = ag._yaw_proxy(node.color.copy(), det)
        if y is None: none+=1; continue
        ys.append(y)
    node.destroy_node(); rclpy.shutdown()
    if not ys: print("YAW no valid frames"); return
    a=np.array(ys)
    print(f"YAW valid={len(ys)} face_lost={none} noframe={noframe}")
    print(f"  proxy min={a.min():.3f} max={a.max():.3f} SWING={a.max()-a.min():.3f}")
    print(f"  p05={np.percentile(a,5):.3f} p50={np.percentile(a,50):.3f} p95={np.percentile(a,95):.3f}")
if __name__=="__main__": main()
