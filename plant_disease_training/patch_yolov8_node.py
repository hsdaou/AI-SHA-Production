#!/usr/bin/env python3
"""
Patches yolov8_node.py in-place to add PlantDiseaseEngine integration.
Idempotent: skips each patch if the marker string is already present.
Run on the Jetson as:
    python3 patch_yolov8_node.py
"""

import shutil
from pathlib import Path

TARGET = Path('~/robot_ws/src/yolov8_ros/yolov8_ros/yolov8_node.py').expanduser()
BACKUP = TARGET.with_suffix('.py.bak')

# ── Read original ──────────────────────────────────────────────────────────────
src = TARGET.read_text()

# ── Backup ────────────────────────────────────────────────────────────────────
shutil.copy2(TARGET, BACKUP)
print(f"Backup written to {BACKUP}")

patches_applied = 0


def patch(src: str, anchor: str, insertion: str, after: bool = True,
          idempotent_check: str = None) -> str:
    """Insert `insertion` immediately before/after the first occurrence of `anchor`."""
    global patches_applied
    check = idempotent_check or insertion.split('\n')[1]  # first non-empty line
    if check.strip() in src:
        print(f"  [SKIP] already applied: {repr(check.strip()[:60])}")
        return src
    if anchor not in src:
        print(f"  [WARN] anchor not found: {repr(anchor[:60])}")
        return src
    idx = src.index(anchor)
    if after:
        idx += len(anchor)
    src = src[:idx] + insertion + src[idx:]
    patches_applied += 1
    print(f"  [OK]   applied: {repr(check.strip()[:60])}")
    return src


# ══════════════════════════════════════════════════════════════════════════════
# Patch 1 — import PlantDiseaseEngine (after `import json`)
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor='import json\n',
    insertion="""
# ── Plant disease classifier ──────────────────────────────────────────────────
try:
    from yolov8_ros.plant_disease_engine import PlantDiseaseEngine
    _DISEASE_ENGINE_AVAILABLE = True
except ImportError:
    _DISEASE_ENGINE_AVAILABLE = False
""",
    idempotent_check='_DISEASE_ENGINE_AVAILABLE'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 2 — add disease_label / disease_confidence fields to DetectionResult
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor='    x2: float = 0\n    y2: float = 0\n',
    insertion="""    disease_label: str = ''
    disease_confidence: float = 0.0
""",
    idempotent_check='disease_label: str'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 3 — declare disease parameters (after depth_sample_ratio)
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor="self.declare_parameter('depth_sample_ratio', 0.5)\n",
    insertion="""        # Disease classifier parameters
        self.declare_parameter('enable_disease_classifier', True)
        self.declare_parameter('disease_engine_path',
                               '~/plant_disease_models/plant_disease_classifier.engine')
        self.declare_parameter('disease_class_mapping_path',
                               '~/plant_disease_models/class_mapping.json')
        self.declare_parameter('disease_confidence_threshold', 0.60)
        self.declare_parameter('disease_trigger_classes',
                               ['potted plant', 'banana', 'apple', 'orange',
                                'broccoli', 'carrot'])
        self.declare_parameter('disease_roi_padding', 0.10)
""",
    idempotent_check='enable_disease_classifier'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 4 — load disease parameters (after depth_sample_ratio is loaded)
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor="self.depth_sample_ratio = self.get_parameter('depth_sample_ratio').value\n",
    insertion="""        # Disease classifier config
        self.enable_disease = self.get_parameter('enable_disease_classifier').value
        _disease_engine_path = os.path.expanduser(
            self.get_parameter('disease_engine_path').value)
        _disease_mapping_path = os.path.expanduser(
            self.get_parameter('disease_class_mapping_path').value)
        self.disease_conf_threshold = self.get_parameter(
            'disease_confidence_threshold').value
        self.disease_trigger_classes = set(
            self.get_parameter('disease_trigger_classes').value)
        self.disease_roi_padding = self.get_parameter('disease_roi_padding').value
        self.disease_timing: 'deque' = deque(maxlen=30)
""",
    idempotent_check='self.enable_disease'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 5 — initialise disease engine (after YOLO warmup / torch.cuda.synchronize)
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor='torch.cuda.synchronize()\n        self.get_logger().info("YOLO ready")\n',
    insertion="""
        # ── Disease classifier ───────────────────────────────────────────────
        self.disease_engine = None
        if self.enable_disease and _DISEASE_ENGINE_AVAILABLE:
            try:
                self.disease_engine = PlantDiseaseEngine(
                    engine_path=_disease_engine_path,
                    class_mapping_path=_disease_mapping_path,
                    confidence_threshold=self.disease_conf_threshold,
                )
                self.get_logger().info(
                    f"Plant disease classifier ready: {_disease_engine_path}")
            except Exception as _e:
                self.get_logger().warn(
                    f"Disease classifier failed to load ({_e}). Running without it.")
        elif not _DISEASE_ENGINE_AVAILABLE and self.enable_disease:
            self.get_logger().warn(
                "plant_disease_engine.py not found on PYTHONPATH. "
                "Disease classification disabled.")
""",
    idempotent_check='self.disease_engine = None'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 6 — add disease publisher (after stats_pub)
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor="self.stats_pub = self.create_publisher(String, '/detection/stats', reliable_qos)\n",
    insertion="""        self.disease_pub = self.create_publisher(
            String, '/detection/disease_simple', reliable_qos)
""",
    idempotent_check='disease_simple'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 7 — log the disease topic in startup banner
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor='self.get_logger().info("  /detection/closest_object")\n',
    insertion="""        self.get_logger().info("  /detection/disease_simple")
""",
    idempotent_check='/detection/disease_simple'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 8 — run disease classification inside detect_objects
#           (before self.timing_stats['yolo'].append(...))
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor="        self.timing_stats['yolo'].append(time.perf_counter() - t0)\n        return results\n",
    insertion="""        # ── Disease classification on plant / food ROIs ───────────────────
        if self.disease_engine is not None and results:
            plant_dets = [r for r in results
                          if r.class_id in self.disease_trigger_classes
                          and (r.x2 - r.x1) >= 32 and (r.y2 - r.y1) >= 32]
            if plant_dets:
                img_h, img_w = frame.shape[:2]
                rois = []
                for det in plant_dets:
                    pad_x = int((det.x2 - det.x1) * self.disease_roi_padding)
                    pad_y = int((det.y2 - det.y1) * self.disease_roi_padding)
                    x1 = max(0, int(det.x1) - pad_x)
                    y1 = max(0, int(det.y1) - pad_y)
                    x2 = min(img_w, int(det.x2) + pad_x)
                    y2 = min(img_h, int(det.y2) + pad_y)
                    roi = frame[y1:y2, x1:x2]
                    rois.append(self.disease_engine.preprocess(roi))
                t_d = time.perf_counter()
                disease_results = self.disease_engine.infer_batch(rois)
                self.disease_timing.append(time.perf_counter() - t_d)
                for det, dr in zip(plant_dets, disease_results):
                    if not dr['below_threshold']:
                        det.disease_label = f"{dr['species']}: {dr['disease']}"
                        det.disease_confidence = dr['confidence']
        # ────────────────────────────────────────────────────────────────────
""",
    idempotent_check='plant_dets = [r for r in results'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 9 — publish disease results in publish_detections
#           (after ocr_simple_pub.publish)
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor="        self.ocr_simple_pub.publish(String(data=make_simple_msg(ocr)))\n",
    insertion="""
        # Disease detections
        diseased = [o for o in objects if o.disease_label]
        disease_items = []
        for det in diseased:
            disease_items.append({
                'class': det.class_id,
                'disease': det.disease_label,
                'conf': round(det.disease_confidence, 2),
                'bbox': [round(det.x1, 1), round(det.y1, 1),
                         round(det.x2, 1), round(det.y2, 1)],
            })
            if det.depth is not None:
                disease_items[-1]['depth'] = round(det.depth, 2)
        self.disease_pub.publish(String(
            data=json.dumps(disease_items, separators=(',', ':'))))
""",
    idempotent_check='disease_items = []'
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 10 — add disease timing to stats dict
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor="'ocr': round(np.mean(self.timing_stats['ocr']) * 1000, 1) if self.timing_stats['ocr'] else 0,\n",
    insertion="""                'disease': round(np.mean(self.disease_timing) * 1000, 1)
                         if self.disease_timing else 0,
""",
    idempotent_check="'disease': round(np.mean(self.disease_timing)"
)

# ══════════════════════════════════════════════════════════════════════════════
# Patch 11 — draw disease labels in visualize()
#            after "for det in objects:" draw loop, before faces loop
# ══════════════════════════════════════════════════════════════════════════════
src = patch(src,
    anchor='        for det in objects:\n            draw_detection(canvas, det, (0, 255, 0))  # Green\n',
    insertion="""        # Draw disease overlays on top of YOLO boxes
        for det in objects:
            if det.disease_label:
                cx, cy = int(det.cx), int(det.y1) - 22
                label = f"[{det.disease_label}  {det.disease_confidence:.0%}]"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
                cx = max(0, min(cx - lw // 2, canvas.shape[1] - lw - 4))
                cy = max(lh + 4, cy)
                # Coral background pill
                cv2.rectangle(canvas, (cx - 2, cy - lh - 3), (cx + lw + 2, cy + 3),
                              (32, 80, 220), -1)
                cv2.rectangle(canvas, (cx - 2, cy - lh - 3), (cx + lw + 2, cy + 3),
                              (255, 255, 255), 1)
                cv2.putText(canvas, label, (cx, cy), font, 0.45, (255, 255, 255), 1)
""",
    idempotent_check='det.disease_label'
)

# ══════════════════════════════════════════════════════════════════════════════
# Write result
# ══════════════════════════════════════════════════════════════════════════════
TARGET.write_text(src)
print(f"\n{patches_applied} patch(es) applied → {TARGET}")
print("Original backed up at:", BACKUP)
