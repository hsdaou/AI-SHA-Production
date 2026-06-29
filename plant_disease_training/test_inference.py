#!/usr/bin/env python3
"""
Plant Disease Classifier — workstation visual test.

Tests the full ROI→TRT→result pipeline using held-out test images.
No ROS, no camera required.

Usage:
    source /home/robot-wst/plant_disease_env/bin/activate
    cd /home/robot-wst/plant_disease_training
    python test_inference.py                          # interactive gallery
    python test_inference.py --mode accuracy          # accuracy report only
    python test_inference.py --mode pipeline          # simulate YOLO ROI pipeline
    python test_inference.py --mode benchmark         # latency numbers
    python test_inference.py --save_dir /tmp/results  # save annotated images
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
ENGINE   = ROOT / 'checkpoints/plant_disease_classifier_ws.engine'
MAPPING  = ROOT / 'checkpoints/class_mapping.json'
TEST_DIR = ROOT / 'data/test'

sys.path.insert(0, str(ROOT))
from plant_disease_engine import PlantDiseaseEngine

# ── Colors (BGR) ───────────────────────────────────────────────────────────────
GREEN  = (50, 200, 50)
RED    = (50, 50, 220)
GOLD   = (0, 180, 255)
BLUE   = (220, 80, 32)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GRAY   = (120, 120, 120)


# ══════════════════════════════════════════════════════════════════════════════
# Mode 1 — Accuracy sweep across the full test set
# ══════════════════════════════════════════════════════════════════════════════

def run_accuracy(engine: PlantDiseaseEngine, max_per_class: int = 50):
    print("\n" + "═" * 64)
    print("  ACCURACY TEST — held-out test split")
    print("═" * 64)

    # Build list of (image_path, true_class_folder_name)
    samples = []
    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        imgs = sorted(class_dir.glob('*.jpg')) + sorted(class_dir.glob('*.JPG')) + \
               sorted(class_dir.glob('*.jpeg')) + sorted(class_dir.glob('*.png'))
        chosen = imgs[:max_per_class]
        for p in chosen:
            samples.append((p, class_dir.name))

    random.shuffle(samples)

    # Run inference in batches of 8
    correct = 0
    per_class_correct = {}
    per_class_total   = {}
    batch_size = 8
    t_start = time.perf_counter()

    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        preprocessed = []
        for path, _ in chunk:
            img = cv2.imread(str(path))
            if img is None:
                preprocessed.append(None)
            else:
                preprocessed.append(engine.preprocess(img))

        results = engine.infer_batch(preprocessed)

        for (path, true_name), result in zip(chunk, results):
            predicted = result['class_name']
            is_correct = (predicted == true_name)
            correct += int(is_correct)
            per_class_correct[true_name] = per_class_correct.get(true_name, 0) + int(is_correct)
            per_class_total[true_name]   = per_class_total.get(true_name, 0) + 1

        done = min(i + batch_size, len(samples))
        print(f"\r  {done:5d}/{len(samples)} images  "
              f"acc={correct/done:.1%}", end='', flush=True)

    elapsed = time.perf_counter() - t_start
    overall = correct / len(samples)

    print(f"\r  {len(samples)}/{len(samples)} images  "
          f"acc={overall:.1%}  ({elapsed:.1f}s, "
          f"{len(samples)/elapsed:.0f} img/s)")

    print(f"\n  Overall accuracy: {overall:.4f}  ({overall*100:.2f}%)")
    print()

    # Per-class breakdown
    print(f"  {'Class':<46}  {'Acc':>6}  {'Corr':>5}/{len(samples)//len(per_class_total):>3}")
    print("  " + "─" * 60)
    low = []
    for cls in sorted(per_class_correct):
        c, t = per_class_correct[cls], per_class_total[cls]
        acc = c / t if t else 0
        flag = "  ← LOW" if acc < 0.85 else ""
        print(f"  {cls:<46}  {acc:>5.1%}  {c:>4}/{t:<3}{flag}")
        if acc < 0.85:
            low.append((cls, acc))

    if low:
        print(f"\n  WARNING: {len(low)} class(es) below 85% accuracy:")
        for cls, acc in low:
            print(f"    {cls}: {acc:.1%}")
    else:
        print("\n  All classes ≥ 85%. ✓")

    print("═" * 64)
    return overall


# ══════════════════════════════════════════════════════════════════════════════
# Mode 2 — Interactive image gallery (one image per class)
# ══════════════════════════════════════════════════════════════════════════════

def run_gallery(engine: PlantDiseaseEngine, n_per_class: int = 3,
                save_dir: Path = None):
    print("\n═" * 32)
    print("  GALLERY TEST — sample images from each class")
    print("  Press any key to advance, Q/ESC to quit")
    print("═" * 32)

    classes = sorted(TEST_DIR.iterdir())
    correct_count = 0
    total_count   = 0

    for class_dir in classes:
        if not class_dir.is_dir():
            continue
        imgs = sorted(class_dir.glob('*.jpg')) + sorted(class_dir.glob('*.JPG')) + \
               sorted(class_dir.glob('*.jpeg')) + sorted(class_dir.glob('*.png'))
        if not imgs:
            continue

        chosen = random.sample(imgs, min(n_per_class, len(imgs)))

        for img_path in chosen:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            prep = engine.preprocess(img_bgr)
            t0 = time.perf_counter()
            result = engine.infer_batch([prep])[0]
            latency_ms = (time.perf_counter() - t0) * 1000

            predicted  = result['class_name']
            true_name  = class_dir.name
            is_correct = (predicted == true_name)
            correct_count += int(is_correct)
            total_count   += 1

            canvas = _make_result_card(img_bgr, result, true_name, latency_ms)

            if save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                out = save_dir / f"{true_name}_{img_path.stem}_pred.jpg"
                cv2.imwrite(str(out), canvas)
                print(f"  Saved {out.name}")

            try:
                cv2.imshow("Plant Disease Test  [Q=quit, any key=next]", canvas)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord('q'), 27):
                    cv2.destroyAllWindows()
                    print(f"\n  Exited early. Accuracy so far: {correct_count/total_count:.1%}")
                    return
            except cv2.error:
                # No display available (headless) — just save
                if not save_dir:
                    print(f"  No display. Use --save_dir to save images. Result: {result['class_name']}")

    cv2.destroyAllWindows()
    print(f"\n  Gallery done. Final accuracy: {correct_count/total_count:.1%}")


def _make_result_card(img_bgr: np.ndarray, result: dict,
                      true_name: str, latency_ms: float) -> np.ndarray:
    """Render a 640×480 annotated result card."""
    display = cv2.resize(img_bgr, (400, 400))
    canvas = np.zeros((480, 700, 3), dtype=np.uint8)
    canvas[:, :] = (30, 30, 30)
    canvas[:400, :400] = display

    is_correct = (result['class_name'] == true_name)
    status_color = GREEN if is_correct else RED
    status_text  = "CORRECT ✓" if is_correct else "WRONG ✗"

    font   = cv2.FONT_HERSHEY_SIMPLEX
    panel_x = 415

    # Status banner
    cv2.rectangle(canvas, (panel_x - 5, 5), (695, 40), status_color, -1)
    cv2.putText(canvas, status_text, (panel_x, 30), font, 0.8, WHITE, 2)

    y = 60

    def row(label, value, color=WHITE, scale=0.45):
        nonlocal y
        cv2.putText(canvas, label, (panel_x, y), font, 0.38, GRAY, 1)
        y += 16
        cv2.putText(canvas, str(value)[:32], (panel_x, y), font, scale, color, 1)
        y += 22

    row("TRUE CLASS",   true_name.replace('_', ' ')[:28])
    row("PREDICTED",    result['class_name'].replace('_', ' ')[:28],
        GREEN if is_correct else RED)
    row("SPECIES",      result['species'])
    row("DISEASE",      result['disease'],
        GREEN if result['is_healthy'] else GOLD)
    row("CONFIDENCE",   f"{result['confidence']:.2%}",
        GREEN if result['confidence'] >= 0.60 else RED)
    row("LATENCY",      f"{latency_ms:.1f} ms")

    y += 5
    cv2.putText(canvas, "TOP-3", (panel_x, y), font, 0.38, GRAY, 1)
    y += 16
    for cls_name, conf in result['top3']:
        bar_w = int(conf * 260)
        cv2.rectangle(canvas, (panel_x, y - 10), (panel_x + bar_w, y), BLUE, -1)
        short = cls_name.split('___')[-1].replace('_', ' ')[:24]
        cv2.putText(canvas, f"{short}  {conf:.1%}", (panel_x + 2, y - 1),
                    font, 0.37, WHITE, 1)
        y += 18

    # Bottom info bar
    cv2.rectangle(canvas, (0, 450), (700, 480), (20, 20, 20), -1)
    cv2.putText(canvas, "PlantVillage Classifier — TensorRT FP16 (RTX 5080)",
                (10, 470), font, 0.38, GRAY, 1)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Mode 3 — Simulate the YOLO ROI pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline_sim(engine: PlantDiseaseEngine, save_dir: Path = None):
    """
    Simulates what the ROS node does:
      1. A 'scene' image containing multiple leaf crops tiled together
      2. Simulated YOLO bounding boxes around each crop
      3. ROI extraction → batch disease classification
      4. Annotated output with per-box disease labels
    """
    print("\n═" * 32)
    print("  PIPELINE SIMULATION — multi-ROI batched inference")
    print("  (Simulates YOLO → ROI crop → disease classifier)")
    print("═" * 32)

    # Pick one image from each of 6 different classes for a synthetic scene
    sample_classes = random.sample(sorted([d for d in TEST_DIR.iterdir() if d.is_dir()]), 6)
    scene_crops = []
    true_labels  = []
    for cls_dir in sample_classes:
        imgs = sorted(cls_dir.glob('*.jpg')) + sorted(cls_dir.glob('*.JPG'))
        if not imgs:
            continue
        img = cv2.imread(str(random.choice(imgs)))
        if img is None:
            continue
        scene_crops.append(cv2.resize(img, (300, 300)))
        true_labels.append(cls_dir.name)

    if not scene_crops:
        print("  No images found.")
        return

    # Tile into a 3×2 scene image
    cols = 3
    rows = (len(scene_crops) + cols - 1) // cols
    # Pad to full grid
    while len(scene_crops) < rows * cols:
        scene_crops.append(np.zeros((300, 300, 3), dtype=np.uint8))
        true_labels.append('(empty)')

    rows_imgs = [np.hstack(scene_crops[i*cols:(i+1)*cols]) for i in range(rows)]
    scene = np.vstack(rows_imgs)

    # Simulated YOLO bboxes (one per tile)
    fake_bboxes = []
    pad = 15
    for idx in range(len(true_labels)):
        if true_labels[idx] == '(empty)':
            continue
        col = idx % cols
        row = idx // cols
        x1, y1 = col * 300 + pad, row * 300 + pad
        x2, y2 = x1 + 300 - 2*pad, y1 + 300 - 2*pad
        fake_bboxes.append((x1, y1, x2, y2, true_labels[idx]))

    # Extract ROIs and preprocess (same logic as the ROS node)
    rois = []
    for x1, y1, x2, y2, _ in fake_bboxes:
        roi_pad = 5
        rx1 = max(0, x1 - roi_pad)
        ry1 = max(0, y1 - roi_pad)
        rx2 = min(scene.shape[1], x2 + roi_pad)
        ry2 = min(scene.shape[0], y2 + roi_pad)
        roi = scene[ry1:ry2, rx1:rx2]
        rois.append(engine.preprocess(roi))

    # Batch inference (all 6 ROIs in one call)
    t0 = time.perf_counter()
    results = engine.infer_batch(rois)
    batch_ms = (time.perf_counter() - t0) * 1000
    print(f"  Batch of {len(rois)} ROIs processed in {batch_ms:.1f} ms  "
          f"({batch_ms/len(rois):.1f} ms/ROI)")

    # Draw annotated scene
    annotated = scene.copy()
    correct = 0
    for (x1, y1, x2, y2, true_name), result in zip(fake_bboxes, results):
        is_correct = (result['class_name'] == true_name)
        correct += int(is_correct)
        box_color = GREEN if is_correct else RED

        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 3)

        label_text = f"{result['species']}: {result['disease']}"
        conf_text  = f"{result['confidence']:.0%}"
        full_label = f"{label_text}  {conf_text}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        (lw, lh), _ = cv2.getTextSize(full_label, font, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 6, y1), box_color, -1)
        cv2.putText(annotated, full_label, (x1 + 3, y1 - 4), font, 0.5, WHITE, 1)

        # True class label at bottom of box
        cv2.putText(annotated, f"True: {true_name.split('___')[-1][:20]}",
                    (x1 + 4, y2 - 6), font, 0.4, WHITE, 1)

    # Stats bar at bottom
    stats_h = 36
    bar = np.zeros((stats_h, annotated.shape[1], 3), dtype=np.uint8)
    bar[:] = (20, 20, 20)
    summary = (f"Batch={len(rois)} ROIs  |  {batch_ms:.1f}ms total  |  "
               f"{batch_ms/len(rois):.1f}ms/ROI  |  "
               f"Correct: {correct}/{len(fake_bboxes)}")
    cv2.putText(bar, summary, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GOLD, 1)
    final = np.vstack([annotated, bar])

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / 'pipeline_sim.jpg'
        cv2.imwrite(str(out), final)
        print(f"  Saved {out}")

    try:
        cv2.imshow("Pipeline Simulation — Q to close", final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        if not save_dir:
            print("  No display available. Use --save_dir to save the result.")

    return correct / len(fake_bboxes)


# ══════════════════════════════════════════════════════════════════════════════
# Mode 4 — Benchmark latency
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(engine: PlantDiseaseEngine):
    print("\n═" * 32)
    print("  LATENCY BENCHMARK — RTX 5080 TensorRT FP16")
    print("═" * 32)
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    prep  = engine.preprocess(dummy)

    for bs in [1, 2, 4, 8]:
        batch = [prep] * bs
        # Warmup
        for _ in range(20):
            engine.infer_batch(batch)
        # Timed
        times = []
        for _ in range(300):
            t0 = time.perf_counter()
            engine.infer_batch(batch)
            times.append((time.perf_counter() - t0) * 1000)
        arr = np.array(times)
        print(f"  batch={bs:<2}  mean={arr.mean():.2f}ms  "
              f"p50={np.percentile(arr,50):.2f}ms  "
              f"p95={np.percentile(arr,95):.2f}ms  "
              f"p99={np.percentile(arr,99):.2f}ms  "
              f"→ {1000/arr.mean()*bs:.0f} img/s")
    print("═" * 32)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PlantDiseaseEngine workstation test')
    parser.add_argument('--mode', choices=['gallery', 'accuracy', 'pipeline', 'benchmark', 'all'],
                        default='gallery')
    parser.add_argument('--engine',   default=str(ENGINE))
    parser.add_argument('--mapping',  default=str(MAPPING))
    parser.add_argument('--save_dir', default=None, help='Save annotated images here')
    parser.add_argument('--n_per_class', type=int, default=3,
                        help='Images per class in gallery mode')
    parser.add_argument('--max_per_class', type=int, default=100,
                        help='Max images per class in accuracy mode')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    save_dir = Path(args.save_dir) if args.save_dir else None

    print(f"\nEngine  : {args.engine}")
    print(f"Mapping : {args.mapping}")
    print(f"Test dir: {TEST_DIR}")
    print()

    print("Loading TensorRT engine...")
    engine = PlantDiseaseEngine(
        engine_path=args.engine,
        class_mapping_path=args.mapping,
    )
    print()

    if args.mode in ('accuracy', 'all'):
        run_accuracy(engine, max_per_class=args.max_per_class)

    if args.mode in ('benchmark', 'all'):
        run_benchmark(engine)

    if args.mode in ('pipeline', 'all'):
        run_pipeline_sim(engine, save_dir=save_dir)

    if args.mode in ('gallery', 'all'):
        run_gallery(engine, n_per_class=args.n_per_class, save_dir=save_dir)


if __name__ == '__main__':
    main()
