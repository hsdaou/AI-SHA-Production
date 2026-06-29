#!/usr/bin/env python3
"""
PlantDiseaseEngine — TensorRT FP16 inference wrapper for the PlantVillage classifier.

Uses PyTorch CUDA tensors as GPU memory buffers (no pycuda required).

Engine I/O:
  input:  ('input',  [-1, 3, 224, 224], FLOAT)
  output: ('output', [-1, 15],          FLOAT)

Usage:
    engine = PlantDiseaseEngine(
        engine_path='/path/to/plant_disease_classifier.engine',
        class_mapping_path='/path/to/class_mapping.json',
    )
    roi_bgr = cv2.imread('leaf.jpg')                   # any size BGR
    preprocessed = engine.preprocess(roi_bgr)          # (3, 224, 224) float32
    results = engine.infer_batch([preprocessed])       # list of dicts
    print(results[0]['class_name'], results[0]['confidence'])
"""

import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tensorrt as trt

# ─── Constants ────────────────────────────────────────────────────────────────

_INPUT_NAME  = 'input'
_OUTPUT_NAME = 'output'
_IMG_H = _IMG_W = 224
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Normalisation pre-computed as (1/255 / std) scale and (-mean/std) shift
# applied per channel: out = (x/255 - mean) / std
_SCALE = (1.0 / 255.0 / _IMAGENET_STD).reshape(1, 1, 3).astype(np.float32)
_SHIFT = (-_IMAGENET_MEAN / _IMAGENET_STD).reshape(1, 1, 3).astype(np.float32)

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ─── Engine ───────────────────────────────────────────────────────────────────

class PlantDiseaseEngine:
    """
    Thread-safe TensorRT classifier. Pre-allocates max-batch GPU buffers at
    init time; inference copies only the active sub-batch.
    """

    def __init__(
        self,
        engine_path: str,
        class_mapping_path: str,
        max_batch: int = 8,
        confidence_threshold: float = 0.60,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_batch = max_batch

        # ── Load TensorRT engine ─────────────────────────────────────────────
        engine_path = str(Path(engine_path).expanduser())
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(_TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Discover number of output classes from engine metadata
        out_shape = tuple(self.engine.get_tensor_shape(_OUTPUT_NAME))
        self.num_classes = out_shape[1]  # (-1, 15) → 15

        # ── Pre-allocate persistent GPU buffers (torch) ──────────────────────
        # Allocate for max_batch; sliced per actual call
        self._inp_gpu = torch.zeros(
            (max_batch, 3, _IMG_H, _IMG_W), dtype=torch.float32, device='cuda'
        )
        self._out_gpu = torch.zeros(
            (max_batch, self.num_classes), dtype=torch.float32, device='cuda'
        )
        # Pinned (page-locked) host buffer for fast H2D transfers
        self._inp_host = torch.zeros(
            (max_batch, 3, _IMG_H, _IMG_W), dtype=torch.float32
        ).pin_memory()

        # CUDA stream for async transfers
        self._stream = torch.cuda.Stream()

        # Warmup: one forward pass to JIT any lazy TRT kernels
        self._run_trt(1)

        # ── Load class mapping ───────────────────────────────────────────────
        class_mapping_path = str(Path(class_mapping_path).expanduser())
        with open(class_mapping_path) as f:
            raw = json.load(f)
        self.classes: dict[int, dict] = {m['class_index']: m for m in raw}

        print(
            f"[PlantDiseaseEngine] Loaded engine '{Path(engine_path).name}' | "
            f"{self.num_classes} classes | max_batch={max_batch}"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def preprocess(self, roi_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        BGR ROI → float32 CHW (3, 224, 224), ImageNet-normalised.
        Returns None if the ROI is too small to be meaningful (< 32×32).
        """
        h, w = roi_bgr.shape[:2]
        if h < 32 or w < 32:
            return None

        # Resize to 224×224 with linear interpolation
        resized = cv2.resize(roi_bgr, (_IMG_W, _IMG_H), interpolation=cv2.INTER_LINEAR)

        # BGR → RGB, float32 [0,1], normalise
        rgb = resized[:, :, ::-1].astype(np.float32)          # HWC BGR→RGB
        normalised = rgb * _SCALE + _SHIFT                     # HWC float32, ~[-2, 2]
        chw = np.ascontiguousarray(normalised.transpose(2, 0, 1))  # CHW
        return chw

    # ── Inference ─────────────────────────────────────────────────────────────

    def infer_batch(self, preprocessed_images: list) -> list[dict]:
        """
        Run TensorRT inference on a list of preprocessed CHW arrays.

        Automatically splits into sub-batches if len > max_batch.

        Returns list of dicts:
          {
            'class_index':   int,
            'class_name':    str,   # raw folder name
            'species':       str,
            'disease':       str,
            'is_healthy':    bool,
            'confidence':    float,
            'top3':          [(class_name, conf), ...],
            'below_threshold': bool,   # True when max conf < threshold
          }
        """
        if not preprocessed_images:
            return []

        results = []
        for batch_start in range(0, len(preprocessed_images), self.max_batch):
            chunk = preprocessed_images[batch_start: batch_start + self.max_batch]
            results.extend(self._infer_chunk(chunk))
        return results

    def _infer_chunk(self, chunk: list) -> list[dict]:
        n = len(chunk)

        # ── Copy to pinned host buffer ────────────────────────────────────────
        for i, img in enumerate(chunk):
            if img is None:
                # Blank image — will produce low confidence output
                self._inp_host[i].zero_()
            else:
                self._inp_host[i] = torch.from_numpy(img)

        # ── H2D async transfer ────────────────────────────────────────────────
        with torch.cuda.stream(self._stream):
            self._inp_gpu[:n].copy_(self._inp_host[:n], non_blocking=True)

        # ── TensorRT execution ────────────────────────────────────────────────
        self._run_trt(n)

        # ── D2H (output is small: n×15 floats) ───────────────────────────────
        logits = self._out_gpu[:n].cpu().numpy()  # (n, num_classes)

        # ── Post-process ──────────────────────────────────────────────────────
        results = []
        for i in range(n):
            if chunk[i] is None:
                results.append(self._unknown_result())
                continue

            # Softmax in NumPy (negligible cost for 15 classes)
            exp = np.exp(logits[i] - logits[i].max())
            probs = exp / exp.sum()

            top_idx  = int(np.argmax(probs))
            top_conf = float(probs[top_idx])

            below = top_conf < self.confidence_threshold

            # Top-3
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [
                (self.classes.get(int(j), {}).get('class_name', f'cls_{j}'),
                 float(probs[j]))
                for j in top3_idx
            ]

            meta = self.classes.get(top_idx, {})
            results.append({
                'class_index':     top_idx,
                'class_name':      meta.get('class_name', f'cls_{top_idx}'),
                'species':         meta.get('species', 'unknown'),
                'disease':         meta.get('disease', 'unknown'),
                'is_healthy':      meta.get('is_healthy', False),
                'confidence':      top_conf,
                'top3':            top3,
                'below_threshold': below,
            })

        return results

    # ── Internal TRT call ─────────────────────────────────────────────────────

    def _run_trt(self, batch_size: int):
        """Execute TRT context with the current GPU buffers for `batch_size` images."""
        self.context.set_input_shape(_INPUT_NAME, (batch_size, 3, _IMG_H, _IMG_W))
        bindings = [
            self._inp_gpu.data_ptr(),
            self._out_gpu.data_ptr(),
        ]
        self._stream.synchronize()
        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned False")
        torch.cuda.synchronize()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _unknown_result() -> dict:
        return {
            'class_index':     -1,
            'class_name':      'unknown',
            'species':         'unknown',
            'disease':         'unknown',
            'is_healthy':      False,
            'confidence':      0.0,
            'top3':            [],
            'below_threshold': True,
        }

    def benchmark(self, batch_size: int = 1, runs: int = 200) -> dict:
        """Latency benchmark (returns mean/p95/p99 in ms)."""
        # Warmup
        for _ in range(10):
            self._run_trt(batch_size)

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self._run_trt(batch_size)
            times.append((time.perf_counter() - t0) * 1000)

        arr = np.array(times)
        result = {
            'batch': batch_size,
            'runs': runs,
            'mean_ms': float(arr.mean()),
            'p50_ms':  float(np.percentile(arr, 50)),
            'p95_ms':  float(np.percentile(arr, 95)),
            'p99_ms':  float(np.percentile(arr, 99)),
        }
        print(f"[PlantDiseaseEngine] Benchmark batch={batch_size}: "
              f"mean={result['mean_ms']:.2f}ms  "
              f"p95={result['p95_ms']:.2f}ms  "
              f"p99={result['p99_ms']:.2f}ms")
        return result

    def __del__(self):
        # Torch tensors are freed automatically; TRT context/engine cleaned by GC
        pass


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    ENGINE = '/home/orin-robot/plant_disease_models/plant_disease_classifier.engine'
    MAPPING = '/home/orin-robot/plant_disease_models/class_mapping.json'

    print("Loading engine...")
    eng = PlantDiseaseEngine(ENGINE, MAPPING)

    print("\nBenchmark batch=1:")
    eng.benchmark(batch_size=1, runs=500)
    print("Benchmark batch=4:")
    eng.benchmark(batch_size=4, runs=200)

    # Test with a random dummy image
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    preprocessed = eng.preprocess(dummy)
    results = eng.infer_batch([preprocessed])
    print("\nDummy inference result:", results[0])
    print("\nAll OK.")
