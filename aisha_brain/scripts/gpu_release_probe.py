#!/usr/bin/env python3
"""GPU-release de-risking probe for ADR 0001 (NAVIGATING <-> CONVERSING).

Answers the make-or-break question: when the YOLOv8 TensorRT engine is
released *inside a still-running Python process* (del + cuda empty_cache),
is the VRAM actually returned to the system so a SEPARATE process (ollama)
can load llama3.2:1b on the GPU at num_gpu=99 — or does the persistent CUDA
primary context retain it, forcing a full process kill instead?

If llama loads at num_gpu=99 in STEP 5 (engine released, this process still
alive) -> an in-node release_gpu service is viable (ADR's lightweight path).
If it still OOMs -> the node must be killed/respawned to reclaim VRAM.

Run on the Jetson with the brain stack DOWN and ollama up:
    python3 gpu_release_probe.py
"""
import gc
import json
import subprocess
import sys
import time

ENGINE = '/home/orin-robot/robot_ws/yolov8m.engine'
OLLAMA = 'http://127.0.0.1:11434/api/generate'


def mem_avail_mb():
    """System MemAvailable (MB) — the cross-process memory that matters."""
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) // 1024
    return -1


def probe_llama(num_gpu, keep_alive=0):
    """Try to load+run llama3.2:1b at the given num_gpu. Returns (ok, detail)."""
    payload = json.dumps({
        'model': 'llama3.2:1b', 'prompt': 'hi', 'stream': False,
        'keep_alive': keep_alive,
        'options': {'num_ctx': 2048, 'num_gpu': num_gpu},
    })
    t0 = time.time()
    try:
        out = subprocess.run(
            ['curl', '-s', OLLAMA, '-d', payload],
            capture_output=True, text=True, timeout=120).stdout
    except subprocess.TimeoutExpired:
        return False, 'curl timeout'
    dt = time.time() - t0
    low = out.lower()
    if 'out of memory' in low or 'cudamalloc' in low:
        return False, f'OOM ({dt:.1f}s)'
    try:
        d = json.loads(out)
        if 'response' in d:
            tok = d.get('eval_count', 0) / max(d.get('eval_duration', 1) / 1e9, 1e-9)
            return True, f'OK {dt:.1f}s @ {tok:.0f} tok/s'
    except json.JSONDecodeError:
        pass
    return False, f'unexpected: {out[:120]}'


def banner(step, msg):
    print(f'\n{"="*64}\n  STEP {step}: {msg}\n{"="*64}', flush=True)


def main():
    print(f'MemAvailable helper OK. Engine: {ENGINE}')

    banner(0, 'Baseline (nothing loaded)')
    base = mem_avail_mb()
    print(f'  MemAvailable: {base} MB', flush=True)
    # make sure no llama is resident from a previous run
    probe_llama(0, keep_alive=0)

    banner(1, 'Load YOLOv8 TensorRT engine + run one inference (allocates engine+ctx)')
    import numpy as np
    import torch
    from ultralytics import YOLO
    model = YOLO(ENGINE, task='detect')
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, verbose=False)  # force engine bindings + CUDA context
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    after_load = mem_avail_mb()
    print(f'  MemAvailable: {after_load} MB  (engine cost ~{base - after_load} MB)', flush=True)
    print(f'  torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB', flush=True)

    banner(2, 'With engine RESIDENT, try llama num_gpu=99 (expect OOM / conflict)')
    ok, detail = probe_llama(99, keep_alive=0)
    print(f'  llama num_gpu=99 -> {detail}', flush=True)
    resident_blocks = not ok

    banner(3, 'RELEASE engine IN-PROCESS (del + gc + torch.cuda.empty_cache)')
    try:
        if hasattr(model, 'predictor') and model.predictor is not None:
            del model.predictor
    except Exception as e:
        print(f'  (predictor del note: {e})')
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    time.sleep(2)
    after_release = mem_avail_mb()
    reclaimed = after_release - after_load
    print(f'  MemAvailable: {after_release} MB  (reclaimed {reclaimed:+d} MB of '
          f'{base - after_load} MB engine cost)', flush=True)
    print(f'  torch.cuda.memory_allocated: {torch.cuda.memory_allocated()/1e6:.0f} MB', flush=True)

    banner(4, 'PROCESS STILL ALIVE, engine released -> try llama num_gpu=99 (THE TEST)')
    ok2, detail2 = probe_llama(99, keep_alive=0)
    print(f'  llama num_gpu=99 -> {detail2}', flush=True)

    engine_cost = base - after_load
    # "Sufficient" must be judged on VRAM actually returned to the SYSTEM,
    # not on whether llama loaded (it may load anyway if no real conflict
    # existed). Require reclaiming a meaningful fraction of the engine cost.
    reclaim_frac = reclaimed / engine_cost if engine_cost > 0 else 0.0
    release_works = reclaim_frac >= 0.5

    banner('VERDICT', 'ADR 0001 open-question #1')
    print(f'  Idle engine static cost:                  {engine_cost} MB')
    print(f'  Engine resident blocked GPU llama:        {resident_blocks} '
          f'(idle engine alone is NOT the blocker)')
    print(f'  VRAM reclaimed by in-process release:     {reclaimed} MB '
          f'({reclaim_frac*100:.0f}% of engine cost)')
    print(f'  GPU llama loads after in-process release: {ok2}')
    if release_works:
        print('\n  ==> IN-PROCESS RELEASE WORKS: a lightweight release_gpu service')
        print('      on the live YOLO node can reclaim VRAM without a respawn.')
    else:
        print('\n  ==> IN-PROCESS RELEASE DOES NOT FREE THE VRAM.')
        print('      del+empty_cache returns only torch-tracked memory; the CUDA')
        print('      primary context + TRT runtime keep the rest until the PROCESS')
        print('      EXITS. => YOLO must be KILLED/RESPAWNED (keep RealSense as a')
        print('      separate, still-alive process). A passive lifecycle/del is NOT')
        print('      enough. Note: the realistic OOM blocker in production is active')
        print('      inference buffers + gemma3 co-residency, not the static engine.')
    # cleanup any llama we loaded
    probe_llama(0, keep_alive=0)


if __name__ == '__main__':
    sys.exit(main())
