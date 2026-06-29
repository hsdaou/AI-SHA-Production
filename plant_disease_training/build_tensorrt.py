"""
Build a TensorRT engine from ONNX on this workstation (RTX 5080 / CUDA 12.8).

NOTE: The resulting .engine file is tied to THIS GPU architecture.
      To run on the Jetson Orin Nano you must rebuild the engine on the Jetson
      using trtexec. The ONNX file is portable.

Usage:
    python build_tensorrt.py \
        --onnx ./checkpoints/plant_disease_classifier_sim.onnx \
        --engine ./checkpoints/plant_disease_classifier_ws.engine \
        [--fp16] [--int8] [--min_batch 1] [--opt_batch 4] [--max_batch 8]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    int8: bool = False,
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 8,
    workspace_mb: int = 1024,
) -> None:
    print(f"TensorRT version: {trt.__version__}")
    print(f"Building engine from {onnx_path}")
    print(f"  fp16={fp16}  int8={int8}  batch=[{min_batch},{opt_batch},{max_batch}]")
    print(f"  workspace={workspace_mb} MiB")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")
    elif fp16:
        print("  WARNING: FP16 requested but not supported on this GPU, using FP32")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("  INT8 enabled")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Dynamic batch profile
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        (min_batch, 3, 224, 224),  # min
        (opt_batch, 3, 224, 224),  # opt
        (max_batch, 3, 224, 224),  # max
    )
    config.add_optimization_profile(profile)

    print("  Building TensorRT engine (this may take several minutes) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.time() - t0
    size_mb = engine_path.stat().st_size / 1e6
    print(f"  Engine saved to {engine_path}  ({size_mb:.1f} MB, built in {elapsed:.0f}s)")


def benchmark_engine(engine_path: Path, batch: int = 1, runs: int = 100):
    """Quick latency benchmark using the built engine."""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    print(f"\nBenchmarking engine (batch={batch}, {runs} runs) ...")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate I/O buffers
    inp_shape = (batch, 3, 224, 224)
    out_shape = (batch, 38)
    inp_host = np.random.randn(*inp_shape).astype(np.float32)
    out_host = np.empty(out_shape, dtype=np.float32)
    inp_gpu  = cuda.mem_alloc(inp_host.nbytes)
    out_gpu  = cuda.mem_alloc(out_host.nbytes)

    bindings = [int(inp_gpu), int(out_gpu)]

    # Warmup
    for _ in range(5):
        cuda.memcpy_htod(inp_gpu, inp_host)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(out_host, out_gpu)

    # Timed runs
    times = []
    for _ in range(runs):
        cuda.memcpy_htod(inp_gpu, inp_host)
        t0 = time.perf_counter()
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(out_host, out_gpu)
        times.append((time.perf_counter() - t0) * 1000)

    times_arr = np.array(times)
    print(f"  Latency (ms): mean={times_arr.mean():.2f}  "
          f"p50={np.percentile(times_arr,50):.2f}  "
          f"p95={np.percentile(times_arr,95):.2f}  "
          f"p99={np.percentile(times_arr,99):.2f}")

    inp_gpu.free()
    out_gpu.free()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="./checkpoints/plant_disease_classifier_sim.onnx")
    parser.add_argument("--engine", default="./checkpoints/plant_disease_classifier_ws.engine")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--int8", action="store_true", default=False)
    parser.add_argument("--min_batch", type=int, default=1)
    parser.add_argument("--opt_batch", type=int, default=4)
    parser.add_argument("--max_batch", type=int, default=8)
    parser.add_argument("--workspace_mb", type=int, default=1024)
    parser.add_argument("--benchmark", action="store_true", default=True)
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)

    if not onnx_path.exists():
        print(f"ERROR: ONNX file not found: {onnx_path}")
        print("Run export_onnx.py first.")
        return

    build_engine(
        onnx_path, engine_path,
        fp16=args.fp16, int8=args.int8,
        min_batch=args.min_batch, opt_batch=args.opt_batch, max_batch=args.max_batch,
        workspace_mb=args.workspace_mb,
    )

    if args.benchmark:
        try:
            benchmark_engine(engine_path, batch=1, runs=200)
            benchmark_engine(engine_path, batch=4, runs=100)
        except ImportError:
            print("  pycuda not available — skipping benchmark. Install with: pip install pycuda")
        except Exception as e:
            print(f"  Benchmark skipped: {e}")

    print("\nDone.")
    print(f"\nWorkstation engine (RTX 5080 / Blackwell architecture):")
    print(f"  {engine_path}")
    print("\nIMPORTANT: This engine runs ONLY on this workstation's GPU.")
    print("  Copy the ONNX file to the Jetson and rebuild the engine there:")
    print("    /usr/src/tensorrt/bin/trtexec \\")
    print("        --onnx=plant_disease_classifier_sim.onnx \\")
    print("        --saveEngine=plant_disease_classifier.engine \\")
    print("        --fp16 --workspace=1024 \\")
    print("        --minShapes=input:1x3x224x224 \\")
    print("        --optShapes=input:4x3x224x224 \\")
    print("        --maxShapes=input:8x3x224x224")


if __name__ == "__main__":
    main()
