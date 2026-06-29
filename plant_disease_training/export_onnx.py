"""
Export trained model to ONNX (opset 17), then simplify with onnxsim.

Usage:
    python export_onnx.py --checkpoint ./checkpoints/best_model.pth \
                          --output ./checkpoints/plant_disease_classifier.onnx
"""

import argparse
import json
import shutil
from pathlib import Path

import onnx
import torch

from model import PlantDiseaseClassifier


def export_onnx(checkpoint_path: Path, onnx_path: Path, data_dir: Path):
    device = torch.device("cpu")  # Export on CPU for portability
    ckpt = torch.load(checkpoint_path, map_location=device)
    classes = ckpt.get("classes", None)

    # Infer num_classes from checkpoint
    # Find the last linear layer's output features
    state = ckpt["state_dict"]
    last_w = [v for k, v in state.items() if k.endswith(".weight") and v.ndim == 2][-1]
    num_classes = last_w.shape[0]
    print(f"  num_classes inferred from checkpoint: {num_classes}")

    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Exported to {onnx_path}")

    # Validate
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print("  ONNX model check passed")

    # Simplify
    try:
        from onnxsim import simplify
        m_sim, ok = simplify(m)
        if ok:
            sim_path = onnx_path.with_name(onnx_path.stem + "_sim.onnx")
            onnx.save(m_sim, str(sim_path))
            print(f"  Simplified model saved to {sim_path}")
        else:
            print("  WARNING: onnxsim simplification returned ok=False, using original")
            sim_path = onnx_path
    except Exception as e:
        print(f"  WARNING: onnxsim failed ({e}), using original ONNX")
        sim_path = onnx_path

    # Copy class_mapping.json next to the onnx if available
    src_map = data_dir / "class_mapping.json"
    if src_map.exists():
        dst_map = onnx_path.parent / "class_mapping.json"
        shutil.copy2(src_map, dst_map)
        print(f"  class_mapping.json copied to {dst_map}")
    else:
        print("  WARNING: class_mapping.json not found at", src_map)

    return sim_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth")
    parser.add_argument("--output", default="./checkpoints/plant_disease_classifier.onnx")
    parser.add_argument("--data", default="./data",
                        help="Data dir containing class_mapping.json")
    args = parser.parse_args()

    print("Exporting to ONNX ...")
    sim_path = export_onnx(
        Path(args.checkpoint),
        Path(args.output),
        Path(args.data),
    )
    print(f"\nFinal ONNX model: {sim_path}")
    print("\nNext step (on this workstation):")
    print("  python build_tensorrt.py --onnx", sim_path)
    print("\nNext step (on Jetson Orin Nano):")
    print("  /usr/src/tensorrt/bin/trtexec \\")
    print("      --onnx=plant_disease_classifier_sim.onnx \\")
    print("      --saveEngine=plant_disease_classifier.engine \\")
    print("      --fp16 \\")
    print("      --workspace=1024 \\")
    print("      --minShapes=input:1x3x224x224 \\")
    print("      --optShapes=input:4x3x224x224 \\")
    print("      --maxShapes=input:8x3x224x224")


if __name__ == "__main__":
    main()
