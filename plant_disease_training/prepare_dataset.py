"""
Prepare PlantVillage dataset for training.

Usage:
    python prepare_dataset.py --src ./plantvillage_raw --dst ./data

Outputs:
    data/train/  data/val/  data/test/   — ImageFolder-compatible symlinks
    data/class_mapping.json              — 38-class metadata
    data/splits.json                     — reproducible split file lists
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASSES: list[str] = []  # Populated from actual dataset folders; order doesn't matter

# Known species prefixes in the dataset (longest match wins)
_KNOWN_SPECIES = [
    "Pepper__bell",
    "Pepper,_bell",
    "Tomato",
    "Potato",
    "Apple",
    "Blueberry",
    "Cherry_(including_sour)",
    "Corn_(maize)",
    "Grape",
    "Orange",
    "Peach",
    "Raspberry",
    "Soybean",
    "Squash",
    "Strawberry",
]


def parse_class_name(folder: str) -> dict:
    """Parse any folder naming variant into species + disease components."""
    species_raw = None
    disease_raw = None

    # Try triple-underscore separator first (most specific)
    if "___" in folder:
        sp, dis = folder.split("___", 1)
        species_raw = sp
        disease_raw = dis
    else:
        # Fall back to known species prefix matching
        for sp in sorted(_KNOWN_SPECIES, key=len, reverse=True):
            if folder.startswith(sp):
                species_raw = sp
                # strip leading underscores from remainder
                remainder = folder[len(sp):].lstrip("_")
                disease_raw = remainder if remainder else "unknown"
                break
        if species_raw is None:
            # Last resort: treat first word as species
            parts = folder.split("_", 1)
            species_raw = parts[0]
            disease_raw = parts[1] if len(parts) > 1 else "unknown"

    # Normalise display strings
    species = species_raw.replace("__", " ").replace("_", " ").strip()
    disease = disease_raw.replace("_", " ").strip()
    is_healthy = disease.lower() == "healthy"

    return {
        "class_name": folder,
        "species": species,
        "disease": "healthy" if is_healthy else disease,
        "is_healthy": is_healthy,
    }


def find_source_dir(src: Path) -> Path:
    """Locate the directory that contains class subfolders."""
    # Dataset may be nested: plantvillage_raw/PlantVillage/  or  plantvillage_raw/
    for candidate in [src, src / "PlantVillage", src / "plant_disease_recognition_dataset"]:
        if candidate.exists():
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if any("___" in d.name for d in subdirs):
                return candidate
    # Recursive search
    for d in src.rglob("*"):
        if d.is_dir() and "___" in d.name:
            return d.parent
    raise FileNotFoundError(f"Could not find class folders under {src}")


def build_file_index(src_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for d in sorted(src_dir.iterdir()):
        if not d.is_dir():
            continue
        files = [f for f in d.iterdir() if f.suffix in exts]
        if files:
            index[d.name] = sorted(files)
    return dict(index)


def stratified_split(files: list[Path], train: float, val: float, seed: int = 42):
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def symlink_files(files: list[Path], dst_class_dir: Path):
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    for src_file in files:
        dst_link = dst_class_dir / src_file.name
        if not dst_link.exists():
            os.symlink(src_file.resolve(), dst_link)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="./plantvillage_raw", help="Raw dataset root")
    parser.add_argument("--dst", default="./data", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Locating source images under {src} ...")
    src_dir = find_source_dir(src)
    print(f"  Found class folders at: {src_dir}")

    index = build_file_index(src_dir)
    print(f"  Discovered {len(index)} class folders")

    # Build class mapping — ordered by CLASSES list, then any extras
    known = {c: i for i, c in enumerate(CLASSES)}
    all_classes = sorted(index.keys(), key=lambda c: known.get(c, 999))
    class_mapping = []
    for idx, name in enumerate(all_classes):
        meta = parse_class_name(name)
        meta["class_index"] = idx
        class_mapping.append(meta)

    with open(dst / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"  Saved class_mapping.json ({len(class_mapping)} classes)")

    # Split and symlink/copy
    splits: dict[str, dict[str, list[str]]] = {"train": {}, "val": {}, "test": {}}
    total_counts = defaultdict(int)
    low_count_classes = []

    for class_name in all_classes:
        files = index[class_name]
        train_f, val_f, test_f = stratified_split(
            files, args.train_ratio, args.val_ratio, seed=args.seed
        )
        for split, file_list in [("train", train_f), ("val", val_f), ("test", test_f)]:
            split_dir = dst / split / class_name
            if args.copy:
                split_dir.mkdir(parents=True, exist_ok=True)
                for f in file_list:
                    shutil.copy2(f, split_dir / f.name)
            else:
                symlink_files(file_list, split_dir)
            splits[split][class_name] = [str(f) for f in file_list]
            total_counts[split] += len(file_list)

        total = len(files)
        if total < 200:
            low_count_classes.append((class_name, total))

        print(
            f"  {class_name[:45]:<45}  "
            f"total={total:5d}  train={len(train_f):4d}  val={len(val_f):4d}  test={len(test_f):4d}"
        )

    with open(dst / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print("\nSplit summary:")
    for split, count in total_counts.items():
        print(f"  {split}: {count:,} images")

    if low_count_classes:
        print("\nWARNING — classes with < 200 samples:")
        for name, count in low_count_classes:
            print(f"  {name}: {count}")
    else:
        print("\nAll classes have >= 200 samples. OK.")

    print(f"\nDataset prepared at: {dst}/")


if __name__ == "__main__":
    main()
