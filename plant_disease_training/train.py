"""
Train PlantVillage disease classifier.

Usage:
    python train.py --data ./data --output ./checkpoints [--epochs 50] [--batch 64]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import PlantDiseaseClassifier, count_parameters


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(dataset: datasets.ImageFolder, device: torch.device) -> torch.Tensor:
    counts = np.zeros(len(dataset.classes), dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total


# ── Confusion matrix ──────────────────────────────────────────────────────────

def save_confusion_matrix(model, loader, class_names, device, out_path: Path):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="computing confusion matrix", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                logits = model(imgs)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm, annot=False, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax, linewidths=0.3)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (validation set)")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {out_path}")

    return all_labels, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--output", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms()
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)

    num_classes = len(train_ds.classes)
    print(f"\nClasses: {num_classes}")
    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PlantDiseaseClassifier(num_classes=num_classes, pretrained=True).to(device)
    print(f"\nParameters: {count_parameters(model):,}")

    # Differential learning rates: higher for head, lower for backbone
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params     = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )

    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    csv_path = out_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "time_s"])

    print(f"\nStarting training for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler, device, train=False
        )
        scheduler.step(epoch)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        row = [epoch, f"{train_loss:.4f}", f"{train_acc:.4f}",
               f"{val_loss:.4f}", f"{val_acc:.4f}", f"{lr:.6f}", f"{elapsed:.1f}"]
        history.append({k: v for k, v in zip(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"], row[:5]
        )})

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        flag = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(),
                 "val_acc": val_acc, "optimizer": optimizer.state_dict(),
                 "classes": train_ds.classes},
                out_dir / "best_model.pth",
            )
            flag = "  ← best"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={lr:.2e}  {elapsed:.0f}s{flag}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    # ── Final evaluation ──────────────────────────────────────────────────────
    print("\nLoading best model for final evaluation ...")
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    all_labels, all_preds = save_confusion_matrix(
        model, val_loader, train_ds.classes, device,
        out_dir / "confusion_matrix.png"
    )

    report = classification_report(
        all_labels, all_preds, target_names=train_ds.classes, output_dict=True
    )
    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Per-class CSV
    with open(out_dir / "per_class_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for cls in train_ds.classes:
            m = report[cls]
            writer.writerow({
                "class": cls,
                "precision": f"{m['precision']:.4f}",
                "recall":    f"{m['recall']:.4f}",
                "f1":        f"{m['f1-score']:.4f}",
                "support":   int(m["support"]),
            })

    # Flag low-F1 classes
    low_f1 = [(cls, report[cls]["f1-score"]) for cls in train_ds.classes
              if report[cls]["f1-score"] < 0.85]
    if low_f1:
        print("\nWARNING — classes with F1 < 0.85:")
        for cls, f1 in sorted(low_f1, key=lambda x: x[1]):
            print(f"  {cls}: F1={f1:.4f}")
    else:
        print("\nAll classes have F1 >= 0.85. Target met.")

    overall_acc = report["accuracy"]
    print(f"\nOverall accuracy (val): {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    if overall_acc >= 0.96:
        print("Target accuracy >= 96% MET.")
    else:
        print(f"WARNING: target accuracy 96% NOT met (got {overall_acc*100:.2f}%). Consider more epochs.")

    # Training curve plot
    epochs_done = [int(h["epoch"]) for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_done, [float(h["train_loss"]) for h in history], label="train")
    axes[0].plot(epochs_done, [float(h["val_loss"]) for h in history], label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("epoch")
    axes[1].plot(epochs_done, [float(h["train_acc"]) for h in history], label="train")
    axes[1].plot(epochs_done, [float(h["val_acc"]) for h in history], label="val")
    axes[1].axhline(0.96, color="r", linestyle="--", label="target 96%")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("epoch")
    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    print(f"\nAll outputs saved to {out_dir}/")
    print("  best_model.pth")
    print("  training_log.csv")
    print("  confusion_matrix.png")
    print("  classification_report.json")
    print("  per_class_metrics.csv")
    print("  training_curves.png")


if __name__ == "__main__":
    main()
