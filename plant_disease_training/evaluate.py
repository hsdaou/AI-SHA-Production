"""
Evaluate a saved checkpoint on the held-out test set.

Usage:
    python evaluate.py --data ./data --checkpoint ./checkpoints/best_model.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import PlantDiseaseClassifier


def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="evaluating"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                logits = model(imgs)
            preds = logits.argmax(1)
            correct += (preds.cpu() == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_labels, all_preds, correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--checkpoint", default="./checkpoints/best_model.pth")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="./checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = datasets.ImageFolder(Path(args.data) / args.split, transform=get_val_transform())
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = PlantDiseaseClassifier(num_classes=len(ds.classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    print(f"Evaluating on {args.split} set ({len(ds):,} images) ...")
    all_labels, all_preds, acc = evaluate(model, loader, device)

    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

    report = classification_report(all_labels, all_preds, target_names=ds.classes, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=ds.classes))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"eval_{args.split}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(cm, annot=False, xticklabels=ds.classes, yticklabels=ds.classes,
                cmap="Blues", ax=ax, linewidths=0.3)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {args.split} set")
    plt.xticks(rotation=90, fontsize=7); plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(out / f"confusion_matrix_{args.split}.png", dpi=150)
    plt.close()
    print(f"Saved results to {out}/")


if __name__ == "__main__":
    main()
