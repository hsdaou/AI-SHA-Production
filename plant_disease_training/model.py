"""
PlantVillage Disease Classifier — MobileNetV3-Small backbone
Input:  (B, 3, 224, 224) normalized RGB
Output: (B, 38) raw logits
~2.5M parameters
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class PlantDiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        in_features = backbone.classifier[0].in_features  # 576
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PlantDiseaseClassifier(num_classes=38, pretrained=False)
    total = count_parameters(model)
    print(f"Trainable parameters: {total:,}  ({total/1e6:.2f}M)")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    assert total < 3_000_000, f"Model too large: {total:,} params"
    print("Model OK")
