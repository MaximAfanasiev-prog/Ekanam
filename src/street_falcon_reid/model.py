from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


class ReIDModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: str,
        embedding_dim: int,
        num_classes: int,
        pretrained: bool,
    ) -> None:
        super().__init__()
        if backbone == "resnet50":
            network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        elif backbone == "resnet18":
            network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Неподдерживаемый backbone: {backbone}")

        features = network.fc.in_features
        network.fc = nn.Identity()
        self.backbone = network
        self.embedding = nn.Linear(features, embedding_dim, bias=False)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
        nn.init.kaiming_normal_(self.embedding.weight, mode="fan_out")
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        embedding = self.bnneck(self.embedding(features))
        logits = self.classifier(embedding)
        return F.normalize(embedding, p=2, dim=1), logits


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    similarity = embeddings @ embeddings.T
    distance = 1.0 - similarity
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & ~diagonal
    negatives = ~same

    hard_positive = distance.masked_fill(~positives, float("-inf")).max(dim=1).values
    hard_negative = distance.masked_fill(~negatives, float("inf")).min(dim=1).values
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    return F.relu(hard_positive[valid] - hard_negative[valid] + margin).mean()
