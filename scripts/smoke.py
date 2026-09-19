from __future__ import annotations

import torch

from street_falcon_reid.model import ReIDModel, batch_hard_triplet_loss


def main() -> None:
    torch.manual_seed(17)
    model = ReIDModel(
        backbone="resnet18",
        embedding_dim=32,
        num_classes=2,
        pretrained=False,
    )
    model.eval()
    with torch.inference_mode():
        embeddings, logits = model(torch.randn(4, 3, 64, 64))
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    if embeddings.shape != (4, 32) or logits.shape != (4, 2) or not loss.isfinite():
        raise SystemExit("model smoke failed")
    print("model smoke passed")


if __name__ == "__main__":
    main()
