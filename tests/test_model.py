from unittest.mock import patch

import torch

from street_falcon_reid.model import ReIDModel, batch_hard_triplet_loss
from street_falcon_reid.train import extract_embeddings


def test_model_and_triplet_loss_smoke() -> None:
    model = ReIDModel(
        backbone="resnet18",
        embedding_dim=32,
        num_classes=2,
        pretrained=False,
    )
    embeddings, logits = model(torch.randn(4, 3, 64, 64))
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)

    assert embeddings.shape == (4, 32)
    assert logits.shape == (4, 2)
    assert torch.allclose(torch.linalg.vector_norm(embeddings, dim=1), torch.ones(4), atol=1e-5)
    assert loss.isfinite()


def test_temporary_embedding_loader_does_not_persist_workers() -> None:
    class FakeModel:
        def eval(self) -> None:
            return None

        def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return images, torch.empty((len(images), 1))

    batch = {
        "image": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        "image_id": ["image-1"],
    }
    with patch("street_falcon_reid.train.DataLoader", return_value=[batch]) as loader:
        embeddings, image_ids = extract_embeddings(
            FakeModel(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            batch_size=1,
            workers=8,
            device=torch.device("cpu"),
        )

    assert loader.call_args.kwargs["persistent_workers"] is False
    assert embeddings.shape == (1, 2)
    assert image_ids == ["image-1"]
