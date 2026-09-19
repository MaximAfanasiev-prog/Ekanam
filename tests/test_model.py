import torch

from street_falcon_reid.model import ReIDModel, batch_hard_triplet_loss


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
