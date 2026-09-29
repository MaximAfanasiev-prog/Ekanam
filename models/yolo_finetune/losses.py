"""Batch-hard triplet loss (Hermans et al. 2017) with online hardest positive / hardest negative mining."""

import torch.nn.functional as F


def euclidean_dist(x):
    sq = (x * x).sum(1)
    return (sq[:, None] + sq[None, :] - 2 * x @ x.T).clamp_min(1e-12).sqrt()


def batch_hard_triplet(feat, labels, margin=0.3):
    """For every anchor: the farthest same-id sample and the closest other-id sample in the batch.

    Returns (loss, fraction of anchors whose hardest negative is already farther than the hardest positive).
    """
    d = euclidean_dist(feat.float())
    same = labels[:, None] == labels[None, :]
    d_ap = d.masked_fill(~same, float("-inf")).max(1).values  # the diagonal (d=0) never wins when K >= 2
    d_an = d.masked_fill(same, float("inf")).min(1).values
    return F.relu(d_ap - d_an + margin).mean(), (d_an > d_ap).float().mean()
