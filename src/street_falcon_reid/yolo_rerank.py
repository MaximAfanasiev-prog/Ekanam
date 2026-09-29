"""K-reciprocal kernel adapted from YOLO branch 711294d (Zhong et al., 2017)."""

import numpy as np
import torch

RERANK_DEFAULT = {"k1": 10, "k2": 3, "lam": 0.5, "top_k": 100}


def _rerank_batch(d_qg, d_gg, k1, k2, lam):
    b, m = d_qg.shape
    n = m + 1
    dist = torch.zeros(b, n, n, device=d_qg.device)
    dist[:, 0, 1:] = d_qg
    dist[:, 1:, 0] = d_qg
    dist[:, 1:, 1:] = d_gg
    dist = dist / dist.amax(dim=2, keepdim=True).clamp_min(1e-12)
    order = dist.topk(k1 + 1, dim=2, largest=False).indices

    def reciprocal(k):
        fwd = torch.zeros(b, n, n, dtype=torch.bool, device=dist.device).scatter_(
            2, order[..., :k], True
        )
        result = fwd & fwd.transpose(1, 2)
        result.diagonal(dim1=1, dim2=2).fill_(True)
        return result

    recip = reciprocal(k1 + 1)
    recip_h = reciprocal(int(np.around(k1 / 2)) + 1).float()
    overlap = recip.float() @ recip_h.transpose(1, 2)
    add = recip & (overlap > 2 / 3 * recip_h.sum(2)[:, None, :])
    expanded = recip | (add.float() @ recip_h > 0)
    v = torch.exp(-dist) * expanded
    v = v / v.sum(2, keepdim=True)
    if k2 != 1:
        avg = torch.zeros(b, n, n, device=dist.device).scatter_(2, order[..., :k2], 1 / k2)
        v = avg @ v
    acc = torch.minimum(v[:, :1], v).sum(2)
    jaccard = 1 - acc / (2 - acc)
    return (jaccard * (1 - lam) + dist[:, 0] * lam)[:, 1:]
