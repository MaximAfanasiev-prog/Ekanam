"""Streaming k-reciprocal re-ranking (Zhong et al. 2017): every query is re-ranked on its own, against the gallery.

The reference algorithm runs on the joint distance matrix of ALL queries and the gallery, so one query's result
depends on the other queries - not allowed here (queries arrive as a stream). This version runs the same algorithm
once per query on the matrix of {this query} + gallery (or + its top_k gallery candidates only), so it uses only the
query itself and the fixed gallery; the gallery-gallery distances are computed once. Up to ties it is exactly the
reference implementation called with a single query (checked against posteval's numpy port).

The whole per-query computation is dense (n x n boolean / float matrices, n = gallery size + 1), vectorized over a
batch of queries on the GPU.
"""

import numpy as np
import torch
from models.yolo_embedding.retrieval import search, top10

# Chosen on mini-val seed 42 (posteval): 0.3 mAP@10 points below the best full-gallery setting there, but the best
# candidates F1 on all three splits and 3x cheaper on CPU.
RERANK_DEFAULT = {"k1": 10, "k2": 3, "lam": 0.5, "top_k": 100}


def _rerank_batch(d_qg, d_gg, k1, k2, lam):
    """d_qg: B x m squared euclidean query-gallery distances, d_gg: B x m x m (or m x m) gallery-gallery.
    -> B x m re-ranked distances in [0, 1] (smaller = closer)."""
    b, m = d_qg.shape
    n = m + 1
    dist = torch.zeros(b, n, n, device=d_qg.device)
    dist[:, 0, 1:] = d_qg
    dist[:, 1:, 0] = d_qg
    dist[:, 1:, 1:] = d_gg
    dist = dist / dist.amax(dim=2, keepdim=True)  # each row by its max, as (d / d.max(axis=0)).T of the reference

    order = dist.topk(k1 + 1, dim=2, largest=False).indices  # B x n x (k1 + 1), order[:, i, 0] == i

    def reciprocal(k):
        fwd = torch.zeros(b, n, n, dtype=torch.bool, device=dist.device).scatter_(2, order[..., :k], True)
        return fwd & fwd.transpose(1, 2)

    recip = reciprocal(k1 + 1)  # recip[i, j]: j in top-(k1+1) of i and i in top-(k1+1) of j
    recip_h = reciprocal(int(np.around(k1 / 2)) + 1).float()
    # expansion: add the half-size reciprocal set of a reciprocal neighbour c if 2/3 of it is already in recip[i]
    overlap = recip.float() @ recip_h.transpose(1, 2)  # [i, c] = |recip_h[c] & recip[i]|
    add = recip & (overlap > 2 / 3 * recip_h.sum(2)[:, None, :])
    expanded = recip | (add.float() @ recip_h > 0)

    v = torch.exp(-dist) * expanded
    v = v / v.sum(2, keepdim=True)
    if k2 != 1:  # local query expansion: average v over each row's top-k2 neighbours
        avg = torch.zeros(b, n, n, device=dist.device).scatter_(2, order[..., :k2], 1 / k2)
        v = avg @ v
    acc = torch.minimum(v[:, :1], v).sum(2)  # sum_k min(v[q, k], v[j, k]) for every j
    jaccard = 1 - acc / (2 - acc)
    return (jaccard * (1 - lam) + dist[:, 0] * lam)[:, 1:]


@torch.no_grad()
def rerank_sims(q_emb, g_emb, k1, k2, lam, top_k=None, batch_size=32, device=None):
    """L2-normalized embeddings (numpy Q x d, G x d) -> Q x G re-ranked similarity, 1 - re-ranked distance, in [0, 1].
    Queries are batched only for speed; each row depends on its own query and the gallery alone.

    top_k: re-rank only each query's top_k cosine candidates (neighbourhoods are then computed inside that set);
    the rest of the gallery keeps its cosine order after them (similarity cos - 2 < 0). None = the whole gallery.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    q, g = torch.as_tensor(q_emb, device=device).float(), torch.as_tensor(g_emb, device=device).float()
    d_gg = (2 - 2 * g @ g.T).clamp_(min=0)
    out = []
    for i in range(0, len(q), batch_size):
        cos = q[i : i + batch_size] @ g.T
        d_qg = (2 - 2 * cos).clamp(min=0)
        if top_k is None or top_k >= len(g):
            out.append(1 - _rerank_batch(d_qg, d_gg, k1, k2, lam))
            continue
        idx = cos.topk(top_k, dim=1).indices
        sims = cos - 2
        sims.scatter_(1, idx, 1 - _rerank_batch(d_qg.gather(1, idx), d_gg[idx[:, :, None], idx[:, None, :]],
                                                k1, k2, lam))
        out.append(sims)
    return torch.cat(out).cpu().numpy()


def retrieve(q_emb, g_emb, rerank=None, device=None):
    """The retrieval step of predict.py, shared with the mini-val evaluation so both write the same artifacts.

    -> (sims, idx, sub_idx): FAISS cosine top-10 with its scores, which candidates.csv is always written from, and
    the gallery order of submission.csv (the same top-10, or re-ranked with `rerank` params).
    """
    sims, idx = search(q_emb, g_emb)
    sub_idx = idx if rerank is None else top10(rerank_sims(q_emb, g_emb, **rerank, device=device))[1]
    return sims, idx, sub_idx
