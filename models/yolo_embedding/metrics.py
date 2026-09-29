"""Metrics that follow the organizers' protocol (source/evaluate.py) exactly, in array form for threshold sweeps.

Protocol recap (from the official evaluate.py docstring):
  * junk = gallery items with the same vehicle_id AND the same camera_id as the query; removed from ranking;
  * queries with no valid positives left are excluded from mAP/Rank and only count in candidate mode;
  * AP@10 is normalized by min(n_pos, 10);
  * candidate mode is decided per query by the top-confidence candidate;
  * mAP_full / mINP come from embeddings.npy, which evaluate.py L2-normalizes itself (so never from a re-ranked
    similarity), and PR-AUC ranks queries by the confidence as written to candidates.csv (clipped, 6 decimals).
Inputs are the very arrays the artifacts are written from (the lists of submission.csv, the FAISS scores of
candidates.csv, the embeddings), produced by the same retrieval code as the test submission.
`run_official` additionally runs the organizers' script itself on written artifacts as a cross-check.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .data import DATASET_DIR

EVALUATE_PY = Path(os.environ.get("FALCON_EVALUATE_PY", DATASET_DIR.parent / "source" / "evaluate.py"))


def _pos_junk(q_vid, q_cam, g_vid, g_cam):
    same_vid = q_vid[:, None] == g_vid[None, :]
    same_cam = q_cam[:, None] == g_cam[None, :]
    return same_vid & ~same_cam, same_vid & same_cam


def ranking_metrics(sub_idx, q_vid, q_cam, g_vid, g_cam, top_k=10):
    """mAP@10 / Rank-1 / Rank-5 of submission.csv: `sub_idx` are the gallery indices written per query (Q x 10),
    junk is stripped from that list before it is cut to top_k, as evaluate.py does."""
    pos, junk = _pos_junk(q_vid, q_cam, g_vid, g_cam)
    n_pos = pos.sum(1)
    order = np.asarray(sub_idx)
    aps, r1, r5 = [], [], []
    for i in np.flatnonzero(n_pos > 0):
        top = order[i][~junk[i, order[i]]][:top_k]
        rel = pos[i, top]
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append((prec * rel).sum() / min(n_pos[i], top_k) if rel.any() else 0.0)
        r1.append(rel[:1].any())
        r5.append(rel[:5].any())
    return {
        "n_scored": len(aps),
        "n_openset_excluded": int((n_pos == 0).sum()),
        "mAP@10": float(np.mean(aps)),
        "Rank-1": float(np.mean(r1)),
        "Rank-5": float(np.mean(r5)),
    }


def per_query_ap(sub_idx, q_vid, q_cam, g_vid, g_cam, top_k=10):
    """AP@10 per scored query of submission.csv (NaN for open-set queries) - for paired bootstrap between models."""
    pos, junk = _pos_junk(q_vid, q_cam, g_vid, g_cam)
    n_pos = pos.sum(1)
    order = np.asarray(sub_idx)
    out = np.full(len(order), np.nan)
    for i in np.flatnonzero(n_pos > 0):
        rel = pos[i, order[i][~junk[i, order[i]]][:top_k]]
        out[i] = (np.cumsum(rel) / (np.arange(len(rel)) + 1) * rel).sum() / min(n_pos[i], top_k)
    return out


def l2_normalize(emb):
    """The L2 normalization evaluate.py applies to embeddings.npy, bit for bit."""
    emb = np.asarray(emb).astype(np.float32, copy=False)
    return emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)


def full_ranking_metrics(q_emb, g_emb, q_vid, q_cam, g_vid, g_cam):
    """mAP over the full ranking and mINP, from the embeddings exactly as evaluate.py computes them from
    embeddings.npy: its own L2 normalization, plain cosine (a re-ranked order does not count here)."""
    sims = l2_normalize(q_emb) @ l2_normalize(g_emb).T
    pos, junk = _pos_junk(q_vid, q_cam, g_vid, g_cam)
    aps, inps = [], []
    for i in range(len(sims)):
        keep = ~junk[i]
        rel = pos[i, keep][np.argsort(-sims[i, keep], kind="stable")]
        n = rel.sum()
        if n == 0:
            continue
        prec = np.cumsum(rel) / (np.arange(len(rel)) + 1)
        aps.append((prec * rel).sum() / n)
        inps.append(n / (np.flatnonzero(rel).max() + 1))
    return {"mAP_full": float(np.mean(aps)), "mINP": float(np.mean(inps))}


def top1_decision_inputs(cand_sims, cand_idx, q_vid, q_cam, g_vid, g_cam):
    """Per query: top-1 confidence, whether top-1 has the query's vehicle_id, whether a valid positive exists.
    `cand_sims` / `cand_idx` are the candidates candidates.csv is written from (retrieval.search, best first)."""
    pos, _ = _pos_junk(q_vid, q_cam, g_vid, g_cam)
    top1 = np.asarray(cand_idx)[:, 0]
    return np.asarray(cand_sims)[:, 0], g_vid[top1] == q_vid, pos.any(1)


def candidate_metrics(score, correct, has_match, threshold):
    """Precision/Recall/F1/TNR/PR-AUC of the accept/refuse decision at a threshold (accept if score >= thr)."""
    acc = score >= threshold
    tp = int((acc & has_match & correct).sum())
    fp = int((acc & ~(has_match & correct)).sum())
    fn = int((~acc & has_match).sum())
    tn = int((~acc & ~has_match).sum())
    fp_open = int((acc & ~has_match).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "n_openset_queries": tn + fp_open,
        "Precision": p,
        "Recall": r,
        "F1": 2 * p * r / (p + r) if p + r else 0.0,
        "TNR": tn / (tn + fp_open) if tn + fp_open else float("nan"),
        "PR-AUC": pr_auc(np.where(acc, written_confidence(score), -np.inf), has_match.astype(int)),
    }


def confidence_text(s):
    """How retrieval.write_candidates stores a similarity in candidates.csv."""
    return f"{float(np.clip(s, 0.0, 1.0)):.6f}"


def written_confidence(score):
    """Scores as evaluate.py reads them back from candidates.csv: rounding creates ties that change PR-AUC."""
    return np.array([float(confidence_text(s)) for s in np.asarray(score).tolist()])


def pr_auc(scores, labels):
    """Same computation as pr_auc() in the official evaluate.py."""
    finite = np.isfinite(scores)
    if labels.sum() == 0 or not finite.any():
        return float("nan")
    s = np.where(finite, scores, scores[finite].min() - 1.0)
    y = labels[np.argsort(-s, kind="stable")]
    cum = np.cumsum(y)
    prec = cum / (np.arange(len(y)) + 1)
    rec = cum / labels.sum()
    ap, prev = 0.0, 0.0
    # the same summation order as evaluate.py, so the float result is identical
    for p, r in zip(prec, rec, strict=False):
        ap += p * (r - prev)
        prev = r
    return float(ap)


def sweep_threshold(score, correct, has_match):
    """Picks the threshold maximizing F1 on the given split. Candidate thresholds are the observed top-1 scores.

    The returned threshold is the midpoint between the chosen score and the next lower observed score, so a
    small shift of the score distribution on unseen data does not flip the boundary query.
    """
    cand = np.unique(score)[::-1]
    best = max((candidate_metrics(score, correct, has_match, t) for t in cand), key=lambda m: (m["F1"], m["threshold"]))
    lower = cand[cand < best["threshold"]]
    thr = (best["threshold"] + lower[0]) / 2 if len(lower) else best["threshold"]
    return candidate_metrics(score, correct, has_match, thr), [
        candidate_metrics(score, correct, has_match, t) for t in np.round(np.arange(0.30, 1.0001, 0.01), 2)
    ]


def run_official(gt, submission, candidates, embeddings, query_csv, gallery_csv, out_json):
    """Runs the organizers' evaluate.py on written artifacts; returns its JSON report."""
    cmd = [sys.executable, str(EVALUATE_PY), "--gt", gt, "--submission", submission, "--candidates", candidates,
           "--embeddings", embeddings, "--query", query_csv, "--gallery", gallery_csv, "--json", out_json]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(Path(out_json).read_text())
