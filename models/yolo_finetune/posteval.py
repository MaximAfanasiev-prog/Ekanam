"""Inference-time tricks on already fine-tuned checkpoints, no training: which embedding to use (pre/post BNNeck),
horizontal-flip TTA, a squash + center_crop ensemble and streaming k-reciprocal re-ranking (rerank.py: each query
against the gallery only), on every mini-val split.

    python -m models.yolo_finetune.posteval --out runs/yolo_finetune/eval/posteval.json

Each split gets its own pair of checkpoints trained on exactly that split (see RUNS). Re-ranking parameters are
chosen on seed 42 only; seeds 7 / 2026 are reported with those parameters as they are.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from models.yolo_embedding.data import REPO_DIR

from .data import load_split, preprocessed_train
from .evaluate import score
from .model import load_embedding_net

RUN_DIR = REPO_DIR / "runs/yolo_finetune"
# split dir -> {mode: run name}; the squash run is the current best config, center_crop is its baseline
RUNS = {
    "data": {"squash": "p1_26l_tronly_squash", "center_crop": "p1_26l_tronly"},
    "data/splits/seed7": {"squash": "ho_26l_tronly_squash_s7", "center_crop": "ho_26l_tronly_s7"},
    "data/splits/seed2026": {"squash": "ho_26l_tronly_squash_s2026", "center_crop": "ho_26l_tronly_s2026"},
}
RERANK_GRID = [dict(k1=k1, k2=k2, lam=lam, top_k=top_k) for top_k in (None, 100, 50)
               for k1, k2, lam in [(20, 6, 0.3), (10, 3, 0.3), (8, 3, 0.3), (6, 2, 0.3), (10, 3, 0.5), (10, 3, 0.1)]]


@torch.no_grad()
def raw_features(net, images, rows, bs=256):
    """-> {"feat", "emb", "feat_flip", "emb_flip"}: unnormalized pre-BN / post-BN outputs, plain and mirrored."""
    out = {k: [] for k in ("feat", "emb", "feat_flip", "emb_flip")}
    for i in range(0, len(rows), bs):
        x = images[rows[i : i + bs]].cuda().float() / 255
        for suffix, xi in (("", x), ("_flip", x.flip(-1))):
            feat, emb = net(xi)
            out["feat" + suffix].append(feat.float().cpu())
            out["emb" + suffix].append(emb.float().cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def embedding(f, kind, flip):
    """L2-normalized embedding; with flip TTA the plain and mirrored outputs are averaged before normalization."""
    x = f[kind] + f[kind + "_flip"] if flip else f[kind]
    return F.normalize(x, dim=1).numpy()


def concat(*embs):
    return F.normalize(torch.from_numpy(np.concatenate(embs, axis=1)), dim=1).numpy()


def summary(res):
    return {k: round(res[k], 4) for k in ("mAP@10", "Rank-1", "Rank-5", "mAP_full", "mINP")} | {
        "F1": round(res["candidates"]["F1"], 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    images = {m: preprocessed_train(m) for m in ("squash", "center_crop")}
    results = {}
    for split, runs in RUNS.items():
        sp = load_split(REPO_DIR / split)
        seed = sp["split"]["seed"]
        feats = {}
        for mode, run in runs.items():
            net, _ = load_embedding_net(RUN_DIR / run / "best.pt", "cuda")
            feats[mode] = {part: raw_features(net, images[mode], sp[f"{part}_rows"]) for part in ("q", "g")}
            del net
            torch.cuda.empty_cache()

        def emb(mode, kind="emb", flip=False):
            return embedding(feats[mode]["q"], kind, flip), embedding(feats[mode]["g"], kind, flip)

        variants = {
            "squash emb (current)": emb("squash"),
            "squash feat": emb("squash", "feat"),
            "squash emb + flip": emb("squash", flip=True),
            "squash feat + flip": emb("squash", "feat", True),
            "center_crop emb": emb("center_crop"),
            "center_crop emb + flip": emb("center_crop", flip=True),
        }
        for kind in ("emb", "feat"):
            (qs, gs), (qc, gc) = emb("squash", kind, True), emb("center_crop", kind, True)
            variants[f"ensemble {kind} + flip"] = (concat(qs, qc), concat(gs, gc))

        res = {name: summary(score(q, g, sp)) for name, (q, g) in variants.items()}
        best_name = max(res, key=lambda n: res[n]["mAP@10"] if n.startswith("squash") else -1)
        for base in (best_name, max(res, key=lambda n: res[n]["mAP@10"])):
            q, g = variants[base]
            for rr in RERANK_GRID:
                t0 = time.time()
                r = summary(score(q, g, sp, rerank=rr))
                name = " ".join(f"{k}={v}" for k, v in rr.items())
                res[f"{base} + rerank {name}"] = r | {"seconds": round(time.time() - t0, 1)}
        results[seed] = res
        print(f"\n=== seed {seed} ({split})")
        for name, r in res.items():
            print(f"  {name:72s} mAP@10={r['mAP@10']:.4f} R1={r['Rank-1']:.4f} mAP_full={r['mAP_full']:.4f} "
                  f"mINP={r['mINP']:.4f} F1={r['F1']:.4f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    main()
