"""Evaluates frozen and fine-tuned embedders on the open-set mini-val splits with the frozen-baseline metrics.

    python -m models.yolo_finetune.evaluate \
        --embedders yolo26l-cls runs/yolo_finetune/yolo26l_s42/best.pt \
        --splits data data/splits/seed7 data/splits/seed2026 --official --out docs/results/yolo_finetune_minival.json

An embedder is either a frozen model name (models.yolo_embedding.extractor.YoloEmbedder) or a fine-tuned
checkpoint path (FinetunedEmbedder, by default with its default feature and flip TTA; see --feature / --no-flip).
`--rerank` additionally scores every embedder with streaming k-reciprocal re-ranking (rerank.py, RERANK_DEFAULT).
Artifacts come from rerank.retrieve, the retrieval step of predict.py: re-ranking changes only submission.csv,
candidates.csv stays cosine (FAISS) and embeddings.npy stays the same, so the re-ranked row differs from the plain
one only in mAP@10 / Rank-1 / Rank-5. Metrics are models.yolo_embedding.metrics (a copy of the organizers'
protocol) computed from those very arrays; `--official` also writes the artifacts and runs the organizers'
evaluate.py on them.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from models.yolo_embedding import metrics as M
from models.yolo_embedding.data import REPO_DIR
from models.yolo_embedding.extractor import YoloEmbedder
from models.yolo_embedding.retrieval import write_candidates, write_embeddings, write_submission

from .data import load_split, preprocessed_train
from .model import FinetunedEmbedder
from .rerank import RERANK_DEFAULT, retrieve


def make_embedder(spec, device=None, feature=None, flip=True):
    return FinetunedEmbedder(spec, device, feature, flip) if spec.endswith(".pt") else YoloEmbedder(spec, device)


def inference_settings(emb):
    """What exactly produced the embeddings of a fine-tuned checkpoint (predict.py matches its threshold on it)."""
    return {"feature": emb.feature, "flip": emb.flip} if isinstance(emb, FinetunedEmbedder) else {}


def embed_rows(embedder, images_u8, rows, batch_size=128):
    return np.concatenate([
        embedder(images_u8[rows[i : i + batch_size]].float() / 255).cpu().numpy()
        for i in range(0, len(rows), batch_size)
    ]).astype(np.float32)


def score(q_emb, g_emb, sp, with_curve=False, rerank=None):
    """All metrics of the frozen-baseline report for one split, from L2-normalized embeddings, for the artifacts
    predict.py would write from them (with `rerank` params: re-ranked submission.csv, cosine candidates.csv)."""
    return score_retrieved(q_emb, g_emb, sp, retrieve(q_emb, g_emb, rerank), with_curve)


def score_retrieved(q_emb, g_emb, sp, retrieved, with_curve=False):
    """The same from the output of rerank.retrieve: every metric is computed from exactly what goes into the files."""
    sims, idx, sub_idx = retrieved
    gt = (sp["q_vid"], sp["q_cam"], sp["g_vid"], sp["g_cam"])
    rank, full = M.ranking_metrics(sub_idx, *gt), M.full_ranking_metrics(q_emb, g_emb, *gt)
    cand, curve = M.sweep_threshold(*M.top1_decision_inputs(sims, idx, *gt))
    res = {**rank, **full, "candidates": cand}
    if with_curve:
        res["threshold_curve"] = curve
        res["per_query_ap10"] = M.per_query_ap(sub_idx, *gt).tolist()
    return res


def write_artifacts(out, sp, q_emb, g_emb, retrieved, threshold):
    """submission.csv / candidates.csv / embeddings.npy exactly as predict.py writes them on the test."""
    sims, idx, sub_idx = retrieved
    out.mkdir(parents=True, exist_ok=True)
    write_submission(out / "submission.csv", sp["q_ids"], sp["g_ids"], sub_idx)
    write_candidates(out / "candidates.csv", sp["q_ids"], sp["g_ids"], sims, idx, threshold)
    write_embeddings(out / "embeddings.npy", q_emb, g_emb)


def run_official(out, sp):
    d = Path(sp["dir"])
    return M.run_official(str(d / "mini_gt.csv"), str(out / "submission.csv"), str(out / "candidates.csv"),
                          str(out / "embeddings.npy"), str(d / "mini_query.csv"), str(d / "mini_gallery.csv"),
                          str(out / "report.json"))


def evaluate(spec, split_dir, mode, official_dir=None, images=None, feature=None, flip=True, rerank=False):
    """-> [result without re-ranking] (+ [result with re-ranking] if rerank), from one set of embeddings."""
    sp = load_split(split_dir)
    images = preprocessed_train(mode) if images is None else images
    emb = make_embedder(spec, feature=feature, flip=flip)
    settings = inference_settings(emb)
    q_emb, g_emb = embed_rows(emb, images, sp["q_rows"]), embed_rows(emb, images, sp["g_rows"])
    del emb
    torch.cuda.empty_cache()
    results = []
    for rr in (None, RERANK_DEFAULT) if rerank else (None,):
        retrieved = retrieve(q_emb, g_emb, rr)
        res = {"embedder": spec, **settings, "rerank": rr, "split": str(split_dir), "seed": sp["split"]["seed"],
               "mode": mode, "dim": int(q_emb.shape[1]), **score_retrieved(q_emb, g_emb, sp, retrieved, True)}
        if official_dir:
            out = Path(official_dir) if rr is None else Path(official_dir + "_rerank")
            write_artifacts(out, sp, q_emb, g_emb, retrieved, res["candidates"]["threshold"])
            res["official"] = run_official(out, sp)
        results.append(res)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", required=True, help="frozen model names and/or fine-tuned .pt")
    ap.add_argument("--splits", nargs="+", default=["data"])
    ap.add_argument("--mode", default="center_crop")
    ap.add_argument("--feature", choices=["feat", "emb"], default=None, help="default: by checkpoint config")
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal-flip TTA")
    ap.add_argument("--rerank", action="store_true", help="also score with streaming k-reciprocal re-ranking")
    ap.add_argument("--official", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    images = preprocessed_train(args.mode)
    results = []
    for split in args.splits:
        for spec in args.embedders:
            tag = Path(spec).parent.name if spec.endswith(".pt") else spec
            off = str(REPO_DIR / "runs/yolo_finetune/eval" / Path(split).name / tag) if args.official else None
            for r in evaluate(spec, split, args.mode, off, images, args.feature, not args.no_flip, args.rerank):
                results.append(r)
                report(r, tag)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print("->", args.out)


def report(r, tag):
    c = r["candidates"]
    chk = ""
    if "official" in r:
        o = r["official"]
        chk = (f" | official mAP@10={o['ranking']['mAP@10']:.4f} R1={o['ranking']['Rank-1']:.4f} "
               f"F1={o['candidates']['F1']:.4f}")
    if "feature" in r:
        tag += f" [{r['feature']}{'+flip' if r['flip'] else ''}]"
    if r["rerank"]:
        tag += " +rerank"
    print(f"seed {r['seed']:<5} {tag:36s} mAP@10={r['mAP@10']:.4f} R1={r['Rank-1']:.4f} R5={r['Rank-5']:.4f} "
          f"mAP_full={r['mAP_full']:.4f} mINP={r['mINP']:.4f} F1={c['F1']:.4f} TNR={c['TNR']:.4f} "
          f"PR-AUC={c['PR-AUC']:.4f} thr={c['threshold']:.4f}{chk}", flush=True)


if __name__ == "__main__":
    main()
