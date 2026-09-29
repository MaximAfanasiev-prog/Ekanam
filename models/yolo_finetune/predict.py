"""submission.csv, embeddings.npy and candidates.csv on the official test with a fine-tuned checkpoint.

    python -m models.yolo_finetune.predict --checkpoint models/yolo_finetune/weights/yolo26l-cls-reid.pt \
        --out submissions/yolo_finetune/yolo26l-cls-reid

Same pipeline as the frozen baseline (models.yolo_embedding.predict): bbox crop with margin 0, the same
preprocessing, chunked embedding, L2-normalization, FAISS IndexFlatIP top-10, the same artifact writers. Only the
embedder differs: by default its default feature and horizontal-flip TTA (see FinetunedEmbedder). With --rerank the
submission top-10 comes from streaming k-reciprocal re-ranking (rerank.py: each query against its top-100 gallery
candidates only, never other queries; allowed by the organizers, and mAP@10 is scored on this final order).
It adds 3.4-4.7 mAP@10 points on the three mini-val splits. embeddings.npy is the same either way.

candidates.csv always comes from the plain cosine top-10, also with --rerank: at the chosen refuse rate 0.25 the
cosine top-1 score separates queries better than the re-ranked one (mean F1 0.884 vs 0.873, TNR 0.76 vs 0.73 over
the three splits), and its confidence stays the cosine of embeddings.npy. The candidates threshold defaults to the
F1-optimal one of this checkpoint on mini-val seed 42 with the same feature / flip, without re-ranking
(docs/results/yolo_finetune_minival.json); pass --threshold or --refuse-rate to override. Works offline.

--refuse-rate R replaces the fixed threshold by a rank rule: refuse the R share of test queries with the lowest
top-1 similarity (threshold = R-quantile of the test top-1 scores). A fixed cosine threshold does not transfer from
mini-val to the test: with the same checkpoint the median top-1 score is 0.908 on mini-val and 0.958 on the test,
so the mini-val F1-optimal 0.7756 refuses only 4% of test queries although ~20% of them have no match. The share
of queries without a match is the same (20-22% on the three mini-val splits, 20% on the test), and mean F1 over
the three splits is flat for R in 0.20-0.27 (max 0.884 at 0.25-0.26); R = 0.25 gives TNR 0.76 instead of 0.63 at 0.20.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from models.yolo_embedding.data import DATASET_DIR, REPO_DIR, read_table
from models.yolo_embedding.predict import embed_table
from models.yolo_embedding.retrieval import write_candidates, write_embeddings, write_submission

from .model import FinetunedEmbedder
from .rerank import RERANK_DEFAULT, retrieve

MINIVAL_JSON = REPO_DIR / "docs/results/yolo_finetune_minival.json"


def minival_threshold(checkpoint, mode, feature, flip):
    """The similarity scale depends on the feature and flip TTA, so a threshold is only reused when they match.
    Candidates are always cosine, so only the result without re-ranking applies."""
    for r in json.loads(MINIVAL_JSON.read_text()) if MINIVAL_JSON.exists() else []:
        if (Path(r["embedder"]).resolve() == Path(checkpoint).resolve() and r["seed"] == 42 and r["mode"] == mode
                and r.get("feature") == feature and r.get("flip") == flip and r.get("rerank") is None):
            return r["candidates"]["threshold"]
    raise SystemExit(f"no seed-42 mini-val result for {checkpoint} (mode={mode}, feature={feature}, flip={flip}, "
                     f"no re-ranking) in {MINIVAL_JSON}; run evaluate or pass --threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", default=None, help="default: the preprocessing the checkpoint was trained with")
    ap.add_argument("--feature", choices=["feat", "emb"], default=None, help="default: by checkpoint config")
    ap.add_argument("--no-flip", action="store_true", help="disable horizontal-flip TTA")
    ap.add_argument("--rerank", action="store_true",
                    help="submission order by streaming k-reciprocal re-ranking (RERANK_DEFAULT); "
                         "candidates stay cosine")
    thr_group = ap.add_mutually_exclusive_group()
    thr_group.add_argument("--threshold", type=float, default=None)
    thr_group.add_argument("--refuse-rate", type=float, default=None,
                           help="refuse this share of queries with the lowest top-1 similarity (see module doc)")
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds, out = Path(args.dataset_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q_df, g_df = read_table(ds / "test_query.csv"), read_table(ds / "test_gallery.csv")
    embedder = FinetunedEmbedder(args.checkpoint, args.device, args.feature, not args.no_flip)
    mode = args.mode or embedder.meta["config"]["mode"]
    rerank = RERANK_DEFAULT if args.rerank else None
    thr = args.threshold
    if thr is None and args.refuse_rate is None:
        thr = minival_threshold(args.checkpoint, mode, embedder.feature, embedder.flip)
    t = time.time()
    q_emb = embed_table(embedder, q_df, mode, 0.0)
    g_emb = embed_table(embedder, g_df, mode, 0.0)
    sims, idx, sub_idx = retrieve(q_emb, g_emb, rerank, args.device)  # candidates: cosine sims/idx; submission: sub_idx
    elapsed = time.time() - t
    if args.refuse_rate is not None:
        thr = float(np.quantile(sims[:, 0], args.refuse_rate))  # top-1 scores below it are refused

    q_ids, g_ids = q_df.image_id.tolist(), g_df.image_id.tolist()
    write_submission(out / "submission.csv", q_ids, g_ids, sub_idx)
    write_embeddings(out / "embeddings.npy", q_emb, g_emb)
    n_rows = write_candidates(out / "candidates.csv", q_ids, g_ids, sims, idx, thr)
    n_accepted = int((sims[:, 0] >= thr).sum())

    meta = {
        "model": embedder.name, "checkpoint": str(args.checkpoint), "trained_epochs": embedder.meta["epoch"],
        "mode": mode, "feature": embedder.feature, "flip": embedder.flip, "rerank": rerank, "margin": 0.0,
        "threshold": thr, "refuse_rate": args.refuse_rate,
        "n_query": len(q_ids), "n_gallery": len(g_ids), "embedding_dim": int(q_emb.shape[1]),
        "candidates_from": "cosine", "rerank_changed_top10_rows": int((sub_idx != idx).any(1).sum()),
        "queries_accepted": n_accepted, "queries_refused": len(q_ids) - n_accepted, "candidate_rows": n_rows,
        "end_to_end_seconds": round(elapsed, 2),
        "end_to_end_img_per_s": round((len(q_ids) + len(g_ids)) / elapsed, 1),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
