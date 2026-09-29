"""Builds submission.csv, embeddings.npy and candidates.csv for the official test split.

    python -m models.yolo_embedding.predict --model yolo26l-cls --out submissions/yolo_embedding

The candidates threshold defaults to the F1-optimal one found for the same model/mode on the mini-val split
(docs/results/yolo_minival.json); pass --threshold to override. Works fully offline (weights are local).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import DATASET_DIR, REPO_DIR, load_crops, preprocess_batch, read_table
from .extractor import YoloEmbedder
from .retrieval import search, write_candidates, write_embeddings, write_submission


def embed_table(embedder, df, mode, margin, chunk=512):
    """Crops and embeds a table chunk by chunk, so memory does not grow with the table size."""
    out = []
    for i in range(0, len(df), chunk):
        crops = load_crops(df.iloc[i : i + chunk], margin=margin)
        out.append(embedder.embed(preprocess_batch(crops, mode)))
    return np.concatenate(out)


def minival_threshold(model, mode, margin):
    results = json.loads((REPO_DIR / "docs/results/yolo_minival.json").read_text())
    for r in results:
        if r["model"] == model and r["mode"] == mode and r["margin"] == margin:
            return r["candidates"]["threshold"]
    raise SystemExit(f"no mini-val result for {model}/{mode}/margin={margin}; run run_minival or pass --threshold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26l-cls")
    ap.add_argument("--mode", default="center_crop")
    ap.add_argument("--margin", type=float, default=0.0, help="own optional bbox padding, off by default")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(REPO_DIR / "submissions/yolo_embedding"))
    args = ap.parse_args()

    ds, out = Path(args.dataset_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    q_df, g_df = read_table(ds / "test_query.csv"), read_table(ds / "test_gallery.csv")
    thr = args.threshold if args.threshold is not None else minival_threshold(args.model, args.mode, args.margin)

    embedder = YoloEmbedder(args.model, args.device)
    t = time.time()
    q_emb = embed_table(embedder, q_df, args.mode, args.margin)
    g_emb = embed_table(embedder, g_df, args.mode, args.margin)
    elapsed = time.time() - t

    sims, idx = search(q_emb, g_emb)
    q_ids, g_ids = q_df.image_id.tolist(), g_df.image_id.tolist()
    write_submission(out / "submission.csv", q_ids, g_ids, idx)
    write_embeddings(out / "embeddings.npy", q_emb, g_emb)
    n_rows = write_candidates(out / "candidates.csv", q_ids, g_ids, sims, idx, thr)
    n_accepted = int((sims[:, 0] >= thr).sum())

    meta = {
        "model": args.model, "mode": args.mode, "margin": args.margin, "threshold": thr,
        "n_query": len(q_ids), "n_gallery": len(g_ids), "embedding_dim": int(q_emb.shape[1]),
        "queries_accepted": n_accepted, "queries_refused": len(q_ids) - n_accepted, "candidate_rows": n_rows,
        "end_to_end_seconds": round(elapsed, 2),
        "end_to_end_img_per_s": round((len(q_ids) + len(g_ids)) / elapsed, 1),
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
