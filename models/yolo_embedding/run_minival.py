"""Runs every model x preprocessing mode on the mini-val split and stores metrics.

Everything except the extractor is shared: the same crops, the same preprocessing per mode, the same FAISS
index type, the same threshold selection rule. Results -> docs/results/yolo_minival.json.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from . import metrics as M
from .data import PREPROCESS_MODES, REPO_DIR, load_crops, preprocess_batch, read_table
from .extractor import MODELS, YoloEmbedder, weights_size_mb
from .retrieval import search, write_candidates, write_embeddings, write_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--modes", nargs="+", default=list(PREPROCESS_MODES))
    ap.add_argument("--margin", type=float, default=0.0, help="own optional bbox padding, off by default")
    ap.add_argument("--official", action="store_true", help="also run the organizers' evaluate.py on artifacts")
    ap.add_argument("--data-dir", default=str(REPO_DIR / "data"), help="dir with mini_query/mini_gallery/mini_gt csv")
    ap.add_argument("--out", default=str(REPO_DIR / "docs/results/yolo_minival.json"))
    args = ap.parse_args()

    data = Path(args.data_dir)
    q_df, g_df = read_table(data / "mini_query.csv"), read_table(data / "mini_gallery.csv")
    gt = read_table(data / "mini_gt.csv").set_index("image_id")
    q_vid, q_cam = gt.loc[q_df.image_id, "vehicle_id"].to_numpy(), gt.loc[q_df.image_id, "camera_id"].to_numpy()
    g_vid, g_cam = gt.loc[g_df.image_id, "vehicle_id"].to_numpy(), gt.loc[g_df.image_id, "camera_id"].to_numpy()
    q_ids, g_ids = q_df.image_id.tolist(), g_df.image_id.tolist()

    t = time.time()
    q_crops, g_crops = load_crops(q_df, margin=args.margin), load_crops(g_df, margin=args.margin)
    print(f"crops loaded: {len(q_crops)} query + {len(g_crops)} gallery in {time.time() - t:.1f}s")

    results = []
    for mode in args.modes:
        q_x, g_x = preprocess_batch(q_crops, mode), preprocess_batch(g_crops, mode)
        for name in args.models:
            emb = YoloEmbedder(name)
            q_emb, g_emb = emb.embed(q_x), emb.embed(g_x)
            del emb
            torch.cuda.empty_cache()

            top_s, top_i = search(q_emb, g_emb)  # as predict.py: submission.csv and candidates.csv come from it
            rank = M.ranking_metrics(top_i, q_vid, q_cam, g_vid, g_cam)
            full = M.full_ranking_metrics(q_emb, g_emb, q_vid, q_cam, g_vid, g_cam)
            score, correct, has_match = M.top1_decision_inputs(top_s, top_i, q_vid, q_cam, g_vid, g_cam)
            cand, curve = M.sweep_threshold(score, correct, has_match)
            res = {"model": name, "mode": mode, "margin": args.margin, "weights_mb": round(weights_size_mb(name), 1),
                   "dim": q_emb.shape[1], **rank, **full, "candidates": cand, "threshold_curve": curve}

            if args.official:
                run = REPO_DIR / "runs/yolo_embedding/minival" / mode / name
                run.mkdir(parents=True, exist_ok=True)
                write_submission(run / "submission.csv", q_ids, g_ids, top_i)
                write_candidates(run / "candidates.csv", q_ids, g_ids, top_s, top_i, cand["threshold"])
                write_embeddings(run / "embeddings.npy", q_emb, g_emb)
                res["official"] = M.run_official(
                    str(data / "mini_gt.csv"), str(run / "submission.csv"), str(run / "candidates.csv"),
                    str(run / "embeddings.npy"), str(data / "mini_query.csv"), str(data / "mini_gallery.csv"),
                    str(run / "report.json"))

            res["per_query_ap10"] = M.per_query_ap(top_i, q_vid, q_cam, g_vid, g_cam).tolist()
            results.append(res)
            print(f"{mode:12s} {name:12s} mAP@10={rank['mAP@10']:.4f} R1={rank['Rank-1']:.4f} "
                  f"R5={rank['Rank-5']:.4f} mAP_full={full['mAP_full']:.4f} mINP={full['mINP']:.4f} "
                  f"F1={cand['F1']:.4f} TNR={cand['TNR']:.4f} thr={cand['threshold']:.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
