"""Builds the self-made open-set mini-val split from train.csv (the only table with vehicle_id).

1. Unique vehicle_ids are split 80/20 into train/val parts without overlap (seeded).
2. Val keeps only vehicle_ids with >= 2 images (in this dataset every id has 4-8, so nothing is dropped).
3. For each val id the images are shuffled; floor(n/2) go to query, the rest to gallery.
4. `--openset-frac` of val ids are "distractors": their gallery images are removed, so their queries have
   no pair in the gallery. Without them TNR (refusal quality) cannot be measured at all.

Outputs (under data/):
  mini_val_split.json - seed, parameters, id lists, query/gallery image_ids (reusable by other branches)
  mini_query.csv, mini_gallery.csv - same columns as test_query.csv / test_gallery.csv
  mini_gt.csv - ground truth in the official evaluate.py format: image_id,vehicle_id,camera_id,split
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASET_DIR, REPO_DIR, read_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--openset-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", default=str(REPO_DIR / "data"))
    args = ap.parse_args()

    train = read_table(DATASET_DIR / "train.csv")
    rng = np.random.default_rng(args.seed)

    ids = np.sort(train.vehicle_id.unique())
    rng.shuffle(ids)
    n_val = int(round(len(ids) * args.val_frac))
    val_ids, train_ids = np.sort(ids[:n_val]), np.sort(ids[n_val:])

    counts = train[train.vehicle_id.isin(val_ids)].groupby("vehicle_id").size()
    dropped = counts[counts < 2].index.tolist()
    val_ids = val_ids[~np.isin(val_ids, dropped)]

    shuffled = val_ids.copy()
    rng.shuffle(shuffled)
    openset_ids = np.sort(shuffled[: int(round(len(val_ids) * args.openset_frac))])

    query, gallery = [], []
    for vid in val_ids:
        imgs = train[train.vehicle_id == vid].image_id.to_numpy().copy()
        rng.shuffle(imgs)
        n_q = len(imgs) // 2
        query += imgs[:n_q].tolist()
        if vid not in openset_ids:
            gallery += imgs[n_q:].tolist()

    rows = train.set_index("image_id")
    q_df, g_df = rows.loc[query].reset_index(), rows.loc[gallery].reset_index()
    cols = ["image_id", "x", "y", "w", "h"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q_df[cols].to_csv(out / "mini_query.csv", index=False)
    g_df[cols].to_csv(out / "mini_gallery.csv", index=False)
    gt = pd.concat([q_df.assign(split="query"), g_df.assign(split="gallery")])
    gt[["image_id", "vehicle_id", "camera_id", "split"]].to_csv(out / "mini_gt.csv", index=False)

    split = {
        "description": "Self-made open-set mini-val from train.csv (NOT the official test). "
                       "query/gallery lists are in the order of mini_query.csv / mini_gallery.csv.",
        "source": str(DATASET_DIR / "train.csv"),
        "seed": args.seed,
        "val_frac": args.val_frac,
        "openset_frac": args.openset_frac,
        "query_rule": "per val id: shuffle images, floor(n/2) -> query, rest -> gallery; open-set ids: gallery dropped",
        "n_train_ids": len(train_ids),
        "n_val_ids": len(val_ids),
        "n_openset_ids": len(openset_ids),
        "dropped_val_ids_lt2_images": dropped,
        "n_query": len(query),
        "n_gallery": len(gallery),
        "train_vehicle_ids": train_ids.tolist(),
        "val_vehicle_ids": val_ids.tolist(),
        "openset_vehicle_ids": openset_ids.tolist(),
        "query_image_ids": query,
        "gallery_image_ids": gallery,
    }
    (out / "mini_val_split.json").write_text(json.dumps(split, indent=1))
    print({k: v for k, v in split.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
