"""Converts another branch's open-set validation split into the make_split format, read-only.

    python -m models.yolo_finetune.import_split \
        --run /home/andrey/datasets/lct26-street-falcon-reid/runs/resnet50-baseline-93f4287 \
        --out data/splits/resnet50-baseline

The ResNet-branch run stores split.json (train / matched / open-set vehicle_ids) and validation-query.csv /
validation-gallery.csv (with vehicle_id, camera_id). Training a YOLO model on exactly its train ids and scoring
on exactly its query/gallery makes the two branches directly comparable. Nothing in the source run is modified.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from models.yolo_embedding.data import DATASET_DIR, read_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    src = json.loads((run / "split.json").read_text())
    q, g = read_table(run / "validation-query.csv"), read_table(run / "validation-gallery.csv")

    train_ids = sorted(int(v) for v in src["train_identities"])
    matched = sorted(int(v) for v in src["matched_validation_identities"])
    openset = sorted(int(v) for v in src["open_set_validation_identities"])
    val_ids = sorted(matched + openset)
    assert not set(train_ids) & set(val_ids)
    assert set(q.vehicle_id) | set(g.vehicle_id) <= set(val_ids)

    # the bbox and labels must be the ones of train.csv (same images, same annotation)
    ref = read_table(DATASET_DIR / "train.csv").set_index("image_id")
    for df in (q, g):
        cols = ["x", "y", "w", "h", "vehicle_id", "camera_id"]
        assert (ref.loc[df.image_id, cols].to_numpy() == df[cols].to_numpy()).all()

    out.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "x", "y", "w", "h"]
    q[cols].to_csv(out / "mini_query.csv", index=False)
    g[cols].to_csv(out / "mini_gallery.csv", index=False)
    gt = pd.concat([q.assign(split="query"), g.assign(split="gallery")])
    gt[["image_id", "vehicle_id", "camera_id", "split"]].to_csv(out / "mini_gt.csv", index=False)
    split = {
        "description": f"Imported read-only from {run} (other branch's validation split), converted to the "
                       "make_split format.",
        "source": str(run),
        "seed": src["seed"],
        "n_train_ids": len(train_ids),
        "n_val_ids": len(val_ids),
        "n_openset_ids": len(openset),
        "n_query": len(q),
        "n_gallery": len(g),
        "train_vehicle_ids": train_ids,
        "val_vehicle_ids": val_ids,
        "openset_vehicle_ids": openset,
        "query_image_ids": q.image_id.tolist(),
        "gallery_image_ids": g.image_id.tolist(),
    }
    (out / "mini_val_split.json").write_text(json.dumps(split, indent=1))
    print({k: v for k, v in split.items() if not isinstance(v, list)})


if __name__ == "__main__":
    main()
