"""Speed of embedding extraction (reference numbers; the official measurement is done by the organizers).

  latency_gpu_ms  - batch=1, from an in-memory PIL crop: preprocess + H2D + forward + L2-norm + D2H, median
  latency_cpu_ms  - same on CPU (torch intra-op threads = --cpu-threads)
  fps_gpu         - batched throughput (--batch) from preprocessed CPU tensors incl. H2D/D2H, images/s
  peak_mem_mb     - peak CUDA memory allocated during the batched run
JPEG decode + crop is excluded from all of these and measured once separately (decode_crop_ms).
Results -> docs/results/yolo_speed.json
"""

import argparse
import json
import time

import numpy as np
import torch

from .data import REPO_DIR, load_crop, load_crops, preprocess, read_table
from .extractor import MODELS, YoloEmbedder


def timed(fn, n, warmup, cuda):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--mode", default="center_crop")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--out", default=str(REPO_DIR / "docs/results/yolo_speed.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = True

    df = read_table(REPO_DIR / "data/mini_query.csv").head(args.batch)
    crops = load_crops(df)
    rows = list(df[["image_id", "x", "y", "w", "h"]].itertuples(index=False, name=None))
    decode_ms = timed(lambda: load_crop(*rows[0]), 50, 5, False) * 1e3
    batch = torch.stack([preprocess(c, args.mode) for c in crops]).pin_memory()

    out = {"device": torch.cuda.get_device_name(0), "cpu_threads": args.cpu_threads, "batch": args.batch,
           "mode": args.mode, "decode_crop_ms": decode_ms, "models": []}
    print(f"decode+crop of one 1920x1080 JPEG: {decode_ms:.2f} ms")
    for name in args.models:
        gpu = YoloEmbedder(name, "cuda")
        one = lambda e: e(preprocess(crops[0], args.mode)[None]).cpu()
        lat_gpu = timed(lambda: one(gpu), args.iters, 30, True) * 1e3
        torch.cuda.reset_peak_memory_stats()
        fps = args.batch / timed(lambda: gpu(batch).cpu(), 30, 5, True)
        mem = torch.cuda.max_memory_allocated() / 2**20
        del gpu
        torch.cuda.empty_cache()
        cpu = YoloEmbedder(name, "cpu")
        lat_cpu = timed(lambda: one(cpu), max(20, args.iters // 5), 5, False) * 1e3
        r = {"model": name, "latency_gpu_ms": lat_gpu, "latency_cpu_ms": lat_cpu, "fps_gpu": fps, "peak_mem_mb": mem}
        out["models"].append(r)
        print(f"{name:12s} gpu b1 {lat_gpu:6.2f} ms | cpu b1 {lat_cpu:7.2f} ms | gpu b{args.batch} {fps:7.0f} img/s "
              f"| peak {mem:.0f} MB", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
