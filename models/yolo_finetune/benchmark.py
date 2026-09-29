"""Speed of frozen vs fine-tuned embedders, measured side by side under the same conditions.

    python -m models.yolo_finetune.benchmark --embedders yolo26l-cls models/yolo_finetune/weights/yolo26l-cls-reid.pt

Same quantities and procedure as models.yolo_embedding.benchmark (reuses its `timed`): batch=1 latency from an
in-memory PIL crop on GPU and CPU, batched GPU throughput, peak CUDA memory. The fine-tuned net differs from the
frozen one only by a BatchNorm1d on the 1280-d vector, so speeds should match; they are measured together
because the absolute numbers depend on what else runs on the GPU at the time. Fine-tuned embedders run with
flip TTA (one forward pass on a doubled batch) unless --no-flip is given.
"""

import argparse
import json
import subprocess

import torch
from models.yolo_embedding.benchmark import timed
from models.yolo_embedding.data import REPO_DIR, load_crops, preprocess, read_table

from .evaluate import make_embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedders", nargs="+", required=True)
    ap.add_argument("--mode", default="center_crop")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--no-flip", action="store_true", help="fine-tuned embedders without flip TTA")
    ap.add_argument("--out", default=str(REPO_DIR / "docs/results/yolo_finetune_speed.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = True

    crops = load_crops(read_table(REPO_DIR / "data/mini_query.csv").head(args.batch))
    batch = torch.stack([preprocess(c, args.mode) for c in crops]).pin_memory()
    gpu_load = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip()
    out = {"device": torch.cuda.get_device_name(0), "gpu_load_before_start": gpu_load,
           "cpu_threads": args.cpu_threads, "batch": args.batch, "mode": args.mode, "models": []}
    print("GPU utilization / memory before start:", gpu_load)
    for spec in args.embedders:
        gpu = make_embedder(spec, "cuda", flip=not args.no_flip)
        one = lambda e: e(preprocess(crops[0], args.mode)[None]).cpu()
        lat_gpu = timed(lambda: one(gpu), args.iters, 30, True) * 1e3
        torch.cuda.reset_peak_memory_stats()
        fps = args.batch / timed(lambda: gpu(batch).cpu(), 30, 5, True)
        mem = torch.cuda.max_memory_allocated() / 2**20
        del gpu
        torch.cuda.empty_cache()
        cpu = make_embedder(spec, "cpu", flip=not args.no_flip)
        lat_cpu = timed(lambda: one(cpu), max(20, args.iters // 5), 5, False) * 1e3
        r = {"embedder": spec, "flip": spec.endswith(".pt") and not args.no_flip, "latency_gpu_ms": lat_gpu,
             "latency_cpu_ms": lat_cpu, "fps_gpu": fps, "peak_mem_mb": mem}
        out["models"].append(r)
        print(f"{spec:55s} gpu b1 {lat_gpu:6.2f} ms | cpu b1 {lat_cpu:7.2f} ms | gpu b{args.batch} {fps:7.0f} img/s "
              f"| peak {mem:.0f} MB", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
