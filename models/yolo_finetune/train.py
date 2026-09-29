"""Fine-tunes a YOLO-cls backbone for vehicle Re-ID: margin ID loss + batch-hard triplet, P x K sampling.

    python -m models.yolo_finetune.train --model yolo26l-cls --split-dir data --run-name yolo26l_s42

Loss = w_id * CosFace/ArcFace(emb, vehicle_id) + w_tri * BatchHardTriplet(feat, vehicle_id).
Val mAP@10 (same metric code as the frozen baseline, organizers' protocol) is computed every --eval-every
epochs; the best epoch is kept as best.pt, training stops after --patience epochs without improvement.
Everything needed to continue after a crash is in last.pt (--resume).

--full-train trains the final model on every vehicle of train.csv: no validation, no early stopping, and best.pt
is the last epoch. The epoch count is fixed in advance: with this config the last of 60 epochs is within 0.005
mAP@10 of the best one on all three mini-val splits (a plateau from epoch ~54 on), so no selection is needed.

Run dir (runs/yolo_finetune/<run-name>/): config.json, history.jsonl, best.pt (embedding net only, fp32),
last.pt (+ ID head, optimizer, scheduler, RNG), summary.json.
"""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from models.yolo_embedding.data import REPO_DIR

from .data import GpuAugment, full_train_split, load_split, preprocessed_train
from .evaluate import score
from .losses import batch_hard_triplet
from .model import MarginHead, ReIDNet, save_embedding_net
from .sampler import CameraAwarePKSampler


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26l-cls")
    ap.add_argument("--split-dir", default="data")
    ap.add_argument("--full-train", action="store_true",
                    help="train on all of train.csv, no validation; best.pt = last epoch (see module doc)")
    ap.add_argument("--mode", default="center_crop", help="deterministic preprocessing, same as inference")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--no-camera-aware", action="store_true", help="ablation: plain random K images per id")
    ap.add_argument("--lr", type=float, default=3.5e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--id-loss", choices=["cosface", "arcface"], default="cosface")
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--margin", type=float, default=None, help="default 0.35 for cosface, 0.5 for arcface")
    ap.add_argument("--w-id", type=float, default=1.0)
    ap.add_argument("--w-tri", type=float, default=1.0)
    ap.add_argument("--tri-margin", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=16, help="epochs without val mAP@10 improvement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if args.margin is None:
        args.margin = 0.35 if args.id_loss == "cosface" else 0.5
    return args


def lr_at(step, total, warmup, base):
    """Linear warmup from 0.1 * base, then cosine decay to 0 (per iteration)."""
    if step < warmup:
        return base * (0.1 + 0.9 * step / warmup)
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def val_metrics(net, images, sp, bs=256):
    net.eval()
    emb = lambda rows: torch.cat([
        F.normalize(net(images[rows[i : i + bs]].float() / 255)[1].float(), dim=1).cpu()
        for i in range(0, len(rows), bs)]).numpy()
    res = score(emb(sp["q_rows"]), emb(sp["g_rows"]), sp)
    net.train()
    return res


def main(argv=None):
    args = parse_args(argv)
    run = REPO_DIR / "runs/yolo_finetune" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    sp = full_train_split() if args.full_train else load_split(args.split_dir)
    images = preprocessed_train(args.mode).to(dev)  # uint8, ~1.4 GB, whole train.csv
    classes = np.unique(sp["train_vid"])
    labels_np = np.searchsorted(classes, sp["train_vid"])
    train_rows = torch.as_tensor(np.array(sp["train_rows"]), device=dev)
    labels = torch.as_tensor(labels_np, device=dev)
    sampler = CameraAwarePKSampler(labels_np, sp["train_cam"], args.P, args.K, args.seed, not args.no_camera_aware)

    net = ReIDNet(args.model).to(dev, memory_format=torch.channels_last).train()
    head = MarginHead(net.dim, len(classes), args.id_loss, args.scale, args.margin).to(dev)
    params = [p for p in list(net.parameters()) + list(head.parameters()) if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    aug, gen = GpuAugment(), torch.Generator(device=dev).manual_seed(args.seed)

    steps_per_epoch = len(sampler.batches(0))
    total, warmup = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    cfg = {**vars(args), "n_train_ids": len(classes), "n_train_images": len(labels_np),
           "steps_per_epoch": steps_per_epoch, "val_seed": sp["split"]["seed"],
           "split_dir": str(sp["dir"]) if sp["dir"] else None}
    start_epoch, best, best_epoch, step = 1, -1.0, 0, 0

    if args.resume and (run / "last.pt").exists():
        st = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        net.load_state_dict(st["net"]), head.load_state_dict(st["head"]), opt.load_state_dict(st["opt"])
        gen.set_state(st["gen"].cpu() if hasattr(st["gen"], "cpu") else st["gen"])
        torch.set_rng_state(st["torch_rng"].cpu())
        start_epoch, best, best_epoch, step = st["epoch"] + 1, st["best"], st["best_epoch"], st["step"]
        print(f"resumed from epoch {st['epoch']} (best mAP@10 {best:.4f} @ {best_epoch})", flush=True)
    else:
        (run / "config.json").write_text(json.dumps(cfg, indent=1))
        (run / "history.jsonl").write_text("")
        if not args.full_train:
            m0 = val_metrics(net, images, sp)  # epoch 0 = frozen ImageNet embedding + untrained BNNeck (identity)
            with open(run / "history.jsonl", "a") as f:
                f.write(json.dumps({"epoch": 0, "val": m0}) + "\n")
            print(f"epoch 0 (no training): val mAP@10={m0['mAP@10']:.4f} R1={m0['Rank-1']:.4f}", flush=True)
    print(json.dumps(cfg), flush=True)

    t_run = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        batches = sampler.batches(epoch)
        cross = sampler.cross_camera_pair_rate(batches)
        sums = {"loss": 0.0, "id": 0.0, "tri": 0.0, "id_acc": 0.0, "tri_acc": 0.0}
        for b in batches:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, args.lr)
            b = torch.as_tensor(b, device=dev)
            x = aug(images[train_rows[b]].float() / 255, gen).contiguous(memory_format=torch.channels_last)
            y = labels[b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feat, emb = net(x)
            logits, cos = head(emb, y)
            l_id = F.cross_entropy(logits, y)
            l_tri, tri_acc = batch_hard_triplet(feat, y, args.tri_margin)
            loss = args.w_id * l_id + args.w_tri * l_tri
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            for k, v in (("loss", loss), ("id", l_id), ("tri", l_tri), ("tri_acc", tri_acc),
                         ("id_acc", (cos.argmax(1) == y).float().mean())):
                sums[k] += v.item()
        rec = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0, 1),
               "cross_camera_pos_pairs": round(float(cross), 4),
               **{k: round(v / len(batches), 5) for k, v in sums.items()}}
        if not np.isfinite(rec["loss"]):
            raise SystemExit(f"non-finite loss at epoch {epoch}")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if args.full_train:
                if epoch == args.epochs:
                    best_epoch = epoch
                    save_embedding_net(net, run / "best.pt", {"epoch": epoch, "val": None, "config": cfg})
            else:
                m = val_metrics(net, images, sp)
                rec["val"] = m
                if m["mAP@10"] > best:
                    best, best_epoch = m["mAP@10"], epoch
                    save_embedding_net(net, run / "best.pt", {"epoch": epoch, "val": m, "config": cfg})
            torch.save({"net": net.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                        "gen": gen.get_state(), "torch_rng": torch.get_rng_state(), "epoch": epoch, "best": best,
                        "best_epoch": best_epoch, "step": step}, run / "last.pt")
        with open(run / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        msg = (f"epoch {epoch:3d} {rec['seconds']:5.1f}s lr={rec['lr']:.2e} loss={rec['loss']:.4f} "
               f"id={rec['id']:.4f} tri={rec['tri']:.4f} id_acc={rec['id_acc']:.3f} tri_acc={rec['tri_acc']:.3f} "
               f"xcam={cross:.3f}")
        if "val" in rec:
            v = rec["val"]
            msg += (f" | val mAP@10={v['mAP@10']:.4f} R1={v['Rank-1']:.4f} mINP={v['mINP']:.4f} "
                    f"F1={v['candidates']['F1']:.4f} thr={v['candidates']['threshold']:.3f} "
                    f"(best {best:.4f} @ {best_epoch})")
        print(msg, flush=True)
        if not args.full_train and epoch - best_epoch >= args.patience:
            print(f"early stop: no val mAP@10 improvement for {args.patience} epochs", flush=True)
            break

    summary = {"run": args.run_name, "best_mAP@10": None if args.full_train else best, "best_epoch": best_epoch,
               "last_epoch": epoch,
               "train_minutes_this_session": round((time.time() - t_run) / 60, 1),
               "best_pt_mb": round((run / "best.pt").stat().st_size / 2**20, 1)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
