from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import IdentityBatchSampler, VehicleDataset, build_transform
from .metrics import calibrate_open_set, cosine_similarity, ranking_metrics
from .model import ReIDModel, batch_hard_triplet_loss
from .prepare import sha256_file
from .records import make_identity_split, read_records, save_split, write_records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def extract_embeddings(
    model: ReIDModel,
    dataset: VehicleDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    embeddings: list[np.ndarray] = []
    image_ids: list[str] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            vectors, _ = model(images)
            embeddings.append(vectors.float().cpu().numpy())
            image_ids.extend(batch["image_id"])
    return np.concatenate(embeddings, axis=0), image_ids


def train_baseline(
    data_dir: str | Path,
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    config_path: str | Path,
) -> Path:
    data_root = Path(data_dir).resolve()
    output = Path(run_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "config.toml")

    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    inference_config = config["inference"]
    debug_config = config["debug"]
    seed = int(data_config["seed"])
    seed_everything(seed)

    all_records = read_records(data_root / "train.csv", require_labels=True)
    split = make_identity_split(
        all_records,
        seed=seed,
        validation_fraction=float(data_config["validation_fraction"]),
        open_set_fraction=float(data_config["open_set_fraction"]),
        max_train_identities=int(debug_config["max_train_identities"]),
        max_validation_identities=int(debug_config["max_validation_identities"]),
    )
    save_split(output / "split.json", split)
    write_records(output / "validation-query.csv", split.query)
    write_records(output / "validation-gallery.csv", split.gallery)

    label_map = {identity: index for index, identity in enumerate(split.train_ids)}
    train_transform = build_transform(
        int(data_config["image_height"]),
        int(data_config["image_width"]),
        training=True,
    )
    eval_transform = build_transform(
        int(data_config["image_height"]),
        int(data_config["image_width"]),
        training=False,
    )
    train_dataset = VehicleDataset(
        split.train,
        data_root,
        train_transform,
        crop_margin=float(data_config["crop_margin"]),
        label_map=label_map,
    )
    sampler = IdentityBatchSampler(
        split.train,
        identities_per_batch=int(train_config["identities_per_batch"]),
        instances_per_identity=int(train_config["instances_per_identity"]),
        seed=seed,
        steps_per_epoch=int(train_config["steps_per_epoch"]),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(train_config["workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(train_config["workers"]) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReIDModel(
        backbone=str(model_config["backbone"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_classes=len(label_map),
        pretrained=bool(model_config["pretrained"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_config["epochs"])
    )
    use_amp = bool(train_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    classification_loss = nn.CrossEntropyLoss(
        label_smoothing=float(train_config["label_smoothing"])
    )

    validation_query = VehicleDataset(
        split.query,
        data_root,
        eval_transform,
        crop_margin=float(data_config["crop_margin"]),
    )
    validation_gallery = VehicleDataset(
        split.gallery,
        data_root,
        eval_transform,
        crop_margin=float(data_config["crop_margin"]),
    )

    history_path = output / "history.jsonl"
    checkpoint_path = output / "checkpoint-best.pt"
    best_key = (-1.0, -1.0)
    epochs = int(train_config["epochs"])
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                embeddings, logits = model(images)
                ce_loss = classification_loss(logits, labels)
                triplet_loss = batch_hard_triplet_loss(
                    embeddings,
                    labels,
                    margin=float(train_config["triplet_margin"]),
                )
                loss = ce_loss + float(train_config["triplet_weight"]) * triplet_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
        scheduler.step()

        query_embeddings, _ = extract_embeddings(
            model,
            validation_query,
            batch_size=int(inference_config["batch_size"]),
            workers=int(inference_config["workers"]),
            device=device,
        )
        gallery_embeddings, _ = extract_embeddings(
            model,
            validation_gallery,
            batch_size=int(inference_config["batch_size"]),
            workers=int(inference_config["workers"]),
            device=device,
        )
        similarities = cosine_similarity(query_embeddings, gallery_embeddings)
        ranking = ranking_metrics(
            split.query,
            split.gallery,
            similarities,
            top_k=int(inference_config["top_k"]),
        )
        threshold, refusal = calibrate_open_set(split.query, split.gallery, similarities)
        elapsed = time.perf_counter() - epoch_started
        report = {
            "epoch": epoch,
            "loss": running_loss / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "ranking": ranking,
            "refusal": refusal,
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)

        key = (float(ranking[f"mAP@{inference_config['top_k']}"]), float(refusal["F1"]))
        if key > best_key:
            best_key = key
            checkpoint = {
                "schema_version": 1,
                "epoch": epoch,
                "model": model.state_dict(),
                "model_config": {
                    "backbone": str(model_config["backbone"]),
                    "embedding_dim": int(model_config["embedding_dim"]),
                    "num_classes": len(label_map),
                },
                "data_config": dict(data_config),
                "open_set_threshold": threshold,
                "validation": report,
                "split": split.manifest(),
                "source_commit": _source_commit(),
            }
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)

    validation_report = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_key": {"mAP_at_10": best_key[0], "F1": best_key[1]},
        "device": str(device),
        "torch": torch.__version__,
    }
    (output / "validation-report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return checkpoint_path


def _source_commit() -> str:
    explicit = os.environ.get("GIT_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
