from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VehicleRecord:
    image_id: str
    x: int
    y: int
    w: int
    h: int
    vehicle_id: str | None = None
    camera_id: str | None = None


@dataclass(frozen=True)
class IdentitySplit:
    train: list[VehicleRecord]
    query: list[VehicleRecord]
    gallery: list[VehicleRecord]
    train_ids: list[str]
    matched_validation_ids: list[str]
    open_set_validation_ids: list[str]
    seed: int

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "identity_disjoint": True,
            "train_identities": self.train_ids,
            "matched_validation_identities": self.matched_validation_ids,
            "open_set_validation_identities": self.open_set_validation_ids,
            "counts": {
                "train_records": len(self.train),
                "query_records": len(self.query),
                "gallery_records": len(self.gallery),
            },
        }


def read_records(path: str | Path, *, require_labels: bool) -> list[VehicleRecord]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "x", "y", "w", "h"}
        if require_labels:
            required.update({"vehicle_id", "camera_id"})
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В {csv_path} отсутствуют колонки: {sorted(missing)}")

        records: list[VehicleRecord] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                record = VehicleRecord(
                    image_id=str(row["image_id"]).strip(),
                    x=int(row["x"]),
                    y=int(row["y"]),
                    w=int(row["w"]),
                    h=int(row["h"]),
                    vehicle_id=_optional(row.get("vehicle_id")),
                    camera_id=_optional(row.get("camera_id")),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Некорректная строка {row_number} в {csv_path}") from exc
            if not record.image_id or record.w <= 0 or record.h <= 0:
                raise ValueError(f"Некорректная строка {row_number} в {csv_path}")
            records.append(record)
    return records


def _optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def write_records(path: str | Path, records: list[VehicleRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    include_labels = any(item.vehicle_id is not None for item in records)
    fields = ["image_id", "x", "y", "w", "h"]
    if include_labels:
        fields.extend(["vehicle_id", "camera_id"])
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            writer.writerow({field: row[field] for field in fields})


def save_split(path: str | Path, split: IdentitySplit) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(split.manifest(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_identity_split(
    records: list[VehicleRecord],
    *,
    seed: int,
    validation_fraction: float,
    open_set_fraction: float,
    max_train_identities: int = 0,
    max_validation_identities: int = 0,
) -> IdentitySplit:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction должна быть между 0 и 1")
    if not 0.0 <= open_set_fraction < 1.0:
        raise ValueError("open_set_fraction должна быть в диапазоне [0, 1)")

    groups: dict[str, list[VehicleRecord]] = defaultdict(list)
    for record in records:
        if record.vehicle_id is None or record.camera_id is None:
            raise ValueError("Для identity split нужны vehicle_id и camera_id")
        groups[record.vehicle_id].append(record)

    all_ids = sorted(groups)
    if len(all_ids) < 4:
        raise ValueError("Для разбиения требуется не менее четырёх идентичностей")

    rng = random.Random(seed)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)
    validation_count = max(2, round(len(all_ids) * validation_fraction))
    if max_validation_identities:
        validation_count = min(validation_count, max_validation_identities)

    eligible_matched = [
        identity for identity in shuffled if len({item.camera_id for item in groups[identity]}) >= 2
    ]
    open_count = round(validation_count * open_set_fraction)
    if open_set_fraction > 0:
        open_count = max(1, open_count)
    matched_count = validation_count - open_count
    if matched_count < 1 or len(eligible_matched) < matched_count:
        raise ValueError("Недостаточно межкамерных идентичностей для validation")

    matched_ids = eligible_matched[:matched_count]
    matched_set = set(matched_ids)
    remaining = [identity for identity in shuffled if identity not in matched_set]
    open_ids = remaining[:open_count]
    validation_ids = matched_set.union(open_ids)
    train_ids = [identity for identity in shuffled if identity not in validation_ids]
    if max_train_identities:
        train_ids = train_ids[:max_train_identities]
    if not train_ids:
        raise ValueError("После разбиения обучающая часть пуста")

    train = [record for identity in train_ids for record in groups[identity]]
    query: list[VehicleRecord] = []
    gallery: list[VehicleRecord] = []

    for identity in matched_ids:
        candidates = groups[identity].copy()
        rng.shuffle(candidates)
        query_record = candidates[0]
        if not any(item.camera_id != query_record.camera_id for item in candidates[1:]):
            query_record = next(
                item
                for item in candidates
                if any(other.camera_id != item.camera_id for other in candidates)
            )
        query.append(query_record)
        gallery.extend(item for item in candidates if item.image_id != query_record.image_id)

    for identity in open_ids:
        query.append(rng.choice(groups[identity]))

    rng.shuffle(train)
    rng.shuffle(query)
    rng.shuffle(gallery)
    return IdentitySplit(
        train=train,
        query=query,
        gallery=gallery,
        train_ids=sorted(train_ids),
        matched_validation_ids=sorted(matched_ids),
        open_set_validation_ids=sorted(open_ids),
        seed=seed,
    )
