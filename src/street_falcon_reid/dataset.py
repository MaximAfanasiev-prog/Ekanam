from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .records import VehicleRecord

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(height: int, width: int, *, training: bool) -> transforms.Compose:
    operations: list[object] = [transforms.Resize((height, width), antialias=True)]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if training:
        operations.append(
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random")
        )
    return transforms.Compose(operations)


class VehicleDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        records: list[VehicleRecord],
        data_dir: str | Path,
        transform: transforms.Compose,
        *,
        crop_margin: float,
        label_map: dict[str, int] | None = None,
    ) -> None:
        self.records = records
        self.images_dir = Path(data_dir) / "images"
        self.transform = transform
        self.crop_margin = crop_margin
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = self.images_dir / f"{record.image_id}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image = crop_vehicle(image, record, self.crop_margin)
            tensor = self.transform(image)
        label = self.label_map.get(record.vehicle_id or "", -1)
        return {"image": tensor, "label": label, "image_id": record.image_id}


def crop_vehicle(image: Image.Image, record: VehicleRecord, margin: float) -> Image.Image:
    dx = round(record.w * margin)
    dy = round(record.h * margin)
    left = max(0, record.x - dx)
    top = max(0, record.y - dy)
    right = min(image.width, record.x + record.w + dx)
    bottom = min(image.height, record.y + record.h + dy)
    if right <= left or bottom <= top:
        raise ValueError(f"Bounding box {record.image_id} не пересекается с изображением")
    return image.crop((left, top, right, bottom))


class IdentityBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: list[VehicleRecord],
        *,
        identities_per_batch: int,
        instances_per_identity: int,
        seed: int,
        steps_per_epoch: int = 0,
    ) -> None:
        self.by_identity: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            if record.vehicle_id is None:
                raise ValueError("PK sampler требует vehicle_id")
            self.by_identity[record.vehicle_id].append(index)
        self.identities = sorted(self.by_identity)
        if len(self.identities) < identities_per_batch:
            raise ValueError("Идентичностей меньше, чем identities_per_batch")
        self.identities_per_batch = identities_per_batch
        self.instances_per_identity = instances_per_identity
        self.seed = seed
        batch_size = identities_per_batch * instances_per_identity
        self.steps = steps_per_epoch or math.ceil(len(records) / batch_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.steps):
            identities = rng.sample(self.identities, self.identities_per_batch)
            batch: list[int] = []
            for identity in identities:
                options = self.by_identity[identity]
                if len(options) >= self.instances_per_identity:
                    batch.extend(rng.sample(options, self.instances_per_identity))
                else:
                    batch.extend(rng.choices(options, k=self.instances_per_identity))
            rng.shuffle(batch)
            yield batch
