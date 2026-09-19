from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path

EXPECTED_TABLE_ROWS = {
    "train.csv": 9_556,
    "test_query.csv": 1_110,
    "test_gallery.csv": 750,
}
EXPECTED_IMAGES = 11_416


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    root = destination.resolve()
    members = archive.infolist()
    for member in members:
        target = (destination / member.filename).resolve()
        if os.path.commonpath([root, target]) != str(root):
            raise ValueError(f"Небезопасный путь в ZIP: {member.filename}")
        if member.is_dir():
            continue
        mode = member.external_attr >> 16
        if mode and (mode & 0o170000) == 0o120000:
            raise ValueError(f"Символическая ссылка в ZIP запрещена: {member.filename}")
    return members


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader, None)
        return sum(1 for row in reader if row)


def validate_prepared_dataset(data_dir: str | Path) -> dict[str, object]:
    root = Path(data_dir)
    missing = [name for name in EXPECTED_TABLE_ROWS if not (root / name).is_file()]
    if not (root / "images").is_dir():
        missing.append("images/")
    if missing:
        raise ValueError(f"В подготовленном датасете отсутствуют: {missing}")

    rows = {name: _row_count(root / name) for name in EXPECTED_TABLE_ROWS}
    unexpected = {
        name: {"expected": EXPECTED_TABLE_ROWS[name], "actual": count}
        for name, count in rows.items()
        if count != EXPECTED_TABLE_ROWS[name]
    }
    if unexpected:
        raise ValueError(f"Неожиданное число строк: {unexpected}")

    image_count = sum(1 for path in (root / "images").iterdir() if path.suffix.lower() == ".jpg")
    if image_count != EXPECTED_IMAGES:
        raise ValueError(f"Ожидалось {EXPECTED_IMAGES} JPEG, найдено {image_count}")
    return {"table_rows": rows, "images": image_count}


def prepare_archive(
    archive_path: str | Path,
    data_dir: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    archive = Path(archive_path).resolve()
    destination = Path(data_dir).resolve()
    marker = destination / ".prepared.json"

    if marker.is_file():
        existing = json.loads(marker.read_text(encoding="utf-8"))
        if existing.get("archive_sha256") == expected_sha256:
            validate_prepared_dataset(destination)
            return existing

    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SHA-256 архива не совпал: ожидался {expected_sha256}, получен {actual_sha256}"
        )

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"Каталог {destination} не пуст и не содержит подходящего .prepared.json"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.extracting"
    if temporary.exists():
        raise FileExistsError(
            f"Остался временный каталог {temporary}; проверьте его и удалите вручную"
        )
    temporary.mkdir(mode=0o700)

    try:
        with zipfile.ZipFile(archive) as source:
            members = _safe_members(source, temporary)
            source.extractall(temporary, members=members)
        validation = validate_prepared_dataset(temporary)
        metadata = {
            "schema_version": 1,
            "archive": archive.name,
            "archive_sha256": actual_sha256,
            **validation,
        }
        (temporary / ".prepared.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            destination.rmdir()
        temporary.rename(destination)
        return metadata
    except Exception:
        raise
