from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)

    required_sections = {"data", "model", "train", "inference", "debug"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"В конфигурации отсутствуют секции: {sorted(missing)}")
    return config
