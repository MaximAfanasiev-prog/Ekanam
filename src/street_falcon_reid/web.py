from __future__ import annotations

import csv
import io
import json
import threading
import tomllib
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from .records import VehicleRecord, read_records

STATIC_DIR = Path(__file__).with_name("static")
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; style-src 'self'; "
        "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


class DashboardData:
    def __init__(self, data_dir: str | Path, run_dir: str | Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.submission_dir = self.run_dir / "submission"

        self.history = self._read_json_lines(self.run_dir / "history.jsonl")
        self.validation = self._read_json(self.run_dir / "validation-report.json")
        self.split = self._read_json(self.run_dir / "split.json")
        self.metadata = self._read_json(self.submission_dir / "run-metadata.json")
        self.verification = self._read_json(self.submission_dir / "verification.json")
        self.config = self._read_toml(self.run_dir / "config.toml")
        self.query = read_records(self.data_dir / "test_query.csv", require_labels=False)
        self.gallery = read_records(self.data_dir / "test_gallery.csv", require_labels=False)
        self.query_by_id = {record.image_id: record for record in self.query}
        self.gallery_by_id = {record.image_id: record for record in self.gallery}
        self.query_index = {record.image_id: index for index, record in enumerate(self.query)}
        self.gallery_index = {record.image_id: index for index, record in enumerate(self.gallery)}
        self.rankings = self._read_rankings(self.submission_dir / "submission.csv")
        self.candidates = self._read_candidates(self.submission_dir / "candidates.csv")
        self._embeddings: np.ndarray | None = None

        if not self.history:
            raise ValueError("История обучения пуста")
        expected_epochs = int(self.config.get("train", {}).get("epochs", len(self.history)))
        if len(self.history) != expected_epochs:
            raise ValueError(
                f"Ожидалось {expected_epochs} эпох, найдено {len(self.history)}"
            )
        if set(self.rankings) != set(self.query_by_id):
            raise ValueError("submission.csv не соответствует test_query.csv")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Ожидался JSON object: {path}")
        return value

    @staticmethod
    def _read_json_lines(path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"Некорректная строка в {path}")
                    rows.append(value)
        return rows

    @staticmethod
    def _read_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as stream:
            return tomllib.load(stream)

    @staticmethod
    def _read_rankings(path: Path) -> dict[str, list[str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return {row[0]: row[1:] for row in csv.reader(stream) if row}

    @staticmethod
    def _read_candidates(path: Path) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                query_id = str(row["query_id"])
                output.setdefault(
                    query_id,
                    {
                        "galleryId": str(row["gallery_id"]),
                        "confidence": float(row["confidence"]),
                    },
                )
        return output

    def summary(self) -> dict[str, Any]:
        top_k = int(self.config["inference"]["top_k"])
        best = max(self.history, key=lambda item: float(item["ranking"][f"mAP@{top_k}"]))
        completed_at = datetime.fromtimestamp(
            (self.submission_dir / "run-metadata.json").stat().st_mtime, tz=UTC
        ).isoformat()
        return {
            "title": "Street Falcon Vehicle ReID",
            "release": self.run_dir.name,
            "completedAt": completed_at,
            "boundary": (
                "Identity-disjoint validation. Это не результат скрытого leaderboard."
            ),
            "best": best,
            "history": self.history,
            "split": self.split["counts"],
            "submission": {
                **self.verification,
                "acceptedQueries": len(self.candidates),
                "refusedQueries": len(self.query) - len(self.candidates),
            },
            "model": {
                "backbone": self.config["model"]["backbone"],
                "embeddingDim": self.metadata["embedding_dim"],
                "checkpointSha256": self.metadata["checkpoint_sha256"],
                "sourceCommit": self.metadata["source_commit"],
                "threshold": self.metadata["open_set_threshold"],
                "device": self.metadata["device"],
                "torch": self.metadata["torch"],
            },
        }

    def query_list(self) -> list[dict[str, Any]]:
        return [
            {"imageId": record.image_id, "accepted": record.image_id in self.candidates}
            for record in self.query
        ]

    def query_detail(self, image_id: str) -> dict[str, Any]:
        if image_id not in self.query_by_id:
            raise KeyError(image_id)
        ranking = self.rankings.get(image_id, [])
        embeddings = self._load_embeddings()
        query_vector = embeddings[self.query_index[image_id]]
        gallery_offset = len(self.query)
        gallery_items = []
        for rank, gallery_id in enumerate(ranking, start=1):
            index = self.gallery_index[gallery_id]
            score = float(query_vector @ embeddings[gallery_offset + index])
            gallery_items.append({"rank": rank, "imageId": gallery_id, "score": score})
        candidate = self.candidates.get(image_id)
        return {
            "imageId": image_id,
            "decision": "accepted" if candidate else "refused",
            "candidate": candidate,
            "threshold": self.metadata["open_set_threshold"],
            "gallery": gallery_items,
            "groundTruthAvailable": False,
        }

    def image(self, image_id: str) -> bytes:
        record = self.query_by_id.get(image_id) or self.gallery_by_id.get(image_id)
        if record is None:
            raise KeyError(image_id)
        image_path = self.data_dir / "images" / f"{record.image_id}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            crop = self._crop(image, record)
            crop.thumbnail((960, 720), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            crop.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue()

    def _load_embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            values = np.load(self.submission_dir / "embeddings.npy", allow_pickle=False)
            values = values.astype(np.float32, copy=False)
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            self._embeddings = values / np.clip(norms, 1e-12, None)
        return self._embeddings

    @staticmethod
    def _crop(image: Image.Image, record: VehicleRecord) -> Image.Image:
        margin_x = round(record.w * 0.05)
        margin_y = round(record.h * 0.05)
        left = max(0, record.x - margin_x)
        top = max(0, record.y - margin_y)
        right = min(image.width, record.x + record.w + margin_x)
        bottom = min(image.height, record.y + record.h + margin_y)
        if right <= left or bottom <= top:
            raise ValueError(f"Некорректный bounding box: {record.image_id}")
        return image.crop((left, top, right, bottom))


class DashboardStore:
    def __init__(self, data_dir: str | Path, run_dir: str | Path) -> None:
        self.data_dir = data_dir
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.data: DashboardData | None = None
        self.error: str | None = None
        self.reload()

    def reload(self) -> None:
        with self.lock:
            try:
                self.data = DashboardData(self.data_dir, self.run_dir)
                self.error = None
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self.data = None
                self.error = str(exc)

    def require(self) -> DashboardData:
        if self.data is None:
            raise RuntimeError(self.error or "Результаты недоступны")
        return self.data


class DashboardHandler(BaseHTTPRequestHandler):
    store: DashboardStore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self._json(
                    {"status": "ok" if self.store.data else "degraded", "error": self.store.error},
                    HTTPStatus.OK if self.store.data else HTTPStatus.SERVICE_UNAVAILABLE,
                )
            elif parsed.path == "/api/summary":
                self.store.reload()
                self._json(self.store.require().summary())
            elif parsed.path == "/api/queries":
                self._json({"items": self.store.require().query_list()})
            elif parsed.path.startswith("/api/query/"):
                image_id = unquote(parsed.path.removeprefix("/api/query/"))
                self._json(self.store.require().query_detail(image_id))
            elif parsed.path.startswith("/api/image/"):
                image_id = unquote(parsed.path.removeprefix("/api/image/"))
                self._bytes(self.store.require().image(image_id), "image/jpeg", no_store=True)
            else:
                self._static(parsed.path)
        except KeyError:
            self._json({"error": "Объект не найден"}, HTTPStatus.NOT_FOUND)
        except RuntimeError as exc:
            self._json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
        except (OSError, ValueError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _static(self, path: str) -> None:
        routes = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path not in routes:
            self._json({"error": "Страница не найдена"}, HTTPStatus.NOT_FOUND)
            return
        filename, content_type = routes[path]
        self._bytes((STATIC_DIR / filename).read_bytes(), content_type)

    def _json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self._bytes(payload, "application/json; charset=utf-8", status=status, no_store=True)

    def _bytes(
        self,
        payload: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        no_store: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if no_store:
            self.send_header("Cache-Control", "no-store")
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"dashboard: {self.address_string()} {format % args}", flush=True)


def serve_dashboard(
    data_dir: str | Path,
    run_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    store = DashboardStore(data_dir, run_dir)
    handler = type("ConfiguredDashboardHandler", (DashboardHandler,), {"store": store})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Street Falcon dashboard: http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
