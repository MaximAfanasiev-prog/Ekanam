from __future__ import annotations

import hashlib
import io
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from .http_support import RequestContext, RequestSizeLimit, error_response
from .predictor import BBox, InvalidBBox
from .search import SearchBusy, SearchService

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
logger = logging.getLogger("uvicorn.error")


class Match(BaseModel):
    rank: int
    image_id: str
    score: float


class SearchResponse(BaseModel):
    request_id: str
    accepted: bool
    threshold: float
    matches: list[Match]
    model_version: str
    gallery_version: str


def _load_service() -> SearchService:
    bundle = os.getenv("LCT_GALLERY_DIR")
    if bundle:
        return SearchService.load_bundle(
            Path(os.environ["LCT_CHECKPOINT"]), Path(bundle), os.getenv("LCT_DEVICE", "cpu")
        )
    return SearchService.load(
        Path(os.environ["LCT_DATA_DIR"]) / "extracted",
        Path(os.environ["LCT_RUN_DIR"]),
        os.getenv("LCT_DEVICE", "cpu"),
    )


def create_app(service_factory: Callable[[], SearchService] | None = None) -> FastAPI:
    factory = service_factory or _load_service

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = factory()
        app.state.gallery_ids = frozenset(app.state.service.gallery.ids)
        thumbnail_dir = os.getenv("LCT_THUMBNAIL_DIR")
        app.state.thumbnail_dir = Path(thumbnail_dir).resolve() if thumbnail_dir else None
        try:
            yield
        finally:
            app.state.service = None

    app = FastAPI(
        title="Street Falcon Search", version="1.0.0", lifespan=lifespan,
        root_path=os.getenv("LCT_ROOT_PATH", ""),
    )
    app.add_middleware(RequestSizeLimit, limit=MAX_REQUEST_BYTES)
    app.add_middleware(RequestContext)

    ui_dir = Path(__file__).with_name("search_ui")
    app.mount("/ui-assets", StaticFiles(directory=ui_dir), name="search-ui")

    @app.get("/", include_in_schema=False)
    def search_page():
        return FileResponse(
            ui_dir / "index.html",
            headers={
                "Content-Security-Policy": (
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' blob:; connect-src 'self'; "
                    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(request.scope, exc.status_code, str(exc.detail), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = [
            {"field": ".".join(str(part) for part in error["loc"]), "type": error["type"]}
            for error in exc.errors()
        ]
        return error_response(request.scope, 422, "Invalid request fields.", fields=fields)

    @app.get("/api/v1/live")
    def live():
        return {"status": "alive"}

    @app.get("/api/v1/ready")
    def ready(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Search service is not ready.")
        return {
            "status": "ok", "model_version": service.predictor.model_version,
            "gallery_version": service.gallery.version, "gallery_count": len(service.gallery.ids),
            "device": str(service.predictor.device),
        }

    @app.get("/api/v1/info")
    def info(request: Request):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Search service is not ready.")
        return {
            **ready(request), "embedding_dim": service.predictor.embedding_dim,
            "threshold": service.predictor.threshold,
            "bbox_format": "x,y,w,h", "bbox_source": "client",
            "coordinate_space": "original image pixels, no EXIF rotation",
            "crop_margin": service.predictor.data_config["crop_margin"],
            "formats": ["JPEG", "PNG"], "max_image_bytes": MAX_IMAGE_BYTES,
            "max_image_pixels": MAX_IMAGE_PIXELS, "max_top_k": 10,
            "max_concurrent_uploads": 2, "upload_timeout_seconds": 15,
        }

    @app.get("/api/v1/gallery/{image_id}/thumbnail", response_class=FileResponse)
    def gallery_thumbnail(image_id: str, request: Request):
        directory = request.app.state.thumbnail_dir
        if directory is None or image_id not in request.app.state.gallery_ids:
            raise HTTPException(404, "Gallery image unavailable.")
        filename = hashlib.sha256(image_id.encode()).hexdigest() + ".jpg"
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise HTTPException(404, "Gallery image unavailable.")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/v1/health")
    def health(request: Request):
        if getattr(request.app.state, "service", None) is None:
            raise HTTPException(503, "Search service is not ready.")
        return {"status": "ok"}

    @app.post(
        "/api/v1/search", response_model=SearchResponse,
        responses={
            status: {"description": message}
            for status, message in {
                408: "Upload timeout", 413: "Upload too large", 415: "Unsupported image",
                422: "Invalid client bbox or fields", 429: "Busy; retry after one second",
                500: "Inference failed", 503: "Not ready",
            }.items()
        },
    )
    def search(
        request: Request,
        image: Annotated[UploadFile, File()],
        x: Annotated[int, Form(ge=0)],
        y: Annotated[int, Form(ge=0)],
        w: Annotated[int, Form(gt=0)],
        h: Annotated[int, Form(gt=0)],
        top_k: Annotated[int, Form(ge=1, le=10)] = 10,
    ):
        service = getattr(request.app.state, "service", None)
        if service is None:
            raise HTTPException(503, "Search service is not ready.")
        payload = image.file.read(MAX_IMAGE_BYTES + 1)
        if len(payload) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Image exceeds 10 MiB.")
        try:
            with Image.open(io.BytesIO(payload)) as source:
                if source.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG and PNG are supported.")
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image exceeds 20 million pixels.")
                if getattr(source, "n_frames", 1) != 1:
                    raise HTTPException(415, "Animated images are not supported.")
                bbox = BBox(x, y, w, h)
                bbox.validate(source)
                source.load()
                decoded = source.convert("RGB")
        except InvalidBBox as exc:
            raise HTTPException(422, str(exc)) from exc
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, "Image dimensions are too large.") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(415, "Image cannot be decoded.") from exc
        try:
            with decoded:
                return {
                    **service.search(decoded, bbox, top_k),
                    "request_id": request.state.request_id,
                }
        except SearchBusy as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Search inference failed request_id=%s", request.state.request_id)
            raise HTTPException(500, "Search failed. Check server logs.") from exc

    return app
