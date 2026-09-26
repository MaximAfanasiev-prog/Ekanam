from __future__ import annotations

import io
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .predictor import BBox, InvalidBBox
from .search import SearchBusy, SearchService

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_IMAGE_BYTES + 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
logger = logging.getLogger(__name__)


class RequestSizeLimit:
    """Bound the complete upload before multipart parsing, even without Content-Length."""

    def __init__(self, app: ASGIApp, limit: int) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                await JSONResponse(
                    {"detail": "Request is too large."}, status_code=413
                )(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay() -> dict:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


class Match(BaseModel):
    rank: int
    image_id: str
    score: float


class SearchResponse(BaseModel):
    accepted: bool
    threshold: float
    matches: list[Match]
    model_version: str
    gallery_version: str


def _load_service() -> SearchService:
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
        try:
            yield
        finally:
            app.state.service = None

    app = FastAPI(title="Street Falcon Search", lifespan=lifespan)
    app.add_middleware(RequestSizeLimit, limit=MAX_REQUEST_BYTES)

    @app.get("/api/v1/health")
    def health(request: Request):
        if getattr(request.app.state, "service", None) is None:
            raise HTTPException(503, "Search service is not ready.")
        return {"status": "ok"}

    @app.post("/api/v1/search", response_model=SearchResponse)
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
                return service.search(decoded, bbox, top_k)
        except SearchBusy as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Search inference failed")
            raise HTTPException(500, "Search failed. Check server logs.") from exc

    return app
