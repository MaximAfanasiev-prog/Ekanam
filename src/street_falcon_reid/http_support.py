from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("uvicorn.error")
ERROR_CODES = {
    400: "bad_request", 404: "not_found", 405: "method_not_allowed",
    408: "upload_timeout", 413: "payload_too_large", 415: "unsupported_image",
    422: "validation_error", 429: "busy", 500: "internal_error", 503: "not_ready",
}


def error_response(
    scope: Scope, status: int, message: str, headers: dict[str, str] | None = None,
    fields: list[dict] | None = None,
) -> JSONResponse:
    payload = {
        "detail": message,
        "error": {"code": ERROR_CODES.get(status, "request_error"), "message": message},
        "request_id": scope.get("state", {}).get("request_id", ""),
    }
    if fields is not None:
        payload["error"]["fields"] = fields
    return JSONResponse(payload, status_code=status, headers=headers)


class RequestContext:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        start = time.monotonic()
        status = 500
        started = False

        async def send_with_context(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                status = message["status"]
                started = True
                headers = list(message.get("headers", []))
                headers.extend([
                    (b"x-request-id", request_id.encode()),
                    (b"x-process-time-ms", f"{(time.monotonic() - start) * 1000:.2f}".encode()),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ])
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        except Exception:
            logger.exception("Unhandled request failure request_id=%s", request_id)
            if started:
                raise
            await error_response(scope, 500, "Request failed. Contact the service operator.")(
                scope, receive, send_with_context
            )
        finally:
            route = scope.get("route")
            logger.info(json.dumps({
                "event": "http_request", "request_id": request_id,
                "method": scope["method"],
                "route": getattr(route, "path", "unmatched"),
                "status": status, "duration_ms": round((time.monotonic() - start) * 1000, 2),
            }))


class RequestSizeLimit:
    """Bound upload count, time and total bytes before parsing multipart data."""

    def __init__(
        self, app: ASGIApp, limit: int, max_uploads: int = 2, upload_timeout: float = 15
    ) -> None:
        self.app = app
        self.limit = limit
        self.max_uploads = max_uploads
        self.upload_timeout = upload_timeout
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        if self.active >= self.max_uploads:
            await error_response(scope, 429, "Search uploads are busy. Retry shortly.",
                                 {"Retry-After": "1"})(scope, receive, send)
            return
        self.active += 1
        try:
            headers = dict(scope.get("headers", []))
            if b"content-length" in headers:
                try:
                    length = int(headers[b"content-length"])
                    if length < 0:
                        raise ValueError
                except ValueError:
                    await error_response(
                        scope, 400, "Invalid Content-Length."
                    )(scope, receive, send)
                    return
                if length > self.limit:
                    await error_response(scope, 413, "Request is too large.")(scope, receive, send)
                    return
            body = bytearray()
            try:
                async with asyncio.timeout(self.upload_timeout):
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return
                        chunk = message.get("body", b"")
                        if len(body) + len(chunk) > self.limit:
                            await error_response(
                                scope, 413, "Request is too large."
                            )(scope, receive, send)
                            return
                        body.extend(chunk)
                        if not message.get("more_body", False):
                            break
            except TimeoutError:
                await error_response(scope, 408, "Upload timed out.")(scope, receive, send)
                return
            delivered = False

            async def replay() -> Message:
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        finally:
            self.active -= 1
