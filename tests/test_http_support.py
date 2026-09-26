from __future__ import annotations

import asyncio

from street_falcon_reid.http_support import RequestContext, RequestSizeLimit


def scope(headers=()):
    return {"type": "http", "method": "POST", "path": "/api/v1/search", "headers": list(headers)}


def test_upload_timeout_and_slot_release():
    async def scenario():
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        limit = RequestSizeLimit(app, 100, max_uploads=1, upload_timeout=0.02)
        sent = []

        async def slow():
            await asyncio.sleep(1)

        async def send(message):
            sent.append(message)

        await RequestContext(limit)(scope(), slow, send)
        assert sent[0]["status"] == 408
        assert dict(sent[0]["headers"])[b"x-request-id"]
        assert limit.active == 0
    asyncio.run(scenario())


def test_upload_admission_is_bounded_and_disconnect_releases_slot():
    async def scenario():
        async def app(scope, receive, send):
            raise AssertionError("Disconnected upload must not reach the app")

        limit = RequestSizeLimit(app, 100, max_uploads=1)
        started = asyncio.Event()
        release = asyncio.Event()

        async def receive():
            started.set()
            await release.wait()
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        first = asyncio.create_task(limit(scope(), receive, send))
        await started.wait()
        await limit(scope(), receive, send)
        assert sent[0]["status"] == 429
        assert limit.active == 1
        release.set()
        await first
        assert limit.active == 0
    asyncio.run(scenario())


def test_declared_length_rejected_before_read():
    async def scenario():
        async def unexpected(*args):
            raise AssertionError("Should reject without reading or calling the app")

        for length, status in [(b"101", 413), (b"-1", 400), (b"invalid", 400)]:
            sent = []

            async def send(message, sent=sent):
                sent.append(message)

            limit = RequestSizeLimit(unexpected, 100)
            await limit(scope([(b"content-length", length)]), unexpected, send)
            assert sent[0]["status"] == status
            assert limit.active == 0
    asyncio.run(scenario())


def test_unhandled_error_is_sanitized_and_has_request_id():
    async def scenario():
        async def app(*args):
            raise RuntimeError("private internal path")

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        sent = []

        async def send(message):
            sent.append(message)

        await RequestContext(app)(scope(), receive, send)
        assert sent[0]["status"] == 500
        assert b"private internal path" not in sent[1]["body"]
        assert dict(sent[0]["headers"])[b"x-request-id"]
    asyncio.run(scenario())
