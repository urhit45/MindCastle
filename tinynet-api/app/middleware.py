"""
ASGI middleware stack:
  1. RequestIDMiddleware  — outermost; injects/propagates X-Request-Id
  2. BodySizeLimitMiddleware — rejects oversized payloads before parsing
  3. RateLimitMiddleware  — sliding-window per (IP, bucket), stdlib only
"""
import logging
import time
import uuid as uuid_lib
from collections import defaultdict, deque
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

# ─── Request ID ───────────────────────────────────────────────────────────────

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = request.headers.get("X-Request-Id") or str(uuid_lib.uuid4())
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response


# ─── Body size guard ──────────────────────────────────────────────────────────

MAX_BODY_BYTES = 64 * 1024  # 64 KB — well above any valid request


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        cl = request.headers.get("content-length")
        if cl and int(cl) > MAX_BODY_BYTES:
            rid = getattr(request.state, "request_id", "unknown")
            log.warning("body_too_large request_id=%s content_length=%s", rid, cl)
            return JSONResponse(
                status_code=413,
                content={"error": {
                    "code": "INVALID_INPUT",
                    "message": "Request body too large.",
                    "requestId": rid,
                }},
                headers={"X-Request-Id": rid},
            )
        return await call_next(request)


# ─── Rate limiter (sliding window, in-memory, no external deps) ───────────────

_WINDOW_SECS = 60

# Endpoint buckets → max requests per window
_LIMITS: dict[str, int] = {
    "classify": 20,   # 20 classify calls / 60 s
    "mutate":   30,   # 30 write ops  / 60 s
}

# {ip: {bucket: deque[timestamp]}}
_counters: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))


def clear_rate_limits() -> None:
    """Reset all counters — call from test teardown."""
    _counters.clear()


def _bucket(path: str, method: str) -> str | None:
    if path.startswith("/classify"):
        return "classify"
    if method in ("POST", "PATCH", "PUT", "DELETE") and path.startswith("/nodes"):
        return "mutate"
    return None


def _allowed(ip: str, bucket: str) -> bool:
    now = time.monotonic()
    window = _counters[ip][bucket]
    cutoff = now - _WINDOW_SECS
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= _LIMITS[bucket]:
        return False
    window.append(now)
    return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        bucket = _bucket(request.url.path, request.method)
        if bucket:
            ip = (request.client.host if request.client else "unknown")
            if not _allowed(ip, bucket):
                rid = getattr(request.state, "request_id", "unknown")
                log.warning("rate_limited ip=%s bucket=%s request_id=%s", ip, bucket, rid)
                return JSONResponse(
                    status_code=429,
                    content={"error": {
                        "code": "RATE_LIMITED",
                        "message": "Too many requests. Please slow down.",
                        "requestId": rid,
                    }},
                    headers={"X-Request-Id": rid},
                )
        return await call_next(request)
