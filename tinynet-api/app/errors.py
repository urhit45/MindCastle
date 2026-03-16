"""
Uniform error envelope — no raw internal details leak to clients.
All responses share the shape: {"error": {"code", "message", "requestId"}}.
"""
import logging
from enum import Enum

from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

log = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    VALIDATION_ERROR    = "VALIDATION_ERROR"
    INVALID_INPUT       = "INVALID_INPUT"
    NOT_FOUND           = "NOT_FOUND"
    FORBIDDEN           = "FORBIDDEN"
    RATE_LIMITED        = "RATE_LIMITED"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTERNAL_ERROR      = "INTERNAL_ERROR"


_STATUS_TO_CODE: dict[int, ErrorCode] = {
    400: ErrorCode.INVALID_INPUT,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.SERVICE_UNAVAILABLE,
}


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def error_body(code: ErrorCode, message: str, request_id: str) -> dict:
    return {"error": {"code": code.value, "message": message, "requestId": request_id}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = _rid(request)
        issues = "; ".join(
            f"{' → '.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        log.warning("validation_error request_id=%s issues=%r", rid, issues)
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_body(ErrorCode.VALIDATION_ERROR, f"Invalid request: {issues}", rid),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        rid = _rid(request)
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.INTERNAL_ERROR)
        log.info("http_error request_id=%s status=%d code=%s", rid, exc.status_code, code.value)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, str(exc.detail), rid),
            headers={"X-Request-Id": rid},
        )

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = _rid(request)
        log.exception("unhandled_error request_id=%s type=%s", rid, type(exc).__name__)
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body(
                ErrorCode.INTERNAL_ERROR,
                "An unexpected error occurred. Please try again.",
                rid,
            ),
            headers={"X-Request-Id": rid},
        )
