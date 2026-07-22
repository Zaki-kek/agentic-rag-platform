"""Domain errors and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Uniform error body returned by the API.

    Attributes:
        error: Human-readable error message.
        code: Machine-readable error identifier (e.g. "validation_error").
    """

    error: str
    code: str


class AppError(Exception):
    """Base application error mapped to an HTTP response."""

    status_code: int = 400
    code: str = "app_error"

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded document has an unsupported extension."""

    status_code = 415
    code = "unsupported_media_type"


class ProviderConfigError(AppError):
    """Raised when a provider is selected but not configured (missing key)."""

    status_code = 500
    code = "provider_config_error"


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that turn errors into uniform ``{error, code}`` JSON."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        body = ErrorResponse(error=exc.message, code=exc.code)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        body = ErrorResponse(error="Request validation failed", code="validation_error")
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        body = ErrorResponse(error="Internal server error", code="internal_error")
        return JSONResponse(status_code=500, content=body.model_dump())
