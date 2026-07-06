"""Domain errors and FastAPI exception handlers."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base application error mapped to an HTTP response."""

    status_code: int = 400

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class UnsupportedFileTypeError(AppError):
    """Raised when an uploaded document has an unsupported extension."""

    status_code = 415


class ProviderConfigError(AppError):
    """Raised when a provider is selected but not configured (missing key)."""

    status_code = 500


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that turn AppError into clean JSON responses."""

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
