"""Core cross-cutting utilities: logging and error handling."""

from .errors import (
    AppError,
    ErrorResponse,
    ProviderConfigError,
    UnsupportedFileTypeError,
    register_exception_handlers,
)
from .logging import configure_logging, get_logger

__all__ = [
    "AppError",
    "ErrorResponse",
    "ProviderConfigError",
    "UnsupportedFileTypeError",
    "register_exception_handlers",
    "configure_logging",
    "get_logger",
]
