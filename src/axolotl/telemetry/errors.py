"""Telemetry utilities for exception and traceback information."""

import re
import traceback
from contextvars import ContextVar
from functools import wraps
from inspect import getmodule
from typing import Any, Callable

from axolotl.telemetry.manager import TelemetryManager
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_REPORTED_ERRORS: ContextVar[list[Exception] | None] = ContextVar(
    "telemetry_reported_errors", default=None
)


def sanitize_filename(filename: str) -> str:
    """Keep package-relative paths or a local file's basename."""
    parts = filename.replace("\\", "/").split("/")
    for marker in ("site-packages", "dist-packages", "axolotl"):
        if marker in parts:
            return "/".join(parts[parts.index(marker) :])
    return parts[-1]


def sanitize_stack_trace(stack_trace: str) -> str:
    """Retain only sanitized frame locations from a formatted traceback."""
    frames = []
    for line in stack_trace.splitlines():
        match = re.fullmatch(r'\s*File "([^"]+)", line (\d+), in (.+)', line)
        if match:
            filename, lineno, function = match.groups()
            frames.append(
                f'  File "{sanitize_filename(filename)}", line {lineno}, in {function}'
            )
    return "\n".join(frames)


def exception_properties(exception: Exception) -> dict[str, Any]:
    """Collect frame metadata without formatting messages or reading source code."""
    frames = [
        {
            "filename": sanitize_filename(frame.f_code.co_filename),
            "function": frame.f_code.co_name,
            "lineno": lineno,
        }
        for frame, lineno in traceback.walk_tb(exception.__traceback__)
    ]
    return {
        "exception": {
            "type": type(exception).__name__,
            "module": type(exception).__module__,
        },
        "stack_trace": {"frames": frames},
    }


def send_errors(func: Callable) -> Callable:
    """Report each exception once within a nested chain of decorated calls."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        telemetry_manager = TelemetryManager.get_instance()
        if not telemetry_manager.enabled:
            return func(*args, **kwargs)

        reported = _REPORTED_ERRORS.get()
        token = None
        if reported is None:
            reported = []
            token = _REPORTED_ERRORS.set(reported)
        try:
            return func(*args, **kwargs)
        except Exception as exception:
            if not any(previous is exception for previous in reported):
                reported.append(exception)
                try:
                    module = getmodule(func)
                    module_path = (
                        f"{module.__name__}.{func.__name__}"
                        if module
                        else func.__name__
                    )
                    telemetry_manager.send_event(
                        event_type=f"{module_path}-error",
                        properties=exception_properties(exception),
                    )
                    LOG.error(
                        "Error captured in telemetry. Run ID: %s",
                        telemetry_manager.run_id,
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    LOG.warning("Could not report error telemetry")
            raise
        finally:
            if token is not None:
                _REPORTED_ERRORS.reset(token)

    return wrapper
