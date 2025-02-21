"""Telemetry utilities for exception and traceback information."""

import logging
import re
import traceback
from functools import wraps
from inspect import getmodule
from typing import Any, Callable

from axolotl.telemetry.manager import TelemetryManager

LOG = logging.getLogger(__name__)

ERROR_HANDLED = False


def sanitize_stack_trace(stack_trace: str) -> str:
    """
    Remove personal information from stack trace messages while keeping Axolotl codepaths.

    Args:
        stack_trace: The original stack trace string.

    Returns:
        A sanitized version of the stack trace with only axolotl paths preserved.
    """
    # Split the stack trace into lines to process each file path separately
    lines = stack_trace.split("\n")
    sanitized_lines = []

    # Regular expression to find file paths in the stack trace
    path_pattern = re.compile(r'(?:File ")(.*?)(?:")')

    for line in lines:
        # Check if this line contains a file path
        path_match = path_pattern.search(line)

        if path_match:
            full_path = path_match.group(1)

            if "axolotl/" in full_path:
                # Keep only the 'axolotl' part and onward
                axolotl_idx = full_path.rfind("axolotl/")
                if axolotl_idx >= 0:
                    # Replace the original path with the sanitized one
                    sanitized_path = full_path[axolotl_idx:]
                    line = line.replace(full_path, sanitized_path)
            else:
                # For non-axolotl paths, replace with an empty string or a placeholder
                line = line.replace(full_path, "")

        sanitized_lines.append(line)

    return "\n".join(sanitized_lines)


def send_errors(func: Callable) -> Callable:
    """
    Decorator to send exception info in a function. If an exception is raised, we send
    telemetry containing the stack trace and error message.

    If an error occurs in a decorated function that is called by another decorated
    function, we'll only send telemetry corresponding to the lower-level function.

    Args:
        func: Function to decorate.

    Returns:
        Decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        telemetry_manager = TelemetryManager.get_instance()
        if not telemetry_manager.enabled:
            return func(*args, **kwargs)

        try:
            return func(*args, **kwargs)
        except Exception as exception:
            # Only track if we're not already handling an error. This prevents us from
            # capturing an error more than once in nested decorated function calls.
            global ERROR_HANDLED  # pylint: disable=global-statement
            if not ERROR_HANDLED:
                ERROR_HANDLED = True

                # Get function module path
                module = getmodule(func)
                module_path = (
                    f"{module.__name__}.{func.__name__}" if module else func.__name__
                )

                # Get stack trace
                stack_trace = "".join(
                    traceback.format_exception(
                        type(exception), exception, exception.__traceback__
                    )
                )
                stack_trace = sanitize_stack_trace(stack_trace)

                # Send error telemetry
                telemetry_manager.send_event(
                    event_type=f"{module_path}-error",
                    properties={
                        "exception": str(exception),
                        "stack_trace": stack_trace,
                    },
                )

            raise

    return wrapper
