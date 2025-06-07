"""Telemetry utilities for exception and traceback information."""

import logging
import os
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
    Remove personal information from stack trace messages while keeping Python package codepaths.

    This function identifies Python packages by looking for common patterns in virtual environment
    and site-packages directories, preserving the package path while removing user-specific paths.

    Args:
        stack_trace: The original stack trace string.

    Returns:
        A sanitized version of the stack trace with Python package paths preserved.
    """
    # Split the stack trace into lines to process each file path separately
    lines = stack_trace.split("\n")
    sanitized_lines = []

    # Regular expression to find file paths in the stack trace
    path_pattern = re.compile(r'(?:File ")(.*?)(?:")')

    # Regular expression to identify paths in site-packages or dist-packages
    # This matches path segments like "site-packages/package_name" or "dist-packages/package_name"
    site_packages_pattern = re.compile(
        r"(?:site-packages|dist-packages)[/\\]([\w\-\.]+)"
    )

    # Additional common virtual environment patterns
    venv_lib_pattern = re.compile(
        r"(?:lib|Lib)[/\\](?:python\d+(?:\.\d+)?[/\\])?(?:site-packages|dist-packages)[/\\]([\w\-\.]+)"
    )

    for line in lines:
        # Check if this line contains a file path
        path_match = path_pattern.search(line)

        if path_match:
            full_path = path_match.group(1)
            sanitized_path = ""

            # Try to match site-packages pattern
            site_packages_match = site_packages_pattern.search(full_path)
            venv_lib_match = venv_lib_pattern.search(full_path)

            if site_packages_match:
                # Find the index where the matched pattern starts
                idx = full_path.find("site-packages")
                if idx == -1:
                    idx = full_path.find("dist-packages")

                # Keep from 'site-packages' onward
                if idx >= 0:
                    sanitized_path = full_path[idx:]
            elif venv_lib_match:
                # For other virtual environment patterns, find the package directory
                match_idx = venv_lib_match.start(1)
                if match_idx > 0:
                    # Keep from the package name onward
                    package_name = venv_lib_match.group(1)
                    idx = full_path.rfind(
                        package_name, 0, match_idx + len(package_name)
                    )
                    if idx >= 0:
                        sanitized_path = full_path[idx:]

            # If we couldn't identify a package pattern but path contains 'axolotl'
            elif "axolotl" in full_path:
                idx = full_path.rfind("axolotl")
                if idx >= 0:
                    sanitized_path = full_path[idx:]

            # Apply the sanitization to the line
            if sanitized_path:
                line = line.replace(full_path, sanitized_path)
            else:
                # If we couldn't identify a package pattern, just keep the filename
                filename = os.path.basename(full_path)
                if filename:
                    line = line.replace(full_path, filename)
                else:
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
