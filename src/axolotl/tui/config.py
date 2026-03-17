"""TUI configuration — Pydantic model for TUI settings."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TUIConfig(BaseModel):
    """Configuration for the Axolotl Training TUI dashboard."""

    enabled: bool = Field(
        default=False,
        json_schema_extra={"description": "Enable the TUI dashboard"},
    )
    refresh_rate: int = Field(
        default=4,
        json_schema_extra={"description": "Renders per second"},
    )
    log_level: str = Field(
        default="debug",
        json_schema_extra={
            "description": "Minimum log level shown in events panel"
        },
    )
    panels: list[str] = Field(
        default_factory=lambda: ["progress", "training", "hardware", "events", "debug"],
        json_schema_extra={
            "description": "Ordered list of panels to display"
        },
    )
    hardware_poll_interval: int = Field(
        default=2,
        json_schema_extra={"description": "Seconds between pynvml GPU queries"},
    )
    stdout_log_path: str = Field(
        default="axolotl_stdout.log",
        json_schema_extra={
            "description": "File path for captured stdout/stderr log"
        },
    )
    parser_plugins: list[str] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "List of extra parser classes to load"
        },
    )
