"""Pydantic models for Multi-teacher On-Policy Distillation (MOPD)."""

from typing import Annotated

from annotated_types import MinLen
from pydantic import BaseModel, Field


class MOPDTeacherConfig(BaseModel):
    """A domain teacher served by a vLLM OpenAI-compatible server."""

    url: str = Field(
        json_schema_extra={
            "description": "Base URL of the teacher's vLLM server, e.g. http://localhost:8001"
        },
    )
    model: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Served model name to request. Defaults to the server's only model."
        },
    )


class MOPDConfig(BaseModel):
    """Multi-teacher On-Policy Distillation (arXiv:2606.30406), on top of `rl: grpo`."""

    teachers: Annotated[dict[str, MOPDTeacherConfig], MinLen(1)] = Field(
        json_schema_extra={
            "description": "Mapping from domain name to its teacher. Each rollout is "
            "scored by the teacher named by its row's `domain_field` value."
        },
    )
    domain_field: str = Field(
        default="domain",
        json_schema_extra={
            "description": "Dataset column holding each prompt's domain name"
        },
    )
    advantage_clip: float = Field(
        default=5.0,
        gt=0,
        json_schema_extra={
            "description": "Two-sided clip on the per-token advantage "
            "log pi_teacher - log pi_student"
        },
    )
    reward_weight: float = Field(
        default=0.0,
        json_schema_extra={
            "description": "Weight of the GRPO reward advantage added to the MOPD "
            "advantage. 0 means pure distillation (reward_funcs optional)."
        },
    )
    timeout: float = Field(
        default=600.0,
        gt=0,
        json_schema_extra={"description": "Teacher request timeout in seconds"},
    )
