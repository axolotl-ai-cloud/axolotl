"""Configuration for offline projection sampling."""

from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROPOSAL_TEMPLATE = """You are given a question, an expert solution, and a partial response.
Use the expert solution to identify the necessary facts and reasoning. Continue
the partial response in your own words, preserving that information and reaching
the same correct final answer. Output only the continuation.

Question: {question}
Expert solution: {expert_response}
Partial response: {prefix}
"""


class ProjectionSamplingConfig(BaseModel):
    """Sampling settings under `projection_sampling:`."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cache_dir: str = "./last_run_prepared/projection-sampling"
    question_field: str = "prompt"
    response_field: str = "response"
    block_size: int = Field(32, ge=1)
    max_new_tokens: int = Field(1856, ge=1)
    mcmc_steps: int = Field(10, ge=0)
    temperature: float = Field(0.6, gt=0)
    repetition_penalty: float = Field(1.0, gt=0)
    acceptance: Literal["metropolis_hastings", "greedy"] = "metropolis_hastings"
    prompt_format: Literal["chat", "raw"] = "chat"
    proposal_template: str = PROPOSAL_TEMPLATE
    seed: int = Field(42, ge=0)
    device: str = Field("cuda", pattern=r"^(cpu|cuda(:\d+)?)$")
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    verifier: str | None = None

    @model_validator(mode="after")
    def validate_template(self):
        fields = {
            name
            for _, name, _, _ in Formatter().parse(self.proposal_template)
            if name is not None
        }
        if fields != {"question", "expert_response", "prefix"}:
            raise ValueError(
                "proposal_template must contain exactly the fields "
                "{question}, {expert_response}, and {prefix}"
            )
        self.proposal_template.format(question="", expert_response="", prefix="")
        return self


class ProjectionSamplingArgs(BaseModel):
    """Plugin configuration mixin."""

    projection_sampling: ProjectionSamplingConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_sft(cls, data):
        if data.get("projection_sampling") is not None:
            for key in (
                "rl",
                "streaming",
                "pretraining_dataset",
                "skip_prepare_dataset",
                "processor_type",
            ):
                if data.get(key):
                    raise ValueError(f"projection_sampling does not support {key}")
        return data
