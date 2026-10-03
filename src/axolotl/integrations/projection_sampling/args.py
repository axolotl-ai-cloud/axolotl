"""Configuration for offline projection sampling."""

from string import Formatter
from typing import Any, Literal

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

    backend: str = "transformers"
    backend_kwargs: dict[str, Any] = Field(default_factory=dict)
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
    device: str = Field("cuda", pattern=r"^(cpu|cuda(:\d+)?)$")
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    verifier: str | None = None

    @model_validator(mode="after")
    def validate_backend(self):
        if self.backend == "transformers" and self.backend_kwargs:
            raise ValueError("The Transformers backend does not accept backend_kwargs")
        if self.backend == "vllm":
            if self.device != "cuda":
                raise ValueError(
                    "vLLM uses visible GPUs; set device: cuda and select GPUs with CUDA_VISIBLE_DEVICES"
                )
            if self.temperature < 0.01:
                raise ValueError(
                    "vLLM temperature must be >= 0.01 to avoid runtime clamping"
                )
            VLLMBackendOptions.model_validate(self.backend_kwargs)
        elif self.backend != "transformers" and "." not in self.backend:
            raise ValueError(
                "backend must be transformers, vllm, or a dotted SamplingBackend subclass"
            )
        return self

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


class VLLMBackendOptions(BaseModel):
    """Engine controls accepted by the bundled vLLM backend."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    tensor_parallel_size: int = Field(1, ge=1)
    gpu_memory_utilization: float = Field(0.8, gt=0, le=1)
    max_model_len: int | None = Field(None, ge=1)
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    attention_backend: str | None = None
    score_batch_size: int = Field(4, ge=1)


def get_seed(cfg) -> int:
    """Use Axolotl's run seed, including its fallback when unset."""
    seed = cfg.get("seed")
    return 42 if seed is None else seed
