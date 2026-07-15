"""SAR (Subspace-Aligned Rewiring) config models."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SARConfig(BaseModel):
    """SAR post-training spectral projection settings."""

    base_model: Annotated[
        str | None,
        Field(
            default=None,
            description="Spectral reference model (local dir or HF hub id). Defaults to the top-level base_model.",
        ),
    ]
    base_model_revision: Annotated[
        str | None,
        Field(
            default=None,
            description="HF hub revision for base_model. Defaults to the top-level revision_of_model when base_model is inherited.",
        ),
    ]
    trained_model: Annotated[
        str | None,
        Field(
            default=None,
            description="Post-trained model supplying the weight delta. Defaults to the top-level output_dir.",
        ),
    ]
    trained_model_revision: Annotated[
        str | None,
        Field(
            default=None,
            description="HF hub revision for trained_model.",
        ),
    ]
    merge_target: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional expert model to merge the projected delta into. Defaults to base_model.",
        ),
    ]
    merge_target_revision: Annotated[
        str | None,
        Field(
            default=None,
            description="HF hub revision for merge_target.",
        ),
    ]
    output_dir: Annotated[
        str | None,
        Field(
            default=None,
            description="Directory for the projected model. Defaults to {output_dir}/sar.",
        ),
    ]

    rank_ratio: Annotated[
        float | list[float],
        Field(
            default=[0.01],
            description="Fraction(s) of min(dout, din) used as the per-layer projection rank, each in (0, 1]. A list emits one output per ratio.",
        ),
    ]
    delta_rank_ratio: Annotated[
        float | None,
        Field(
            default=None,
            gt=0,
            le=1,
            description="Rank ratio for truncating the weight delta before projection. Defaults to rank_ratio.",
        ),
    ]
    projection: Annotated[
        Literal["spectral", "none"],
        Field(
            default="spectral",
            description="Projection mode. `none` skips the spectral projection and applies the low-rank delta directly (ablation).",
        ),
    ]
    rewiring: Annotated[
        Literal["full", "diagonal", "off_diagonal"],
        Field(
            default="full",
            description="Masking applied to the rewiring matrix M before reconstruction (ablations).",
        ),
    ]
    scale: Annotated[
        float,
        Field(
            default=1.0,
            gt=0,
            description="Coefficient applied to the projected delta at reconstruction.",
        ),
    ]

    target_modules: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Substrings selecting the 2D `.weight` parameters to project. Defaults to the attention and MLP linears.",
        ),
    ]
    exclude_modules: Annotated[
        list[str],
        Field(
            default=[],
            description="Substrings excluding parameters even when matched by target_modules.",
        ),
    ]

    svd_device: Annotated[
        Literal["auto", "cuda", "cpu"],
        Field(
            default="auto",
            description="Device for SVD computation. `auto` uses CUDA when available.",
        ),
    ]
    save_dtype: Annotated[
        Literal["float16", "bfloat16", "float32"],
        Field(
            default="float16",
            description="Dtype for saved output tensors.",
        ),
    ]
    save_rewiring_matrix: Annotated[
        bool,
        Field(
            default=False,
            description="Persist the compact per-layer rewiring matrices under {output_dir}/rewiring/.",
        ),
    ]
    run_after_training: Annotated[
        bool,
        Field(
            default=True,
            description="Run SAR automatically in post_train_unload when training completes.",
        ),
    ]

    @field_validator("rank_ratio", mode="before")
    @classmethod
    def normalize_rank_ratio(cls, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        ratios = sorted({float(ratio) for ratio in value})
        if not ratios:
            raise ValueError("rank_ratio requires at least one value")
        for ratio in ratios:
            if not 0 < ratio <= 1:
                raise ValueError(f"rank_ratio values must be in (0, 1], got {ratio}")
        return ratios

    @model_validator(mode="after")
    def validate_projection_rewiring(self):
        if self.projection == "none" and self.rewiring != "full":
            raise ValueError(
                "rewiring ablations require projection: spectral; no rewiring matrix exists with projection: none"
            )
        return self


class SARArgs(BaseModel):
    """Input args for the SAR plugin."""

    sar: Annotated[
        SARConfig | None,
        Field(
            default=None,
            description="SAR post-training spectral projection settings.",
        ),
    ]
