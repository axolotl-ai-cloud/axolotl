# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the ringmaster context-parallel plugin."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator


class ContextParallelConfig(BaseModel):
    """Nested ``context_parallel:`` config block for the ringmaster plugin."""

    model_config = ConfigDict(extra="forbid")

    size: int = Field(default=1, ge=1)
    """Total context-parallel degree. 1 = disabled."""

    backend: Literal["auto", "ulysses", "ring", "usp"] = "auto"

    ulysses_size: Optional[int] = Field(default=None, ge=1)
    """All-to-all (head) degree. None = auto-select."""

    ring_size: Optional[int] = Field(default=None, ge=1)
    """Ring degree. None = auto-select. ulysses_size * ring_size must == size."""

    rotate_method: Literal["allgather", "alltoall"] = "allgather"
    load_balance: Literal["auto", "none", "head_tail", "distflash"] = "auto"
    ring_impl: Literal["auto", "torch_native", "hf_kernels"] = "auto"
    """Ring block-kernel provider. auto -> hf_kernels (FA2/3/4 via HF kernels) when
    the model uses a flash kernel, else torch_native (SDPA/flex)."""

    @model_validator(mode="after")
    def _validate(self):
        for name in ("ulysses_size", "ring_size"):
            degree = getattr(self, name)
            if degree is not None and self.size % degree:
                raise ValueError(
                    f"{name} ({degree}) must divide context_parallel.size ({self.size})"
                )
        if self.backend == "ulysses" and (
            self.ring_size not in (None, 1)
            or self.ulysses_size not in (None, self.size)
        ):
            raise ValueError(
                "backend=ulysses requires ring_size=1 and ulysses_size=size"
            )
        if self.backend == "ring" and (
            self.ulysses_size not in (None, 1)
            or self.ring_size not in (None, self.size)
        ):
            raise ValueError("backend=ring requires ulysses_size=1 and ring_size=size")
        if self.ulysses_size and self.ring_size:
            if self.ulysses_size * self.ring_size != self.size:
                raise ValueError(
                    f"ulysses_size({self.ulysses_size}) * ring_size({self.ring_size}) "
                    f"!= context_parallel.size({self.size})"
                )
        return self


class ContextParallelArgs(BaseModel):
    """Input args contributed by the ringmaster context-parallel plugin."""

    context_parallel: Optional[ContextParallelConfig] = None

    @field_serializer("context_parallel")
    def serialize_context_parallel(self, cp: Optional[ContextParallelConfig]):
        # Strategy resolution distinguishes omitted options from explicit defaults.
        return cp.model_dump(exclude_unset=True) if cp is not None else None

    @model_validator(mode="before")
    @classmethod
    def sync_context_parallel_size(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        block = data.get("context_parallel") or {}
        if isinstance(block, ContextParallelConfig):
            block = block.model_dump(exclude_unset=True)
        if not isinstance(block, dict):
            return data
        size = block.get("size")
        flat = data.get("context_parallel_size")
        if flat is None:
            flat = data.get("sequence_parallel_degree")
        if size is not None and flat is not None and flat != size:
            raise ValueError(
                f"context_parallel.size ({size}) conflicts with "
                f"context_parallel_size ({flat}); set only one"
            )
        size = size if size is not None else flat
        if size is not None:
            data["context_parallel_size"] = size
            data["context_parallel"] = {**block, "size": size}
        return data
