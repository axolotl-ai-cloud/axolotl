"""Pure latent-slot initialization contracts for decision canvases."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Sequence

from axolotl.model_support import DiffusionNoise, DiffusionSpec

SlotMode = Literal["none", "pad", "pinned", "learned", "free", "mask", "prompt"]
SlotPlacement = Literal["none", "thought", "after_turn", "prompt"]


@dataclass(frozen=True)
class SlotPlan:
    """Initial token values and immutable mask facts for one latent-slot arm."""

    ids: tuple[int, ...]
    placement: SlotPlacement
    pinned_mask: tuple[bool, ...]
    update_mask: tuple[bool, ...]
    loss_mask: tuple[bool, ...]
    trainable_token_ids: tuple[int, ...]


@dataclass(frozen=True, init=False)
class SlotInit:
    """Resolve a configured slot mode without changing a model or a canvas."""

    mode: SlotMode
    num_slots: int
    token_ids: tuple[int, ...]
    vocab_size: int | None
    pad_id: int | None
    spec: DiffusionSpec | None
    mask_token_id: int | None

    def __init__(
        self,
        mode: SlotMode = "none",
        ids: Sequence[int] = (),
        *,
        token_ids: Sequence[int] | None = None,
        num_slots: int = 0,
        vocab_size: int | None = None,
        pad_id: int | None = None,
        spec: DiffusionSpec | None = None,
        mask_token_id: int | None = None,
    ) -> None:
        if token_ids is not None and ids:
            raise ValueError("provide either ids or token_ids, not both")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "num_slots", num_slots)
        object.__setattr__(
            self, "token_ids", tuple(ids if token_ids is None else token_ids)
        )
        object.__setattr__(self, "vocab_size", vocab_size)
        object.__setattr__(self, "pad_id", pad_id)
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "mask_token_id", mask_token_id)

    @property
    def ids(self) -> tuple[int, ...]:
        """Compatibility alias for the configured token IDs."""
        return self.token_ids

    def build(
        self,
        *,
        seed: int | None = None,
        generator: random.Random | None = None,
    ) -> SlotPlan:
        """Validate the mode and resolve its deterministic initial slot values."""
        self._validate_common()
        if self.mode == "none":
            return SlotPlan((), "none", (), (), (), ())
        assert self.vocab_size is not None
        if self.mode == "pad":
            assert self.pad_id is not None
            self._require_no_token_ids()
            return self._fixed_plan(
                (self.pad_id,) * self.num_slots, placement="after_turn"
            )
        if self.mode == "pinned":
            values = self._pinned_values()
            return self._fixed_plan(values, placement="thought")
        if self.mode in {"learned", "prompt"}:
            self._require_exact_token_count(self.num_slots)
            if len(set(self.token_ids)) != self.num_slots:
                raise ValueError(f"{self.mode} slots require distinct token_ids")
            return self._fixed_plan(
                self.token_ids,
                placement="prompt" if self.mode == "prompt" else "thought",
                trainable=True,
            )
        if self.mode == "mask":
            self._require_no_token_ids()
            mask_token_id = self._absorbing_mask_token_id()
            return self._fixed_plan(
                (mask_token_id,) * self.num_slots, placement="thought"
            )
        if self.mode == "free":
            self._require_no_token_ids()
            assert self.spec is not None
            if self.spec.noise is DiffusionNoise.ABSORBING:
                values = (self._absorbing_mask_token_id(),) * self.num_slots
            else:
                values = self._uniform_values(seed=seed, generator=generator)
            return SlotPlan(
                values,
                "thought",
                (False,) * self.num_slots,
                (True,) * self.num_slots,
                (False,) * self.num_slots,
                (),
            )
        raise ValueError(f"unknown slot mode: {self.mode!r}")

    def _validate_common(self) -> None:
        if self.mode not in {
            "none",
            "pad",
            "pinned",
            "learned",
            "free",
            "mask",
            "prompt",
        }:
            raise ValueError(f"unknown slot mode: {self.mode!r}")
        if isinstance(self.num_slots, bool) or not isinstance(self.num_slots, int):
            raise ValueError("num_slots must be an integer")
        if self.num_slots < 0:
            raise ValueError("num_slots must be nonnegative")
        if self.mode == "none":
            if self.num_slots or self.token_ids:
                raise ValueError("mode=none requires num_slots=0 and no token_ids")
            return
        if self.num_slots < 1:
            raise ValueError(f"mode={self.mode} requires num_slots to be positive")
        if isinstance(self.vocab_size, bool) or not isinstance(self.vocab_size, int):
            raise ValueError("slot modes require an integer vocab_size")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.mode == "pad":
            self._validate_token_id(self.pad_id, "pad_id")
        if not isinstance(self.spec, DiffusionSpec):
            raise TypeError("slot modes require a DiffusionSpec")
        for token_id in self.token_ids:
            self._validate_token_id(token_id, "token_ids")

    def _validate_token_id(self, token_id: int | None, name: str) -> None:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"{name} must be an integer")
        assert self.vocab_size is not None
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"{name} must be within the model vocabulary")

    def _require_no_token_ids(self) -> None:
        if self.token_ids:
            raise ValueError(f"mode={self.mode} does not accept token_ids")

    def _require_exact_token_count(self, count: int) -> None:
        if len(self.token_ids) != count:
            raise ValueError(f"mode={self.mode} requires exactly {count} token_ids")

    def _pinned_values(self) -> tuple[int, ...]:
        if len(self.token_ids) == 1:
            return self.token_ids * self.num_slots
        if len(self.token_ids) != self.num_slots:
            raise ValueError(
                "mode=pinned requires exactly 1 token_id or exactly num_slots distinct token_ids"
            )
        if len(set(self.token_ids)) != self.num_slots:
            raise ValueError("mode=pinned distinct token_ids must not repeat")
        return self.token_ids

    def _absorbing_mask_token_id(self) -> int:
        assert self.spec is not None
        if self.spec.noise is not DiffusionNoise.ABSORBING:
            raise ValueError(f"mode={self.mode} requires absorbing diffusion noise")
        self._validate_token_id(self.mask_token_id, "mask_token_id")
        assert self.mask_token_id is not None
        return self.mask_token_id

    def _uniform_values(
        self, *, seed: int | None, generator: random.Random | None
    ) -> tuple[int, ...]:
        if seed is not None and generator is not None:
            raise ValueError("free uniform slots accept either seed or generator")
        if generator is None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError("free uniform slots require an explicit integer seed")
            generator = random.Random(seed)  # nosec B311 - caller supplied deterministic seed.
        assert self.vocab_size is not None
        return tuple(
            generator.randrange(self.vocab_size) for _ in range(self.num_slots)
        )

    def _fixed_plan(
        self,
        values: tuple[int, ...],
        *,
        placement: SlotPlacement,
        trainable: bool = False,
    ) -> SlotPlan:
        return SlotPlan(
            values,
            placement,
            (True,) * self.num_slots,
            (False,) * self.num_slots,
            (False,) * self.num_slots,
            values if trainable else (),
        )
