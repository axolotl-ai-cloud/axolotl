"""Async on-policy data for KD healing, on TRL's rollout-worker contract.

A generation worker runs the (quantized) student on its own device and streams
prompt+continuation samples into a queue while the trainer keeps stepping — the
`trl.experimental.async_grpo` architecture with the GRPO-specific parts (rewards,
advantages, vLLM server) replaced by plain student generation. The trainer side
consumes the queue as an iterable dataset with a bounded weight-staleness window.
"""

from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch.utils.data import IterableDataset

from axolotl.utils.logging import get_logger

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

LOG = get_logger(__name__)

IGNORE_INDEX: int = -100


@dataclass
class KDSample:
    """One prompt+continuation pair with the weight version that generated it."""

    input_ids: list[int]
    prompt_len: int
    model_version: int


@dataclass
class HFGenerateRolloutWorker:
    """`RolloutWorkerProtocol` backend that generates with `model.generate`.

    Runs a daemon thread over `prompts` (token-id lists, cycled), generating with
    the student copy owned by this worker. `sync_weights` replaces that copy's
    parameters from the live trainer between optimizer steps; generation of the
    in-flight batch finishes on the old weights, matching the bounded-staleness
    contract of the async-GRPO worker this mirrors.

    Attributes:
        rollout_buffer: Queue the dataset side drains.
    """

    model: torch.nn.Module
    prompts: list[list[int]]
    pad_token_id: int
    max_new_tokens: int = 256
    temperature: float = 0.0
    batch_size: int = 8
    buffer_size: int = 64
    rollout_buffer: queue.Queue = field(init=False)

    def __post_init__(self) -> None:
        if not self.prompts:
            raise ValueError("async KD worker needs a non-empty prompt pool")
        self.rollout_buffer = queue.Queue(maxsize=self.buffer_size)
        self._version = 0
        self._stop = threading.Event()
        self._sync_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="ternary-async-kd-worker", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=60)

    def update_model_version(self, version: int) -> None:
        self._version = version

    def check_health(self, stale_after_s: float) -> None:
        if self._failure is not None:
            raise RuntimeError("async KD worker died") from self._failure
        if self._thread is not None and not self._thread.is_alive():
            raise RuntimeError("async KD worker thread exited without a failure")

    def sync_weights(self, state_dict: dict[str, torch.Tensor], version: int) -> None:
        """Load fresh student weights; generation in flight finishes on the old ones."""
        with self._sync_lock:
            self.model.load_state_dict(state_dict, strict=False)
            self._version = version

    def _run(self) -> None:
        try:
            device = next(self.model.parameters()).device
            self.model.eval()
            cursor = 0
            while not self._stop.is_set():
                batch = [
                    self.prompts[(cursor + i) % len(self.prompts)]
                    for i in range(self.batch_size)
                ]
                cursor = (cursor + self.batch_size) % len(self.prompts)

                width = max(len(p) for p in batch)
                ids = torch.full(
                    (len(batch), width), self.pad_token_id, dtype=torch.long
                )
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, prompt in enumerate(batch):
                    ids[row, width - len(prompt) :] = torch.tensor(prompt)
                    mask[row, width - len(prompt) :] = 1

                sampling: dict[str, Any] = (
                    {"do_sample": True, "temperature": self.temperature}
                    if self.temperature > 0
                    else {"do_sample": False}
                )
                with self._sync_lock, torch.no_grad():
                    version = self._version
                    generated = self.model.generate(
                        input_ids=ids.to(device),
                        attention_mask=mask.to(device),
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.pad_token_id,
                        **sampling,
                    )

                for row, prompt in enumerate(batch):
                    completion = generated[row, width:].tolist()
                    while completion and completion[-1] == self.pad_token_id:
                        completion.pop()
                    if not completion:
                        continue
                    sample = KDSample(
                        input_ids=list(prompt) + completion,
                        prompt_len=len(prompt),
                        model_version=version,
                    )
                    while not self._stop.is_set():
                        try:
                            self.rollout_buffer.put(sample, timeout=1.0)
                            break
                        except queue.Full:
                            continue
        except BaseException as exc:  # noqa: BLE001 - surfaced via check_health
            self._failure = exc


class KDRolloutDataset(IterableDataset):
    """Drains a rollout buffer into prompt-masked KD training samples.

    Samples older than `max_staleness` weight versions are dropped; `version_fn`
    supplies the trainer's current version at read time.
    """

    def __init__(
        self,
        buffer: queue.Queue,
        version_fn: Callable[[], int],
        max_staleness: int = 4,
        max_samples: int | None = None,
    ) -> None:
        self.buffer = buffer
        self.version_fn = version_fn
        self.max_staleness = max_staleness
        self.max_samples = max_samples
        self.dropped_stale = 0

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        emitted = 0
        while self.max_samples is None or emitted < self.max_samples:
            sample: KDSample = self.buffer.get()
            if self.version_fn() - sample.model_version > self.max_staleness:
                self.dropped_stale += 1
                continue
            labels = list(sample.input_ids)
            labels[: sample.prompt_len] = [IGNORE_INDEX] * sample.prompt_len
            emitted += 1
            yield {"input_ids": sample.input_ids, "labels": labels}


@dataclass
class KDRolloutCollator:
    """Right-pads rollout samples; padding is masked from both loss terms."""

    pad_token_id: int

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        width = max(len(e["input_ids"]) for e in examples)
        input_ids = torch.full(
            (len(examples), width), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
        for row, example in enumerate(examples):
            n = len(example["input_ids"])
            input_ids[row, :n] = torch.tensor(example["input_ids"])
            labels[row, :n] = torch.tensor(example["labels"])
            attention_mask[row, :n] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
