"""Rollout-process extensions for TRL's async GRPO worker.

Everything here runs inside the CUDA-free child process TRL's
``AsyncRolloutWorker`` spawns, so it keeps its imports light.
"""

from __future__ import annotations

import asyncio
import functools
import multiprocessing as mp
import queue
import time
import warnings
from typing import Any

import numpy as np

from axolotl.core.trainers.grpo.trl_compat import (
    AsyncRolloutLoop,
    AsyncRolloutWorker,
    RolloutGroup,
    RolloutSample,
    TrainingSequence,
    add_response_schema,
    is_async_callable,
    print_prompt_completions_sample,
    sampler_fields,
)
from axolotl.utils.logging import get_logger
from axolotl.utils.reward_worker import reward_worker_main

LOG = get_logger(__name__)

AGGREGATIONS = ("sum_then_normalize", "normalize_then_sum")
REWARD_SCALINGS = ("group", "none")
# Key in `RolloutSample.metrics` carrying what the trainer needs to score a sample whose
# advantage is still pending; the trainer pops it before metrics are logged.
SCORE_PAYLOAD = "_trainer_score"


def compute_group_advantages(
    rewards_per_func: np.ndarray,
    reward_weights: np.ndarray,
    aggregation: str = "sum_then_normalize",
    scale_rewards: str = "group",
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted rewards and the advantages of one prompt group.

    Args:
        rewards_per_func: ``(num_funcs, group_size)`` rewards, NaN where a function
            did not score a completion.
        reward_weights: ``(num_funcs,)`` weight of each reward function.
        aggregation: ``sum_then_normalize`` normalizes the weighted sum;
            ``normalize_then_sum`` (GDPO) normalizes each function within the group
            before the weighted sum.
        scale_rewards: ``group`` divides by the group's standard deviation,
            ``none`` only centers.

    A completion no function scored keeps a NaN reward and a zero advantage. A group
    whose learning signal is constant gets exactly zero advantages, so it can be
    told apart from a group that merely has small ones.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"multi_objective_aggregation must be one of {AGGREGATIONS}")
    if scale_rewards not in REWARD_SCALINGS:
        raise ValueError(f"scale_rewards must be one of {REWARD_SCALINGS}")
    weights = reward_weights[:, None]
    unscored = np.all(np.isnan(rewards_per_func), axis=0)
    rewards = np.nansum(rewards_per_func * weights, axis=0)
    rewards[unscored] = np.nan
    advantages = np.zeros_like(rewards)
    scored = ~unscored
    if not scored.any():
        return rewards, advantages

    if aggregation == "normalize_then_sum":
        per_func = rewards_per_func[:, scored]
        with warnings.catch_warnings():
            # A function that scored none of the group yields an all-NaN row.
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(per_func, axis=1, keepdims=True)
            std = np.nanstd(per_func, axis=1, keepdims=True)
        signal = np.nansum((per_func - mean) / (std + 1e-8) * weights, axis=0)
    else:
        signal = rewards[scored]

    if np.ptp(signal) == 0:
        return rewards, advantages
    centered = signal - signal.mean()
    if scale_rewards == "group":
        centered = centered / (signal.std() + 1e-8)
    advantages[scored] = centered
    return rewards, advantages


def _round_trip(conn, message):
    conn.send(message)
    try:
        return conn.recv()
    except EOFError as error:
        raise RuntimeError("A reward worker process exited mid-call") from error


class RewardProcessPool:
    """Long-lived processes that run synchronous reward functions.

    Each call runs on a worker's main thread, so CPU-bound rewards scale past the
    GIL and ``signal.alarm`` timeouts (math_verify) stay enabled; TRL's default
    ``asyncio.to_thread`` has to switch them off.
    """

    def __init__(self, reward_funcs: list, num_workers: int):
        ctx = mp.get_context("spawn")
        # TRL starts the rollout process as a daemon, and multiprocessing refuses to
        # let a daemon start children. The workers are daemons themselves and exit on
        # EOF when this process dies, so none outlives it.
        mp.current_process()._config["daemon"] = False  # type: ignore[attr-defined]
        self._workers = []
        for _ in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=reward_worker_main,
                args=(child_conn, reward_funcs),
                name="grpo-reward-worker",
                daemon=True,
            )
            process.start()
            child_conn.close()
            self._workers.append((parent_conn, process))
        self._idle: asyncio.Queue | None = None

    def __len__(self) -> int:
        return len(self._workers)

    async def call(self, func_index: int, kwargs: dict[str, Any]) -> list:
        if self._idle is None:
            self._idle = asyncio.Queue()
            for worker in self._workers:
                self._idle.put_nowait(worker)
        worker = await self._idle.get()
        try:
            ok, payload = await asyncio.to_thread(
                _round_trip, worker[0], (func_index, kwargs)
            )
        finally:
            self._idle.put_nowait(worker)
        if not ok:
            raise RuntimeError(
                f"Reward function raised in a worker process:\n{payload}"
            )
        return payload

    def close(self) -> None:
        for conn, process in self._workers:
            conn.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        self._workers = []


def _decode_as_message(tokenizer, ids, *args, **kwargs) -> dict[str, str]:
    return {
        "role": "assistant",
        "content": tokenizer.decode(ids, skip_special_tokens=True),
    }


class AxolotlRolloutLoop(AsyncRolloutLoop):
    """TRL's rollout loop with GRPOTrainer's reward options and parallel scoring.

    Subclasses that score each rollout themselves set ``_provides_rollout_reward``
    and return the reward from ``_generate_one``; it is scored as one more reward
    column, named ``_rollout_reward_name``, after the reward functions.

    With ``trainer_scored``, the trainer adds reward columns that need the GPU, so
    this process scores its own columns but leaves the advantage pending: samples
    carry a NaN advantage and a payload under ``SCORE_PAYLOAD`` (the group's row
    count, the columns scored here and ``_trainer_score_payload``).
    """

    _provides_rollout_reward: bool = False
    _rollout_reward_name = "rollout_reward"
    # Whether completions go through the chat template's response parser, which reads
    # tool calls out of them. Loops that decode text themselves, or get messages from
    # an agent, set this to False and need no parser.
    _parses_responses = True

    def __init__(
        self,
        *,
        reward_weights: list[float] | None = None,
        multi_objective_aggregation: str = "sum_then_normalize",
        scale_rewards: str = "group",
        reward_num_workers: int = 1,
        trainer_scored: bool = False,
        **kwargs,
    ):
        self._trainer_scored = trainer_scored
        provides_rollout_reward = self._provides_rollout_reward
        if trainer_scored:
            # TRL refuses a loop without a reward source; the trainer is one here.
            self._provides_rollout_reward = True
        tokenizer = kwargs["processing_class"]
        has_parser = (
            getattr(tokenizer, "response_template", None) is not None
            or getattr(tokenizer, "response_schema", None) is not None
        )
        decode_responses = False
        if (
            not has_parser
            and self._parses_responses
            and not (kwargs.get("tools") or kwargs.get("environment_factory"))
        ):
            # TRL ships response parsers, which read tool calls out of completions, for
            # some chat templates only and refuses to start on any other. Without
            # tools, the decoded completion is the whole assistant message.
            try:
                kwargs["processing_class"] = add_response_schema(tokenizer)
                has_parser = True
            except ValueError:
                decode_responses = True
        skip_parser = not has_parser and (
            decode_responses or not self._parses_responses
        )
        if skip_parser:
            tokenizer.response_template = ""  # passes TRL's check for a parser
        try:
            super().__init__(**kwargs)
        finally:
            if skip_parser:
                tokenizer.response_template = None
        self._provides_rollout_reward = provides_rollout_reward
        if decode_responses:
            tokenizer.parse_response = functools.partial(_decode_as_message, tokenizer)
            LOG.info(
                "TRL has no response parser for this chat template; completions are "
                "decoded as plain assistant messages."
            )
        if self._provides_rollout_reward:
            self.reward_func_names.append(self._rollout_reward_name)
        # `_generate_one` gets only the prompt; subclasses that need the whole row
        # look it up here by group.
        self._group_rows: dict[int, dict] = {}
        if trainer_scored:
            # The trainer owns the weights and advantages, and logs completions once
            # they are scored.
            reward_weights = None
            self.log_completions = False
        num_columns = len(self.reward_func_names)
        if reward_weights is None:
            reward_weights = [1.0] * num_columns
        if len(reward_weights) != num_columns:
            raise ValueError(
                f"reward_weights has {len(reward_weights)} entries but there are "
                f"{num_columns} reward sources ({self.reward_func_names})"
            )
        self._reward_weights = np.asarray(reward_weights, dtype=float)
        self._aggregation = multi_objective_aggregation
        self._scale_rewards = scale_rewards
        # Validates the options before any rollout is generated.
        compute_group_advantages(
            np.zeros((num_columns, 1)),
            self._reward_weights,
            self._aggregation,
            self._scale_rewards,
        )
        self._reward_pool = (
            RewardProcessPool(self.reward_funcs, reward_num_workers)
            if reward_num_workers > 1 and self.reward_funcs
            else None
        )

    def run(self) -> None:
        try:
            super().run()
        finally:
            if self._reward_pool is not None:
                self._reward_pool.close()

    def _repeat_iterator(self):
        for group_id, row in super()._repeat_iterator():
            self._group_rows[group_id] = row
            yield group_id, row

    def _trainer_score_payload(
        self, group: RolloutGroup, index: int, sequence: TrainingSequence
    ) -> dict[str, Any]:
        """What the trainer's scorers need for one training row of rollout ``index``."""
        return {}

    async def _call_reward_func(self, func_index: int, kwargs: dict[str, Any]):
        func = self.reward_funcs[func_index]
        if is_async_callable(func):
            return await func(**kwargs)
        if self._reward_pool is not None:
            return await self._reward_pool.call(func_index, kwargs)
        return await asyncio.to_thread(func, **kwargs)

    async def _score_loop(self, stop_event: asyncio.Event) -> None:
        if self._reward_pool is None:
            await super()._score_loop(stop_event)
            return
        # Several groups score at once; TRL's loop scores one at a time.
        max_concurrent = len(self._reward_pool)
        tasks: set[asyncio.Task] = set()
        try:
            while not stop_event.is_set():
                self._heartbeat_value.value = time.time()
                for task in [task for task in tasks if task.done()]:
                    tasks.discard(task)
                    task.result()
                if len(tasks) >= max_concurrent:
                    await asyncio.wait(
                        tasks, timeout=0.5, return_when=asyncio.FIRST_COMPLETED
                    )
                    continue
                try:
                    group = await asyncio.wait_for(
                        self._groups_to_score.get(), timeout=0.5
                    )
                except TimeoutError:
                    continue
                if group is None:
                    break
                tasks.add(asyncio.create_task(self._score_and_push(group, stop_event)))
            if tasks:
                await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()

    async def _score_and_push(
        self, group: RolloutGroup, stop_event: asyncio.Event
    ) -> None:
        wait_scoring = time.monotonic() - group.queued_at
        t0 = time.monotonic()
        samples = await self._score_group(group)
        now = time.monotonic()
        self._push_metrics(
            {
                "rollout/score_s": now - t0,
                "rollout/score_wait_s": wait_scoring,
                "rollout/score_queue_size": float(self._groups_to_score.qsize()),
                "rollout/inflight": float(self._inflight),
                "rollout/generated_tok_s": (
                    float(
                        self._total_completion_tokens - self._pushed_completion_tokens  # type: ignore[has-type]
                    ),
                    now - self._pushed_at,  # type: ignore[has-type]
                ),
            }
        )
        self._pushed_completion_tokens = self._total_completion_tokens
        self._pushed_at = now

        if self.log_completions and samples:
            print_prompt_completions_sample(
                prompts=[s.prompt for s in samples],
                completions=[s.completion for s in samples],
                rewards={"reward": [s.metrics["reward"] for s in samples]},
                advantages=[s.advantage for s in samples],
                step=self._total_groups_scored,
                num_samples=self.num_completions_to_print,
            )
        self._total_groups_scored += 1

        for sample in samples:
            t_blocked = None
            while True:
                try:
                    sample.enqueued_at = time.time()
                    self.rollout_buffer.put_nowait(sample)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        return
                    if t_blocked is None:
                        t_blocked = time.monotonic()
                    await asyncio.sleep(0.1)
            if t_blocked is not None:
                self._push_metrics(
                    {"rollout/backpressure_s": time.monotonic() - t_blocked}
                )

    async def _score_group(self, group: RolloutGroup) -> list[RolloutSample]:
        self._group_rows.pop(group.group_id, None)
        kwargs = dict(
            completions=group.completions,
            prompts=group.prompts,
            completion_ids=group.completions_ids,
            **group.reward_kwargs,
        )
        all_rewards = list(
            await asyncio.gather(
                *[
                    self._call_reward_func(index, kwargs)
                    for index in range(len(self.reward_funcs))
                ]
            )
        )
        for env_type in self._env_reward_types:
            all_rewards.append(
                [
                    r[1] if (r is not None and r[0] is env_type) else None
                    for r in group.env_rewards
                ]
            )
        if self._provides_rollout_reward:
            all_rewards.append(list(group.rollout_rewards))

        rewards_per_func = np.array(
            [[np.nan if r is None else r for r in column] for column in all_rewards],
            dtype=float,
        ).reshape(len(all_rewards), len(group.completions))
        if self._trainer_scored:
            return self._pending_samples(group, rewards_per_func)
        rewards, advantages = compute_group_advantages(
            rewards_per_func,
            self._reward_weights,
            self._aggregation,
            self._scale_rewards,
        )
        scored_rewards = rewards[~np.isnan(rewards)]
        reward_std = float(scored_rewards.std()) if scored_rewards.size else np.nan

        tool_metrics = self._tool_metrics(group)
        samples = []
        for i, (prompt, completion, sequences, advantage, reward, extra) in enumerate(
            zip(
                group.prompts,
                group.completions,
                group.completions_sequences,
                advantages,
                rewards,
                tool_metrics,
                strict=True,
            )
        ):
            metrics = {
                "reward": float(reward),
                "reward_std": reward_std,
                **{
                    f"rewards/{name}": float(value)
                    for name, value in zip(
                        self.reward_func_names, rewards_per_func[:, i], strict=True
                    )
                },
                **extra,
            }
            for seq in sequences:
                samples.append(
                    self._make_sample(
                        group, prompt, completion, seq, advantage, metrics
                    )
                )
        return samples

    def _pending_samples(
        self, group: RolloutGroup, rewards_per_func: np.ndarray
    ) -> list[RolloutSample]:
        tool_metrics = self._tool_metrics(group)
        group_rows = sum(len(sequences) for sequences in group.completions_sequences)
        samples = []
        for i, (prompt, completion, sequences, extra) in enumerate(
            zip(
                group.prompts,
                group.completions,
                group.completions_sequences,
                tool_metrics,
                strict=True,
            )
        ):
            for seq in sequences:
                payload = {
                    "group_rows": group_rows,
                    "reward_names": list(self.reward_func_names),
                    "rewards": rewards_per_func[:, i].tolist(),
                    **self._trainer_score_payload(group, i, seq),
                }
                metrics = {**extra, SCORE_PAYLOAD: payload}
                samples.append(
                    self._make_sample(
                        group, prompt, completion, seq, float("nan"), metrics
                    )
                )
        return samples

    @staticmethod
    def _make_sample(
        group, prompt, completion, seq, advantage, metrics
    ) -> RolloutSample:
        return RolloutSample(
            prompt=prompt,
            completion=completion,
            input_ids=np.asarray(seq.input_ids, dtype=np.int64),
            completion_mask=np.asarray(seq.completion_mask, dtype=np.int64),
            old_log_probs=np.asarray(seq.old_log_probs, dtype=np.float32),
            **sampler_fields(seq),
            advantage=float(advantage),
            model_version=group.model_version,
            group_id=group.group_id,
            metrics=dict(metrics),
        )

    @staticmethod
    def _tool_metrics(group: RolloutGroup) -> list[dict[str, float]]:
        if sum(group.tool_call_counts) > 0:
            return [
                {
                    "tools/call_frequency": float(n_calls),
                    "tools/failure_frequency": (n_failures / n_calls)
                    if n_calls > 0
                    else 0.0,
                }
                for n_calls, n_failures in zip(
                    group.tool_call_counts, group.tool_failure_counts, strict=True
                )
            ]
        return [{}] * len(group.completions)


class AxolotlAsyncRolloutWorker(AsyncRolloutWorker):
    """``AsyncRolloutWorker`` whose child process runs :class:`AxolotlRolloutLoop`,
    or the subclass passed as ``loop_cls``."""

    _loop_cls = AxolotlRolloutLoop

    def __init__(
        self, *, loop_cls: type[AxolotlRolloutLoop] | None = None, **kwargs: Any
    ):
        super().__init__(**kwargs)
        if loop_cls is not None:
            self._loop_cls = loop_cls

    def stop(self) -> None:
        # A child that died on its own (it raised) stays counted as a waiter on the
        # stop event, and multiprocessing's Event.set() waits for every waiter to
        # wake, so TRL's stop() would hang the trainer instead of re-raising.
        process = self._process  # type: ignore[has-type]
        if (
            process is not None
            and process._popen is not None
            and not process.is_alive()
        ):
            process.join()
            self._process = None
            return
        super().stop()
