"""Blockwise projection sampling using expert-conditioned proposals."""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from .args import ProjectionSamplingConfig
from .backend import SamplingBackend


@dataclass
class SamplingResult:
    """Final chain state and acceptance statistics."""

    token_ids: list[int]
    target_logprob: float
    attempts: int
    accepted: int
    finished: bool


class ProjectionSampler:
    """Algorithm 1 with explicitly rescored forward and reverse proposals."""

    def __init__(
        self, backend: SamplingBackend, config: ProjectionSamplingConfig, seed: int = 42
    ):
        self.backend = backend
        self.config = config
        self.rng = random.Random(seed)  # nosec B311

    @property
    def _improvement_acceptance(self) -> bool:
        return self.config.acceptance in ("logprob_improvement", "greedy")

    def prompt_ids(self, text: str) -> list[int]:
        tokenizer = self.backend.tokenizer
        if self.config.prompt_format == "chat":
            return list(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if not tokens:
            if tokenizer.bos_token_id is None:
                raise ValueError("Raw prompts must tokenize to at least one token")
            tokens = [tokenizer.bos_token_id]
        return tokens

    def proposal_ids(
        self,
        question: str,
        expert: str,
        prefix: list[int],
        prompt_builder: Callable[[str], list[int]] | None = None,
    ) -> list[int]:
        text = self.config.proposal_template.format(
            question=question,
            expert_response=expert,
            prefix=self.backend.tokenizer.decode(prefix, skip_special_tokens=True),
        )
        return (prompt_builder or self.prompt_ids)(text) + prefix

    def sample(
        self,
        question: str,
        expert: str,
        *,
        target_context: list[int] | None = None,
        prompt_builder: Callable[[str], list[int]] | None = None,
    ) -> SamplingResult:
        if target_context is None:
            target_context = self.prompt_ids(question)
        current: list[int] = []
        attempts = accepted = 0
        target = 0.0
        for horizon in range(
            self.config.block_size,
            self.config.max_new_tokens + self.config.block_size,
            self.config.block_size,
        ):
            horizon = min(horizon, self.config.max_new_tokens)
            if self._improvement_acceptance:
                horizon = min(
                    len(current) + self.config.block_size, self.config.max_new_tokens
                )
            current += self.backend.sample(
                self.proposal_ids(question, expert, current, prompt_builder),
                horizon - len(current),
            )
            self._validate_tokens(current, horizon)
            target = self.backend.target_logprob(target_context, current)
            for _ in range(self.config.mcmc_steps):
                index = self.rng.randrange(len(current))
                prefix = current[:index]
                context = self.proposal_ids(question, expert, prefix, prompt_builder)
                proposal_horizon = (
                    len(current) if self._improvement_acceptance else horizon
                )
                if self.config.proposal_batch_size > 1:
                    current, target, accept = self._batched_step(
                        target_context,
                        current,
                        target,
                        context,
                        prefix,
                        proposal_horizon,
                    )
                    attempts += 1
                    accepted += int(accept)
                    continue
                candidate = prefix + self.backend.sample(
                    context, proposal_horizon - index
                )
                self._validate_tokens(candidate, proposal_horizon)
                proposed_target = self.backend.target_logprob(target_context, candidate)
                attempts += 1
                if self._improvement_acceptance:
                    accept = proposed_target / len(candidate) > target / len(current)
                else:
                    forward = self.backend.proposal_logprob(context, candidate[index:])
                    reverse = self.backend.proposal_logprob(context, current[index:])
                    # EOS can change length, hence the reverse cut-index probability.
                    log_ratio = (
                        proposed_target
                        - target
                        + reverse
                        - forward
                        + math.log(len(current) / len(candidate))
                    )
                    if math.isnan(log_ratio):
                        raise ValueError(
                            "Undefined Metropolis-Hastings acceptance ratio"
                        )
                    uniform = self.rng.random()
                    log_uniform = math.log(uniform) if uniform > 0 else -math.inf
                    accept = log_uniform < min(0.0, log_ratio)
                if accept:
                    current, target = candidate, proposed_target
                    accepted += 1
            if current[-1] in self.backend.eos_token_ids:
                break
        return SamplingResult(
            current,
            target,
            attempts,
            accepted,
            current[-1] in self.backend.eos_token_ids,
        )

    def _draw_candidates(self, context, prefix, horizon, count):
        suffixes = self.backend.sample_batch(
            [context] * count, [horizon - len(prefix)] * count
        )
        self._check_batch(suffixes, count)
        candidates = [prefix + suffix for suffix in suffixes]
        for candidate in candidates:
            self._validate_tokens(candidate, horizon)
        return candidates, suffixes

    def proposal_statistics(self, result: SamplingResult) -> dict[str, int]:
        count = self.config.proposal_batch_size
        if count == 1:
            return {}
        return {
            "proposal_batch_size": count,
            "forward_proposals": result.attempts * count,
            "balancing_proposals_reused": result.attempts * (count - 1)
            if self.config.acceptance == "metropolis_hastings"
            else 0,
        }

    @staticmethod
    def _check_batch(values, count):
        if len(values) != count:
            raise ValueError("Backend returned a misaligned proposal batch")

    @staticmethod
    def _logsumexp(values):
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Multiple-try proposal weights must be finite")
        maximum = max(values)
        return maximum + math.log(
            math.fsum(math.exp(value - maximum) for value in values)
        )

    def _batched_step(self, target_context, current, target, context, prefix, horizon):
        count = self.config.proposal_batch_size
        candidates, suffixes = self._draw_candidates(context, prefix, horizon, count)
        targets = self.backend.target_logprob_batch(
            [target_context] * count, candidates
        )
        self._check_batch(targets, count)
        if self._improvement_acceptance:
            selected = max(
                range(count), key=lambda index: targets[index] / len(candidates[index])
            )
            accept = targets[selected] / len(candidates[selected]) > target / len(
                current
            )
        else:
            proposals = self.backend.proposal_logprob_batch(
                [context] * (count + 1), suffixes + [current[len(prefix) :]]
            )
            self._check_batch(proposals, count + 1)
            forward, reverse = proposals[:-1], proposals[-1]
            # A uniformly chosen cut weights each state by 1 / length.
            weights = [
                value - proposal - math.log(len(candidate))
                for value, proposal, candidate in zip(
                    targets, forward, candidates, strict=True
                )
            ]
            total = self._logsumexp(weights)
            threshold = self.rng.random()
            selected = count - 1
            for index, weight in enumerate(weights):
                threshold -= math.exp(weight - total)
                if threshold <= 0:
                    selected = index
                    break
            # Fixed-prefix independence permits reusing the unselected trials.
            reverse_weights = weights[:selected] + weights[selected + 1 :]
            reverse_weights.append(target - reverse - math.log(len(current)))
            log_ratio = total - self._logsumexp(reverse_weights)
            uniform = self.rng.random()
            accept = (math.log(uniform) if uniform > 0 else -math.inf) < min(
                0.0, log_ratio
            )
        if accept:
            return candidates[selected], targets[selected], True
        return current, target, False

    def _validate_tokens(self, tokens: list[int], horizon: int):
        if not tokens or len(tokens) > horizon:
            raise ValueError("Proposal returned an empty or overlong trajectory")
        if len(tokens) < horizon and tokens[-1] not in self.backend.eos_token_ids:
            raise ValueError("Proposal stopped before its horizon without EOS")
        if any(token in self.backend.eos_token_ids for token in tokens[:-1]):
            raise ValueError("Proposal returned tokens after EOS")
