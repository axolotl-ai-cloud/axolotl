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
                candidate = prefix + self.backend.sample(context, horizon - index)
                self._validate_tokens(candidate, horizon)
                proposed_target = self.backend.target_logprob(target_context, candidate)
                attempts += 1
                if self.config.acceptance == "greedy":
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

    def _validate_tokens(self, tokens: list[int], horizon: int):
        if not tokens or len(tokens) > horizon:
            raise ValueError("Proposal returned an empty or overlong trajectory")
        if len(tokens) < horizon and tokens[-1] not in self.backend.eos_token_ids:
            raise ValueError("Proposal stopped before its horizon without EOS")
        if any(token in self.backend.eos_token_ids for token in tokens[:-1]):
            raise ValueError("Proposal returned tokens after EOS")
