"""Multi-teacher On-Policy Distillation (MOPD, arXiv:2606.30406).

The student samples its own rollouts; each rollout is routed by domain to a frozen
teacher served by vLLM, which prefills it to give per-token log-probabilities. The
GRPO advantage is replaced by the clipped token-level reverse-KL signal
``clip(log pi_teacher(y_t) - log pi_student(y_t), -A, A)``.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
import torch
from trl.models.utils import disable_gradient_checkpointing

from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer
from axolotl.utils.schemas.mopd import MOPDConfig


def zero_reward(completions, **kwargs) -> list[float]:
    return [0.0] * len(completions)


def mopd_advantages(
    teacher_logps: torch.Tensor,
    student_logps: torch.Tensor,
    mask: torch.Tensor,
    clip: float,
) -> torch.Tensor:
    return (teacher_logps - student_logps).clamp(-clip, clip) * mask


class MOPDTeacherClient:
    """Scores token sequences with a teacher behind vLLM's OpenAI completions API."""

    def __init__(self, url: str, model: str | None = None, timeout: float = 600.0):
        self.endpoint = url.rstrip("/") + "/v1/completions"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def sequence_logprobs(
        self, sequences: list[list[int]], prompt_lengths: list[int]
    ) -> list[list[float]]:
        """Teacher log-prob of each completion token (positions >= prompt length)."""
        payload: dict[str, Any] = {
            "prompt": sequences,
            "max_tokens": 1,
            "prompt_logprobs": 0,
            "temperature": 1.0,
        }
        if self.model:
            payload["model"] = self.model
        response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()
        choices = sorted(response.json()["choices"], key=lambda c: c["index"])
        return [
            [
                entry[str(token)]["logprob"]
                for token, entry in zip(
                    seq[plen:], choice["prompt_logprobs"][plen:], strict=True
                )
            ]
            for seq, plen, choice in zip(
                sequences, prompt_lengths, choices, strict=True
            )
        ]


class AxolotlMOPDTrainer(AxolotlGRPOTrainer):
    """GRPO trainer whose advantages come from per-domain teachers."""

    _tag_names = ["trl", "grpo", "mopd", "axolotl"]

    def __init__(self, *args, mopd_config: MOPDConfig | dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.mopd = MOPDConfig.model_validate(mopd_config)
        self.mopd_teachers = {
            domain: MOPDTeacherClient(t.url, t.model, self.mopd.timeout)
            for domain, t in self.mopd.teachers.items()
        }

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        mode = "train" if self.model.training else "eval"

        mask = output["completion_mask"]
        if output.get("tool_mask") is not None:
            mask = mask * output["tool_mask"]
        domains = [x[self.mopd.domain_field] for x in inputs]
        teacher_logps = self._mopd_teacher_logps(output, domains)
        student_logps = self._mopd_student_logps(output, mode)

        advantages = mopd_advantages(
            teacher_logps, student_logps, mask, self.mopd.advantage_clip
        )
        if self.mopd.reward_weight:
            advantages = advantages + self.mopd.reward_weight * output[
                "advantages"
            ].unsqueeze(1)
        output["advantages"] = advantages

        n_tokens = mask.sum().clamp(min=1)
        log_ratio = (student_logps - teacher_logps) * mask
        clipped = log_ratio.abs() > self.mopd.advantage_clip
        metrics = self.accelerator.gather(
            torch.stack([log_ratio.sum() / n_tokens, clipped.sum() / n_tokens])
        ).view(-1, 2)
        self._metrics[mode]["mopd/reverse_kl"].append(metrics[:, 0].mean().item())
        self._metrics[mode]["mopd/clip_frac"].append(metrics[:, 1].mean().item())
        return output

    def _mopd_teacher_logps(
        self, output: dict[str, Any], domains: list[str]
    ) -> torch.Tensor:
        prompt_ids, prompt_mask = output["prompt_ids"], output["prompt_mask"]
        completion_ids, completion_mask = (
            output["completion_ids"],
            output["completion_mask"],
        )
        sequences, prompt_lengths = [], []
        for i in range(len(domains)):
            prompt = prompt_ids[i][prompt_mask[i].bool()].tolist()
            valid = completion_mask[i].nonzero()
            n_completion = int(valid.max()) + 1 if valid.numel() else 0
            sequences.append(prompt + completion_ids[i, :n_completion].tolist())
            prompt_lengths.append(len(prompt))

        unknown = set(domains) - self.mopd_teachers.keys()
        if unknown:
            raise ValueError(
                f"MOPD: no teacher for domain(s) {sorted(unknown)}; "
                f"configured: {sorted(self.mopd_teachers)}"
            )
        by_domain: dict[str, list[int]] = {}
        for i, domain in enumerate(domains):
            by_domain.setdefault(domain, []).append(i)

        with ThreadPoolExecutor(max_workers=len(by_domain)) as pool:
            futures = {
                domain: pool.submit(
                    self.mopd_teachers[domain].sequence_logprobs,
                    [sequences[i] for i in rows],
                    [prompt_lengths[i] for i in rows],
                )
                for domain, rows in by_domain.items()
            }
        teacher_logps = torch.zeros(completion_ids.shape, dtype=torch.float32)
        for domain, rows in by_domain.items():
            for i, logps in zip(rows, futures[domain].result(), strict=True):
                teacher_logps[i, : len(logps)] = torch.tensor(logps)
        return teacher_logps.to(completion_ids.device)

    def _mopd_student_logps(self, output: dict[str, Any], mode: str) -> torch.Tensor:
        # Present whenever TRL already ran the training model over the rollout
        # (vLLM importance-sampling correction or misaligned generation steps).
        if output.get("old_per_token_logps") is not None:
            return output["old_per_token_logps"]
        completion_ids = output["completion_ids"]
        input_ids = torch.cat([output["prompt_ids"], completion_ids], dim=1)
        attention_mask = torch.cat(
            [output["prompt_mask"], output["completion_mask"]], dim=1
        )
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )
        with (
            torch.no_grad(),
            disable_gradient_checkpointing(
                self.model, self.args.gradient_checkpointing_kwargs
            ),
        ):
            logps, _, _ = self._get_per_token_logps_and_entropies(
                self.model,
                input_ids,
                attention_mask,
                completion_ids.size(1),
                batch_size,
            )
        return logps
