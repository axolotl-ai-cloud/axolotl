"""Unit tests for Multi-teacher On-Policy Distillation (MOPD) on GRPO."""

from unittest.mock import MagicMock

import pytest
import torch

from axolotl.core.trainers.grpo.args import AxolotlGRPOConfig, AxolotlMOPDConfig
from axolotl.core.trainers.grpo.mopd import (
    AxolotlMOPDTrainer,
    MOPDTeacherClient,
    mopd_advantages,
)
from axolotl.utils.schemas.mopd import MOPDConfig
from axolotl.utils.schemas.validation import MOPDValidationMixin, RLValidationMixin

MOPD = {"teachers": {"math": {"url": "http://math:8000"}}}


def test_advantages_are_clipped_and_masked():
    teacher = torch.tensor([[0.0, -10.0, -1.0]])
    student = torch.tensor([[-1.0, 0.0, -1.0]])
    mask = torch.tensor([[1, 1, 0]])
    adv = mopd_advantages(teacher, student, mask, clip=5.0)
    assert adv.tolist() == [[1.0, -5.0, 0.0]]


def test_teacher_client_reads_completion_token_logprobs():
    client = MOPDTeacherClient("http://teacher:8000/")
    response = MagicMock()
    response.json.return_value = {
        "choices": [
            {
                "index": 1,
                "prompt_logprobs": [None, {"7": {"logprob": -0.5}}],
            },
            {
                "index": 0,
                "prompt_logprobs": [
                    None,
                    {"2": {"logprob": -9.0}},
                    {"3": {"logprob": -0.1}, "4": {"logprob": -2.0}},
                    {"4": {"logprob": -0.2}},
                ],
            },
        ]
    }
    client.session.post = MagicMock(return_value=response)

    logps = client.sequence_logprobs([[1, 2, 3, 4], [5, 7]], [2, 1])

    assert logps == [[-0.1, -0.2], [-0.5]]
    (url,) = client.session.post.call_args.args
    assert url == "http://teacher:8000/v1/completions"
    assert client.session.post.call_args.kwargs["json"]["prompt_logprobs"] == 0


def test_teacher_logps_route_each_row_to_its_domain():
    trainer = object.__new__(AxolotlMOPDTrainer)
    trainer.mopd = MOPDConfig.model_validate(
        {"teachers": {"math": {"url": "u"}, "code": {"url": "u"}}}
    )
    calls = {}

    def fake_teacher(name, value):
        def score(sequences, prompt_lengths):
            calls[name] = (sequences, prompt_lengths)
            return [
                [value] * (len(s) - p)
                for s, p in zip(sequences, prompt_lengths, strict=True)
            ]

        return MagicMock(sequence_logprobs=score)

    trainer.mopd_teachers = {
        "math": fake_teacher("math", -1.0),
        "code": fake_teacher("code", -2.0),
    }
    output = {
        "prompt_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
        "prompt_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        "completion_ids": torch.tensor([[13, 14, 0], [24, 0, 0]]),
        "completion_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
    }

    logps = trainer._mopd_teacher_logps(output, ["code", "math"])

    assert calls["code"] == ([[11, 12, 13, 14]], [2])
    assert calls["math"] == ([[21, 22, 23, 24]], [3])
    assert logps.tolist() == [[-2.0, -2.0, 0.0], [-1.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match=r"no teacher for domain\(s\) \['chat'\]"):
        trainer._mopd_teacher_logps(output, ["chat", "math"])


def test_mopd_training_args_allow_single_generation(tmp_path):
    kwargs = {
        "output_dir": str(tmp_path),
        "num_generations": 1,
        "per_device_train_batch_size": 1,
        "report_to": "none",
        "use_cpu": True,
    }
    with pytest.raises(ValueError, match="at least 2 generations"):
        AxolotlGRPOConfig(**kwargs)
    assert AxolotlMOPDConfig(**kwargs).num_generations == 1


def test_config_validation():
    base = {
        "rl": "grpo",
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "trl": {"num_generations": 1},
    }
    with pytest.raises(ValueError, match=r"num_generations >= 2"):
        RLValidationMixin.check_grpo_batch_size_divisibility(base)
    RLValidationMixin.check_grpo_batch_size_divisibility({**base, "mopd": MOPD})

    with pytest.raises(ValueError, match="requires `rl: grpo`"):
        MOPDValidationMixin.check_mopd({"rl": "dpo", "mopd": MOPD})
    with pytest.raises(ValueError, match="trl.temperature: 1.0"):
        MOPDValidationMixin.check_mopd(
            {"rl": "grpo", "mopd": MOPD, "trl": {"temperature": 0.7}}
        )
    with pytest.raises(ValueError, match="trl.async_prefetch"):
        MOPDValidationMixin.check_mopd(
            {"rl": "grpo", "mopd": MOPD, "trl": {"async_prefetch": True}}
        )
