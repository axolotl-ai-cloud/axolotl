"""The correction gate must not activate TRL's generation backend or leak state."""

from types import SimpleNamespace

import pytest

from axolotl.integrations.coreweave.correction import install_sampler_correction


def test_nested_generation_gate_and_exception_restore():
    trainer = SimpleNamespace(
        use_vllm=False,
        loss_type="grpo",
        use_liger_kernel=False,
        vllm_importance_sampling_correction=True,
    )

    def generate():
        assert trainer.use_vllm is False
        raise RuntimeError("generation failed")

    def score():
        assert trainer.use_vllm is True
        trainer._generate()

    def loss():
        assert trainer.use_vllm is True
        raise RuntimeError("loss failed")

    trainer._generate = generate
    trainer._generate_and_score_completions = score
    trainer._compute_loss = loss
    install_sampler_correction(trainer)
    install_sampler_correction(trainer)
    with pytest.raises(RuntimeError, match="generation failed"):
        trainer._generate_and_score_completions()
    assert trainer.use_vllm is False
    with pytest.raises(RuntimeError, match="loss failed"):
        trainer._compute_loss()
    assert trainer.use_vllm is False
