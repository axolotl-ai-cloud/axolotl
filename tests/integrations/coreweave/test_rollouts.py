"""Rollout synchronization and the pinned TRL generation contract."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.coreweave import rollouts

from .test_checkpoints import config


def trainer(tmp_path):
    model = MagicMock()
    model.config.to_dict.return_value = {"model_type": "test"}
    accelerator = MagicMock(is_main_process=True)
    accelerator.get_state_dict.return_value = {"weight": torch.ones(2)}
    accelerator.unwrap_model.return_value = model
    return SimpleNamespace(
        accelerator=accelerator,
        model_wrapped=model,
        state=SimpleNamespace(global_step=0),
        args=SimpleNamespace(output_dir=str(tmp_path)),
        _tokenize_prompts=lambda prompts: ([[1, 2] for _ in prompts], None, {}),
        max_completion_length=8,
        temperature=1.0,
        top_p=0.9,
    )


def test_sync_precedes_sampling_once_per_step(tmp_path, monkeypatch):
    events = []
    client = MagicMock()
    client.sample.side_effect = lambda *a, **k: (
        events.append("sample") or [{"tokens": [3, 4], "logprobs": [-0.1, -0.2]}]
    )
    publisher = MagicMock(identity="identity")
    publisher.publish.side_effect = lambda *args: events.append("publish")
    monkeypatch.setattr(rollouts, "CoreWeaveClient", lambda cfg: client)
    monkeypatch.setattr(rollouts, "CheckpointPublisher", lambda *a: publisher)
    generate = rollouts.CoreWeaveRollouts(config(rollouts=True))
    tr = trainer(tmp_path)
    prompts = [[{"role": "user", "content": "test"}]] * 2
    output = generate(prompts, tr)
    assert events == ["publish", "sample", "sample"]
    assert output == {
        "prompt_ids": [[1, 2], [1, 2]],
        "completion_ids": [[3, 4], [3, 4]],
        "logprobs": [[-0.1, -0.2], [-0.1, -0.2]],
    }
    generate(prompts, tr)
    assert publisher.publish.call_count == 1
    tr.state.global_step = 1
    generate(prompts, tr)
    assert publisher.publish.call_count == 2
    assert client.ready_identity == "identity"
    assert client.sample.call_args.kwargs["n"] == 1
    assert client.sample.call_args.args[0] == [1, 2]


def test_parallel_sampling_keeps_prompt_order(tmp_path, monkeypatch):
    import threading
    import time

    barrier = threading.Barrier(4)

    def sample(tokens, **kwargs):
        barrier.wait(timeout=5)
        time.sleep(0.01 * (4 - tokens[0]))
        return [{"tokens": [tokens[0] * 10], "logprobs": [-float(tokens[0])]}]

    client = MagicMock()
    client.sample.side_effect = sample
    monkeypatch.setattr(rollouts, "CoreWeaveClient", lambda cfg: client)
    monkeypatch.setattr(
        rollouts, "CheckpointPublisher", lambda *a: MagicMock(identity="identity")
    )
    tr = trainer(tmp_path)
    tr._tokenize_prompts = lambda prompts: (
        [[i] for i in range(len(prompts))],
        None,
        {},
    )
    output = rollouts.CoreWeaveRollouts(config(rollouts=True))(
        [[{"role": "user", "content": "test"}]] * 4, tr
    )
    assert output["completion_ids"] == [[0], [10], [20], [30]]
    assert output["logprobs"] == [[-0.0], [-1.0], [-2.0], [-3.0]]


def test_sync_failure_blocks_sampling(tmp_path, monkeypatch):
    client = MagicMock()
    publisher = MagicMock()
    publisher.publish.side_effect = TimeoutError("not converged")
    monkeypatch.setattr(rollouts, "CoreWeaveClient", lambda cfg: client)
    monkeypatch.setattr(rollouts, "CheckpointPublisher", lambda *a: publisher)
    generate = rollouts.CoreWeaveRollouts(config(rollouts=True))
    with pytest.raises(RuntimeError, match="not converged"):
        generate([[{"role": "user", "content": "test"}]], trainer(tmp_path))
    client.sample.assert_not_called()
    assert generate.step is None


def test_peer_failure_propagates(monkeypatch):
    monkeypatch.setattr(rollouts.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(rollouts.dist, "get_world_size", lambda: 2)

    def gather(result, error):
        result[:] = [error, "peer request failed"]

    monkeypatch.setattr(rollouts.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="peer request failed"):
        rollouts.check_rank_errors(None)


@pytest.mark.parametrize(
    "prompts", [["raw text"], [[{"role": "user", "content": [{"type": "image"}]}]]]
)
def test_unsupported_prompt_fails_before_publication(tmp_path, prompts):
    tr = trainer(tmp_path)
    with pytest.raises(RuntimeError, match="conversational"):
        rollouts.CoreWeaveRollouts(config())(prompts, tr)
    tr.accelerator.get_state_dict.assert_not_called()


@pytest.mark.parametrize(
    "correction_mode",
    ["token_truncate", "token_mask", "sequence_truncate", "sequence_mask"],
)
def test_cpu_grpo_trains_with_remote_rollouts(tmp_path, monkeypatch, correction_mode):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    from axolotl.integrations.coreweave.plugin import CoreWeavePlugin

    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "yes": 3, "no": 4, "test": 5}
    raw = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            eos_token_id=2,
            bos_token_id=2,
            pad_token_id=0,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    client = MagicMock()
    counter = iter(range(100))
    client.sample.side_effect = lambda *a, **k: [
        {"tokens": [3 + next(counter) % 2, 2], "logprobs": [-0.7, -0.2]}
    ]
    snapshots = []
    publisher = MagicMock(identity="confirmed")
    publisher.publish.side_effect = lambda weights, *args: snapshots.append(
        {name: value.detach().clone() for name, value in weights.items()}
    )
    monkeypatch.setattr(rollouts, "CoreWeaveClient", lambda cfg: client)
    monkeypatch.setattr(rollouts, "CheckpointPublisher", lambda *a: publisher)

    def reward(completions, **kwargs):
        return [
            float(completion[0]["content"].startswith("yes"))
            for completion in completions
        ]

    tr = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward,
        train_dataset=Dataset.from_list(
            [{"prompt": [{"role": "user", "content": "test"}]}] * 4
        ),
        args=GRPOConfig(
            output_dir=str(tmp_path),
            use_cpu=True,
            bf16=False,
            fp16=False,
            per_device_train_batch_size=2,
            num_generations=2,
            max_steps=2,
            max_completion_length=4,
            beta=0,
            report_to=[],
            save_strategy="no",
            disable_tqdm=True,
            gradient_checkpointing=False,
            vllm_importance_sampling_mode=correction_mode,
            learning_rate=0.01,
        ),
    )
    CoreWeavePlugin().post_trainer_create(
        SimpleNamespace(coreweave=config(rollouts=True)), tr
    )
    scored = []
    score = tr._generate_and_score_completions

    def capture(inputs):
        batch = score(inputs)
        difference = (
            batch["old_per_token_logps"] - batch["sampling_per_token_logps"]
        ) * batch["completion_mask"]
        if correction_mode.startswith("sequence"):
            difference = difference.sum(dim=-1, keepdim=True)
        expected = difference.exp()
        if correction_mode.endswith("truncate"):
            expected = expected.clamp(
                min=tr.vllm_importance_sampling_clip_min,
                max=tr.vllm_importance_sampling_clip_max,
            )
        else:
            expected = expected.masked_fill(
                expected > tr.vllm_importance_sampling_clip_max, 0
            )
        torch.testing.assert_close(batch["importance_sampling_ratio"], expected)
        scored.append(batch)
        assert not tr.use_vllm
        return batch

    tr._generate_and_score_completions = capture
    result = tr.train()
    assert not tr.use_vllm
    assert not hasattr(tr, "vllm_generation")
    assert scored
    batch = {**scored[-1], "advantages": torch.ones_like(scored[-1]["advantages"])}
    loss = tr.compute_loss(tr.model, batch)
    half = {
        **batch,
        "importance_sampling_ratio": batch["importance_sampling_ratio"] * 0.5,
    }
    half_loss = tr.compute_loss(tr.model, half)
    assert loss.abs().item() > 0
    torch.testing.assert_close(half_loss, loss * 0.5)
    assert not tr.use_vllm
    assert result.global_step == 2
    assert len(snapshots) == 2
    assert any(
        not torch.equal(snapshots[0][key], snapshots[1][key]) for key in snapshots[0]
    )
