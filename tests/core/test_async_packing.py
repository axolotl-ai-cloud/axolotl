"""Async GRPO packing for models with recurrent layers (tree-recurrence in axolotl)."""

import types

import numpy as np
import pytest
import torch
from async_test_utils import (
    StubTransfer,
    StubWorker,
    assert_tree_rows_match_alone,
    branching_samples,
    reference_tree_attention,
)
from tree_recurrence import Support, uninstall
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from axolotl.core.trainers.grpo.async_scoring import pack_sequences
from axolotl.core.trainers.grpo.async_trainer import (
    AxolotlAsyncGRPOConfig,
    AxolotlAsyncGRPOTrainer,
)
from axolotl.core.trainers.grpo.trl_compat import unsampled_fields
from axolotl.monkeypatch.models.tree_packing import packing_support


def test_pack_sequences_without_sharing():
    sequences = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])]
    shared = pack_sequences(sequences)
    chains = pack_sequences(sequences, share_prefixes=False)
    assert len(shared["input_ids"]) == 6
    assert chains["input_ids"].tolist() == [1, 2, 3, 4, 1, 2, 3, 5]
    assert chains["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert chains["tree_cu_seqlens_q"].tolist() == [0, 4, 8]
    assert chains["positions"].tolist() == list(range(8))


@pytest.fixture
def tree_hooks(monkeypatch):
    """tree-recurrence's hooks as packing_support installs them, off after the test."""
    from trl.experimental.async_grpo import packing

    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    uninstall()
    yield
    uninstall()


class _Hybrid(torch.nn.Module):
    """A model type no tree family and no axolotl packing patch covers."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            model_type="my_hybrid", layer_types=["linear_attention", "full_attention"]
        )
        self.config.get_text_config = lambda: self.config


@pytest.mark.usefixtures("tree_hooks")
class TestPackingSupport:
    def test_attention_models_share(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        support = packing_support(LlamaForCausalLM(config))
        assert support.mode == "tree" and support.reason == "no recurrent layers"

    def test_gated_delta_nets_follow_trees_through_the_hooks(self):
        support = packing_support(_tiny_qwen3_5(64))
        assert support.mode == "tree"
        assert support.owners == {"Qwen3_5GatedDeltaNet": "tree_recurrence"}

    def test_recurrent_models_with_a_packing_patch_pack_chains(self):
        from transformers import MambaConfig, MambaForCausalLM

        config = MambaConfig(
            vocab_size=64, hidden_size=32, state_size=4, num_hidden_layers=1
        )
        assert packing_support(MambaForCausalLM(config)).mode == "chains"

    def test_other_recurrent_models_raise(self):
        with pytest.raises(ValueError, match="cannot keep its packed samples apart"):
            packing_support(_Hybrid())


@pytest.mark.usefixtures("tree_hooks")
def test_packed_hybrid_rows_match_each_sample_alone():
    """Trie and chain rows of a Qwen3.5 hybrid, following trees through
    tree-recurrence's hooks on its stock forward, score every sample as it scores
    alone; state leaking across samples or branches would show up as drift."""
    model = _tiny_qwen3_5(64)
    assert packing_support(model).shares_prefixes
    shared_len, chain_len = assert_tree_rows_match_alone(model, branching_samples())
    assert shared_len < chain_len


def _tiny_qwen3_5(vocab_size):
    """A tiny Qwen3.5 checkpoint as axolotl loads it: the multimodal class."""
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    config = Qwen3_5Config(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
        },
    )
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config)


@pytest.mark.usefixtures("tree_hooks")
@pytest.mark.parametrize("routes_trie", [True, False])
def test_trains_hybrid_model_on_cpu(
    download_smollm2_135m_instruct_model, tmp_path, monkeypatch, routes_trie
):
    """Rollouts share prompts when the GatedDeltaNets follow the trie, else each is
    its own chain; either way only the adapter trains."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import async_grpo_trainer

    from axolotl.core.trainers.grpo import async_trainer

    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    if not routes_trie:

        def chains(model):
            packing_support(model)
            return Support("chains", "test")

        monkeypatch.setattr(async_trainer, "packing_support", chains)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = _tiny_qwen3_5(len(tokenizer))
    layer_types = list(model.config.get_text_config().layer_types)
    model = get_peft_model(
        model,
        LoraConfig(r=4, target_modules=r".*language_model.*\.(q_proj|v_proj|out_proj)"),
    )
    trainable_before = {n: p.requires_grad for n, p in model.named_parameters()}

    prompt = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    rng = np.random.default_rng(0)
    samples = []
    for group_id in range(16):
        for g in range(4):
            completion = [*rng.integers(100, 1000, size=2 + g), tokenizer.eos_token_id]
            ids = np.asarray(prompt_ids + completion, dtype=np.int64)
            samples.append(
                RolloutSample(
                    prompt=prompt,
                    completion=[{"role": "assistant", "content": "x"}],
                    input_ids=ids,
                    completion_mask=np.asarray(
                        [0] * len(prompt_ids) + [1] * len(completion), dtype=np.int64
                    ),
                    old_log_probs=np.zeros(len(ids), np.float32),
                    **unsampled_fields(len(ids)),
                    advantage=float(g) - 1.5,
                    model_version=0,
                    group_id=group_id,
                    metrics={"reward": float(g), "reward_std": 1.0},
                )
            )

    args = AxolotlAsyncGRPOConfig(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        bf16=False,
        use_cpu=True,
        token_budget=0,
        report_to=[],
        save_strategy="no",
        gradient_checkpointing=False,
    )
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 16}),
        processing_class=tokenizer,
        rollout_worker=StubWorker(samples),
        weight_transfer=StubTransfer(),
    )
    assert trainer._share_prefixes is routes_trie
    text_config = trainer.model.get_base_model().config.get_text_config()
    assert text_config.layer_types == layer_types
    # TRL's VLM branch would have made the whole text model trainable.
    params = dict(trainer.model.named_parameters())
    assert {n: p.requires_grad for n, p in params.items()} == trainable_before
    assert all("lora_" in n for n, p in params.items() if p.requires_grad)

    trained = {n: p.detach().clone() for n, p in params.items() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in params.items() if not p.requires_grad}
    trainer.train()

    ratios = [
        log["batch/packing_ratio"]
        for log in trainer.state.log_history
        if "batch/packing_ratio" in log
    ]
    assert ratios
    if routes_trie:
        assert min(ratios) > 1
    else:
        assert set(ratios) == {1.0}
    params = dict(trainer.model.named_parameters())
    assert all(not torch.equal(trained[n], params[n].detach()) for n in trained)
    assert all(torch.equal(frozen[n], params[n].detach()) for n in frozen)
