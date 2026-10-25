"""Async GRPO packing for models with recurrent layers."""

import types
from collections import defaultdict

import numpy as np
import pytest
import torch
from trl.experimental.async_grpo.async_grpo_trainer import (
    FixedCountBatcher,
    TokenBudgetBatcher,
)
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample
from trl.experimental.async_grpo.packing import TreeRow
from trl_async_test_utils import StubTransfer, StubWorker, reference_tree_attention

from axolotl.core.trainers.grpo.trl_async_packing import (
    ChainFixedCountBatcher,
    ChainRow,
    ChainTokenBudgetBatcher,
    shares_prefixes,
)
from axolotl.core.trainers.grpo.trl_async_scoring import pack_sequences
from axolotl.core.trainers.grpo.trl_async_trainer import (
    AxolotlAsyncGRPOConfig,
    AxolotlAsyncGRPOTrainer,
)


def _sample(ids, prompt_len, group_id=0):
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "input_ids": ids,
        "completion_mask": np.asarray(
            [0] * prompt_len + [1] * (len(ids) - prompt_len), dtype=np.int64
        ),
        "old_log_probs": np.zeros(len(ids), np.float32),
        "advantage": 1.0,
        "group_id": group_id,
        "metrics": {},
    }


# Two rollouts of one prompt, which a TreeRow would share, and another prompt.
SAMPLES = [
    _sample([1, 2, 3, 4, 5], 3),
    _sample([1, 2, 3, 6], 3),
    _sample([7, 8, 9], 1, group_id=1),
]


class TestChainRow:
    def test_packs_each_sample_as_its_own_chain(self):
        tree, chain = TreeRow(), ChainRow()
        for sample in SAMPLES:
            tree.add(sample)
            chain.add(sample)
        assert len(tree.tokens) == 10 and len(chain.tokens) == 12

        packed = chain.pack()
        assert packed["input_ids"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 8, 9]
        assert packed["position_ids"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2]
        # Tree attention degenerates to varlen attention over the samples.
        assert packed["tree_kv_index"].tolist() == list(range(12))
        assert packed["tree_cu_seqlens_q"].tolist() == [0, 5, 9, 12]
        assert packed["tree_cu_seqlens_k"].tolist() == [0, 5, 9, 12]
        assert packed["tree_max_seqlens"].tolist() == [5, 5]

        # The same loss terms as the trie, scored off unshared positions.
        shared = tree.pack()
        for key in ("old_log_probs", "advantages", "segment_id"):
            assert torch.equal(packed[key], shared[key])
        assert torch.equal(
            packed["shift_labels"][packed["token_index"]],
            shared["shift_labels"][shared["token_index"]],
        )

    def test_budget_counts_every_token(self):
        tree, chain = TreeRow(), ChainRow()
        tree.add(SAMPLES[0])
        chain.add(SAMPLES[0])
        # The trie forks one token early, so [1, 2, 3, 6] shares [1, 2] with the row.
        assert tree.fits(SAMPLES[1], budget=7)
        assert not chain.fits(SAMPLES[1], budget=8)
        assert chain.fits(SAMPLES[1], budget=9)

    def test_pack_sequences_without_sharing(self):
        sequences = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])]
        shared = pack_sequences(sequences)
        chains = pack_sequences(sequences, share_prefixes=False)
        assert len(shared["input_ids"]) == 6
        assert chains["input_ids"].tolist() == [1, 2, 3, 4, 1, 2, 3, 5]
        assert chains["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        assert chains["tree_cu_seqlens_q"].tolist() == [0, 4, 8]
        assert chains["positions"].tolist() == list(range(8))


def _rows(batcher):
    return [[[s["group_id"] for s in row.samples] for row in rows] for rows in batcher]


class TestChainBatchers:
    # Distinct first tokens: no prefix to share, so TRL's batchers place them the same.
    SAMPLES = [
        _sample([10 * i + 1, *range(2, 3 + i % 5)], 1, group_id=i // 2)
        for i in range(12)
    ]

    def test_token_budget_batcher_places_like_trl(self):
        for budget in (6, 9, 20):
            ours = ChainTokenBudgetBatcher(self.SAMPLES, 2, budget, defaultdict(list))
            trl = TokenBudgetBatcher(self.SAMPLES, 2, budget, defaultdict(list))
            assert _rows(ours) == _rows(trl)
            assert all(isinstance(row, ChainRow) for rows in ours for row in rows)

    def test_fixed_count_batcher_places_like_trl(self):
        ours = ChainFixedCountBatcher(self.SAMPLES, 2, 4)
        assert _rows(ours) == _rows(FixedCountBatcher(self.SAMPLES, 2, 4))
        assert all(isinstance(row, ChainRow) for rows in ours for row in rows)


def _config(model_type, layer_types=None):
    config = types.SimpleNamespace(model_type=model_type, layer_types=layer_types)
    config.get_text_config = lambda: config
    return config


class TestSharesPrefixes:
    def test_attention_models_share(self):
        assert shares_prefixes(_config("llama"))
        assert shares_prefixes(
            _config("gemma3", ["sliding_attention", "full_attention"])
        )

    def test_patched_recurrent_models_do_not_share(self):
        assert not shares_prefixes(
            _config("qwen3_5_text", ["linear_attention", "full_attention"])
        )
        assert not shares_prefixes(_config("mamba2"))

    @pytest.mark.parametrize(
        "config",
        [
            _config("jamba"),
            _config("my_hybrid", ["linear_attention", "full_attention"]),
        ],
    )
    def test_unsupported_recurrent_models_raise(self, config):
        with pytest.raises(ValueError, match="state would leak"):
            shares_prefixes(config)


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


def test_trains_hybrid_model_on_unshared_rows_on_cpu(
    download_smollm2_135m_instruct_model, tmp_path, monkeypatch
):
    """Each rollout is its own chain, and its boundaries reach the GatedDeltaNet kernels."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from trl.experimental.async_grpo import async_grpo_trainer, packing

    from axolotl.monkeypatch.models.qwen3_5 import modeling as qwen3_5_packing

    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    for cls in (
        modeling_qwen3_5.Qwen3_5DecoderLayer,
        modeling_qwen3_5.Qwen3_5GatedDeltaNet,
    ):
        monkeypatch.setattr(cls, "forward", cls.forward)
    qwen3_5_packing.patch_qwen3_5_modeling_packing()
    monkeypatch.setattr(qwen3_5_packing, "_FLA_COMPILED_OPS", False)

    # FLA's varlen kernels need a GPU: record the boundaries they would get.
    boundaries = []

    def causal_conv1d(x, weight, bias, activation, cu_seqlens):
        boundaries.append(cu_seqlens.tolist())
        return torch.nn.functional.silu(x), None

    def chunk_gated_delta_rule(query, key, value, *, beta, cu_seqlens, **kwargs):
        boundaries.append(cu_seqlens.tolist())
        return value * beta[..., None], None

    monkeypatch.setattr(qwen3_5_packing, "fla_causal_conv1d", causal_conv1d)
    monkeypatch.setattr(
        qwen3_5_packing, "fla_chunk_gated_delta_rule", chunk_gated_delta_rule
    )

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = _tiny_qwen3_5(len(tokenizer))
    layer_types = list(model.config.get_text_config().layer_types)
    model = get_peft_model(
        model,
        LoraConfig(r=4, target_modules=r".*language_model.*\.(q_proj|v_proj|out_proj)"),
    )

    prompt = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    rng = np.random.default_rng(0)
    samples, lengths = [], set()
    for group_id in range(8):
        for g in range(4):
            completion = [*rng.integers(100, 1000, size=2 + g), tokenizer.eos_token_id]
            ids = np.asarray(prompt_ids + completion, dtype=np.int64)
            lengths.add(len(ids))
            samples.append(
                RolloutSample(
                    prompt=prompt,
                    completion=[{"role": "assistant", "content": "x"}],
                    input_ids=ids,
                    completion_mask=np.asarray(
                        [0] * len(prompt_ids) + [1] * len(completion), dtype=np.int64
                    ),
                    old_log_probs=np.zeros(len(ids), np.float32),
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
        token_budget=4 * max(lengths),
        report_to=[],
        save_strategy="no",
        gradient_checkpointing=False,
    )
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 8}),
        processing_class=tokenizer,
        rollout_worker=StubWorker(samples),
        weight_transfer=StubTransfer(),
    )
    assert not trainer._share_prefixes
    text_config = trainer.model.get_base_model().config.get_text_config()
    assert text_config.layer_types == layer_types
    # TRL's VLM branch would have made the whole text model trainable.
    params = dict(trainer.model.named_parameters())
    assert all("lora_" in n for n, p in params.items() if p.requires_grad)

    trained = {n: p.detach().clone() for n, p in params.items() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in params.items() if not p.requires_grad}
    trainer.train()

    assert boundaries
    for cu_seqlens in boundaries:
        # One segment per rollout; a shared prompt would show up as a shorter one.
        segments = np.diff(cu_seqlens).tolist()
        assert segments and set(segments) <= lengths
    params = dict(trainer.model.named_parameters())
    assert all(not torch.equal(trained[n], params[n].detach()) for n in trained)
    assert all(torch.equal(frozen[n], params[n].detach()) for n in frozen)
