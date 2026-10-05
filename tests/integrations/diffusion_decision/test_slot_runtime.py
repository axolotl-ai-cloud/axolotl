"""Learned decision slots wire only audited embedding rows into PEFT."""

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.diffusion_decision.args import DiffusionDecisionConfig
from axolotl.integrations.diffusion_decision.plugin import DiffusionDecisionPlugin
from axolotl.integrations.diffusion_decision.slot_runtime import (
    merge_trainable_slot_indices,
    resolve_trainable_slot_runtime,
)
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
    DiffusionSpec,
    EosHandling,
    FirstPositionAlignment,
    GenerationAdapter,
    LogitAlignment,
    MaskTokenPolicy,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
)
from axolotl.utils.dict import DictDefault


def _spec():
    return DiffusionSpec(
        noise=DiffusionNoise.ABSORBING,
        layout=DiffusionLayout.FULL_SEQUENCE,
        logit_alignment=LogitAlignment.ALIGNED,
        first_position_alignment=FirstPositionAlignment.REQUIRES_PREDECESSOR,
        self_conditioning=False,
        max_canvas=None,
        max_context=64,
        eos_handling=EosHandling.INDEPENDENT,
        mask_token_policy=MaskTokenPolicy.MODEL,
        default_time_weighting=TimeWeighting.INV_T,
        objective_reduction=ObjectiveReduction.MASKED_TOKEN_MEAN,
        generation_adapter=GenerationAdapter.FULL_SEQUENCE,
        reduction_scope=ReductionScope.GLOBAL_WINDOW,
    )


def _model():
    config = LlamaConfig(
        vocab_size=131072,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=11,
    )
    config.mask_token_id = 100
    return LlamaForCausalLM(config)


def _cfg(*, mode="learned", ids=(936, 937), existing=None):
    return DictDefault(
        {
            "model_config_type": "nemotron_labs_diffusion",
            "diffusion_lm": {"from_causal_lm": False},
            "diffusion_decision": {
                "latent": {"mode": mode, "num_slots": len(ids), "token_ids": ids}
            },
            "peft_trainable_token_indices": existing,
        }
    )


def _decision(cfg):
    return DiffusionDecisionConfig.model_validate(cfg.diffusion_decision)


def test_pre_lora_merges_learned_rows_before_real_peft_embedding_wrap():
    torch.manual_seed(0)
    cfg = _cfg(existing=[900])
    model = _model()
    base_weight = model.get_input_embeddings().weight
    outside_before = base_weight.detach().clone()

    DiffusionDecisionPlugin().pre_lora_load(cfg, model)

    assert cfg.peft_trainable_token_indices == [900, 936, 937]
    wrapped = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj"],
            r=2,
            lora_alpha=2,
            trainable_token_indices=cfg.peft_trainable_token_indices,
        ),
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in wrapped.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    loss = wrapped(
        input_ids=torch.tensor([[936, 937, 5]]),
        labels=torch.tensor([[-100, -100, 5]]),
    ).loss
    loss.backward()
    token_deltas = [
        parameter
        for name, parameter in wrapped.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    assert token_deltas
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in token_deltas
    )
    optimizer.step()
    assert base_weight.requires_grad is False
    assert torch.equal(base_weight.detach(), outside_before)


def test_prompt_rows_use_actual_embedding_path_and_preserve_mapping_indices():
    cfg = _cfg(mode="prompt", existing={"model.embed_tokens": [900]})
    model = _model()

    runtime = resolve_trainable_slot_runtime(cfg, model, _decision(cfg), spec=_spec())
    assert runtime is not None
    assert runtime.embedding_module_path == "model.embed_tokens"
    assert runtime.plan.trainable_token_ids == (936, 937)
    merge_trainable_slot_indices(cfg, runtime)

    assert cfg.peft_trainable_token_indices == {"model.embed_tokens": [900, 936, 937]}


def test_nonlearned_modes_do_not_change_peft_embedding_rows():
    cfg = _cfg(mode="pinned", ids=(936,), existing=[900])
    model = _model()

    DiffusionDecisionPlugin().pre_lora_load(cfg, model)

    assert cfg.peft_trainable_token_indices == [900]


def test_distinct_pinned_rows_do_not_change_peft_embedding_rows():
    cfg = _cfg(mode="pinned", ids=(936, 937), existing=[900])
    model = _model()

    DiffusionDecisionPlugin().pre_lora_load(cfg, model)

    assert cfg.peft_trainable_token_indices == [900]


@pytest.mark.parametrize("token_id", (1, 11, 100))
def test_active_model_control_ids_are_rejected_for_trainable_slots(token_id):
    cfg = _cfg(ids=(token_id, 936))

    with pytest.raises(ValueError, match="active pad, bos, eos, or mask"):
        resolve_trainable_slot_runtime(cfg, _model(), _decision(cfg), spec=_spec())


def test_mapping_without_actual_embedding_path_fails_explicitly():
    cfg = _cfg(existing={"other.embedding": [900]})

    runtime = resolve_trainable_slot_runtime(
        cfg, _model(), _decision(cfg), spec=_spec()
    )
    assert runtime is not None
    with pytest.raises(ValueError, match="actual input embedding path"):
        merge_trainable_slot_indices(cfg, runtime)


def test_runtime_uses_embedding_vocab_not_model_config_vocab():
    model = _model()
    model.config.vocab_size = 4
    cfg = _cfg(ids=(936, 937))

    runtime = resolve_trainable_slot_runtime(cfg, model, _decision(cfg), spec=_spec())

    assert runtime is not None
    assert runtime.plan.ids == (936, 937)


def test_multiple_eos_ids_are_controls_without_requiring_a_pad_token():
    model = _model()
    model.config.eos_token_id = [1, 106]

    runtime = resolve_trainable_slot_runtime(
        _cfg(ids=(936, 937)), model, _decision(_cfg(ids=(936, 937))), spec=_spec()
    )

    assert runtime is not None
    assert runtime.plan.ids == (936, 937)
    with pytest.raises(ValueError, match="active pad, bos, eos, or mask"):
        cfg = _cfg(ids=(106, 936))
        resolve_trainable_slot_runtime(cfg, model, _decision(cfg), spec=_spec())
