"""DPO/APO and KTO score tokens through TRL's fused LM head, with the MoE aux loss intact."""

import pytest
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)
from trl.trainer.utils import FusedCausalLMOutput

from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer

from tests.hf_offline_utils import enable_hf_offline

PAIRS = [
    {"prompt": "Find X.", "chosen": " X is here.", "rejected": " No idea."},
    {"prompt": "Find Y.", "chosen": " Y is over there.", "rejected": " Unknown."},
]


@pytest.fixture(name="tokenizer", scope="module")
@enable_hf_offline
def fixture_tokenizer(download_qwen3_half_billion_model):
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def _tiny_model(tokenizer, seed):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id,
    )
    return Qwen3ForCausalLM(config).float()


def _dpo_args(tmp_path, loss_type):
    return AxolotlDPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        loss_type=loss_type,
        beta=0.5,
        max_length=1024,
        report_to="none",
        save_strategy="no",
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
    )


def _trainer(tokenizer, tmp_path, loss_type, model=_tiny_model):
    return AxolotlDPOTrainer(
        model=model(tokenizer, seed=0),
        ref_model=model(tokenizer, seed=1),
        args=_dpo_args(tmp_path, loss_type),
        train_dataset=Dataset.from_list(PAIRS),
        processing_class=tokenizer,
    )


def _batch(trainer):
    rows = [trainer.train_dataset[i] for i in range(len(trainer.train_dataset))]
    return trainer.data_collator(rows)


def _record_forward(model):
    """Wrap the patched instance forward and record what reaches the fused head."""
    calls = []
    fused_forward = model.forward

    def recording_forward(*args, **kwargs):
        out = fused_forward(*args, **kwargs)
        calls.append((kwargs, out))
        return out

    model.forward = recording_forward
    return calls


def _forbid_lm_head_module(model):
    """Fail if anything projects the hidden states through the `lm_head` module (full logits)."""

    def no_lm_head(*_args, **_kwargs):
        raise AssertionError("lm_head module forward ran: full logits materialized")

    return model.get_output_embeddings().register_forward_hook(no_lm_head)


@pytest.mark.parametrize("loss_type", [["sigmoid"], ["apo_zero"], ["apo_down"]])
def test_policy_and_ref_use_fused_head(tokenizer, tmp_path, loss_type):
    trainer = _trainer(tokenizer, tmp_path, loss_type)
    batch = _batch(trainer)

    policy_calls = _record_forward(trainer.model)
    ref_calls = _record_forward(trainer.ref_model)
    for model in (trainer.model, trainer.ref_model):
        _forbid_lm_head_module(model)
    with pytest.raises(AssertionError, match="full logits"):
        trainer.model(input_ids=batch["input_ids"][:1])

    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss)

    for calls in (policy_calls, ref_calls):
        assert len(calls) == 1
        kwargs, out = calls[0]
        assert kwargs["fused_lm_head"] is True
        assert isinstance(out, FusedCausalLMOutput)
        assert not hasattr(out, "logits")
        assert out.log_probs.shape == batch["input_ids"][:, 1:].shape


def _tiny_moe(tokenizer, seed, coef):
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=coef,
        max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id,
    )
    return Qwen3MoeForCausalLM(config).float()


def test_moe_aux_loss_flows_through_fused_head(tokenizer, tmp_path):
    coef = 0.5

    def moe_trainer(coef):
        return _trainer(
            tokenizer,
            tmp_path,
            ["apo_zero"],
            model=lambda tok, seed: _tiny_moe(tok, seed, coef),
        )

    with_aux = moe_trainer(coef)
    without_aux = moe_trainer(0.0)
    assert with_aux.aux_loss_enabled and not without_aux.aux_loss_enabled

    batch = _batch(with_aux)
    with torch.no_grad():
        loss = with_aux.compute_loss(with_aux.model, batch).item()
        base = without_aux.compute_loss(without_aux.model, batch).item()
    aux = with_aux._metrics["train"]["aux_loss"]
    assert len(aux) == 1 and aux[0] > 0
    assert loss == pytest.approx(base + coef * aux[0], rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
def test_fused_head_peak_memory_below_full_logits(tokenizer, tmp_path):
    """A fused DPO step, reference model included, peaks below the policy alone
    scoring the same batch through full logits. Both pay the vocabulary-sized
    weight gradients, so the difference is the logits."""
    trainer = _trainer(tokenizer, tmp_path, ["sigmoid"])
    model = trainer.model.cuda()
    trainer.ref_model.cuda()
    batch = {k: v.cuda() for k, v in _batch(trainer).items()}

    def full_logits_loss():
        logits = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).logits
        targets = batch["input_ids"][:, 1:, None]
        return -logits[:, :-1].log_softmax(-1).gather(-1, targets).mean()

    def peak(loss_fn):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        loss_fn().backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - start

    fused = peak(lambda: trainer.compute_loss(model, batch))
    assert fused < peak(full_logits_loss)


def test_kto_uses_fused_head(tokenizer, tmp_path):
    from axolotl.core.trainers import AxolotlKTOTrainer
    from axolotl.core.training_args import AxolotlKTOConfig

    rows = [
        {"prompt": "Find X.", "completion": " X is here.", "label": True},
        {"prompt": "Find Y.", "completion": " No idea.", "label": False},
    ] * 2
    args = AxolotlKTOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        max_steps=1,
        max_length=128,
        report_to="none",
        save_strategy="no",
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )
    trainer = AxolotlKTOTrainer(
        model=_tiny_model(tokenizer, seed=0),
        ref_model=_tiny_model(tokenizer, seed=1),
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )
    calls = _record_forward(trainer.model)
    _forbid_lm_head_module(trainer.model)
    _forbid_lm_head_module(trainer.ref_model)

    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(4)])
    loss = trainer.compute_loss(trainer.model, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert calls and all(kwargs.get("fused_lm_head") for kwargs, _ in calls)
