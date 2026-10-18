"""DPO/APO and KTO score tokens through TRL's fused LM head, with loss masks and MoE aux loss intact."""

import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
from trl.trainer.utils import FusedCausalLMOutput

from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer

from tests.core.trainers.test_dpo_loss_mask import _rows, _trainer
from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="tokenizer", scope="module")
@enable_hf_offline
def fixture_tokenizer(download_qwen3_half_billion_model):
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


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
    rows = _rows(tokenizer)
    trainer = _trainer(tokenizer, tmp_path, rows, loss_type)
    batch = trainer.data_collator(rows)

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


def test_loss_mask_reaches_fused_head_labels(tokenizer, tmp_path):
    """Tool results and other masked tokens are -100 in the head's labels, so they are never scored."""
    rows = _rows(tokenizer)
    trainer = _trainer(tokenizer, tmp_path, rows, ["sigmoid"])
    batch = trainer.data_collator(rows)
    assert (batch["completion_mask"] < batch["attention_mask"]).any()

    calls = _record_forward(trainer.model)
    with torch.no_grad():
        trainer.compute_loss(trainer.model, batch)

    kwargs, out = calls[0]
    expected = batch["completion_mask"].bool()
    assert torch.equal(kwargs["labels"] != -100, expected)
    assert torch.equal(out.label_mask, expected[:, 1:])
    assert (out.log_probs[~out.label_mask] == 0).all()


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


def _moe_trainer(tokenizer, tmp_path, rows, coef):
    args = AxolotlDPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=1,
        loss_type=["apo_zero"],
        beta=0.5,
        max_length=1024,
        report_to="none",
        save_strategy="no",
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
    )
    return AxolotlDPOTrainer(
        model=_tiny_moe(tokenizer, seed=0, coef=coef),
        ref_model=_tiny_moe(tokenizer, seed=1, coef=coef),
        args=args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
    )


def test_moe_aux_loss_flows_through_fused_head(tokenizer, tmp_path):
    rows = _rows(tokenizer)
    coef = 0.5
    with_aux = _moe_trainer(tokenizer, tmp_path, rows, coef)
    without_aux = _moe_trainer(tokenizer, tmp_path, rows, 0.0)
    assert with_aux.aux_loss_enabled and not without_aux.aux_loss_enabled

    batch = with_aux.data_collator(rows)
    with torch.no_grad():
        loss = with_aux.compute_loss(with_aux.model, batch).item()
        base = without_aux.compute_loss(without_aux.model, batch).item()
    aux = with_aux._metrics["train"]["aux_loss"]
    assert len(aux) == 1 and aux[0] > 0
    assert loss == pytest.approx(base + coef * aux[0], rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton)")
def test_fused_head_peak_memory_below_full_logits(tokenizer, tmp_path):
    rows = _rows(tokenizer)
    trainer = _trainer(tokenizer, tmp_path, rows, ["sigmoid"])
    trainer.model.cuda()
    trainer.ref_model.cuda()
    batch = {k: v.cuda() for k, v in trainer.data_collator(rows).items()}
    full_logits_bytes = batch["input_ids"].numel() * len(tokenizer) * 4

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    trainer.compute_loss(trainer.model, batch).backward()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - start < full_logits_bytes


def test_kto_uses_fused_head(tokenizer, tmp_path):
    from axolotl.core.trainers import AxolotlKTOTrainer
    from axolotl.core.training_args import AxolotlKTOConfig

    from tests.core.trainers.test_dpo_loss_mask import _tiny_model

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
