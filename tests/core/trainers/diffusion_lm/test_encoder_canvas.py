"""Native DiffusionGemma coverage for packed encoder/canvas execution."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import DiffusionGemmaConfig, DiffusionGemmaForBlockDiffusion
from transformers.cache_utils import DynamicCache

from axolotl.core.trainers.diffusion_lm.attention import (
    decoder_prefix_canvas_mask,
    encoder_causal_mask,
)
from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.core.trainers.diffusion_lm.collator import DiffusionCollator
from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer
from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.model_support.diffusion_gemma.modeling import (
    AxolotlDiffusionGemmaForBlockDiffusion,
    decode_packed_canvas,
    encode_packed_prefix,
    forward_packed_encoder_canvas,
)
from axolotl.utils.dict import DictDefault


def _tiny_config(*, flex: bool = False) -> DiffusionGemmaConfig:
    hidden_size = 64 if flex else 32
    head_dim = 16 if flex else 8
    text = {
        "vocab_size": 32,
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size * 2,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": head_dim,
        "max_position_embeddings": 64,
        "layer_types": ["sliding_attention", "full_attention"],
        "per_layer_config": {
            "0": {"head_dim": head_dim},
            "1": {"head_dim": head_dim},
        },
        "sliding_window": 2,
        "use_bidirectional_attention": "vision",
        "num_experts": 2,
        "top_k_experts": 1,
        "moe_intermediate_size": hidden_size,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
    }
    vision = {
        "model_type": "gemma4_vision",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "patch_size": 16,
        "position_embedding_size": 16,
    }
    return DiffusionGemmaConfig(
        text_config=text,
        vision_config=vision,
        canvas_length=8,
        boi_token_id=31,
        eoi_token_id=30,
        image_token_id=29,
    )


@pytest.fixture
def native_model():
    torch.manual_seed(7)
    return DiffusionGemmaForBlockDiffusion(_tiny_config()).eval()


def _masks():
    prompt_lengths, canvas_lengths, ids = (4, 3), (3, 3), torch.tensor([17, 91])
    encoder_docs = torch.cat(
        [torch.full((n,), doc) for n, doc in zip(prompt_lengths, ids, strict=True)]
    )[None]
    canvas_docs = torch.cat(
        [torch.full((n,), doc) for n, doc in zip(canvas_lengths, ids, strict=True)]
    )[None]
    encoder_pos = torch.cat([torch.arange(n) for n in prompt_lengths])[None]
    canvas_pos = torch.cat(
        [
            torch.arange(p, p + n)
            for p, n in zip(prompt_lengths, canvas_lengths, strict=True)
        ]
    )[None]
    encoder_valid, canvas_valid = (
        torch.ones_like(encoder_docs, dtype=torch.bool),
        torch.ones_like(canvas_docs, dtype=torch.bool),
    )
    full_encoder = encoder_causal_mask(encoder_docs, encoder_valid, encoder_pos)
    full_decoder = decoder_prefix_canvas_mask(
        encoder_docs,
        canvas_docs,
        encoder_valid,
        canvas_valid,
        torch.tensor(prompt_lengths),
        encoder_pos,
        logical_ids=ids,
    )
    return (
        encoder_pos,
        canvas_pos,
        {
            "full_attention": full_encoder,
            "sliding_attention": encoder_causal_mask(
                encoder_docs, encoder_valid, encoder_pos, sliding_window=2
            ),
        },
        {
            "full_attention": full_decoder,
            "sliding_attention": decoder_prefix_canvas_mask(
                encoder_docs,
                canvas_docs,
                encoder_valid,
                canvas_valid,
                torch.tensor(prompt_lengths),
                encoder_pos,
                sliding_window=2,
                logical_ids=ids,
            ),
        },
    )


def _single_masks(prompt_length: int, canvas_length: int):
    docs, canvas_docs = (
        torch.zeros((1, prompt_length), dtype=torch.long),
        torch.zeros((1, canvas_length), dtype=torch.long),
    )
    prompt_pos, canvas_pos = (
        torch.arange(prompt_length)[None],
        torch.arange(prompt_length, prompt_length + canvas_length)[None],
    )
    prompt_valid, canvas_valid = (
        torch.ones_like(docs, dtype=torch.bool),
        torch.ones_like(canvas_docs, dtype=torch.bool),
    )
    return (
        prompt_pos,
        canvas_pos,
        {
            "full_attention": encoder_causal_mask(docs, prompt_valid, prompt_pos),
            "sliding_attention": encoder_causal_mask(
                docs, prompt_valid, prompt_pos, sliding_window=2
            ),
        },
        {
            "full_attention": decoder_prefix_canvas_mask(
                docs,
                canvas_docs,
                prompt_valid,
                canvas_valid,
                torch.tensor([prompt_length]),
                prompt_pos,
            ),
            "sliding_attention": decoder_prefix_canvas_mask(
                docs,
                canvas_docs,
                prompt_valid,
                canvas_valid,
                torch.tensor([prompt_length]),
                prompt_pos,
                sliding_window=2,
            ),
        },
    )


def _packed_inputs():
    return torch.tensor([[2, 3, 4, 5, 2, 6, 7]]), torch.tensor([[8, 9, 10, 11, 12, 13]])


def test_native_decoder_all_off_equals_native_decoder(native_model):
    prompt, canvas = _packed_inputs()
    encoder_pos, canvas_pos, encoder_mask, decoder_mask = _masks()
    sc_logits = torch.randn(1, canvas.shape[1], 32)
    with torch.no_grad():
        packed = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=sc_logits,
            self_conditioning_token_mask=torch.zeros_like(canvas, dtype=torch.bool),
        ).logits
        native = native_model(
            input_ids=prompt,
            attention_mask=encoder_mask,
            position_ids=encoder_pos,
            past_key_values=DynamicCache(),
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
        ).logits
    torch.testing.assert_close(packed, native, rtol=1e-5, atol=1e-6)
    with torch.no_grad():
        all_on = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=sc_logits,
            self_conditioning_token_mask=torch.ones_like(canvas, dtype=torch.bool),
        ).logits
        native_all_on = native_model(
            input_ids=prompt,
            attention_mask=encoder_mask,
            position_ids=encoder_pos,
            past_key_values=DynamicCache(),
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=sc_logits,
            self_conditioning_mask=torch.tensor([True]),
        ).logits
    torch.testing.assert_close(all_on, native_all_on, rtol=1e-5, atol=1e-6)


def test_native_packed_execution_keeps_peft_lora_active(native_model):
    peft_model = get_peft_model(
        native_model,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=(
                r"^(model\.encoder\.language_model\.layers|model\.decoder\.layers)"
                r"\.0\.self_attn\.q_proj$"
            ),
            task_type=None,
        ),
    ).train()
    prompt, canvas = _packed_inputs()
    encoder_pos, canvas_pos, encoder_mask, decoder_mask = _masks()
    output = forward_packed_encoder_canvas(
        peft_model,
        encoder_input_ids=prompt,
        encoder_attention_mask=encoder_mask,
        encoder_position_ids=encoder_pos,
        decoder_input_ids=canvas,
        decoder_attention_mask=decoder_mask,
        decoder_position_ids=canvas_pos,
    ).logits
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in peft_model.named_parameters()
        if "lora_" in name
    )


def test_native_physical_b1_matches_two_unpacked_documents_in_logits_and_gradients(
    native_model,
):
    packed_model, unpacked_model = (
        copy.deepcopy(native_model).train(),
        copy.deepcopy(native_model).train(),
    )
    prompt, canvas = _packed_inputs()
    encoder_pos, canvas_pos, encoder_mask, decoder_mask = _masks()
    packed_logits = forward_packed_encoder_canvas(
        packed_model,
        encoder_input_ids=prompt,
        encoder_attention_mask=encoder_mask,
        encoder_position_ids=encoder_pos,
        decoder_input_ids=canvas,
        decoder_attention_mask=decoder_mask,
        decoder_position_ids=canvas_pos,
    ).logits
    packed_logits.square().sum().backward()
    expected, loss = [], 0.0
    for prompt_ids, canvas_ids in (
        (prompt[:, :4], canvas[:, :3]),
        (prompt[:, 4:], canvas[:, 3:]),
    ):
        prompt_pos, one_canvas_pos, one_encoder_mask, one_decoder_mask = _single_masks(
            prompt_ids.shape[1], canvas_ids.shape[1]
        )
        logits = unpacked_model(
            input_ids=prompt_ids,
            attention_mask=one_encoder_mask,
            position_ids=prompt_pos,
            past_key_values=DynamicCache(),
            decoder_input_ids=canvas_ids,
            decoder_attention_mask=one_decoder_mask,
            decoder_position_ids=one_canvas_pos,
        ).logits
        expected.append(logits)
        loss = loss + logits.square().sum()
    loss.backward()
    torch.testing.assert_close(
        packed_logits, torch.cat(expected, dim=1), rtol=2e-5, atol=2e-6
    )
    for (name, packed_parameter), (_, unpacked_parameter) in zip(
        packed_model.named_parameters(), unpacked_model.named_parameters(), strict=True
    ):
        if packed_parameter.grad is None or unpacked_parameter.grad is None:
            assert packed_parameter.grad is unpacked_parameter.grad is None, name
        else:
            torch.testing.assert_close(
                packed_parameter.grad, unpacked_parameter.grad, rtol=3e-5, atol=3e-6
            )


def test_native_self_conditioning_is_independent_per_packed_document(native_model):
    prompt, canvas = _packed_inputs()
    encoder_pos, canvas_pos, encoder_mask, decoder_mask = _masks()
    logits, gate = (
        torch.randn(1, canvas.shape[1], 32),
        torch.tensor([[True, True, True, False, False, False]]),
    )
    with torch.no_grad():
        mixed = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=logits,
            self_conditioning_token_mask=gate,
        ).logits
        disabled_perturbed = logits.clone()
        disabled_perturbed[:, 3:] += 10 * torch.randn_like(disabled_perturbed[:, 3:])
        still_mixed = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=disabled_perturbed,
            self_conditioning_token_mask=gate,
        ).logits
        enabled = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=logits,
            self_conditioning_token_mask=torch.ones_like(gate),
        ).logits
        enabled_perturbed = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
            self_conditioning_logits=disabled_perturbed,
            self_conditioning_token_mask=torch.ones_like(gate),
        ).logits
    torch.testing.assert_close(mixed, still_mixed, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        enabled[:, :3], enabled_perturbed[:, :3], rtol=2e-5, atol=2e-6
    )
    assert (enabled[:, 3:] - enabled_perturbed[:, 3:]).abs().max() > 1e-5


def test_native_encoder_cache_is_retained_and_reused_after_no_grad_pilot(native_model):
    prompt, canvas = _packed_inputs()
    encoder_pos, canvas_pos, encoder_mask, decoder_mask = _masks()
    encoded = encode_packed_prefix(native_model, prompt, encoder_mask, encoder_pos)
    assert encoded.past_key_values.get_seq_length(layer_idx=0) == prompt.shape[1]
    cached_keys = encoded.past_key_values.layers[0].keys
    cached_keys.retain_grad()
    with torch.no_grad():
        pilot = decode_packed_canvas(
            native_model, canvas, encoded.past_key_values, decoder_mask, canvas_pos
        )
    decode_packed_canvas(
        native_model,
        canvas,
        encoded.past_key_values,
        decoder_mask,
        canvas_pos,
        pilot.detach(),
        torch.ones_like(canvas, dtype=torch.bool),
    ).square().mean().backward()
    assert cached_keys.grad is not None
    assert cached_keys.grad.abs().max() > 0
    assert (
        native_model.model.decoder.self_conditioning.down_proj.weight.grad is not None
    )


def test_native_selected_block_does_not_read_future_clean_encoder_tokens(native_model):
    prompt = torch.tensor([[2, 3, 10, 11, 12, 13, 14, 15, 16]])
    canvas = torch.tensor([[13, 14, 15]])
    docs = torch.zeros_like(prompt)
    canvas_docs = torch.zeros_like(canvas)
    prompt_pos = torch.arange(prompt.shape[1])[None]
    canvas_pos = torch.tensor([[5, 6, 7]])
    prompt_valid = torch.ones_like(prompt, dtype=torch.bool)
    canvas_valid = torch.ones_like(canvas, dtype=torch.bool)
    encoder_mask = {
        "full_attention": encoder_causal_mask(docs, prompt_valid, prompt_pos),
        "sliding_attention": encoder_causal_mask(
            docs, prompt_valid, prompt_pos, sliding_window=2
        ),
    }
    decoder_mask = {
        "full_attention": decoder_prefix_canvas_mask(
            docs,
            canvas_docs,
            prompt_valid,
            canvas_valid,
            torch.tensor([5]),
            prompt_pos,
        ),
        "sliding_attention": decoder_prefix_canvas_mask(
            docs,
            canvas_docs,
            prompt_valid,
            canvas_valid,
            torch.tensor([5]),
            prompt_pos,
            sliding_window=2,
        ),
    }
    future_changed = prompt.clone()
    future_changed[:, 5:] = torch.tensor([[21, 22, 23, 24]])
    with torch.no_grad():
        reference = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=prompt_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
        ).logits
        changed = forward_packed_encoder_canvas(
            native_model,
            encoder_input_ids=future_changed,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=prompt_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
        ).logits
    torch.testing.assert_close(reference, changed, rtol=1e-5, atol=1e-6)


def test_native_eager_dense_fallback_converts_boolean_masks(native_model):
    prompt, canvas = _packed_inputs()
    encoder_pos, canvas_pos, encoder_mask, decoder_mask = _masks()
    sdpa_model = copy.deepcopy(native_model)
    eager_model = copy.deepcopy(native_model)
    _set_attention_backend(sdpa_model, "sdpa")
    _set_attention_backend(eager_model, "eager")
    with torch.no_grad():
        sdpa_logits = forward_packed_encoder_canvas(
            sdpa_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
        ).logits
        eager_logits = forward_packed_encoder_canvas(
            eager_model,
            encoder_input_ids=prompt,
            encoder_attention_mask=encoder_mask,
            encoder_position_ids=encoder_pos,
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_pos,
        ).logits
    torch.testing.assert_close(eager_logits, sdpa_logits, rtol=2e-5, atol=2e-6)


def _set_attention_backend(model, backend: str) -> None:
    for config in (
        model.config,
        model.model.encoder.config,
        model.model.encoder.language_model.config,
        model.model.decoder.config,
        model.model.decoder.text_config,
    ):
        config._attn_implementation = backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dense_backend", ["sdpa", "eager"])
def test_native_flex_packed_logits_loss_and_gradients_match_dense(dense_backend):
    device = torch.device("cuda:0")
    batch = DiffusionCollator(0, 3).build_batch(
        [
            {
                "input_ids": [2, 3, 4, 5, 8, 9, 10],
                "labels": [-100] * 4 + [8, 9, 10],
            },
            {
                "input_ids": [2, 6, 7, 11, 12, 13],
                "labels": [-100] * 3 + [11, 12, 13],
            },
        ]
    )
    logical_ids = torch.tensor([17, 91])
    batch = replace(
        batch,
        logical_ids=logical_ids,
        encoder_document_ids=torch.where(
            batch.encoder_validity,
            logical_ids[:, None].expand_as(batch.encoder_document_ids),
            batch.encoder_document_ids,
        ),
    ).to(device)
    dense = EncoderCanvasBackend(32, 2).pack(batch)
    flex = EncoderCanvasBackend(32, 2, attention_backend="flex_attention").pack(batch)
    assert not isinstance(flex.encoder_attention_mask["full_attention"], torch.Tensor)
    torch.manual_seed(7)
    reference_model = DiffusionGemmaForBlockDiffusion(_tiny_config(flex=True)).eval()
    dense_model = copy.deepcopy(reference_model).to(device).train()
    flex_model = copy.deepcopy(reference_model).to(device).train()
    _set_attention_backend(dense_model, dense_backend)
    _set_attention_backend(flex_model, "flex_attention")
    dense_logits = forward_packed_encoder_canvas(
        dense_model,
        encoder_input_ids=dense.encoder_input_ids,
        encoder_attention_mask=dense.encoder_attention_mask,
        encoder_position_ids=dense.encoder_position_ids,
        decoder_input_ids=dense.canvas_clean_ids,
        decoder_attention_mask=dense.decoder_attention_mask,
        decoder_position_ids=dense.canvas_position_ids,
    ).logits
    flex_logits = forward_packed_encoder_canvas(
        flex_model,
        encoder_input_ids=flex.encoder_input_ids,
        encoder_attention_mask=flex.encoder_attention_mask,
        encoder_position_ids=flex.encoder_position_ids,
        decoder_input_ids=flex.canvas_clean_ids,
        decoder_attention_mask=flex.decoder_attention_mask,
        decoder_position_ids=flex.canvas_position_ids,
    ).logits
    canvas_length = dense_logits.shape[1]
    flex_logits = flex_logits[:, :canvas_length]
    dense_loss, flex_loss = dense_logits.square().mean(), flex_logits.square().mean()
    dense_loss.backward()
    flex_loss.backward()
    torch.testing.assert_close(flex_logits, dense_logits, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(flex_loss, dense_loss, rtol=3e-4, atol=3e-5)
    for (name, dense_parameter), (_, flex_parameter) in zip(
        dense_model.named_parameters(), flex_model.named_parameters(), strict=True
    ):
        if dense_parameter.grad is None or flex_parameter.grad is None:
            assert dense_parameter.grad is flex_parameter.grad is None, name
        else:
            torch.testing.assert_close(
                flex_parameter.grad, dense_parameter.grad, rtol=5e-4, atol=5e-5
            )
    varied_batch = (
        DiffusionCollator(0, 3)
        .build_batch(
            [
                {"input_ids": [2, 3, 4, 5], "labels": [-100, 3, 4, 5]},
                {"input_ids": [2, 6, 7, 8, 9, 10], "labels": [-100, -100, 7, 8, 9, 10]},
            ]
        )
        .to(device)
    )
    varied = EncoderCanvasBackend(32, 2, attention_backend="flex_attention").pack(
        varied_batch
    )
    assert varied.encoder_input_ids.shape == flex.encoder_input_ids.shape == (1, 128)
    assert varied.canvas_clean_ids.shape == flex.canvas_clean_ids.shape == (1, 128)
    counters = torch._dynamo.utils.counters
    before = counters["stats"].get("unique_graphs", 0)
    forward_packed_encoder_canvas(
        flex_model,
        encoder_input_ids=varied.encoder_input_ids,
        encoder_attention_mask=varied.encoder_attention_mask,
        encoder_position_ids=varied.encoder_position_ids,
        decoder_input_ids=varied.canvas_clean_ids,
        decoder_attention_mask=varied.decoder_attention_mask,
        decoder_position_ids=varied.canvas_position_ids,
    )
    assert counters["stats"].get("unique_graphs", 0) == before


def test_gemma_global_canvas_and_ar_window_matches_packed_trainer(tmp_path):
    features = [
        {"input_ids": [2, 3, 4, 5], "labels": [-100, -100, 4, 5]},
        {"input_ids": [2, 6, 7, 8, 9], "labels": [-100, -100, 7, 8, 9]},
    ]
    dataset = Dataset.from_list(features)
    torch.manual_seed(31)
    initial = DiffusionGemmaForBlockDiffusion(_tiny_config()).state_dict()

    def train(batch_size, accumulation, output_dir):
        model = AxolotlDiffusionGemmaForBlockDiffusion(_tiny_config())
        model.load_state_dict(initial)
        collator = DiffusionCollator(0, 8)

        def data_collator(items):
            batch = collator(items)
            batch["canvas_corruptible_mask"] = torch.zeros_like(
                batch["canvas_corruptible_mask"]
            )
            return batch

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = AxolotlDiffusionTrainer(
            model=model,
            args=AxolotlTrainingArguments(
                output_dir=str(output_dir),
                max_steps=1,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=accumulation,
                learning_rate=1e-3,
                max_grad_norm=0.0,
                weight_decay=0.0,
                lr_scheduler_type="constant",
                report_to=[],
                remove_unused_columns=False,
                save_strategy="no",
                use_cpu=True,
            ),
            train_dataset=dataset,
            data_collator=data_collator,
            optimizers=(optimizer, None),
        )
        trainer.axolotl_cfg = DictDefault(
            {
                "diffusion_lm": {
                    "from_causal_lm": False,
                    "self_conditioning": {"p": 0.0},
                    "encoder_ar_weight": 1.0,
                }
            }
        )
        trainer._special_token_ids = set()
        trainer.post_set_axolotl_cfg()
        trainer.train()
        return model

    accumulated = train(1, 2, tmp_path / "accumulated")
    packed = train(2, 1, tmp_path / "packed")
    for accumulated_parameter, packed_parameter in zip(
        accumulated.parameters(), packed.parameters(), strict=True
    ):
        torch.testing.assert_close(
            accumulated_parameter, packed_parameter, rtol=2e-5, atol=2e-7
        )

    expected = DiffusionGemmaForBlockDiffusion(_tiny_config())
    expected.load_state_dict(initial)
    expected_optimizer = torch.optim.SGD(expected.parameters(), lr=1e-3)
    canvas_numerator = torch.zeros(())
    encoder_ar_numerator = torch.zeros(())
    canvas_count = 0
    encoder_ar_count = 0
    for feature in features:
        input_ids = torch.tensor([feature["input_ids"]])
        labels = torch.tensor([feature["labels"]])
        supervised = torch.where(labels[0].ne(-100))[0]
        canvas = input_ids[:, supervised]
        prefix_length = int(supervised[0])
        prompt_position_ids = torch.arange(input_ids.shape[1])[None]
        canvas_position_ids = torch.arange(
            prefix_length, prefix_length + canvas.shape[1]
        )[None]
        encoder_docs = torch.zeros_like(input_ids)
        canvas_docs = torch.zeros_like(canvas)
        encoder_validity = torch.ones_like(input_ids, dtype=torch.bool)
        canvas_validity = torch.ones_like(canvas, dtype=torch.bool)
        encoder_mask = {
            "full_attention": encoder_causal_mask(
                encoder_docs, encoder_validity, prompt_position_ids
            ),
            "sliding_attention": encoder_causal_mask(
                encoder_docs,
                encoder_validity,
                prompt_position_ids,
                sliding_window=2,
            ),
        }
        decoder_mask = {
            "full_attention": decoder_prefix_canvas_mask(
                encoder_docs,
                canvas_docs,
                encoder_validity,
                canvas_validity,
                torch.tensor([prefix_length]),
                prompt_position_ids,
            ),
            "sliding_attention": decoder_prefix_canvas_mask(
                encoder_docs,
                canvas_docs,
                encoder_validity,
                canvas_validity,
                torch.tensor([prefix_length]),
                prompt_position_ids,
                sliding_window=2,
            ),
        }
        outputs = expected(
            input_ids=input_ids,
            attention_mask=encoder_mask,
            position_ids=prompt_position_ids,
            past_key_values=DynamicCache(),
            decoder_input_ids=canvas,
            decoder_attention_mask=decoder_mask,
            decoder_position_ids=canvas_position_ids,
        )
        canvas_numerator = canvas_numerator + F.cross_entropy(
            outputs.logits.float().flatten(0, -2), canvas.flatten(), reduction="sum"
        )
        canvas_count += canvas.numel()

        encoder_outputs = expected.model.encoder(
            input_ids=input_ids,
            attention_mask=encoder_mask,
            position_ids=prompt_position_ids,
            past_key_values=DynamicCache(),
        )
        encoder_logits = expected.lm_head(encoder_outputs.last_hidden_state).float()
        encoder_logits = (
            torch.tanh(encoder_logits / expected.final_logit_softcapping)
            * expected.final_logit_softcapping
        )
        encoder_ar_numerator = encoder_ar_numerator + F.cross_entropy(
            encoder_logits[:, :-1].flatten(0, -2),
            input_ids[:, 1:].flatten(),
            reduction="sum",
        )
        encoder_ar_count += input_ids.shape[1] - 1
    assert (canvas_count, encoder_ar_count) == (5, 7)
    (
        canvas_numerator / canvas_count + encoder_ar_numerator / encoder_ar_count
    ).backward()
    expected_optimizer.step()
    for accumulated_parameter, expected_parameter in zip(
        accumulated.parameters(), expected.parameters(), strict=True
    ):
        torch.testing.assert_close(
            accumulated_parameter, expected_parameter, rtol=3e-5, atol=3e-7
        )


def test_gemma_native_trainer_evaluate_uses_diffusion_loss(tmp_path):
    dataset = Dataset.from_list(
        [
            {"input_ids": [2, 3, 4, 5], "labels": [-100, -100, 4, 5]},
            {"input_ids": [2, 6, 7, 8, 9], "labels": [-100, -100, 7, 8, 9]},
        ]
    )
    torch.manual_seed(7)
    trainer = AxolotlDiffusionTrainer(
        model=AxolotlDiffusionGemmaForBlockDiffusion(_tiny_config()),
        args=AxolotlTrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            report_to=[],
            remove_unused_columns=False,
            use_cpu=True,
        ),
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DiffusionCollator(0, 8),
        eval_data_collator=DiffusionCollator(0, 8),
    )
    trainer.axolotl_cfg = DictDefault(
        {"diffusion_lm": {"from_causal_lm": False, "self_conditioning": {"p": 0.0}}}
    )
    trainer._special_token_ids = set()
    trainer.post_set_axolotl_cfg()

    metrics = trainer.evaluate()

    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
