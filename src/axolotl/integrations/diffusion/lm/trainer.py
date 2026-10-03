"""Trainer for full-sequence diffusion language models."""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.model_support import (
    DiffusionLayout,
    FirstPositionAlignment,
    LogitAlignment,
    ObjectiveReduction,
    ReductionScope,
    TimeWeighting,
    get_model_support,
    resolve_model_support,
)
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler

from .backends import FullSequenceBackend
from .backends.encoder_canvas import EncoderCanvasBackend
from .batch import DiffusionBatch
from .callbacks import DiffusionGenerationCallback
from .config import get_diffusion_config
from .sampling import native_packing_lengths, resolve_native_packing_budget
from .tokens import resolve_mask_token_id
from .unroll import run_unroll
from .weighting import (
    cart_weights,
    focal_weighted_nll,
    reduce_objective,
    rhine_loo_nll,
    time_weights,
)

LOG = get_logger(__name__)


def normalized_sft_loss(
    weighted_loss: torch.Tensor, batch_indices: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    answer_lengths = (labels != -100).sum(dim=1).float().clamp(min=1.0)
    loss_per_sample = torch.zeros(
        labels.shape[0], dtype=weighted_loss.dtype, device=weighted_loss.device
    ).scatter_add(0, batch_indices, weighted_loss)
    return (loss_per_sample / answer_lengths).mean()


class AxolotlDiffusionTrainer(AxolotlTrainer):
    """Trainer that computes the full-sequence diffusion objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._special_token_ids = None

    @property
    def _native_spec(self):
        """Resolve immutable native facts without borrowing legacy defaults."""

        canonical = (
            self.axolotl_cfg.get("diffusion")
            if isinstance(self.axolotl_cfg, dict)
            else getattr(self.axolotl_cfg, "diffusion", None)
        )
        if canonical is None or bool(getattr(canonical, "from_causal_lm", False)):
            return None
        model = getattr(self, "model", None)
        model_config = getattr(model, "config", None)
        if model_config is None:
            model_config = getattr(getattr(model, "module", None), "config", None)
        model_type = getattr(model_config, "model_type", None)
        support = get_model_support(model_type)
        profile = resolve_model_support(support)
        return None if profile is None else profile.diffusion

    def _full_sequence_backend(self) -> FullSequenceBackend:
        diffusion_cfg = get_diffusion_config(self.axolotl_cfg)
        mask_token_id = getattr(diffusion_cfg, "mask_token_id", None)
        if mask_token_id is None and self._native_spec is not None:
            model_cfg = getattr(self.model, "config", None)
            if model_cfg is None:
                model_cfg = getattr(getattr(self.model, "module", None), "config", None)
            mask_token_id = getattr(model_cfg, "mask_token_id", None)
            if mask_token_id is None:
                mask_token_id = getattr(
                    getattr(model_cfg, "text_config", None), "mask_token_id", None
                )
        if mask_token_id is None:
            raise ValueError("absorbing diffusion requires a resolved mask_token_id")
        attention_implementation = getattr(
            self.axolotl_cfg, "attn_implementation", None
        )
        return FullSequenceBackend(
            mask_token_id=int(mask_token_id),
            special_token_ids=self._special_token_ids,
            sample_packing=bool(getattr(self.axolotl_cfg, "sample_packing", False)),
            attention_backend=(
                "varlen"
                if attention_implementation == "varlen"
                else "flex_attention"
                if attention_implementation == "flex_attention"
                else "dense"
            ),
        )

    def post_set_axolotl_cfg(self):
        if self._native_spec is not None:
            self.model_accepts_loss_kwargs = (
                self._native_spec.reduction_scope is ReductionScope.GLOBAL_WINDOW
            )
            if getattr(
                get_diffusion_config(self.axolotl_cfg), "generate_samples", False
            ):
                self.add_callback(DiffusionGenerationCallback(self))
            return
        self._cache_special_token_ids()
        self._resolve_mask_token_id()
        diffusion_cfg = get_diffusion_config(self.axolotl_cfg)
        token_id = int(getattr(diffusion_cfg, "mask_token_id", 0))
        LOG.info("Diffusion: using mask_token_id=%s", token_id)
        if getattr(diffusion_cfg, "generate_samples", True):
            self.add_callback(DiffusionGenerationCallback(self))

    def _resolve_mask_token_id(self) -> None:
        assert self.axolotl_cfg is not None, "axolotl_cfg is not set yet"
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None:
            return
        mid = resolve_mask_token_id(
            tokenizer,
            self.axolotl_cfg,
            allow_add=True,
            model=getattr(self, "model", None),
        )
        try:
            get_diffusion_config(self.axolotl_cfg).mask_token_id = int(mid)
        except Exception:
            pass

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._native_spec is not None:
            loss, outputs = self._compute_native_diffusion_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )
            return (loss, outputs) if return_outputs else loss
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("input_ids is required for diffusion training")
        loss, outputs = self._compute_diffusion_loss(
            model, input_ids, inputs.get("attention_mask"), inputs.get("labels")
        )
        if return_outputs:
            return loss, outputs
        return loss

    def _get_num_items_in_batch(self, batch_samples, device):
        """Count native supervised canvas tokens across a gradient window."""

        if self._native_spec is None:
            return super()._get_num_items_in_batch(batch_samples, device)
        if self._native_spec.layout is DiffusionLayout.ENCODER_CANVAS:
            if not batch_samples:
                return None
            canvas_count = sum(
                batch["canvas_loss_mask"].sum() for batch in batch_samples
            ).to(device)
            ar_count = sum(
                (
                    batch["encoder_ar_valid_mask"][:, 1:]
                    & batch["encoder_ar_valid_mask"][:, :-1]
                    & (
                        batch["encoder_document_ids"][:, 1:]
                        == batch["encoder_document_ids"][:, :-1]
                    )
                ).sum()
                for batch in batch_samples
            ).to(device)
            example_count = sum(
                batch["canvas_loss_mask"].any(dim=1).sum() for batch in batch_samples
            )
            example_count = torch.as_tensor(example_count, device=device)
            if self.args.world_size > 1:
                canvas_count = self.accelerator.gather(canvas_count).sum()
                ar_count = self.accelerator.gather(ar_count).sum()
                example_count = self.accelerator.gather(example_count).sum()
            return {
                "canvas": canvas_count,
                "encoder_ar": ar_count,
                "examples": example_count,
            }
        counts = []
        for batch in batch_samples:
            prepared_count = batch.get("native_num_items")
            if prepared_count is not None:
                counts.append(prepared_count)
                continue
            mask = batch.get("canvas_loss_mask")
            if mask is None:
                mask = batch.get("labels")
                if mask is not None:
                    mask = mask.ne(-100)
            if mask is not None:
                counts.append(mask.sum())
        if not counts:
            return None
        count = sum(counts).to(device)
        if self.args.world_size > 1:
            count = self.accelerator.gather(count).sum()
        return count

    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batches, _ = super().get_batch_samples(epoch_iterator, num_batches, device)
        spec = self._native_spec
        if (
            spec is not None
            and spec.layout is DiffusionLayout.FULL_SEQUENCE
            and spec.reduction_scope is ReductionScope.GLOBAL_WINDOW
            and spec.objective_reduction is ObjectiveReduction.MASKED_TOKEN_MEAN
        ):
            prepared = [self._prepare_global_masked_batch(batch) for batch in batches]
            return prepared, self._get_num_items_in_batch(prepared, device)
        return batches, self._get_num_items_in_batch(batches, device)

    def _create_multipack_sampler(self, base_sampler, dataset):
        spec = self._native_spec
        diffusion_cfg = get_diffusion_config(self.axolotl_cfg)
        if spec is None:
            return super()._create_multipack_sampler(base_sampler, dataset)
        batch_size = 1
        batch_max_len = self.args.max_seq_length * self.args.per_device_train_batch_size
        packing_budget = resolve_native_packing_budget(self.axolotl_cfg)
        if packing_budget is not None:
            batch_max_len = packing_budget.payload_capacity
        sampler = MultipackBatchSampler(
            base_sampler,
            lengths=native_packing_lengths(
                dataset,
                eos_tail=diffusion_cfg.eos_tail,
                logical_sequence_length=self.axolotl_cfg.sequence_len,
                layout=spec.layout.value,
                canvas_width=(
                    diffusion_cfg.canvas_width
                    if diffusion_cfg.canvas_width is not None
                    else spec.max_canvas
                ),
            ),
            packing_efficiency_estimate=self.args.sample_packing_efficiency,
            batch_max_len=batch_max_len,
            batch_size=batch_size,
            group_size=self.args.sample_packing_group_size,
            bin_size=self.args.sample_packing_bin_size,
            sequential=self.args.sample_packing_sequentially,
            drop_last=True,
            num_processes=self.args.dataset_num_proc,
            mp_start_method=self.args.sample_packing_mp_start_method or "fork",
        )
        len(sampler)
        return sampler

    def _prepare_global_masked_batch(self, inputs: dict[str, torch.Tensor]):
        prepared = (
            inputs
            if "input_ids" in inputs
            else self._full_sequence_inputs_from_collator(inputs)
        )
        input_ids = prepared["input_ids"].long()
        semantic = self._native_mask(
            prepared,
            "semantic_validity",
            prepared.get("attention_mask", torch.ones_like(input_ids)).bool(),
        )
        document_ids = prepared.get("document_ids")
        if document_ids is None:
            document_ids = torch.arange(input_ids.shape[0], device=input_ids.device)[
                :, None
            ].expand_as(input_ids)
        backend = self._full_sequence_backend()
        packed = backend.pack(input_ids, document_ids.long(), semantic)
        logical_rows = inputs.get("native_logical_rows")
        if logical_rows is None:
            logical_rows, logical_count = self._logical_row_indices(
                document_ids, semantic
            )
        else:
            logical_count = (
                int(logical_rows.max().item()) + 1 if logical_rows.numel() else 0
            )
        logical_rows = self._pad_full_sequence_metadata(logical_rows, packed, -1)
        times = self._sample_native_times(
            max(logical_count, 1), input_ids.device, self._native_spec.time_floor
        )
        corruptible = (
            self._native_mask(
                prepared,
                "canvas_corruptible_mask",
                prepared.get("canvas_loss_mask", torch.zeros_like(semantic)),
            )
            & semantic
        )
        pinned = self._native_mask(
            prepared, "canvas_input_pinned_mask", torch.zeros_like(semantic)
        )
        corruptible = self._pad_full_sequence_metadata(corruptible, packed, False)
        pinned = self._pad_full_sequence_metadata(pinned, packed, False)
        loss_mask = self._pad_full_sequence_metadata(
            self._native_mask(prepared, "canvas_loss_mask", torch.zeros_like(semantic)),
            packed,
            False,
        )
        noisy, events, _ = backend.corrupt_native(
            packed,
            corruptible & ~pinned,
            times,
            document_time_indices=logical_rows,
            eos_token_id=getattr(
                getattr(self, "processing_class", None), "eos_token_id", None
            ),
            treat_eos_as_one=(
                self._native_spec.eos_handling.value == "treat_eos_as_one"
                if self._native_value("treat_eos_as_one") is None
                else bool(self._native_value("treat_eos_as_one"))
            ),
        )
        return {
            **prepared,
            "native_prepared_noisy_ids": noisy,
            "native_prepared_events": events,
            "native_prepared_times": times,
            "native_logical_rows": logical_rows,
            "native_num_items": (events & loss_mask).sum(),
        }

    def _native_value(self, name: str, default=None):
        cfg = get_diffusion_config(self.axolotl_cfg)
        return getattr(cfg, name, default)

    def _native_time_weighting(self, spec) -> TimeWeighting:
        value = self._native_value("time_weighting")
        return spec.default_time_weighting if value is None else TimeWeighting(value)

    def _native_reduction(self, spec) -> ObjectiveReduction:
        value = self._native_value("objective_reduction")
        return spec.objective_reduction if value is None else ObjectiveReduction(value)

    @staticmethod
    def _native_lm_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
        native_model = getattr(model, "module", model)
        logits = native_model.lm_head(hidden_states).float()
        softcap = getattr(native_model, "final_logit_softcapping", None)
        if softcap is None:
            return logits
        return torch.tanh(logits / softcap) * softcap

    @staticmethod
    def _native_mask(inputs, name: str, fallback: torch.Tensor) -> torch.Tensor:
        value = inputs.get(name)
        return fallback if value is None else value.bool()

    @staticmethod
    def _pad_full_sequence_metadata(
        value: torch.Tensor, packed: dict[str, torch.Tensor], pad_value
    ) -> torch.Tensor:
        expected_shape = packed["input_ids"].shape
        if value.shape == expected_shape:
            return value
        if value.shape[0] != expected_shape[0] or value.shape[1] > expected_shape[1]:
            raise ValueError("full-sequence metadata must fit the packed row")
        return F.pad(value, (0, expected_shape[1] - value.shape[1]), value=pad_value)

    def _compute_native_diffusion_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        *,
        num_items_in_batch=None,
    ):
        spec = self._native_spec
        if spec is None:
            raise RuntimeError(
                "native diffusion loss requires a resolved DiffusionSpec"
            )
        if self._native_value("token_reweighting", False) and (
            spec.layout is not DiffusionLayout.FULL_SEQUENCE
            or spec.first_position_alignment
            is not FirstPositionAlignment.DUPLICATE_FIRST
        ):
            raise ValueError(
                "token_reweighting is currently supported only by native Dream"
            )
        if spec.layout is DiffusionLayout.FULL_SEQUENCE:
            return self._compute_native_full_sequence_loss(
                model, inputs, spec, num_items_in_batch=num_items_in_batch
            )
        return self._compute_native_encoder_canvas_loss(
            model, inputs, spec, num_items_in_batch=num_items_in_batch
        )

    def _sample_native_times(
        self, count: int, device: torch.device, default_eps: float
    ) -> torch.Tensor:
        eps = self._native_value("t_eps")
        eps = default_eps if eps is None else eps
        return torch.rand(count, device=device) * (1.0 - eps) + eps

    def _native_unroll_settings(self) -> tuple[int, bool]:
        unroll = self._native_value("unroll")
        if unroll is None:
            return 1, False
        return int(unroll.k_max), bool(unroll.grad_through_steps)

    @staticmethod
    def _sample_native_unroll_steps(k_max: int, device: torch.device) -> int:
        if k_max < 1:
            raise ValueError("unroll.k_max must be positive")
        if k_max == 1:
            return 1
        return int(torch.randint(1, k_max + 1, (), device=device).item())

    def _run_native_unroll(
        self,
        *,
        state: torch.Tensor,
        update_mask: torch.Tensor,
        supports_self_conditioning: bool,
        k1_conditioning_mask: torch.Tensor | None,
        recurrent_conditioning_mask: torch.Tensor | None,
        forward_step,
        logits_from_outputs,
        update_state,
        forward_final=None,
        final_logits_from_outputs=None,
    ):
        """Run the shared native K-step recurrence and retain only its final loss pass."""

        k_max, grad_through_steps = self._native_unroll_settings()
        steps = self._sample_native_unroll_steps(k_max, state.device)
        outputs, logits, final_state = run_unroll(
            state=state,
            update_mask=update_mask,
            steps=steps,
            grad_through_steps=grad_through_steps,
            supports_self_conditioning=supports_self_conditioning,
            k1_conditioning_mask=k1_conditioning_mask,
            recurrent_conditioning_mask=recurrent_conditioning_mask,
            forward_step=forward_step,
            logits_from_outputs=logits_from_outputs,
            update_state=update_state,
            forward_final=forward_final,
            final_logits_from_outputs=final_logits_from_outputs,
        )
        return outputs, logits, final_state, steps

    @staticmethod
    def _packed_cart_weights(
        events: torch.Tensor,
        semantic: torch.Tensor,
        document_ids: torch.Tensor,
        cart_p: float,
    ) -> torch.Tensor:
        weights = torch.zeros_like(events, dtype=torch.float)
        groups: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for row in range(events.shape[0]):
            for document in document_ids[row][semantic[row]].unique().tolist():
                positions = torch.where(
                    (document_ids[row] == document) & semantic[row]
                )[0]
                groups.setdefault(positions.numel(), []).append((row, positions))
        for documents in groups.values():
            unmasked = torch.stack(
                [~events[row, positions] for row, positions in documents]
            )
            document_weights = cart_weights(unmasked, cart_p).to(weights.dtype)
            for index, (row, positions) in enumerate(documents):
                weights[row, positions] = document_weights[index]
        return weights

    @staticmethod
    def _logical_row_indices(
        document_ids: torch.Tensor, semantic: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        rows = torch.full_like(document_ids, -1)
        next_index = 0
        for document in document_ids[semantic].unique(sorted=True).tolist():
            rows[document_ids == document] = next_index
            next_index += 1
        return rows, next_index

    def _compute_native_full_sequence_loss(
        self, model, inputs, spec, *, num_items_in_batch=None
    ):
        if "input_ids" not in inputs:
            inputs = self._full_sequence_inputs_from_collator(inputs)
        input_ids = inputs["input_ids"].long()
        semantic = self._native_mask(
            inputs,
            "semantic_validity",
            inputs.get("attention_mask", torch.ones_like(input_ids)).bool(),
        )
        document_ids = inputs.get("document_ids")
        if document_ids is None:
            document_ids = torch.arange(input_ids.shape[0], device=input_ids.device)[
                :, None
            ].expand_as(input_ids)
        document_ids = document_ids.long()
        loss_mask = (
            self._native_mask(
                inputs,
                "canvas_loss_mask",
                inputs.get("loss_mask", inputs.get("labels", input_ids) != -100),
            )
            & semantic
        )
        corruptible = (
            self._native_mask(inputs, "canvas_corruptible_mask", loss_mask) & semantic
        )
        pinned = self._native_mask(
            inputs, "canvas_input_pinned_mask", torch.zeros_like(semantic)
        )
        corruptible &= ~pinned
        backend = self._full_sequence_backend()
        packed = backend.pack(
            input_ids, document_ids, semantic, inputs.get("position_ids")
        )
        logical_rows = inputs.get("native_logical_rows")
        if logical_rows is None:
            logical_rows, logical_count = self._logical_row_indices(
                document_ids, semantic
            )
        else:
            logical_count = int(logical_rows.max().item()) + 1
        logical_count = max(logical_count, 1)
        logical_rows = self._pad_full_sequence_metadata(logical_rows, packed, -1)
        loss_mask = self._pad_full_sequence_metadata(loss_mask, packed, False)
        corruptible = self._pad_full_sequence_metadata(corruptible, packed, False)
        pinned = self._pad_full_sequence_metadata(pinned, packed, False)
        input_ids = packed["input_ids"]
        document_ids = packed["document_ids"]
        semantic = packed["semantic_validity"]
        loss_mask &= semantic
        corruptible &= semantic
        times = self._sample_native_times(
            logical_count, input_ids.device, spec.time_floor
        )
        reduction = self._native_reduction(spec)
        targets = input_ids
        if "native_prepared_noisy_ids" in inputs:
            state = inputs["native_prepared_noisy_ids"]
            events = inputs["native_prepared_events"]
            times = inputs["native_prepared_times"]
        else:
            state, events, _ = backend.corrupt_native(
                packed,
                corruptible,
                times,
                document_time_indices=logical_rows,
                eos_token_id=getattr(
                    getattr(self, "processing_class", None), "eos_token_id", None
                ),
                treat_eos_as_one=(
                    spec.eos_handling.value == "treat_eos_as_one"
                    if self._native_value("treat_eos_as_one") is None
                    else bool(self._native_value("treat_eos_as_one"))
                ),
            )
        update_mask = (
            self._pad_full_sequence_metadata(
                self._native_mask(inputs, "canvas_update_mask", corruptible),
                packed,
                False,
            )
            & ~pinned
        )
        cce_enabled = bool(getattr(self.axolotl_cfg, "cut_cross_entropy", False))
        weighting = self._native_time_weighting(spec)
        if cce_enabled:
            from axolotl.model_support.nemotron_diffusion.cut_cross_entropy import (
                get_cce_options,
            )

            model_config = getattr(model, "config", None)
            if model_config is None:
                model_config = getattr(getattr(model, "module", None), "config", None)
            model_type = getattr(model_config, "model_type", None)
            if model_type is None and hasattr(model_config, "get_text_config"):
                model_type = getattr(model_config.get_text_config(), "model_type", None)
            if model_type != "nemotron_labs_diffusion":
                raise ValueError(
                    "cut_cross_entropy native diffusion support is implemented only for Nemotron."
                )
            if spec.logit_alignment is not LogitAlignment.ALIGNED:
                raise ValueError(
                    "cut_cross_entropy native Nemotron requires aligned diffusion logits."
                )
            if weighting is TimeWeighting.LOO:
                raise ValueError(
                    "cut_cross_entropy native Nemotron does not support loo time weighting."
                )
            cce_options = get_cce_options(model)
            cce_enabled = model.training or not bool(
                getattr(cce_options, "train_only", False)
            )

            if cce_enabled:

                def forward_final(current_state, _conditioning, _conditioning_mask):
                    cce_targets = torch.where(
                        events & loss_mask,
                        targets,
                        torch.full_like(targets, -100),
                    )
                    return backend.forward(
                        model,
                        packed,
                        current_state,
                        kernel_options=getattr(
                            self.axolotl_cfg, "flex_attn_compile_kwargs", None
                        ),
                        model_kwargs={"cce_targets": cce_targets},
                    )

            else:
                forward_final = None

        else:
            forward_final = None
        outputs, logits, final_state, _ = self._run_native_unroll(
            state=state,
            update_mask=update_mask,
            supports_self_conditioning=spec.self_conditioning,
            k1_conditioning_mask=None,
            recurrent_conditioning_mask=None,
            forward_step=lambda current_state, _conditioning, _conditioning_mask: (
                backend.forward(
                    model,
                    packed,
                    current_state,
                    kernel_options=getattr(
                        self.axolotl_cfg, "flex_attn_compile_kwargs", None
                    ),
                )
            ),
            logits_from_outputs=lambda current_outputs: backend.canvas_logits(
                current_outputs,
                packed,
                aligned=spec.logit_alignment is LogitAlignment.ALIGNED,
            ),
            update_state=backend.update,
            forward_final=forward_final,
            final_logits_from_outputs=(lambda _outputs: None) if cce_enabled else None,
        )
        if cce_enabled:
            token_loss = getattr(outputs, "loss", None)
            if (
                not isinstance(token_loss, torch.Tensor)
                or token_loss.shape != targets.shape
            ):
                raise ValueError(
                    "Nemotron CCE final forward must return per-token loss matching the canvas."
                )
            token_loss = token_loss.float()
        else:
            token_loss = (
                rhine_loo_nll(
                    logits, targets, final_state, times[logical_rows.clamp_min(0)]
                )
                if weighting is TimeWeighting.LOO
                else F.cross_entropy(
                    logits.flatten(0, -2), targets.flatten(), reduction="none"
                )
                .float()
                .view_as(targets)
            )
        support = events & loss_mask
        if self._native_value("token_reweighting", False):
            token_loss = focal_weighted_nll(
                token_loss,
                float(self._native_value("alpha", 0.25)),
                float(self._native_value("gamma", 2.0)),
            )
        if weighting is TimeWeighting.CART:
            token_loss = token_loss * self._packed_cart_weights(
                events,
                semantic,
                document_ids,
                0.1
                if self._native_value("cart_p") is None
                else self._native_value("cart_p"),
            )
        else:
            token_loss = (
                token_loss * time_weights(times, weighting)[logical_rows.clamp_min(0)]
            )
        denominator = None
        if spec.reduction_scope is ReductionScope.GLOBAL_WINDOW:
            if num_items_in_batch is not None:
                denominator = torch.as_tensor(
                    num_items_in_batch,
                    device=token_loss.device,
                    dtype=token_loss.dtype,
                )
                if self.args.world_size > 1:
                    denominator /= self.args.world_size
            else:
                denominator = support.sum().detach().to(token_loss.dtype)
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.all_reduce(denominator)
                    denominator /= torch.distributed.get_world_size()
        loss = reduce_objective(
            token_loss,
            support,
            reduction,
            logical_ids=logical_rows,
            logical_count=logical_count,
            denominator=denominator,
        )
        self.store_metrics(
            {"loss": loss.detach().item(), "mask_ratio": events.float().mean().item()},
            train_eval="train" if model.training else "eval",
        )
        return loss, outputs

    @staticmethod
    def _full_sequence_inputs_from_collator(inputs: dict[str, torch.Tensor]):
        """Turn logical collator streams into one document-isolated physical row."""

        fields = DiffusionBatch.__dataclass_fields__
        batch = DiffusionBatch(**{name: inputs[name] for name in fields})
        ids, documents, loss, corruptible, pinned, update = ([] for _ in range(6))
        if not batch.encoder_lengths.any():
            for index, length in enumerate(batch.canvas_lengths.tolist()):
                ids.append(batch.canvas_clean_ids[index, :length])
                documents.append(
                    torch.full(
                        (length,),
                        batch.logical_ids[index],
                        dtype=torch.long,
                        device=batch.device,
                    )
                )
                loss.append(batch.canvas_loss_mask[index, :length])
                corruptible.append(batch.canvas_corruptible_mask[index, :length])
                pinned.append(batch.canvas_input_pinned_mask[index, :length])
                update.append(batch.canvas_update_mask[index, :length])
            input_ids = torch.cat(ids)[None]
            return {
                "input_ids": input_ids,
                "document_ids": torch.cat(documents)[None],
                "semantic_validity": torch.cat(
                    [
                        batch.canvas_semantic_validity[index, :length]
                        for index, length in enumerate(batch.canvas_lengths.tolist())
                    ]
                )[None],
                "canvas_loss_mask": torch.cat(loss)[None],
                "canvas_corruptible_mask": torch.cat(corruptible)[None],
                "canvas_input_pinned_mask": torch.cat(pinned)[None],
                "canvas_update_mask": torch.cat(update)[None],
            }
        for index, length in enumerate(batch.encoder_lengths.tolist()):
            ids.append(batch.encoder_input_ids[index, :length])
            documents.append(
                torch.full(
                    (length,),
                    batch.logical_ids[index],
                    dtype=torch.long,
                    device=batch.device,
                )
            )
            prefix = int(batch.decoder_prefix_lengths[index])
            response_length = int(batch.canvas_lengths[index])
            response = slice(prefix, prefix + response_length)
            item_loss = torch.zeros(length, dtype=torch.bool, device=batch.device)
            item_corruptible = torch.zeros_like(item_loss)
            item_pinned = torch.zeros_like(item_loss)
            item_update = torch.zeros_like(item_loss)
            item_loss[response] = batch.canvas_loss_mask[index, :response_length]
            item_corruptible[response] = batch.canvas_corruptible_mask[
                index, :response_length
            ]
            item_pinned[response] = batch.canvas_input_pinned_mask[
                index, :response_length
            ]
            item_update[response] = batch.canvas_update_mask[index, :response_length]
            loss.append(item_loss)
            corruptible.append(item_corruptible)
            pinned.append(item_pinned)
            update.append(item_update)
        input_ids = torch.cat(ids)[None]
        document_ids = torch.cat(documents)[None]
        semantic = torch.ones_like(input_ids, dtype=torch.bool)
        return {
            "input_ids": input_ids,
            "document_ids": document_ids,
            "semantic_validity": semantic,
            "canvas_loss_mask": torch.cat(loss)[None],
            "canvas_corruptible_mask": torch.cat(corruptible)[None],
            "canvas_input_pinned_mask": torch.cat(pinned)[None],
            "canvas_update_mask": torch.cat(update)[None],
        }

    def _compute_native_encoder_canvas_loss(
        self, model, inputs, spec, *, num_items_in_batch=None
    ):
        weighting = self._native_time_weighting(spec)
        reduction = self._native_reduction(spec)
        rhine_objective = weighting in {
            TimeWeighting.LOO,
            TimeWeighting.INV_ONE_MINUS_T,
        }
        if weighting not in {
            TimeWeighting.NONE,
            TimeWeighting.LOO,
            TimeWeighting.INV_ONE_MINUS_T,
        }:
            raise ValueError(
                "DiffusionGemma supports time_weighting: none, loo, or inv_one_minus_t"
            )
        if rhine_objective and reduction is not ObjectiveReduction.EXAMPLE_MEAN:
            raise ValueError(
                "DiffusionGemma Rhine objectives require "
                "objective_reduction: example_mean"
            )
        if (
            not rhine_objective
            and reduction is not ObjectiveReduction.SUPERVISED_TOKEN_MEAN
        ):
            raise ValueError(
                "DiffusionGemma plain CE requires "
                "objective_reduction: supervised_token_mean"
            )
        batch = inputs.get("diffusion_batch")
        if batch is None:
            fields = DiffusionBatch.__dataclass_fields__
            batch = DiffusionBatch(**{name: inputs[name] for name in fields})
        model_config = getattr(model, "config", None)
        if model_config is None:
            model_config = getattr(getattr(model, "module", None), "config", None)
        if model_config is None:
            raise TypeError("native encoder/canvas model must expose a config")
        text_config = getattr(model_config, "text_config", model_config)
        backend = EncoderCanvasBackend(
            vocab_size=int(text_config.vocab_size),
            sliding_window=int(getattr(text_config, "sliding_window", 1024)),
            attention_backend=(
                "flex_attention"
                if getattr(self.axolotl_cfg, "attn_implementation", None)
                == "flex_attention"
                else "dense"
            ),
        )
        packed = backend.pack(batch)
        times = self._sample_native_times(
            batch.logical_ids.numel(), packed.device, spec.time_floor
        )
        corrupted = backend.corrupt(packed, times)
        sc_cfg = self._native_value("self_conditioning")
        sc_probability = float(getattr(sc_cfg, "p", 0.5) if sc_cfg is not None else 0.5)
        k1_conditioning_mask = None
        if sc_probability:
            per_document_gate = (
                torch.rand(batch.logical_ids.numel(), device=packed.device)
                < sc_probability
            )
            sc_gate = per_document_gate[packed.canvas_logical_row_indices]
            k1_conditioning_mask = (
                packed.canvas_sc_eligible_mask
                & ~packed.canvas_input_pinned_mask
                & sc_gate
            )
        recurrent_conditioning_mask = (
            packed.canvas_sc_eligible_mask & ~packed.canvas_input_pinned_mask
        )
        update_mask = packed.canvas_update_mask & ~packed.canvas_input_pinned_mask
        k_max, grad_through_steps = self._native_unroll_settings()
        steps = self._sample_native_unroll_steps(k_max, packed.device)
        outputs = backend.forward(
            model,
            packed,
            corrupted.input_ids,
            unroll_steps=steps,
            grad_through_steps=grad_through_steps,
            k1_conditioning_mask=k1_conditioning_mask,
            recurrent_conditioning_mask=recurrent_conditioning_mask,
            update_mask=update_mask,
            kernel_options=getattr(self.axolotl_cfg, "flex_attn_compile_kwargs", None),
        )
        logits = outputs.logits
        token_loss = (
            rhine_loo_nll(
                logits,
                packed.canvas_clean_ids,
                corrupted.input_ids,
                times[packed.canvas_logical_row_indices],
            )
            if rhine_objective
            else F.cross_entropy(
                logits.float().flatten(0, -2),
                packed.canvas_clean_ids.flatten(),
                reduction="none",
            ).view_as(packed.canvas_clean_ids)
        )
        if weighting is TimeWeighting.INV_ONE_MINUS_T:
            rhine_weights = time_weights(times, weighting)[
                packed.canvas_logical_row_indices
            ]
            rhine_weight_clip = self._native_value("rhine_weight_clip")
            if rhine_weight_clip is not None:
                rhine_weights = rhine_weights.clamp_max(rhine_weight_clip)
            token_loss = token_loss * rhine_weights
        canvas_denominator = None
        if spec.reduction_scope is ReductionScope.GLOBAL_WINDOW:
            if rhine_objective:
                canvas_denominator = (
                    num_items_in_batch["examples"]
                    if isinstance(num_items_in_batch, dict)
                    and "examples" in num_items_in_batch
                    else batch.canvas_loss_mask.any(dim=1).sum().to(token_loss.device)
                )
            else:
                canvas_denominator = (
                    num_items_in_batch["canvas"]
                    if isinstance(num_items_in_batch, dict)
                    else packed.canvas_loss_mask.sum().detach().to(token_loss.dtype)
                )
            if self.args.world_size > 1:
                canvas_denominator = canvas_denominator / self.args.world_size
        loss = reduce_objective(
            token_loss,
            packed.canvas_loss_mask,
            reduction,
            logical_ids=(
                packed.canvas_logical_row_indices if rhine_objective else None
            ),
            logical_count=batch.logical_ids.numel() if rhine_objective else None,
            denominator=canvas_denominator,
        )
        encoder_weight = self._native_value("encoder_ar_weight")
        encoder_weight = 1.0 if encoder_weight is None else encoder_weight
        if encoder_weight:
            encoder_logits = getattr(outputs, "encoder_logits", None)
            if encoder_logits is None:
                if hasattr(model, "module"):
                    raise RuntimeError(
                        "packed DiffusionGemma DDP forward must return encoder_logits"
                    )
                encoder_logits = self._native_lm_logits(
                    model, outputs.encoder_last_hidden_state
                )
            ar_loss = F.cross_entropy(
                encoder_logits[:, :-1].float().flatten(0, -2),
                packed.encoder_input_ids[:, 1:].flatten(),
                reduction="none",
            )
            valid = (
                packed.encoder_ar_valid_mask[:, 1:]
                & packed.encoder_ar_valid_mask[:, :-1]
                & (
                    packed.encoder_document_ids[:, 1:]
                    == packed.encoder_document_ids[:, :-1]
                )
            )
            ar_denominator = None
            if spec.reduction_scope is ReductionScope.GLOBAL_WINDOW:
                ar_denominator = (
                    num_items_in_batch["encoder_ar"]
                    if isinstance(num_items_in_batch, dict)
                    else valid.sum().detach().to(token_loss.dtype)
                )
                if self.args.world_size > 1:
                    ar_denominator = ar_denominator / self.args.world_size
            loss = loss + encoder_weight * reduce_objective(
                ar_loss,
                valid,
                ObjectiveReduction.SUPERVISED_TOKEN_MEAN,
                denominator=ar_denominator,
            )
        return loss, outputs

    def _cache_special_token_ids(self):
        if self.processing_class is None:
            self._special_token_ids = set()
            return
        tokenizer = self.processing_class
        special_tokens = set()
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token_id = getattr(tokenizer, name, None)
            if token_id is not None:
                special_tokens.add(token_id)
        self._special_token_ids = special_tokens

    def _forward_process(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        eps: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._full_sequence_backend().corrupt(
            input_ids, attention_mask, labels, eps
        )

    def _compute_diffusion_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | Any]:
        if input_ids is None or input_ids.numel() == 0 or input_ids.shape[1] == 0:
            return (
                torch.tensor(
                    0.0,
                    device=input_ids.device if input_ids is not None else None,
                    requires_grad=True,
                ),
                {},
            )
        if attention_mask is not None and attention_mask.dim() == 2:
            if (attention_mask.sum(dim=1) == 0).all():
                return torch.tensor(
                    0.0, device=input_ids.device, requires_grad=True
                ), {}

        diffusion_cfg = get_diffusion_config(self.axolotl_cfg)
        noisy_batch, masked_indices, p_mask = self._forward_process(
            input_ids, attention_mask, labels, diffusion_cfg.eps
        )
        backend = self._full_sequence_backend()
        outputs = model(
            input_ids=noisy_batch.long(),
            attention_mask=backend.attention_mask(input_ids, attention_mask),
        )
        logits = backend.align_logits(outputs.logits)

        if masked_indices.sum() > 0:
            batch_indices, sequence_indices = torch.where(masked_indices)
            masked_logits = logits[batch_indices, sequence_indices]
            masked_targets = input_ids[batch_indices, sequence_indices]
            masked_p_mask = p_mask[batch_indices, sequence_indices]
            token_loss = F.cross_entropy(
                masked_logits.float(), masked_targets, reduction="none"
            )
            weighted_loss = (
                token_loss / masked_p_mask.float()
                if diffusion_cfg.importance_weighting
                else token_loss
            )
            if labels is not None:
                loss = normalized_sft_loss(weighted_loss, batch_indices, labels)
            elif diffusion_cfg.importance_weighting:
                loss = weighted_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            else:
                loss = weighted_loss.mean()
            ce_loss = token_loss.mean()
            with torch.no_grad():
                accuracy = (
                    (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
                )
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            accuracy = torch.tensor(0.0, device=input_ids.device)
            ce_loss = torch.tensor(0.0, device=input_ids.device)
            masked_p_mask = torch.tensor(1.0, device=input_ids.device)

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "mask_ratio": masked_indices.float().mean().item(),
            "num_masked_tokens": (masked_indices.sum().item(), "sum"),
            "avg_p_mask": (
                p_mask[masked_indices].mean().item() if masked_indices.any() else 0.0
            ),
            "ce_loss": ce_loss.item(),
        }
        if self.axolotl_cfg.datasets is not None and labels is not None:
            with torch.no_grad():
                answer_mask = labels != -100
                answer_lengths = answer_mask.sum(dim=1).float()
                metrics["answer_ratio"] = answer_mask.sum().item() / max(
                    labels.numel(), 1
                )
                metrics["avg_answer_length"] = answer_lengths.mean().item()
        if diffusion_cfg.importance_weighting:
            metrics["importance_weight_avg"] = (1.0 / masked_p_mask).mean().item()
        train_eval: Literal["train", "eval"] = "train" if model.training else "eval"
        self.store_metrics(metrics, train_eval=train_eval)
        return loss, outputs


DiffusionTrainer = AxolotlDiffusionTrainer
