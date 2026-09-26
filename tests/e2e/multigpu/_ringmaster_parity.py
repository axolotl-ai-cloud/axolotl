"""Two-rank Trainer loss/gradient parity against an unsharded tiny model."""

import copy
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate import ParallelismConfig
from transformers import LlamaConfig, LlamaForCausalLM, Mamba2Config, Mamba2ForCausalLM

from axolotl.core.trainers.base import AxolotlTrainer as Trainer
from axolotl.core.training_args import AxolotlTrainingArguments as TrainingArguments
from axolotl.integrations.context_parallel import (
    ContextParallelConfig,
    ContextParallelPlugin,
)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    backend = os.environ.get("RM_BACKEND", "ulysses")
    inner = os.environ.get("RM_INNER", "sdpa")
    for average in (True, False):
        torch.manual_seed(123)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=4,
                attn_implementation=inner,
            )
        ).cuda()
        if os.environ.get("RM_MODEL") in ("mamba", "mamba2"):
            model_cls, config_cls = Mamba2ForCausalLM, Mamba2Config
            if os.environ.get("RM_FLA") == "1":
                from transformers import MambaConfig

                from axolotl.model_support.mamba.loading import MambaModelLoader

                model_cls = MambaModelLoader
                config_cls = (
                    MambaConfig
                    if os.environ.get("RM_MODEL") == "mamba"
                    else Mamba2Config
                )
            model = model_cls(
                config_cls(
                    mamba_backend="fla"
                    if os.environ.get("RM_FLA") == "1"
                    else "transformers",
                    vocab_size=32,
                    hidden_size=128,
                    expand=2,
                    num_heads=8,
                    head_dim=32,
                    state_size=16,
                    n_groups=2,
                    num_hidden_layers=1,
                    chunk_size=64,
                    use_cache=False,
                )
            ).cuda()
        if inner != "sdpa":
            model = model.to(torch.bfloat16)
        reference = type(model)(copy.deepcopy(model.config)).cuda().to(model.dtype)
        if os.environ.get("RM_MODEL") not in ("mamba", "mamba2"):
            reference.set_attn_implementation("sdpa")
        reference.load_state_dict(model.state_dict())
        if os.environ.get("RM_HUB") == "1":
            model = model.to(torch.bfloat16)
            reference = reference.to(torch.bfloat16)
            from transformers.integrations.hub_kernels import (
                kernelize,
                register_kernel_mapping_transformers,
            )

            register_kernel_mapping_transformers()
            if os.environ.get("RM_FLA") != "1":
                kernelize(model)
                kernelize(reference)
            if (
                os.environ.get("RM_MODEL") == "mamba2"
                and os.environ.get("RM_FLA") != "1"
            ):
                from transformers.models.mamba2.modeling_mamba2 import (
                    mamba2_split_conv1d_scan_combined as fused,
                )

                assert fused.forward.__func__ is not type(fused).forward
        args = TrainingArguments(
            output_dir="/tmp/ringmaster-parity",
            report_to="none",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            average_tokens_across_devices=average,
            parallelism_config=ParallelismConfig(cp_size=2),
            fsdp=True,
            fsdp_config={"version": 2},
            use_cpu=False,
        )
        trainer = Trainer(model=model, args=args)
        cfg = SimpleNamespace(
            context_parallel=ContextParallelConfig(
                size=2, backend=backend, load_balance="none"
            ),
            attn_implementation=inner,
            gradient_accumulation_steps=2,
            rl=None,
        )
        plugin = ContextParallelPlugin()
        plugin.post_trainer_create(cfg, trainer)
        trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer.model, trainer.optimizer = trainer.accelerator.prepare(
            trainer.model, trainer.optimizer
        )
        trainer.current_gradient_accumulation_steps = 2
        batches = []
        for n, masked in [(7, 2), (10, 6)] if average else [(7, 2)]:
            ids = torch.arange(n, device=rank).unsqueeze(0) + 1
            labels = ids.clone()
            labels[:, :masked] = -100
            batch = dict(input_ids=ids, labels=labels)
            if os.environ.get("RM_PACKED") == "1":
                split = 3
                batch["position_ids"] = torch.cat(
                    (
                        torch.arange(split, device=rank),
                        torch.arange(n - split, device=rank),
                    )
                ).unsqueeze(0)
                labels[:, split] = -100
                batch["attention_mask"] = (
                    torch.cat(
                        (
                            torch.ones(split, device=rank),
                            torch.full((n - split,), 2, device=rank),
                        )
                    )
                    .unsqueeze(0)
                    .long()
                )
            batches.append(batch)

        def reference_batch(batch):
            if os.environ.get("RM_PACKED") != "1":
                return batch
            batch = dict(batch)
            segments = batch["attention_mask"]
            n = segments.shape[1]
            causal = torch.ones(n, n, device=segments.device, dtype=torch.bool).tril()
            batch["attention_mask"] = (segments[:, :, None] == segments[:, None, :])[
                :, None
            ] & causal
            return batch

        def reference_forward(batch, num_items=None, reference=reference):
            if (
                os.environ.get("RM_MODEL") not in ("mamba", "mamba2")
                or os.environ.get("RM_PACKED") != "1"
            ):
                return reference(**reference_batch(batch), num_items_in_batch=num_items)
            if num_items is None:
                num_items = (batch["labels"][:, 1:] != -100).sum()
            losses = []
            for start, end in ((0, 3), (3, batch["input_ids"].shape[1])):
                losses.append(
                    reference(
                        input_ids=batch["input_ids"][:, start:end],
                        labels=batch["labels"][:, start:end],
                        num_items_in_batch=num_items,
                    ).loss
                )
            return SimpleNamespace(loss=sum(losses))

        count = trainer._get_num_items_in_batch(batches, torch.device("cuda", rank))
        total = sum((b["labels"][:, 1:] != -100).sum() for b in batches)
        reference.train()
        losses = []
        for b in batches:
            loss = reference_forward(b, total).loss
            loss.backward()
            losses.append(trainer.training_step(trainer.model, dict(b), count))
        loss_sum = torch.stack(losses).sum()
        dist.all_reduce(loss_sum)
        loss_sum /= dist.get_world_size()
        with torch.no_grad():
            expected_loss = sum(reference_forward(b, total).loss for b in batches)
        torch.testing.assert_close(
            loss_sum,
            expected_loss,
            atol=0.03 if inner != "sdpa" or os.environ.get("RM_HUB") == "1" else 1e-6,
            rtol=0.01 if inner != "sdpa" or os.environ.get("RM_HUB") == "1" else 1e-5,
        )
        worst = 0
        for (name, p), (_, r) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            grad = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
            err = (grad - r.grad).abs().max().item()
            worst = max(worst, err)
            if os.environ.get("RM_HUB") == "1":
                # BF16 reductions change accumulation order, especially near cancellation.
                tolerance = 2 * torch.finfo(grad.dtype).eps
                difference = grad.float() - r.grad.float()
                assert difference.norm() <= tolerance * r.grad.float().norm(), name
                torch.testing.assert_close(
                    grad,
                    r.grad,
                    atol=float(tolerance * r.grad.abs().max()),
                    rtol=0,
                    msg=lambda message, name=name: f"{name}: {message}",
                )
            else:
                torch.testing.assert_close(
                    grad,
                    r.grad,
                    atol=2e-3 if inner != "sdpa" else 3e-6,
                    rtol=2e-2 if inner != "sdpa" else 3e-4,
                    msg=lambda message, name=name: f"{name}: {message}",
                )

        print(
            f"PASS rank={rank} backend={backend} average={average} gradient_max_error={worst}",
            flush=True,
        )
        trainer.model.eval()
        reference.eval()
        with torch.no_grad():
            eval_count = trainer._get_num_items_in_batch(
                [batches[0]], torch.device("cuda", rank)
            )
            eval_loss = trainer.compute_loss(
                trainer.model, dict(batches[0]), num_items_in_batch=eval_count
            )
            dist.all_reduce(eval_loss)
            eval_loss /= dist.get_world_size()
            expected_eval = reference_forward(batches[0]).loss
            torch.testing.assert_close(
                eval_loss,
                expected_eval,
                atol=0.03
                if inner != "sdpa" or os.environ.get("RM_HUB") == "1"
                else 1e-6,
                rtol=0.01
                if inner != "sdpa" or os.environ.get("RM_HUB") == "1"
                else 1e-5,
            )
        plugin.post_train_unload(cfg)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
