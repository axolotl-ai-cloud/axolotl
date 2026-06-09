"""Experimental single-GPU NVFP4 training loop for CUDA graph probes."""

from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import dataclass, field
from itertools import cycle
from typing import Any

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from axolotl.common.datasets import load_datasets
from axolotl.loaders import load_tokenizer
from axolotl.train import setup_model_and_tokenizer
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
)
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

LOG = get_logger(__name__)


@dataclass
class GraphLoopOptions:
    mode: str = "auto"
    steps: int = 20
    warmup_steps: int = 5
    capture_warmup_steps: int = 3
    reuse_static_batch: bool = False
    compile_model: bool | None = None
    fullgraph: bool = False
    probe_on_fail: bool = True
    probe_only: bool = False


@dataclass
class GraphProbe:
    name: str
    ok: bool
    reason: str = ""


@dataclass
class GraphLoopResult:
    mode: str
    graph_requested: bool
    graph_captured: bool
    graph_reason: str
    steps: int
    warmup_steps: int
    batch_shape: dict[str, tuple[int, ...]]
    input_tokens_per_step: int
    supervised_tokens_per_step: int
    loss_first: float | None = None
    loss_last: float | None = None
    loss_min: float | None = None
    loss_max: float | None = None
    median_ms: float | None = None
    mean_ms: float | None = None
    tokens_per_second: float | None = None
    active_gib: float | None = None
    notes: list[str] = field(default_factory=list)
    probes: list[GraphProbe] = field(default_factory=list)


class StaticShapeError(RuntimeError):
    pass


def validate_loop_cfg(cfg):
    if cfg.rl:
        raise ValueError("nvfp4 cuda graph loop only supports SFT configs")
    if cfg.fsdp or cfg.fsdp_config or cfg.deepspeed:
        raise ValueError("nvfp4 cuda graph loop is single-GPU only")
    if cfg.gradient_accumulation_steps not in (None, 1):
        raise ValueError("gradient_accumulation_steps must be 1 for this prototype")
    if cfg.adapter not in ("lora", "qlora"):
        raise ValueError("expected a LoRA/QLoRA-style adapter config")
    nvfp4 = getattr(cfg, "nvfp4_training", None)
    if not (nvfp4 and nvfp4.enabled):
        raise ValueError("nvfp4_training.enabled must be true")
    wants_fp4_base = bool(
        cfg.adapter == "qlora"
        or nvfp4.quantize_base
        or getattr(nvfp4, "base_mode", None) in ("storage", "compute")
    )
    if not wants_fp4_base:
        raise ValueError("expected NVFP4 base quantization for the QLoRA path")
    if not cfg.bf16:
        raise ValueError("NVFP4 training requires bf16")


def maybe_compile_model(model: torch.nn.Module, cfg, options: GraphLoopOptions):
    compile_model = (
        cfg.torch_compile if options.compile_model is None else options.compile_model
    )
    if not compile_model:
        return model, "compile=off"

    if getattr(torch, "_dynamo", None):
        torch._dynamo.config.suppress_errors = False
        torch._dynamo.config.accumulated_cache_size_limit = 256
        if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
            torch._dynamo.config.capture_scalar_outputs = True
        if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
            torch._dynamo.config.allow_unspec_int_on_nn_module = True

    backend = cfg.torch_compile_backend or "inductor"
    kwargs: dict[str, Any] = {"backend": backend, "fullgraph": options.fullgraph}
    if cfg.torch_compile_mode:
        kwargs["mode"] = cfg.torch_compile_mode
    return torch.compile(model, **kwargs), f"compile={backend}"


def _collator_kwargs(cfg) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"padding": True, "return_tensors": "pt"}
    multiple = getattr(cfg, "pad_to_multiple_of", None) or 64
    if cfg.pad_to_sequence_len:
        kwargs["pad_to_multiple_of"] = multiple * (
            (cfg.sequence_len + multiple - 1) // multiple
        )
    elif cfg.pad_to_sequence_len is None:
        kwargs["pad_to_multiple_of"] = multiple
    return kwargs


def build_dataloader(cfg, train_dataset, tokenizer) -> DataLoader:
    kwargs = _collator_kwargs(cfg)
    num_workers = int(cfg.dataloader_num_workers or 0)
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataloader_pin_memory),
        "persistent_workers": bool(cfg.dataloader_persistent_workers)
        if num_workers
        else False,
    }
    if num_workers and cfg.dataloader_prefetch_factor:
        loader_kwargs["prefetch_factor"] = cfg.dataloader_prefetch_factor

    if cfg.sample_packing and not cfg.pretraining_dataset:
        if cfg.curriculum_sampling:
            base_sampler = SequentialSampler(train_dataset)
        else:
            base_sampler = RandomSampler(train_dataset)
        if cfg.multipack_real_batches:
            batch_size = cfg.micro_batch_size
            batch_max_len = cfg.sequence_len
        else:
            batch_size = 1
            batch_max_len = cfg.micro_batch_size * cfg.sequence_len
        sampler = MultipackBatchSampler(
            base_sampler,
            lengths=get_dataset_lengths(train_dataset),
            packing_efficiency_estimate=cfg.sample_packing_eff_est,
            batch_max_len=batch_max_len,
            batch_size=batch_size,
            group_size=cfg.sample_packing_group_size,
            bin_size=cfg.sample_packing_bin_size,
            sequential=cfg.sample_packing_sequentially,
            drop_last=True,
            num_processes=cfg.dataset_num_proc,
            mp_start_method=cfg.sample_packing_mp_start_method or "fork",
        )
        len(sampler)
        dataset = train_dataset
        if getattr(dataset, "column_names", None) and "length" in dataset.column_names:
            dataset = dataset.remove_columns(["length"])
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=BatchSamplerDataCollatorForSeq2Seq(tokenizer, **kwargs),
            **loader_kwargs,
        )

    dataset = train_dataset
    if getattr(dataset, "column_names", None) and "length" in dataset.column_names:
        dataset = dataset.remove_columns(["length"])
    return DataLoader(
        dataset,
        batch_size=cfg.micro_batch_size,
        sampler=RandomSampler(dataset),
        drop_last=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, **kwargs),
        **loader_kwargs,
    )


def build_optimizer(
    model: torch.nn.Module, cfg, capturable: bool
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    optimizer_name = str(cfg.optimizer or "adamw_torch_fused")
    fused = "fused" in optimizer_name
    return torch.optim.AdamW(
        params,
        lr=float(cfg.learning_rate),
        betas=(float(cfg.adam_beta1 or 0.9), float(cfg.adam_beta2 or 0.999)),
        eps=float(cfg.adam_epsilon or 1e-8),
        weight_decay=float(cfg.weight_decay or 0.0),
        fused=fused,
        capturable=capturable,
    )


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda", torch.cuda.current_device())


def _tensor_batch(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in batch.items() if isinstance(value, torch.Tensor)
    }


def _trim_batch_for_model(
    batch: dict[str, torch.Tensor], cfg
) -> dict[str, torch.Tensor]:
    out = dict(batch)
    out.pop("length", None)
    if (
        cfg.sample_packing
        and cfg.sample_packing_drop_attention_mask
        and "position_ids" in out
        and "attention_mask" in out
    ):
        del out["attention_mask"]
    return out


def _new_static_batch(cpu_batch: dict[str, Any], cfg, device: torch.device):
    tensor_batch = _trim_batch_for_model(_tensor_batch(cpu_batch), cfg)
    return {
        key: tensor.to(device=device, non_blocking=True).contiguous()
        for key, tensor in tensor_batch.items()
    }


def _copy_to_static(
    static_batch: dict[str, torch.Tensor],
    cpu_batch: dict[str, Any],
    cfg,
):
    tensor_batch = _trim_batch_for_model(_tensor_batch(cpu_batch), cfg)
    for key, dst in static_batch.items():
        src = tensor_batch.get(key)
        if src is None:
            raise StaticShapeError(f"next batch is missing tensor key {key!r}")
        if tuple(src.shape) != tuple(dst.shape):
            raise StaticShapeError(
                f"static shape mismatch for {key}: first={tuple(dst.shape)} "
                f"next={tuple(src.shape)}"
            )
        dst.copy_(src, non_blocking=True)


def _batch_shape(batch: dict[str, torch.Tensor]) -> dict[str, tuple[int, ...]]:
    return {key: tuple(tensor.shape) for key, tensor in batch.items()}


def _loss_from_output(output):
    if isinstance(output, dict):
        loss = output["loss"]
    else:
        loss = output.loss if hasattr(output, "loss") else output[0]
    return loss.mean() if loss.ndim else loss


def _forward_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]):
    return _loss_from_output(model(**batch))


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
):
    optimizer.zero_grad(set_to_none=False)
    loss = _forward_loss(model, batch)
    loss.backward()
    optimizer.step()
    return loss


def _finite_loss(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _supervised_tokens(batch: dict[str, torch.Tensor]) -> int:
    labels = batch.get("labels")
    if labels is None:
        return 0
    return int((labels != -100).sum().item())


def _recorded_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _warm_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    static_batch: dict[str, torch.Tensor],
    steps: int,
):
    for _ in range(max(1, steps)):
        loss = _train_step(model, optimizer, static_batch)
        if not _finite_loss(float(loss.detach().float().item())):
            raise RuntimeError("non-finite loss during graph warmup")
    optimizer.zero_grad(set_to_none=False)
    torch.cuda.synchronize()


def _capture_graph(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    static_batch: dict[str, torch.Tensor],
    capture_warmup_steps: int,
):
    _warm_optimizer_state(model, optimizer, static_batch, capture_warmup_steps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss = _train_step(model, optimizer, static_batch)
    torch.cuda.synchronize()
    return graph, static_loss


def _probe_capture(name: str, body, warmup=None) -> GraphProbe:
    try:
        torch.cuda.synchronize()
        if warmup is not None:
            warmup()
            torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        graph.replay()
        torch.cuda.synchronize()
        return GraphProbe(name=name, ok=True)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        try:
            torch.cuda.synchronize()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        return GraphProbe(name=name, ok=False, reason=f"{type(exc).__name__}: {exc}")


def graph_stage_probes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    static_batch: dict[str, torch.Tensor],
) -> list[GraphProbe]:
    probes: list[GraphProbe] = []

    probes.append(
        _probe_capture(
            "forward_loss",
            lambda: _forward_loss(model, static_batch),
            warmup=lambda: _forward_loss(model, static_batch),
        )
    )

    def backward_body():
        optimizer.zero_grad(set_to_none=False)
        _forward_loss(model, static_batch).backward()

    probes.append(
        _probe_capture(
            "forward_backward",
            backward_body,
            warmup=backward_body,
        )
    )

    def optimizer_warmup():
        optimizer.zero_grad(set_to_none=False)
        _forward_loss(model, static_batch).backward()

    def optimizer_body():
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

    probes.append(
        _probe_capture(
            "optimizer_step",
            optimizer_body,
            warmup=optimizer_warmup,
        )
    )
    return probes


def _measure_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    cfg,
    options: GraphLoopOptions,
    static_batch: dict[str, torch.Tensor],
    first_cpu_batch: dict[str, Any],
    graph: torch.cuda.CUDAGraph | None = None,
    graph_loss: torch.Tensor | None = None,
) -> tuple[list[float], list[float]]:
    iterator = cycle(dataloader)

    losses: list[float] = []
    timings: list[float] = []
    total_steps = options.warmup_steps + options.steps

    for step_idx in range(total_steps):
        cpu_batch = first_cpu_batch if options.reuse_static_batch else next(iterator)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if not options.reuse_static_batch:
            _copy_to_static(static_batch, cpu_batch, cfg)
        if graph is None:
            loss = _train_step(model, optimizer, static_batch)
        else:
            graph.replay()
            loss = graph_loss
        end.record()
        elapsed = _recorded_ms(start, end)
        loss_value = float(loss.detach().float().item())
        if not _finite_loss(loss_value):
            raise RuntimeError(f"non-finite loss at step {step_idx}: {loss_value}")
        if step_idx >= options.warmup_steps:
            losses.append(loss_value)
            timings.append(elapsed)

    return losses, timings


def run_loop(cfg, options: GraphLoopOptions) -> GraphLoopResult:
    validate_loop_cfg(cfg)
    torch.manual_seed(int(cfg.seed or 42))
    torch.cuda.set_device(0)

    notes = [
        "CUDA event timing includes static H2D copies plus fwd/bwd/optimizer; "
        "it excludes DataLoader CPU time, logging, evaluation, and checkpointing.",
        "No LR scheduler is applied in this prototype.",
    ]
    if cfg.sample_packing:
        notes.append(
            "DataLoader uses Axolotl MultipackBatchSampler and packed collator."
        )
    if options.reuse_static_batch:
        notes.append("Reuses one static batch; this is capture feasibility only.")

    tokenizer = load_tokenizer(cfg)
    dataset_meta = load_datasets(cfg=cfg)
    model, _, _, _ = setup_model_and_tokenizer(cfg)
    model.train()
    model, compile_note = maybe_compile_model(model, cfg, options)
    notes.append(compile_note)

    dataloader = build_dataloader(cfg, dataset_meta.train_dataset, tokenizer)
    first_batch = next(iter(dataloader))
    static_batch = _new_static_batch(first_batch, cfg, _model_device(model))
    batch_shape = _batch_shape(static_batch)
    input_tokens = int(static_batch["input_ids"].numel())
    supervised_tokens = _supervised_tokens(static_batch)

    graph_requested = options.mode in ("graph", "auto")
    graph = None
    graph_loss = None
    graph_captured = False
    graph_reason = "not requested"
    optimizer = build_optimizer(model, cfg, capturable=graph_requested)
    probes: list[GraphProbe] = []

    if options.probe_only:
        probes = graph_stage_probes(model, optimizer, static_batch)
        first_blocked = next((probe for probe in probes if not probe.ok), None)
        graph_reason = (
            f"{first_blocked.name} blocked: {first_blocked.reason}"
            if first_blocked
            else "all stage probes captured"
        )
        return GraphLoopResult(
            mode="probe",
            graph_requested=True,
            graph_captured=False,
            graph_reason=graph_reason,
            steps=0,
            warmup_steps=0,
            batch_shape=batch_shape,
            input_tokens_per_step=input_tokens,
            supervised_tokens_per_step=supervised_tokens,
            active_gib=torch.cuda.memory_stats().get("active_bytes.all.peak", 0)
            / 1024.0**3,
            notes=notes,
            probes=probes,
        )

    if graph_requested:
        try:
            graph, graph_loss = _capture_graph(
                model,
                optimizer,
                static_batch,
                options.capture_warmup_steps,
            )
            graph_captured = True
            graph_reason = "captured full train step"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            graph_reason = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__, limit=8)
            ).strip()
            LOG.warning("CUDA graph capture failed: %s", graph_reason.splitlines()[-1])
            probes = (
                graph_stage_probes(model, optimizer, static_batch)
                if options.probe_on_fail
                else []
            )
            if options.mode == "graph":
                return GraphLoopResult(
                    mode="graph",
                    graph_requested=True,
                    graph_captured=False,
                    graph_reason=graph_reason,
                    steps=0,
                    warmup_steps=0,
                    batch_shape=batch_shape,
                    input_tokens_per_step=input_tokens,
                    supervised_tokens_per_step=supervised_tokens,
                    notes=notes,
                    probes=probes,
                )
            notes.append(
                "Graph capture failed; auto mode fell back to eager static loop."
            )
            optimizer = build_optimizer(model, cfg, capturable=False)
    else:
        probes = []

    started = time.perf_counter()
    losses, timings = _measure_loop(
        model,
        optimizer,
        dataloader,
        cfg,
        options,
        static_batch=static_batch,
        first_cpu_batch=first_batch,
        graph=graph if graph_captured else None,
        graph_loss=graph_loss if graph_captured else None,
    )
    wall_s = time.perf_counter() - started

    active_gib = torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / 1024.0**3
    median_ms = statistics.median(timings) if timings else None
    mean_ms = statistics.fmean(timings) if timings else None
    tokens_per_second = (
        input_tokens / (median_ms / 1000.0) if median_ms and input_tokens else None
    )
    if wall_s and timings:
        notes.append(f"host wall for measured loop: {wall_s:.3f}s")

    mode = "graph" if graph_captured else "eager"
    return GraphLoopResult(
        mode=mode,
        graph_requested=graph_requested,
        graph_captured=graph_captured,
        graph_reason=graph_reason,
        steps=len(timings),
        warmup_steps=options.warmup_steps,
        batch_shape=_batch_shape(static_batch),
        input_tokens_per_step=input_tokens,
        supervised_tokens_per_step=supervised_tokens,
        loss_first=losses[0] if losses else None,
        loss_last=losses[-1] if losses else None,
        loss_min=min(losses) if losses else None,
        loss_max=max(losses) if losses else None,
        median_ms=median_ms,
        mean_ms=mean_ms,
        tokens_per_second=tokens_per_second,
        active_gib=active_gib,
        notes=notes,
        probes=probes,
    )


def format_result(result: GraphLoopResult) -> str:
    reason_lines = result.graph_reason.splitlines()
    lines = [
        f"mode={result.mode}",
        f"graph_requested={result.graph_requested}",
        f"graph_captured={result.graph_captured}",
        f"graph_reason={reason_lines[-1] if reason_lines else result.graph_reason}",
        f"steps={result.steps} warmup_steps={result.warmup_steps}",
        f"batch_shape={result.batch_shape}",
        f"input_tokens_per_step={result.input_tokens_per_step} "
        f"supervised_tokens_per_step={result.supervised_tokens_per_step}",
    ]
    if result.median_ms is not None:
        lines.append(
            f"cuda_ms_per_step median={result.median_ms:.3f} mean={result.mean_ms:.3f}"
        )
        lines.append(f"input_tokens_per_second median={result.tokens_per_second:.1f}")
    if result.loss_first is not None:
        lines.append(
            "loss "
            f"first={result.loss_first:.4f} last={result.loss_last:.4f} "
            f"min={result.loss_min:.4f} max={result.loss_max:.4f}"
        )
    if result.active_gib is not None:
        lines.append(f"max_active_gib={result.active_gib:.2f}")
    if result.probes:
        lines.append("graph_stage_probes:")
        for probe in result.probes:
            suffix = "ok" if probe.ok else f"blocked: {probe.reason}"
            lines.append(f"  {probe.name}: {suffix}")
    if result.notes:
        lines.append("notes:")
        lines.extend(f"  {note}" for note in result.notes)
    return "\n".join(lines)
