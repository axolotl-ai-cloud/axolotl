"""
Trainer mixin for layer-wise parameter offloading to CPU.

Offloads frozen (non-trainable) parameters in decoder layers to CPU, then uses
forward/backward hooks to stream them on/off GPU one layer at a time with CUDA
stream prefetching. Trainable parameters (e.g. LoRA weights) stay on GPU always.

Forward:  pre-hook loads layer N's frozen params to GPU (prefetches N+1 on
          transfer stream), post-hook offloads layer N-1's frozen params.
Backward: same in reverse order.
"""

import contextlib

import torch
import torch.nn as nn
from transformers import Trainer

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _find_decoder_layers(model: nn.Module) -> tuple[nn.ModuleList | None, list[str]]:
    """Recursively search the model for the decoder layer ModuleList.

    Finds any ModuleList whose children have 'DecoderLayer' in their class name.
    Handles all common HF architectures including VLM wrappers (e.g. Qwen3.5-MoE
    where layers are at model.language_model.layers).
    """
    # BFS to find the first ModuleList containing decoder layers
    queue = [model]
    while queue:
        m = queue.pop(0)
        for _name, child in m.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                first_type = type(child[0]).__name__
                if "DecoderLayer" in first_type or "TransformerBlock" in first_type:
                    layer_types = list({type(layer).__name__ for layer in child})
                    return child, layer_types
            else:
                queue.append(child)

    return None, []


def _get_frozen_params(layer: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Get all non-trainable parameters in a layer."""
    return [(n, p) for n, p in layer.named_parameters() if not p.requires_grad]


class LayerOffloadManager:
    """Manages offloading frozen decoder layer params to CPU and streaming
    them back during forward/backward with CUDA stream overlap.

    Only frozen (requires_grad=False) parameters are offloaded.
    Trainable parameters (LoRA weights, etc.) remain on GPU at all times.
    """

    def __init__(
        self,
        model: nn.Module,
        num_prefetch: int = 1,
    ):
        self.model = model
        self.num_prefetch = num_prefetch
        self._hooks: list = []
        self._device = None

        # Find decoder layers
        self.layers, layer_types = _find_decoder_layers(model)
        if self.layers is None:
            LOG.warning(
                "LayerOffloadManager: no decoder layers found, offloading disabled"
            )
            self.enabled = False
            return

        self.enabled = True
        self.n_layers = len(self.layers)
        LOG.info(
            f"Layer offloading: found {self.n_layers} layers ({', '.join(layer_types)})"
        )

        # Determine GPU device
        for p in model.parameters():
            if p.device.type == "cuda":
                self._device = p.device
                break
        if self._device is None:
            LOG.warning("LayerOffloadManager: no CUDA parameters found")
            self.enabled = False
            return

        # Transfer stream for async prefetch
        self._transfer_stream = torch.cuda.Stream(device=self._device)

        # Track which layers have their frozen params on GPU
        self._on_gpu: set[int] = set(range(self.n_layers))

        # Cache: frozen param references per layer (list of (name, param) tuples)
        self._frozen_params: list[list[tuple[str, nn.Parameter]]] = [
            _get_frozen_params(self.layers[i]) for i in range(self.n_layers)
        ]

        # CPU storage: pinned tensors for each layer's frozen params
        # Populated on first offload
        self._cpu_data: list[dict[str, torch.Tensor]] = [
            {} for _ in range(self.n_layers)
        ]

        # Offload all layers upfront
        self._offload_all()

        # Release cached memory blocks back to the driver
        torch.cuda.empty_cache()

    def _offload_all(self):
        """Move all frozen params in all decoder layers to CPU."""
        mem_before = torch.cuda.memory_allocated(self._device)
        for i in range(self.n_layers):
            self._offload_layer(i)
        mem_after = torch.cuda.memory_allocated(self._device)
        freed = (mem_before - mem_after) / 1e6
        LOG.info(
            f"Layer offloading: offloaded frozen params from {self.n_layers} layers, "
            f"freed {freed:.0f} MB GPU memory"
        )

    def _offload_layer(self, idx: int):
        """Move frozen params of layer idx to CPU pinned memory."""
        if idx not in self._on_gpu:
            return
        for name, param in self._frozen_params[idx]:
            if param.device.type != "cuda":
                continue
            # Allocate pinned CPU tensor on first offload
            if name not in self._cpu_data[idx]:
                self._cpu_data[idx][name] = torch.empty_like(
                    param.data, device="cpu", pin_memory=True
                )
            cpu_buf = self._cpu_data[idx][name]
            # Async copy GPU -> CPU (on transfer stream for overlap)
            cpu_buf.copy_(param.data, non_blocking=True)
            # Point parameter at a dummy CPU tensor to free GPU memory
            param.data = cpu_buf
        self._on_gpu.discard(idx)

    def _load_layer(self, idx: int, stream=None):
        """Move frozen params of layer idx back to GPU."""
        if idx in self._on_gpu or idx < 0 or idx >= self.n_layers:
            return
        ctx = (
            torch.cuda.stream(stream)
            if stream is not None
            else contextlib.nullcontext()
        )
        with ctx:
            for _name, param in self._frozen_params[idx]:
                if param.device.type == "cuda":
                    continue
                gpu_data = param.data.to(self._device, non_blocking=True)
                param.data = gpu_data
        self._on_gpu.add(idx)

    def _prefetch_layer(self, idx: int):
        """Async prefetch layer idx on the transfer stream."""
        if idx in self._on_gpu or idx < 0 or idx >= self.n_layers:
            return
        self._transfer_stream.wait_stream(torch.cuda.default_stream(self._device))
        self._load_layer(idx, stream=self._transfer_stream)

    def _wait_transfer(self):
        """Make default stream wait for any in-flight transfers."""
        torch.cuda.default_stream(self._device).wait_stream(self._transfer_stream)

    def setup_hooks(self):
        """Register forward and backward hooks on each decoder layer."""
        if not self.enabled:
            return

        for idx in range(self.n_layers):
            layer = self.layers[idx]

            def make_pre_fwd(i):
                def hook(module, args):
                    # Ensure this layer is on GPU
                    if i not in self._on_gpu:
                        self._load_layer(i)
                    self._wait_transfer()
                    # Prefetch next layer(s)
                    for offset in range(1, self.num_prefetch + 1):
                        self._prefetch_layer(i + offset)

                return hook

            def make_post_fwd(i):
                def hook(module, args, output):
                    # Offload previous layer (no longer needed in forward)
                    if i > 0:
                        self._offload_layer(i - 1)
                    # Offload last layer after forward
                    if i == self.n_layers - 1:
                        self._offload_layer(i)

                return hook

            def make_pre_bwd(i):
                def hook(module, grad_output):
                    # Load this layer for backward
                    if i not in self._on_gpu:
                        self._load_layer(i)
                    self._wait_transfer()
                    # Prefetch previous layer(s)
                    for offset in range(1, self.num_prefetch + 1):
                        self._prefetch_layer(i - offset)

                return hook

            def make_post_bwd(i):
                def hook(module, grad_input, grad_output):
                    # Offload the layer above
                    if i < self.n_layers - 1:
                        self._offload_layer(i + 1)
                    # Offload first layer after backward
                    if i == 0:
                        self._offload_layer(i)

                return hook

            h1 = layer.register_forward_pre_hook(make_pre_fwd(idx))
            h2 = layer.register_forward_hook(make_post_fwd(idx))
            h3 = layer.register_full_backward_pre_hook(make_pre_bwd(idx))
            h4 = layer.register_full_backward_hook(make_post_bwd(idx))
            self._hooks.extend([h1, h2, h3, h4])

    def remove_hooks(self):
        """Remove all hooks and restore layers to GPU."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self.enabled:
            for i in range(self.n_layers):
                if i not in self._on_gpu:
                    self._load_layer(i)

    def pre_step(self):
        """Called before each training step — ensure layers start offloaded."""
        if not self.enabled:
            return
        for i in list(self._on_gpu):
            self._offload_layer(i)
        # Prefetch layer 0 for forward
        self._prefetch_layer(0)

    def post_step(self):
        """Called after each training step — ensure layers are offloaded."""
        if not self.enabled:
            return
        for i in list(self._on_gpu):
            self._offload_layer(i)
        # Prefetch layer 0 for next step
        self._prefetch_layer(0)


class _LayerOffloadContext:
    """Context manager wrapping pre_step / post_step around a training step."""

    def __init__(self, manager: LayerOffloadManager):
        self.manager = manager

    def __enter__(self):
        self.manager.pre_step()
        return self

    def __exit__(self, *args):
        self.manager.post_step()


class LayerOffloadingMixin(Trainer):
    """
    Trainer mixin class for layer-wise parameter offloading to CPU.

    Offloads frozen decoder layer params to CPU at init, then streams them
    on/off GPU one layer at a time during each training step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.args, "layer_offloading", False):
            LOG.info("Layer parameter offloading enabled")
            self._layer_offload_manager = LayerOffloadManager(
                model=self.model,
                num_prefetch=1,
            )
            self._layer_offload_manager.setup_hooks()
            self._layer_offload_ctx = _LayerOffloadContext(self._layer_offload_manager)
        else:
            self._layer_offload_manager = None
            self._layer_offload_ctx = contextlib.nullcontext()

    def training_step(self, *args, **kwargs):
        with self._layer_offload_ctx:
            return super().training_step(*args, **kwargs)
