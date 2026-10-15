"""Bridge Ringmaster's sharding to upstream Trainer and Accelerate."""

import contextlib
from types import MethodType

from ringmaster import ContextParallelContextManager


def _cp_context(buffers=None, buffer_seq_dims=None, no_restore_buffers=None):
    return contextlib.nullcontext()


def _prepare_cp(self, *args):
    self._cp_context = _cp_context
    return args


def _prepare_inputs(self, model, inputs):
    return contextlib.nullcontext, inputs


def _num_items(self, batch_samples, device):
    if not batch_samples or not self.model_accepts_loss_kwargs:
        return None
    if "labels" not in batch_samples[0]:
        return None
    labels = [
        batch["shift_labels"]
        if "shift_labels" in batch
        else batch["labels"][..., 1:]
        if self._loss_shifts_labels
        else batch["labels"]
        for batch in batch_samples
    ]
    count = sum(label.ne(-100).sum() for label in labels).to(device)
    pc = self.accelerator.parallelism_config
    if self.args.average_tokens_across_devices:
        count = self.accelerator.gather(count).sum() / pc.non_data_parallel_size
    else:
        # CP gradients are averaged even when token averaging across DP is disabled.
        count = count / pc.cp_size
    return count.clamp_min(1e-8)


def configure_trainer(trainer, *, gather_outputs=False):
    """Install instance-local overrides, returning a function that restores them."""
    if not gather_outputs and (
        not trainer.model_accepts_loss_kwargs
        or trainer.compute_loss_func is not None
        or trainer.label_smoother is not None
    ):
        raise ValueError(
            "Ringmaster SFT requires a model accepting loss kwargs and the model's "
            "causal LM loss (no custom compute_loss_func or label smoothing)."
        )
    accelerator = trainer.accelerator
    originals = [
        (
            accelerator,
            "_cp_context",
            "_cp_context" in vars(accelerator),
            vars(accelerator).get("_cp_context"),
        )
    ]
    overrides = [
        (trainer.accelerator, "_prepare_cp", _prepare_cp),
        (trainer, "_prepare_context_parallel_inputs", _prepare_inputs),
    ]
    if not gather_outputs:
        overrides.append((trainer, "_get_num_items_in_batch", _num_items))
    for obj, name, method in overrides:
        originals.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, MethodType(method, obj))

    def restore():
        for obj, name, existed, value in reversed(originals):
            if existed:
                setattr(obj, name, value)
            elif name in vars(obj):
                delattr(obj, name)

    return restore


class TrainerContextParallelContextManager(ContextParallelContextManager):
    """Preserve Trainer's token count over the complete accumulation window."""

    def _make_pre_hook(self, forward_params):
        hook = super()._make_pre_hook(forward_params)

        def pre_hook(module, args, kwargs):
            kwargs = dict(kwargs)
            num_items = kwargs.pop("num_items_in_batch", None)
            args, kwargs = hook(module, args, kwargs)
            if num_items is not None:
                kwargs["num_items_in_batch"] = num_items
                # With an explicit denominator, Trainer owns train and eval reduction.
                self._local_valid = None
            return args, kwargs

        return pre_hook
