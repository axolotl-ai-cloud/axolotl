"""FSDP2 x TP on transformers >= 5.17.

accelerate's ``_prepare_tp`` (FSDP2 branch) wraps every parameter outside the model's
``tp_plan`` as a replicated DTensor on the TP mesh and asks
``transformers.integrations.tensor_parallel.ReplicateParallel().prepare_module_tp`` to make
the owning module accept it. transformers 5.17 dropped that class and method (its TP layers
now install a forward wrapper and keep activations local between modules), so the import
fails and FSDP2 x TP is unusable. This provides the missing class with the 5.17 mechanics:
the module runs its forward on the local view of its replicated parameters.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _build_replicate_parallel():
    from transformers.distributed.tensor_parallel import TensorParallelLayer

    class ReplicateParallel(TensorParallelLayer):
        """Replicated parameters, local activations: the forward sees plain tensors."""

        def should_use_local_tensors(self, module):
            return True

        def prepare_module_tp(self, module, device_mesh):
            # a module already carrying a plan style (transformers binds its wrapper as an
            # instance attribute) handles its own DTensors; a second wrapper would localise
            # its sharded weight under the style's DTensor inputs
            if "forward" in vars(module):
                return module
            self.install_forward(module, device_mesh)
            return module

    return ReplicateParallel


def _patch_replicated_grad_all_reduce() -> None:
    """Sum the partial grads of replicated-with-all-reduce params (q_norm/k_norm) per backward.

    transformers all-reduces ``param.grad`` from a module backward hook, i.e. the ACCUMULATED
    grad: under gradient accumulation (and FSDP2's deferred reduction) every earlier
    micro-step is summed again. A tensor hook on the parameter sees only this backward's
    incoming grad, and accepts FSDP2's DTensor grads.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from transformers.distributed.tensor_parallel import ReplicatedWithGradAllReduce

    if getattr(ReplicatedWithGradAllReduce, "_axolotl_dtensor_grads", False):
        return

    def install_forward(self, module, mesh):
        group = mesh.get_group()

        def _all_reduce(grad):
            local = grad.to_local() if isinstance(grad, DTensor) else grad
            local = local.clone()
            dist.all_reduce(local, group=group)
            if isinstance(grad, DTensor):
                return DTensor.from_local(
                    local, grad.device_mesh, grad.placements, run_check=False
                )
            return local

        # FSDP2 swaps in a fresh unsharded parameter per unshard, so hook whatever
        # parameter object the forward is about to use (once per object)
        def _hook_params(mod, args):
            for param in mod.parameters(recurse=False):
                if param.requires_grad and not getattr(
                    param, "_axolotl_tp_grad_hook", False
                ):
                    param.register_hook(_all_reduce)
                    param._axolotl_tp_grad_hook = True

        module.register_forward_pre_hook(_hook_params)
        return module

    ReplicatedWithGradAllReduce.install_forward = install_forward
    ReplicatedWithGradAllReduce._axolotl_dtensor_grads = True


def patch_accelerate_prepare_tp() -> bool:
    """Give accelerate the ``ReplicateParallel`` it imports; returns True if installed."""
    import transformers.integrations.tensor_parallel as compat

    _patch_replicated_grad_all_reduce()
    if getattr(compat, "ReplicateParallel", None) is not None:
        return False
    compat.ReplicateParallel = _build_replicate_parallel()
    LOG.debug("Installed ReplicateParallel shim for accelerate FSDP2 x TP.")
    return True


def replicate_plain_params_for_tp(model, tp_mesh) -> int:
    """Turn every parameter outside the ``tp_plan`` into a replicated DTensor on ``tp_mesh``.

    accelerate does this for the FSDP2 branch of ``_prepare_tp`` only; without FSDP the model
    keeps a mix of plain and DTensor parameters, which the foreach optimizers reject. The
    owning module runs on the local view (the ``ReplicateParallel`` shim). Returns the count.
    """
    import torch
    from torch.distributed.tensor import DTensor, Replicate

    patch_accelerate_prepare_tp()
    from transformers.integrations.tensor_parallel import ReplicateParallel

    style = ReplicateParallel()
    n = 0
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, DTensor):
                continue
            style.prepare_module_tp(module, tp_mesh)
            replicated = DTensor.from_local(
                param.data,
                device_mesh=tp_mesh,
                placements=[Replicate()],
                run_check=False,
            )
            setattr(
                module,
                name,
                torch.nn.Parameter(replicated, requires_grad=param.requires_grad),
            )
            n += 1
    if n:
        LOG.debug(f"Replicated {n} parameters outside the tp_plan on the TP mesh.")
    return n
