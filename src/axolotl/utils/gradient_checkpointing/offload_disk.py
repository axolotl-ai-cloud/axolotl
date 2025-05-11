"""Disk offloaded checkpointing"""

import os
import tempfile
import uuid

import torch

torch_cuda_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
torch_cuda_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


class DiskOffloadedGradientCheckpointer(torch.autograd.Function):
    """
    Saves both VRAM and RAM by offloading activations to disk.
    Greater hit to performance than RAM offloading, but useful for extremely memory-constrained environments.
    """

    # Create a temporary directory for storing tensors
    _temp_dir = tempfile.mkdtemp(prefix="disk_checkpoint_")

    @staticmethod
    def _get_temp_file_path():
        """Generate a unique file path for tensor storage"""
        return os.path.join(
            DiskOffloadedGradientCheckpointer._temp_dir, f"{uuid.uuid4()}.pt"
        )

    @staticmethod
    @torch_cuda_amp_custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        # Generate a unique file path for this tensor
        file_path = DiskOffloadedGradientCheckpointer._get_temp_file_path()

        # Save tensor to disk in a non-blocking way (detached from compute)
        # First move to CPU, then save
        cpu_hidden_states = hidden_states.detach().cpu()
        torch.save(cpu_hidden_states, file_path)

        # Free CPU memory
        del cpu_hidden_states

        # Run forward pass
        with torch.no_grad():
            output = forward_function(hidden_states, *args)

        # Store the path instead of the tensor
        ctx.save_for_backward(torch.tensor([0]))  # Dummy tensor
        ctx.file_path = file_path
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch_cuda_amp_custom_bwd
    def backward(ctx, dY):  # pylint: disable=invalid-name
        # Load the hidden states from disk
        hidden_states = torch.load(ctx.file_path, weights_only=True)

        # Move to CUDA and prepare for gradient computation
        hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
        hidden_states.requires_grad = True

        # Clean up the temporary file
        try:
            os.remove(ctx.file_path)
        except FileNotFoundError:
            pass  # Ignore errors in file deletion

        # Compute gradients
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
        # pylint: disable=duplicate-code
        torch.autograd.backward(output, dY)

        return (
            None,
            hidden_states.grad,
        ) + (
            None,
        ) * len(ctx.args)

    @staticmethod
    def cleanup():
        """Clean up the temporary directory when done"""
        import shutil

        try:
            shutil.rmtree(
                DiskOffloadedGradientCheckpointer._temp_dir
            )  # pylint: disable=protected-access
        except FileNotFoundError:
            pass
