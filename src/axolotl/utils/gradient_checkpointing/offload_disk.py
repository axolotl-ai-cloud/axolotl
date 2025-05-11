"""Disk offloaded checkpointing"""

import os
import queue
import tempfile
import threading
import time
import uuid
from collections import deque

import torch

torch_cuda_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
torch_cuda_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


class DiskOffloadManager:
    """
    Manages offloaded tensors and handles prefetching in a separate thread.
    """

    def __init__(self, prefetch_size=3, prefetch_to_gpu=False):
        self.temp_dir = tempfile.mkdtemp(prefix="disk_checkpoint_")
        self.tensor_paths = deque()  # Ordered history of tensor paths (FIFO)
        self.max_prefetch = prefetch_size
        self.prefetch_to_gpu = prefetch_to_gpu

        # Prefetch queue and cache
        self.prefetch_queue = queue.Queue()
        self.prefetch_cache = {}  # Maps file_path -> tensor

        # Start prefetch worker thread
        self.stop_event = threading.Event()
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        """Background thread that loads tensors from disk ahead of time"""
        while not self.stop_event.is_set():
            try:
                file_path = self.prefetch_queue.get(timeout=0.5)
                if file_path is None:
                    continue

                # Load tensor from disk and store in cache
                if file_path not in self.prefetch_cache:
                    try:
                        tensor = torch.load(file_path, weights_only=True)
                        if self.prefetch_to_gpu:
                            tensor = tensor.to("cuda", non_blocking=True)
                        self.prefetch_cache[file_path] = tensor
                    except FileNotFoundError as e:
                        print(f"Prefetch error for {file_path}: {e}")

                self.prefetch_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                continue

    def save_tensor(self, tensor):
        """Save tensor to disk and return file path"""
        file_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}.pt")
        cpu_tensor = tensor.detach().cpu()
        torch.save(cpu_tensor, file_path)

        # Add to history
        self.tensor_paths.append(file_path)
        return file_path

    def load_tensor(self, file_path, target_device="cuda"):
        """Load tensor from disk or prefetch cache"""
        # Check if tensor is already in cache
        if file_path in self.prefetch_cache:
            tensor = self.prefetch_cache[file_path]
            del self.prefetch_cache[file_path]

            # Ensure tensor is on correct device
            if target_device != "cpu" and tensor.device.type == "cpu":
                tensor = tensor.to(target_device, non_blocking=True)

            # Clean up file if possible
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

            return tensor

        # If not in cache, load directly
        tensor = torch.load(file_path, weights_only=True)
        if target_device != "cpu":
            tensor = tensor.to(target_device, non_blocking=True)

        # Clean up file if possible
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

        return tensor

    def trigger_prefetch(self, n=None):
        """Trigger prefetching of the next N tensors"""
        if n is None:
            n = self.max_prefetch

        # Select the n oldest tensors (those that will be needed first in FIFO)
        prefetch_paths = [p for p in self.tensor_paths if p not in self.prefetch_cache][
            :n
        ]

        # Add to prefetch queue
        for path in prefetch_paths:
            self.prefetch_queue.put(path)

    def cleanup(self):
        """Clean up all temp files and stop prefetch thread"""
        self.stop_event.set()
        self.prefetch_thread.join(timeout=2.0)

        # Clear cache and remove any remaining files
        self.prefetch_cache.clear()
        for path in self.tensor_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except FileNotFoundError:
                pass

        # Remove temp directory
        try:
            os.rmdir(self.temp_dir)
        except FileNotFoundError:
            pass


class DiskOffloadedGradientCheckpointer(torch.autograd.Function):
    """
    Advanced disk-based gradient checkpointer with prefetching.
    """

    # Shared manager instance across all checkpointing operations
    _manager = None

    @staticmethod
    def get_manager(prefetch_size=3, prefetch_to_gpu=True):
        """Get or create the offload manager"""
        if DiskOffloadedGradientCheckpointer._manager is None:
            DiskOffloadedGradientCheckpointer._manager = DiskOffloadManager(
                prefetch_size=prefetch_size, prefetch_to_gpu=prefetch_to_gpu
            )
        return DiskOffloadedGradientCheckpointer._manager

    @staticmethod
    @torch_cuda_amp_custom_fwd
    def forward(
        ctx,
        forward_function,
        hidden_states,
        *args,
        prefetch_size=3,
        prefetch_to_gpu=False,
    ):
        # Get or create the manager
        manager = DiskOffloadedGradientCheckpointer.get_manager(
            prefetch_size=prefetch_size, prefetch_to_gpu=prefetch_to_gpu
        )

        # Save tensor to disk
        file_path = manager.save_tensor(hidden_states)

        # Run forward pass
        with torch.no_grad():
            output = forward_function(hidden_states, *args)

        # Register a hook to trigger prefetching just before backward
        def pre_backward_hook(grad_output):
            manager.trigger_prefetch()
            return grad_output

        # Register the hook on the output tensor
        if isinstance(output, torch.Tensor):
            output.register_hook(pre_backward_hook)
        elif (
            isinstance(output, tuple)
            and len(output) > 0
            and isinstance(output[0], torch.Tensor)
        ):
            output[0].register_hook(pre_backward_hook)

        # Store what we need for backward
        ctx.save_for_backward(torch.tensor([0]))  # Dummy tensor
        ctx.file_path = file_path
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch_cuda_amp_custom_bwd
    def backward(ctx, dY):  # pylint: disable=invalid-name
        # Get the manager
        manager = DiskOffloadedGradientCheckpointer._manager

        # Load hidden states from disk or prefetch cache
        hidden_states = manager.load_tensor(ctx.file_path)
        hidden_states.requires_grad = True

        # Compute gradients
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
        torch.autograd.backward(output, dY)

        return (
            None,
            hidden_states.grad,
            None,
            None,
        ) + (
            None,
        ) * len(ctx.args)

    @staticmethod
    def cleanup():
        """Clean up the offload manager"""
        if DiskOffloadedGradientCheckpointer._manager is not None:
            DiskOffloadedGradientCheckpointer._manager.cleanup()
            DiskOffloadedGradientCheckpointer._manager = None
