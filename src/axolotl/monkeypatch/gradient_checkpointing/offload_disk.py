"""
DISCO - DIsk-based Storage and Checkpointing with Optimized prefetching
"""

# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import concurrent.futures
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from typing import Dict

import torch

from axolotl.utils.logging import get_logger

torch_cuda_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
torch_cuda_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")

# Setup logger
logger = get_logger(__name__)


class DiskOffloadManager:
    """
    Manages offloaded tensors and handles prefetching in a separate thread.
    Includes synchronization to prevent race conditions.
    """

    def __init__(
        self,
        prefetch_size: int = 3,
        prefetch_to_gpu: bool = True,
        save_workers: int = 4,
    ):
        """
        Args:
            prefetch_size: Maximum number of tensors to prefetch in the background.
            prefetch_to_gpu: Whether to prefetch tensors directly to GPU memory.
            save_workers: Maximum number of concurrent save operations.
        """
        self.temp_dir = tempfile.mkdtemp(prefix="disco_")

        # Track tensor paths and their status
        self.tensor_paths: deque = deque()  # Ordered history of tensor paths (LIFO)
        self.file_locks: Dict[
            str, threading.Lock
        ] = {}  # Maps file_path -> threading.Lock()
        # Maps file_path -> status ("saving", "ready", "prefetching", "loaded", "deleted")
        self.file_status: Dict[str, str] = {}

        self.max_prefetch = prefetch_size
        self.prefetch_to_gpu = prefetch_to_gpu

        # Thread synchronization
        self.manager_lock = threading.RLock()  # Used for thread-safe operations

        # Prefetch queue and cache
        self.prefetch_queue: queue.Queue = queue.Queue()
        self.prefetch_cache: Dict[str, torch.Tensor] = {}  # Maps file_path -> tensor

        # Save queue and thread pool
        self.save_queue: queue.Queue = queue.Queue()
        self.save_pool = concurrent.futures.ThreadPoolExecutor(max_workers=save_workers)
        self.save_futures: Dict[str, Future] = {}
        self.save_semaphore = threading.Semaphore(
            save_workers * 2
        )  # Limit concurrent save operations

        # Start prefetch worker thread
        self.stop_event = threading.Event()
        # start multiple threads for prefetching
        self.prefetch_worker_count = 2
        self.prefetch_workers = []
        for _ in range(self.prefetch_worker_count):
            worker = threading.Thread(target=self._prefetch_worker, daemon=True)
            worker.start()
            self.prefetch_workers.append(worker)

        # Start save worker thread
        self.save_worker = threading.Thread(target=self._save_worker, daemon=True)
        self.save_worker.start()
        self.idx = 0

        atexit.register(self.cleanup)

    def _save_worker(self):
        """Background thread that processes the save queue"""
        while not self.stop_event.is_set():
            try:
                save_item = self.save_queue.get(timeout=0.5)
                if save_item is None:
                    continue

                tensor, file_path = save_item

                # Submit the save task to the thread pool
                future = self.save_pool.submit(
                    self._save_tensor_to_disk, tensor, file_path
                )
                with self.manager_lock:
                    self.save_futures[file_path] = future

                self.save_queue.task_done()

            except queue.Empty:
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                continue

    def _save_tensor_to_disk(self, tensor: torch.Tensor, file_path: str):
        """Actually save the tensor to disk"""
        try:
            # Save tensor to disk
            cpu_tensor = tensor.detach().cpu()
            torch.save(cpu_tensor, file_path)
            del cpu_tensor

            with self.manager_lock:
                # Mark file as ready
                self.file_status[file_path] = "ready"

            # Release semaphore
            self.save_semaphore.release()

            return True
        except FileNotFoundError as e:
            logger.error(f"Error saving tensor to {file_path}: {e}")
            with self.manager_lock:
                self.file_status[file_path] = "error"

            # Release semaphore
            self.save_semaphore.release()

            return False

    def _prefetch_worker(self):
        """Background thread that loads tensors from disk ahead of time"""
        while not self.stop_event.is_set():
            try:
                file_path = self.prefetch_queue.get(timeout=0.5)
                if file_path is None:
                    continue

                # Check if file is available and not already in cache
                with self.manager_lock:
                    if (
                        file_path not in self.file_status
                        or self.file_status[file_path] == "deleted"
                    ):
                        self.prefetch_queue.task_done()
                    if file_path in self.prefetch_cache:
                        self.prefetch_queue.task_done()
                        continue

                    # If file is still being saved, wait for it
                    if (
                        self.file_status[file_path] == "saving"
                        and file_path in self.save_futures
                    ):
                        # Re-queue this prefetch request with a little delay
                        self.prefetch_queue.task_done()
                        time.sleep(0.1)
                        self.prefetch_queue.put(file_path)
                        continue

                    # Mark file as being prefetched
                    self.file_status[file_path] = "prefetching"

                # Load tensor from disk and store in cache
                try:
                    if os.path.exists(file_path):
                        if self.prefetch_to_gpu:
                            tensor = torch.load(
                                file_path,
                                map_location=torch.device("cuda"),
                                weights_only=True,
                            )
                        else:
                            tensor = torch.load(file_path, weights_only=True)

                        with self.manager_lock:
                            self.prefetch_cache[file_path] = tensor
                            self.file_status[file_path] = "ready"
                    else:
                        with self.manager_lock:
                            if self.file_status.get(file_path) != "deleted":
                                logger.warning(
                                    f"Prefetch error: File not found {file_path}"
                                )
                                self.file_status[file_path] = "missing"

                except FileNotFoundError as e:
                    with self.manager_lock:
                        if self.file_status.get(file_path) != "deleted":
                            logger.warning(f"Prefetch error for {file_path}: {e}")
                            self.file_status[file_path] = "error"

                self.prefetch_queue.task_done()

            except queue.Empty:
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                continue

    def save_tensor(self, tensor: torch.Tensor):
        """Save tensor to disk asynchronously and return file path with thread-safe operations"""
        # Generate unique file path
        self.idx += 1
        file_path: str = os.path.join(
            self.temp_dir, f"{self.idx:06d}-{uuid.uuid4()}.pt"
        )

        with self.manager_lock:
            # Mark file as being saved
            self.file_locks[file_path] = threading.Lock()
            self.file_status[file_path] = "saving"
            # Add to history
            self.tensor_paths.append(file_path)

        # Acquire semaphore to limit concurrent save operations
        self.save_semaphore.acquire()
        # Queue tensor for saving in background
        self.save_queue.put((tensor.detach(), file_path))

        return file_path

    def wait_for_save(self, file_path, timeout=None) -> None:
        """Wait for a tensor to be saved to disk"""
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            with self.manager_lock:
                if self.file_status.get(file_path) == "ready":
                    return
                if self.file_status.get(file_path) in ["error", "missing", "deleted"]:
                    return

                if file_path in self.save_futures:
                    future = self.save_futures[file_path]
                    if future.done():
                        return

            # Small sleep to prevent CPU spinning
            time.sleep(0.01)

        # Timeout
        logger.warning(f"Timeout waiting for tensor to be saved: {file_path}")
        return

    def load_tensor(self, file_path, target_device="cuda"):
        """Load tensor from disk or prefetch cache with proper synchronization"""
        # Wait for tensor to be saved if it's still in progress
        self.wait_for_save(file_path)

        tensor = None

        # Try to get from cache first
        with self.manager_lock:
            # Check if tensor is already in cache
            if file_path in self.prefetch_cache:
                tensor = self.prefetch_cache[file_path]
                del self.prefetch_cache[file_path]
                self.file_status[file_path] = "loaded"

        if tensor is not None:
            # Ensure tensor is on correct device
            if target_device != "cpu" and tensor.device.type == "cpu":
                tensor = tensor.to(target_device, non_blocking=True)
            return tensor

        # If not in cache, load directly from disk
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found for loading: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")

            tensor = torch.load(file_path, weights_only=True)

            with self.manager_lock:
                self.file_status[file_path] = "loaded"

            if target_device != "cpu":
                tensor = tensor.to(target_device, non_blocking=True)

            return tensor

        except Exception as e:
            logger.error(f"Error loading tensor from {file_path}: {e}")
            raise

    def _safe_delete_file(self, file_path):
        """Safely delete a file with proper synchronization"""
        with self.manager_lock:
            # Make sure any save operation is completed
            if file_path in self.save_futures:
                future = self.save_futures[file_path]
                try:
                    if not future.done():
                        future.cancel()
                    del self.save_futures[file_path]
                except FileNotFoundError as e:
                    logger.warning(
                        f"Error canceling save operation for {file_path}: {e}"
                    )

            # Only delete if file exists and is not being prefetched
            status = self.file_status.get(file_path)
            if status in ["ready", "loaded", "error", "missing"]:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    self.file_status[file_path] = "deleted"
                    return True
                except FileNotFoundError as e:
                    logger.warning(f"Error deleting file {file_path}: {e}")
            return False

    def trigger_prefetch(self, n=None):
        """Trigger prefetching of the next N tensors with proper synchronization"""
        if n is None:
            n = self.max_prefetch

        prefetch_paths = []
        with self.manager_lock:
            # Find files that are ready to be prefetched (not already in cache or being prefetched)
            for path in reversed(self.tensor_paths):
                if (
                    path not in self.prefetch_cache
                    and self.file_status.get(path) == "ready"
                ):
                    prefetch_paths.append(path)
                    if len(prefetch_paths) >= n:
                        break

        # Queue files for prefetching
        for path in prefetch_paths:
            self.prefetch_queue.put(path)

    def cleanup_tensor(self, file_path: str):
        """Clean up a specific tensor file after it's been used"""
        with self.manager_lock:
            if file_path in self.tensor_paths:
                self.tensor_paths.remove(file_path)

            # Remove from prefetch cache if present
            if file_path in self.prefetch_cache:
                del self.prefetch_cache[file_path]

            # Remove from save futures if present
            if file_path in self.save_futures:
                future = self.save_futures[file_path]
                if not future.done():
                    future.cancel()
                del self.save_futures[file_path]

        # Try to delete the file
        self._safe_delete_file(file_path)

    def cleanup(self):
        """Clean up all temp files and stop prefetch thread with proper synchronization"""
        self.stop_event.set()

        # Cancel all pending save operations
        with self.manager_lock:
            for _, future in self.save_futures.items():
                if not future.done():
                    future.cancel()
            self.save_futures.clear()

        # Drain the save queue
        while not self.save_queue.empty():
            try:
                self.save_queue.get_nowait()
                self.save_queue.task_done()
            except queue.Empty:
                break

        # Shutdown the save pool
        self.save_pool.shutdown(wait=False)

        # Join the save worker thread
        if self.save_worker.is_alive():
            self.save_worker.join(timeout=2.0)

        # Join the prefetch worker threads
        for thread in self.prefetch_workers:
            if thread.is_alive():
                thread.join(timeout=2.0)

        # Clear cache and remove all temporary files
        with self.manager_lock:
            self.prefetch_cache.clear()
            paths_to_delete = list(self.tensor_paths)
            self.tensor_paths.clear()

        # Delete all temporary files
        for path in paths_to_delete:
            self._safe_delete_file(path)

        # Remove temp directory
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except FileNotFoundError as e:
            logger.warning(f"Error removing temporary directory {self.temp_dir}: {e}")


class Disco(torch.autograd.Function):
    """
    Disco: DIsk-based Storage and Checkpointing with Optimized prefetching
    Advanced disk-based gradient checkpointer with prefetching.
    """

    # Shared manager instance across all checkpointing operations
    _manager = None

    @staticmethod
    def get_instance(prefetch_size=1, prefetch_to_gpu=True, save_workers=4):
        """Get or create the offload manager"""
        if Disco._manager is None:
            Disco._manager = DiskOffloadManager(
                prefetch_size=prefetch_size,
                prefetch_to_gpu=prefetch_to_gpu,
                save_workers=save_workers,
            )
        return Disco._manager

    @staticmethod
    @torch_cuda_amp_custom_fwd
    def forward(
        ctx,
        forward_function,
        hidden_states,
        *args,
        prefetch_size=1,
        prefetch_to_gpu=True,
        save_workers=4,
    ):
        """Forward pass that offloads activations to disk asynchronously"""
        # Get or create the manager
        manager = Disco.get_instance(
            prefetch_size=prefetch_size,
            prefetch_to_gpu=prefetch_to_gpu,
            save_workers=save_workers,
        )

        # Save tensor to disk asynchronously
        file_path = manager.save_tensor(hidden_states)

        # Run forward pass immediately without waiting for save to complete
        with torch.no_grad():
            output = forward_function(hidden_states, *args)

        # Store what we need for backward
        ctx.save_for_backward(torch.tensor([0]))  # Dummy tensor
        ctx.file_path = file_path
        ctx.forward_function = forward_function
        ctx.args = args

        return output

    @staticmethod
    @torch_cuda_amp_custom_bwd
    def backward(ctx, *grad_outputs):
        """Backward pass that loads activations from disk with prefetching"""
        # Get the manager
        manager = Disco._manager

        # Trigger prefetching for future tensors
        # This happens at the start of backward, so should have time to complete
        manager.trigger_prefetch()

        # Load hidden states from disk or prefetch cache
        file_path = ctx.file_path
        try:
            # Ensure the file is saved before we try to load it
            manager.wait_for_save(file_path)

            hidden_states = manager.load_tensor(file_path)
            hidden_states.requires_grad = True

            # Compute gradients
            with torch.enable_grad():
                output = ctx.forward_function(hidden_states, *ctx.args)

                # Handle tuple outputs properly
                if isinstance(output, tuple):
                    if len(grad_outputs) == len(output):
                        torch.autograd.backward(output, grad_outputs)
                    else:
                        torch.autograd.backward(output, grad_outputs[0])
                else:
                    torch.autograd.backward(output, grad_outputs[0])

            # Clean up the file after we're done with it
            manager.cleanup_tensor(file_path)

            return (
                (
                    None,  # forward_function
                    hidden_states.grad,  # hidden_states grad
                )
                + (None,) * len(ctx.args)  # for each arg
                + (
                    None,  # prefetch_size
                    None,  # prefetch_to_gpu
                    None,  # save_workers
                )
            )

        except Exception as e:
            logger.error(f"Error in backward pass: {e}")
            # Clean up the file even on error
            manager.cleanup_tensor(file_path)
            raise
