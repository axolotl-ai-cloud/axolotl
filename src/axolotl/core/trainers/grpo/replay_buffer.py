"""Simple replay buffer for storing and sampling high-signal rollout groups."""

import heapq

import torch


class ReplayBuffer:
    """Min-heap replay buffer that keeps the highest-scoring rollout groups.
    Groups are scored by signal quality (advantage magnitude * reward variance).
    When sampling, groups are drawn proportional to their scores.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._heap: list[tuple[float, int, dict]] = []  # min-heap of (score, id, data)
        self._counter = 0  # unique tiebreaker for heap

    def __len__(self):
        return len(self._heap)

    def add(self, score: float, data: dict):
        """Add a group to the buffer. If full, replaces lowest-scoring entry."""
        if self.max_size <= 0:
            return
        self._counter += 1
        if len(self._heap) < self.max_size:
            heapq.heappush(self._heap, (score, self._counter, data))
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, (score, self._counter, data))

    def sample(self, num_samples: int) -> list[dict] | None:
        """Sample groups weighted by their scores. Returns None if buffer is empty."""
        if self.max_size <= 0 or not self._heap:
            return None

        scores = torch.tensor([item[0] for item in self._heap], dtype=torch.float32)
        scores = scores.clamp(min=1e-8)  # avoid zero probabilities
        probs = scores / scores.sum()
        replacement = num_samples > len(self._heap)
        indices = torch.multinomial(
            probs, num_samples, replacement=replacement
        ).tolist()
        return [self._heap[i][2] for i in indices]
