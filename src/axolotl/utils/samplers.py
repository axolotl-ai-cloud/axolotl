import random
from typing import Iterator, List

from torch.utils.data import BatchSampler


class DemoBatchSampler(BatchSampler):
    def __iter__(self) -> Iterator[List[int]]:
        sampler_iter = iter(self.sampler)
        while True:
            try:
                batch_sz = random.randrange(self.batch_size - 1, self.batch_size + 1)
                batch = [next(sampler_iter) for _ in range(batch_sz)]
                yield batch
            except StopIteration:
                break
