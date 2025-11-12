"""Task-aware distributed samplers for multi-task training."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class MultiTaskDistributedSampler(Sampler[int]):
    """
    Distributed sampler that maintains balanced task sampling across ranks.

    Given a list of per-task dataset sizes, ensures each rank sees the same
    proportion of each task per epoch. Respects distributed training semantics
    (rank, world_size) while maintaining task balance.

    Args:
        task_sizes: List of dataset sizes per task (e.g., [1000, 800, 1200] for 3 tasks)
        num_replicas: Number of processes (default: world_size)
        rank: Rank of current process (default: current rank)
        shuffle: Whether to shuffle samples within each task (default: True)
        seed: Random seed for shuffling (default: 0)
        drop_last: Whether to drop incomplete batches (default: False)

    Example:
        >>> task_sizes = [1000, 800, 1200]  # 3 tasks
        >>> sampler = MultiTaskDistributedSampler(task_sizes, num_replicas=2, rank=0)
        >>> len(sampler)  # 1500 (total 3000 / 2 replicas)
        1500
    """

    def __init__(
        self,
        task_sizes: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank() if dist.is_initialized() else 0

        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, should be in [0, {num_replicas-1}]")

        self.task_sizes = task_sizes
        self.num_tasks = len(task_sizes)
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Compute per-task samples per replica
        self.per_task_num_samples = []
        for size in task_sizes:
            if self.drop_last and size % self.num_replicas != 0:
                num_samples = math.ceil((size - self.num_replicas) / self.num_replicas)
            else:
                num_samples = math.ceil(size / self.num_replicas)
            self.per_task_num_samples.append(num_samples)

        self.total_size = sum(self.per_task_num_samples) * self.num_replicas
        self.num_samples = sum(self.per_task_num_samples)

        # Compute task offsets in ConcatDataset (cumulative sizes)
        self.task_offsets = [0]
        for size in task_sizes[:-1]:
            self.task_offsets.append(self.task_offsets[-1] + size)

    def __iter__(self) -> Iterator[int]:
        """Generate indices for this rank."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []

        # For each task, generate indices and select this rank's subset
        for _task_idx, (size, offset, num_samples) in enumerate(
            zip(self.task_sizes, self.task_offsets, self.per_task_num_samples, strict=False)
        ):
            # Generate task-local indices [0, size)
            if self.shuffle:
                task_indices = torch.randperm(size, generator=g).tolist()
            else:
                task_indices = list(range(size))

            # Pad if needed (for balanced distribution)
            if not self.drop_last:
                padding_size = num_samples * self.num_replicas - len(task_indices)
                if padding_size > 0:
                    task_indices += task_indices[:padding_size]
            else:
                # Drop tail to make evenly divisible
                task_indices = task_indices[: num_samples * self.num_replicas]

            # Select this rank's subset (every num_replicas-th element)
            rank_indices = task_indices[self.rank : len(task_indices) : self.num_replicas]

            # Convert to global indices (add task offset)
            rank_indices = [idx + offset for idx in rank_indices]

            indices.extend(rank_indices)

        # Shuffle across tasks if requested (maintains per-task balance in expectation)
        if self.shuffle:
            # Shuffle with task-preserving property (interleave tasks)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler (for shuffling)."""
        self.epoch = epoch
