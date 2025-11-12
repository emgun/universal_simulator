"""Unit tests for task-aware distributed samplers."""

from __future__ import annotations

import pytest

from ups.data.task_samplers import MultiTaskDistributedSampler


class TestMultiTaskDistributedSampler:
    """Tests for MultiTaskDistributedSampler."""

    def test_basic_initialization(self):
        """Test basic sampler initialization."""
        task_sizes = [100, 80, 120]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=2, rank=0, shuffle=False
        )

        assert sampler.num_tasks == 3
        assert sampler.num_replicas == 2
        assert sampler.rank == 0
        assert len(sampler) == 150  # (100 + 80 + 120) / 2 = 150

    def test_balanced_distribution_two_ranks(self):
        """Test that samples are distributed evenly across 2 ranks."""
        task_sizes = [100, 80, 120]

        # Create samplers for both ranks
        sampler_rank0 = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=2, rank=0, shuffle=False, seed=42
        )
        sampler_rank1 = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=2, rank=1, shuffle=False, seed=42
        )

        # Get indices for both ranks
        indices_rank0 = list(sampler_rank0)
        indices_rank1 = list(sampler_rank1)

        # Check that no indices overlap
        assert len(set(indices_rank0) & set(indices_rank1)) == 0

        # Check that each rank gets roughly half the samples
        assert len(indices_rank0) == 150
        assert len(indices_rank1) == 150

    def test_task_offsets(self):
        """Test that task offsets are computed correctly."""
        task_sizes = [100, 80, 120]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=1, rank=0, shuffle=False
        )

        assert sampler.task_offsets == [0, 100, 180]

    def test_indices_within_range(self):
        """Test that generated indices are within valid range."""
        task_sizes = [100, 80, 120]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=2, rank=0, shuffle=True, seed=42
        )

        indices = list(sampler)
        total_size = sum(task_sizes)

        for idx in indices:
            assert 0 <= idx < total_size

    def test_set_epoch(self):
        """Test that set_epoch changes the sampling order."""
        task_sizes = [100, 80, 120]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=2, rank=0, shuffle=True, seed=42
        )

        # Get indices for epoch 0
        indices_epoch0 = list(sampler)

        # Set epoch and get indices again
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # Indices should be different due to different epoch
        assert indices_epoch0 != indices_epoch1

    def test_per_task_balance_across_ranks(self):
        """Test that each rank sees balanced proportion of each task."""
        task_sizes = [1000, 800, 1200]
        num_replicas = 4

        # Create samplers for all ranks
        samplers = [
            MultiTaskDistributedSampler(
                task_sizes=task_sizes,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                seed=42,
            )
            for rank in range(num_replicas)
        ]

        # Count task samples per rank
        task_counts = [{0: 0, 1: 0, 2: 0} for _ in range(num_replicas)]  # task_id -> count

        for rank, sampler in enumerate(samplers):
            indices = list(sampler)
            for idx in indices:
                # Determine which task this index belongs to
                if idx < 1000:
                    task_id = 0
                elif idx < 1800:
                    task_id = 1
                else:
                    task_id = 2
                task_counts[rank][task_id] += 1

        # Check that each task is balanced across ranks
        for task_id in range(3):
            counts = [task_counts[rank][task_id] for rank in range(num_replicas)]
            mean_count = sum(counts) / len(counts)
            # Allow 10% variation
            for count in counts:
                assert abs(count - mean_count) / mean_count < 0.1

    def test_drop_last_false(self):
        """Test that drop_last=False pads samples."""
        # 101 samples, 2 replicas -> each rank gets 51 samples (one padded)
        task_sizes = [101]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes,
            num_replicas=2,
            rank=0,
            shuffle=False,
            drop_last=False,
        )

        assert len(sampler) == 51  # ceil(101/2) = 51

    def test_drop_last_true(self):
        """Test that drop_last=True drops incomplete samples."""
        # 101 samples, 2 replicas -> each rank gets 50 samples (1 dropped)
        task_sizes = [101]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=2, rank=0, shuffle=False, drop_last=True
        )

        assert len(sampler) == 50  # (101 - 2) / 2 = 49.5 -> ceil = 50

    def test_invalid_rank_raises_error(self):
        """Test that invalid rank raises ValueError."""
        task_sizes = [100, 80]

        with pytest.raises(ValueError, match="Invalid rank"):
            MultiTaskDistributedSampler(
                task_sizes=task_sizes, num_replicas=2, rank=2  # rank >= num_replicas
            )

        with pytest.raises(ValueError, match="Invalid rank"):
            MultiTaskDistributedSampler(
                task_sizes=task_sizes, num_replicas=2, rank=-1  # negative rank
            )

    def test_single_task_single_rank(self):
        """Test sampler with single task and single rank."""
        task_sizes = [100]
        sampler = MultiTaskDistributedSampler(
            task_sizes=task_sizes, num_replicas=1, rank=0, shuffle=False
        )

        indices = list(sampler)
        assert len(indices) == 100
        assert indices == list(range(100))  # Should be identity for no shuffle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
