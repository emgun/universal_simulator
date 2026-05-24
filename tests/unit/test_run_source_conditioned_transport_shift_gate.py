from __future__ import annotations

import argparse

import h5py
import numpy as np
import torch

from scripts.run_source_conditioned_transport_shift_gate import _periodic_shift, run_gate


def _write_source_shifted_split(path, *, split: str, source_shifts: list[tuple[int, int]]) -> None:
    data = torch.zeros(len(source_shifts), 4, 8, 1, dtype=torch.float32)
    source_file_index = []
    source_sample_index = []
    for sample_idx, (source_index, shift) in enumerate(source_shifts):
        data[sample_idx, 0, (sample_idx + 1) % 8, 0] = 1.0
        for step in range(1, data.shape[1]):
            data[sample_idx, step, :, 0] = torch.roll(
                data[sample_idx, step - 1, :, 0], shifts=shift, dims=-1
            )
        source_file_index.append(source_index)
        source_sample_index.append(sample_idx)
    with h5py.File(path / f"advection1d_{split}.h5", "w") as handle:
        handle.create_dataset("data", data=data.numpy())
        handle.create_dataset("source_file_index", data=source_file_index)
        handle.create_dataset("source_sample_index", data=source_sample_index)
        handle.attrs["source_paths"] = ["beta_a.hdf5", "beta_b.hdf5", "beta_c.hdf5"]


def _args(tmp_path, *, reference: float = 1.0):
    return argparse.Namespace(
        data_root=str(tmp_path),
        task="advection1d",
        train_split="train",
        val_split="val",
        test_split="",
        max_samples=4,
        val_max_samples=None,
        test_max_samples=None,
        rollout_steps=3,
        shift=[0, 1, 2],
        metric="nrmse",
        fit_strategy="aggregate",
        refine_radius=0,
        fractional_refine_step=0.0,
        reference_metric_value=reference,
        val_min_relative_improvement=0.0,
    )


def test_source_conditioned_gate_passes_mixed_train_val_sources(tmp_path):
    _write_source_shifted_split(
        tmp_path, split="train", source_shifts=[(0, 1), (0, 1), (1, 2), (1, 2)]
    )
    _write_source_shifted_split(tmp_path, split="val", source_shifts=[(0, 1), (1, 2)])

    record = run_gate(_args(tmp_path))

    assert record["test_eligible"] is True
    assert record["blockers"] == []
    assert record["fit"]["source_shift_map"] == {"0": 1, "1": 2}
    assert record["fit"]["selected_validation"]["nrmse"] == 0.0
    assert record["data_sources"]["train"]["datasets"]["source_file_index"]["shape"] == [4]
    assert record["test"] is None


def test_source_conditioned_gate_blocks_unsupported_validation_source(tmp_path):
    _write_source_shifted_split(tmp_path, split="train", source_shifts=[(0, 1), (0, 1)])
    _write_source_shifted_split(tmp_path, split="val", source_shifts=[(1, 2)])

    record = run_gate(_args(tmp_path))

    assert record["test_eligible"] is False
    assert record["fit"]["selected_validation"] is None
    assert record["diagnostic"]["unsupported_val_source_file_indices"] == [1]
    assert any("absent from train" in blocker for blocker in record["blockers"])


def test_source_conditioned_gate_measures_test_only_after_validation_passes(tmp_path):
    _write_source_shifted_split(
        tmp_path, split="train", source_shifts=[(0, 1), (0, 1), (1, 2), (1, 2)]
    )
    _write_source_shifted_split(tmp_path, split="val", source_shifts=[(0, 1), (1, 2)])
    _write_source_shifted_split(tmp_path, split="test", source_shifts=[(0, 1), (1, 2)])
    args = _args(tmp_path)
    args.test_split = "test"

    record = run_gate(args)

    assert record["test_eligible"] is True
    assert record["test"]["selected_test"]["nrmse"] == 0.0
    assert record["next_action"] == "held-out test measured"


def test_source_conditioned_gate_can_use_sample_mode_strategy(tmp_path):
    _write_source_shifted_split(
        tmp_path, split="train", source_shifts=[(0, 1), (0, 1), (0, 2), (1, 2)]
    )
    _write_source_shifted_split(tmp_path, split="val", source_shifts=[(0, 1), (1, 2)])
    args = _args(tmp_path)
    args.fit_strategy = "sample_mode"

    record = run_gate(args)

    assert record["fit"]["fit_strategy"] == "sample_mode"
    assert record["fit"]["source_shift_map"] == {"0": 1, "1": 2}
    assert record["fit"]["train_groups"]["0"]["sample_votes"]
    assert record["test_eligible"] is True


def test_source_conditioned_gate_refines_sample_mode_shift_from_coarse_grid(tmp_path):
    _write_source_shifted_split(
        tmp_path, split="train", source_shifts=[(0, 3), (0, 3), (1, 1), (1, 1)]
    )
    _write_source_shifted_split(tmp_path, split="val", source_shifts=[(0, 3), (1, 1)])
    args = _args(tmp_path)
    args.shift = [0, 4]
    args.fit_strategy = "sample_mode"
    args.refine_radius = 3

    record = run_gate(args)

    assert record["fit"]["source_shift_map"] == {"0": 3, "1": 1}
    assert record["fit"]["train_groups"]["0"]["refine_radius"] == 3
    assert record["fit"]["train_groups"]["0"]["refined_candidate_scores"]
    assert record["fit"]["selected_validation"]["nrmse"] == 0.0


def test_source_conditioned_gate_fractionally_refines_train_shift(tmp_path):
    width = 32
    samples = [(0, 1.5), (0, 1.5), (1, -0.5), (1, -0.5)]
    for split in ("train", "val"):
        data = np.zeros((len(samples), 4, width, 1), dtype=np.float32)
        source_file_index = []
        for sample_idx, (source_index, shift) in enumerate(samples):
            x = np.arange(width, dtype=np.float32)
            data[sample_idx, 0, :, 0] = np.exp(-0.5 * ((x - (8 + sample_idx)) / 2.0) ** 2)
            for step in range(1, data.shape[1]):
                data[sample_idx, step, :, 0] = _periodic_shift(
                    data[sample_idx : sample_idx + 1, step - 1, :, 0], shift
                )[0]
            source_file_index.append(source_index)
        with h5py.File(tmp_path / f"advection1d_{split}.h5", "w") as handle:
            handle.create_dataset("data", data=data)
            handle.create_dataset("source_file_index", data=source_file_index)
            handle.create_dataset("source_sample_index", data=list(range(len(samples))))
            handle.attrs["source_paths"] = ["beta_a.hdf5", "beta_b.hdf5"]

    args = _args(tmp_path, reference=1.0)
    args.shift = [-2, 0, 2]
    args.refine_radius = 2
    args.fractional_refine_step = 0.5

    record = run_gate(args)

    assert record["fit"]["fractional_refine_step"] == 0.5
    assert record["fit"]["source_shift_map"] == {"0": 1.5, "1": -0.5}
    assert record["fit"]["selected_validation"]["nrmse"] < 1e-5
    assert record["test_eligible"] is True


def test_source_conditioned_gate_fractionally_refines_sample_mode_votes(tmp_path):
    width = 32
    samples = [(0, 1.5), (0, 1.5), (1, -0.5), (1, -0.5)]
    for split in ("train", "val"):
        data = np.zeros((len(samples), 4, width, 1), dtype=np.float32)
        source_file_index = []
        for sample_idx, (source_index, shift) in enumerate(samples):
            x = np.arange(width, dtype=np.float32)
            data[sample_idx, 0, :, 0] = np.exp(-0.5 * ((x - (8 + sample_idx)) / 2.0) ** 2)
            for step in range(1, data.shape[1]):
                data[sample_idx, step, :, 0] = _periodic_shift(
                    data[sample_idx : sample_idx + 1, step - 1, :, 0], shift
                )[0]
            source_file_index.append(source_index)
        with h5py.File(tmp_path / f"advection1d_{split}.h5", "w") as handle:
            handle.create_dataset("data", data=data)
            handle.create_dataset("source_file_index", data=source_file_index)
            handle.create_dataset("source_sample_index", data=list(range(len(samples))))
            handle.attrs["source_paths"] = ["beta_a.hdf5", "beta_b.hdf5"]

    args = _args(tmp_path, reference=1.0)
    args.shift = [-2, 0, 2]
    args.fit_strategy = "sample_mode"
    args.refine_radius = 2
    args.fractional_refine_step = 0.5

    record = run_gate(args)

    assert record["fit"]["source_shift_map"] == {"0": 1.5, "1": -0.5}
    assert record["fit"]["train_groups"]["0"]["sample_votes"][0]["selected_shift"] == 1.5
    assert 1.5 in record["fit"]["train_groups"]["0"]["sample_votes"][0]["candidate_shifts"]
    assert all(
        isinstance(shift, float)
        for shift in record["fit"]["train_groups"]["0"]["sample_votes"][0]["candidate_shifts"]
    )
    assert record["fit"]["selected_validation"]["nrmse"] < 1e-5
