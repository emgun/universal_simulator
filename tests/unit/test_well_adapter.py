from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ups.data.trajectory import validate_trajectory_sample
from ups.data.well import WellConfig, WellTrajectoryDataset


class FakeWellDataset:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples
        self.reads: list[int] = []
        self.metadata = object()
        self.core_field_names = ("density", "pressure")
        self.constant_scalar_names = ("reynolds",)
        self.file_index_offsets = [0, 4]
        self.n_windows_per_trajectory = [2]
        self.files_paths = ["train/sample_000.hdf5"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        self.reads.append(index)
        return self.samples[index]


def _sample() -> dict[str, Any]:
    return {
        "input_fields": torch.zeros(2, 8, 6, 2),
        "output_fields": torch.ones(1, 8, 6, 2),
        "constant_scalars": {"reynolds": torch.tensor(100.0)},
        "constant_fields": torch.zeros(8, 6, 1),
        "boundary_conditions": torch.tensor([[0, 0], [1, 1]]),
        "space_grid": torch.zeros(8, 6, 2),
        "input_time_grid": torch.tensor([0.0, 0.1]),
        "output_time_grid": torch.tensor([0.2]),
    }


def test_adapter_is_lazy_and_respects_max_samples(tmp_path) -> None:
    native = FakeWellDataset([_sample() for _ in range(4)])
    dataset = WellTrajectoryDataset(
        WellConfig("turbulent_radiative_layer_2D", str(tmp_path), max_samples=2),
        native_dataset=native,
    )

    assert len(dataset) == 2
    assert native.reads == []
    assert dataset[-1]["targets"].shape == (1, 8, 6, 2)
    assert native.reads == [1]
    with pytest.raises(IndexError):
        _ = dataset[2]


def test_adapter_maps_native_physics_metadata_and_identity(tmp_path) -> None:
    native = FakeWellDataset([_sample() for _ in range(4)])
    dataset = WellTrajectoryDataset(
        WellConfig(
            "turbulent_radiative_layer_2D",
            str(tmp_path),
            split="valid",
            n_steps_input=2,
        ),
        native_dataset=native,
    )

    sample = dataset[3]
    validate_trajectory_sample(sample)
    assert sample["params"]["reynolds"].item() == 100.0
    assert sample["boundary_conditions"]["padding"].shape == (2, 2)
    assert sample["auxiliary"]["constant_fields"].shape == (8, 6, 1)
    assert sample["metadata"]["field_names"] == ("density", "pressure")
    assert sample["metadata"]["identity"] == {
        "source": "the-well",
        "dataset_name": "turbulent_radiative_layer_2D",
        "split": "valid",
        "sample_index": 3,
        "source_file": "train/sample_000.hdf5",
        "trajectory_index": 1,
        "window_start": 1,
    }


def test_adapter_rejects_bad_native_shapes(tmp_path) -> None:
    native_sample = _sample()
    native_sample["output_fields"] = torch.ones(1, 7, 6, 2)
    dataset = WellTrajectoryDataset(
        WellConfig("turbulent_radiative_layer_2D", str(tmp_path)),
        native_dataset=FakeWellDataset([native_sample]),
    )

    with pytest.raises(ValueError, match="share spatial and channel"):
        _ = dataset[0]


def test_config_rejects_remote_and_invalid_selection() -> None:
    with pytest.raises(ValueError, match="staged locally"):
        WellConfig("dataset", "hf://datasets/polymathic-ai/")
    with pytest.raises(ValueError, match="split"):
        WellConfig("dataset", "/tmp/data", split="validation")
    with pytest.raises(ValueError, match="max_samples"):
        WellConfig("dataset", "/tmp/data", max_samples=0)


def test_missing_optional_dependency_has_actionable_error(tmp_path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fail_the_well(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "the_well.data":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_the_well)
    with pytest.raises(ImportError, match="optional 'the_well' package"):
        WellTrajectoryDataset(WellConfig("dataset", str(tmp_path)))


def test_adapter_enforces_lock_role_and_exact_native_files(tmp_path, monkeypatch) -> None:
    native = FakeWellDataset([_sample()])
    monkeypatch.setattr(
        "ups.data.well.load_data_lock",
        lambda _: SimpleNamespace(
            requested_roles=("valid",),
            objects=(SimpleNamespace(path="sample_000.hdf5", role="valid"),),
        ),
    )
    with pytest.raises(PermissionError, match="does not authorize"):
        WellTrajectoryDataset(
            WellConfig("dataset", str(tmp_path), data_lock_path="lock.json"),
            native_dataset=native,
        )

    monkeypatch.setattr(
        "ups.data.well.load_data_lock",
        lambda _: SimpleNamespace(
            requested_roles=("train",),
            objects=(SimpleNamespace(path="other_file.hdf5", role="train"),),
        ),
    )
    with pytest.raises(PermissionError, match="outside the run data lock"):
        WellTrajectoryDataset(
            WellConfig("dataset", str(tmp_path), data_lock_path="lock.json"),
            native_dataset=native,
        )
