from __future__ import annotations

import h5py
import numpy as np
import torch
import zarr

from scripts import train as train_script
from ups.data.latent_pairs import (
    GridLatentPairDataset,
    infer_channel_count,
    infer_grid_shape,
    make_grid_coords,
    unpack_batch,
)
from ups.data.pdebench import PDEBenchConfig, PDEBenchDataset
from ups.io.enc_grid import GridEncoder, GridEncoderConfig


def test_dataset_loader_encodes_pdebench_grid(tmp_path):
    # Create a minimal Burgers 1D dump with shape (samples, time, spatial)
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    loader = train_script.dataset_loader(cfg)
    batch = next(iter(loader))
    z0, z1, cond = unpack_batch(batch)

    assert z0.shape == (4, 4, 8)
    assert z1.shape == (4, 4, 8)
    assert cond == {}
    # LatentState constructor should accept the batch without shape errors
    train_script.LatentState(z=z0, t=torch.tensor(0.0))


def test_grid_latent_pair_dataset_conditioning_broadcast():
    tensor_data = {
        "fields": torch.randn(1, 4, 2, 2, 1),  # (samples, time, H, W, C)
        "params": {"forcing": torch.linspace(0.0, 0.3, steps=4).view(1, 4, 1)},
        "bc": {
            "left": torch.zeros(1, 4, 1),
            "right": torch.ones(1, 4, 1),
        },
    }
    dataset = PDEBenchDataset(PDEBenchConfig(task="burgers1d"), tensor_data=tensor_data)

    sample_fields = dataset.fields[0]
    grid_shape = infer_grid_shape(sample_fields)
    channels = infer_channel_count(sample_fields, grid_shape)
    encoder_cfg = GridEncoderConfig(
        patch_size=1,
        latent_dim=6,
        latent_len=4,
        field_channels={"u": channels},
    )
    encoder = GridEncoder(encoder_cfg).eval()
    coords = make_grid_coords(grid_shape, torch.device("cpu"))

    latent_dataset = GridLatentPairDataset(dataset, encoder, coords, grid_shape)
    pair = latent_dataset[0]
    z0, z1, cond = pair.z0, pair.z1, pair.cond

    assert z0.shape == (3, 4, 6)
    assert z1.shape == (3, 4, 6)
    assert set(cond.keys()) == {"param_forcing", "bc_left", "bc_right"}
    for value in cond.values():
        assert value.shape[0] == 3


def _make_grid_zarr(tmp_path) -> str:
    path = tmp_path / "toy_grid.zarr"
    store = zarr.open(path, mode="w")
    group = store.create_group("toy")
    group.attrs["kind"] = "grid"
    group.attrs["dt"] = 0.1
    group.attrs["H"] = 2
    group.attrs["W"] = 2

    coords = np.array(
        [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]],
        dtype=np.float32,
    )
    group.create_dataset("coords", data=coords, dtype="f4")
    group.create_dataset("time", data=np.array([0.0, 0.1, 0.2], dtype=np.float32), dtype="f4")
    fields = group.create_group("fields")
    data = np.random.randn(3, 2, 2, 1).astype(np.float32)
    fields.create_dataset("u", data=data, dtype="f4")
    return str(path)


def test_dataset_loader_grid_zarr(tmp_path):
    path = _make_grid_zarr(tmp_path)
    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 6, "tokens": 4},
        "data": {
            "kind": "grid",
            "path": path,
            "group": "toy",
            "patch_size": 1,
        },
    }

    loader = train_script.dataset_loader(cfg)
    batch = next(iter(loader))
    z0, z1, cond = unpack_batch(batch)

    assert z0.shape == (2, 4, 6)
    assert z1.shape == (2, 4, 6)
    assert "dt" in cond and "time" in cond
    assert cond["dt"].shape[0] == z0.shape[0]


def _make_particle_zarr(tmp_path) -> str:
    path = tmp_path / "particles.zarr"
    store = zarr.open(path, mode="w")
    root = store.create_group("particles_advect")
    sample = root.create_group("sample_00000")
    sample.attrs["kind"] = "particles"
    sample.attrs["steps"] = 3
    sample.attrs["radius"] = 0.5

    positions = np.random.randn(3, 5, 2).astype(np.float32)
    velocities = np.random.randn(3, 5, 2).astype(np.float32)
    sample.create_dataset("positions", data=positions, dtype="f4")
    sample.create_dataset("velocities", data=velocities, dtype="f4")

    nbr = sample.create_group("neighbors")
    nbr.attrs["radius"] = 0.5
    indices = []
    indptr = [0]
    edge_set = set()
    for i in range(5):
        nbrs = [(i + 1) % 5]
        indices.extend(nbrs)
        indptr.append(len(indices))
        for j in nbrs:
            if i != j:
                edge_set.add(tuple(sorted((i, j))))
    nbr.create_dataset("indices", data=np.array(indices, dtype=np.int32), dtype="i4")
    nbr.create_dataset("indptr", data=np.array(indptr, dtype=np.int32), dtype="i4")
    edges = np.array(sorted(edge_set), dtype=np.int32)
    nbr.create_dataset("edges", data=edges, dtype="i4")
    return str(path)


def test_dataset_loader_particle_zarr(tmp_path):
    path = _make_particle_zarr(tmp_path)
    cfg = {
        "training": {"batch_size": 1, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 3},
        "data": {
            "kind": "particles",
            "path": path,
            "group": "particles_advect",
            "hidden_dim": 16,
            "message_passing_steps": 1,
            "supernodes": 8,
        },
    }

    loader = train_script.dataset_loader(cfg)
    batch = next(iter(loader))
    z0, z1, cond = unpack_batch(batch)

    assert z0.shape[0] == 2
    assert z0.shape[1:] == (3, 8)
    assert "param_radius" in cond
    assert cond["param_radius"].shape[0] == z0.shape[0]
