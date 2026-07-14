from __future__ import annotations

import h5py
import numpy as np
import torch
import zarr

from scripts import train as train_script
from ups.data.latent_pairs import (
    GridLatentPairDataset,
    collate_latent_pairs_with_sequences,
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


def test_dataset_loader_can_preserve_sequences(tmp_path):
    data = torch.randn(2, 4, 4, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "preserve_sequences": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    batch = next(iter(train_script.dataset_loader(cfg)))
    z0, z1, cond = unpack_batch(batch)

    assert isinstance(batch, dict)
    assert z0.shape == (6, 4, 8)
    assert z1.shape == (6, 4, 8)
    assert cond == {}
    assert batch["cond_seq"] == {}
    assert batch["z_seq"].shape == (2, 4, 4, 8)
    assert torch.equal(batch["seq_lens"], torch.tensor([4, 4]))


def test_dataset_loader_multitask_auto_conditioning(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.randn(2, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    batch = next(iter(train_script.dataset_loader(cfg)))
    z0, _, cond = unpack_batch(batch)

    assert cond["task_id"].shape == (z0.shape[0], 2)
    assert cond["task_family"].shape == (z0.shape[0], 2)
    assert cond["equation_traits"].shape[0] == z0.shape[0]
    assert cond["equation_traits"].shape[1] >= 5
    assert cond["equation_signature"].shape[0] == z0.shape[0]
    assert cond["equation_signature"].shape[1] > cond["equation_traits"].shape[1]
    assert cond["equation_nodes"].shape[0] == z0.shape[0]
    assert cond["equation_nodes"].dim() == 3
    assert cond["resolution"].shape == (z0.shape[0], 2)
    assert cond["spatial_dims"].shape == (z0.shape[0], 1)
    assert torch.allclose(cond["task_id"].sum(dim=-1), torch.ones(z0.shape[0]))
    assert torch.allclose(cond["task_family"].sum(dim=-1), torch.ones(z0.shape[0]))


def test_dataset_loader_reads_configured_pdebench_param_and_bc_keys(tmp_path):
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    beta = torch.full((2, 1), 0.25, dtype=torch.float32)
    left_bc = torch.zeros(2, 1, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())
        handle.create_dataset("beta", data=beta.numpy())
        handle.create_dataset("left_bc", data=left_bc.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "param_keys": ["beta"],
            "bc_keys": ["left_bc"],
        },
    }

    batch = next(iter(train_script.dataset_loader(cfg)))
    _, _, cond = unpack_batch(batch)
    assert "param_beta" in cond
    assert "bc_left_bc" in cond
    assert "param_presence" in cond
    assert "bc_presence" in cond
    assert "parameter_signature" in cond
    assert "boundary_signature" in cond
    assert "parameter_nodes" in cond
    assert "boundary_nodes" in cond
    assert cond["param_presence"].shape == (4, 1)
    assert cond["bc_presence"].shape == (4, 1)
    assert cond["parameter_signature"].shape == (4, 2)
    assert cond["boundary_signature"].shape == (4, 2)
    assert cond["parameter_nodes"].shape == (4, 1, 2)
    assert cond["boundary_nodes"].shape == (4, 1, 2)


def test_make_operator_auto_conditioning_infers_param_and_bc_sources(tmp_path):
    data = torch.randn(2, 3, 4, dtype=torch.float32)
    beta = torch.full((2, 1), 0.25, dtype=torch.float32)
    left_bc = torch.zeros(2, 1, dtype=torch.float32)
    file_path = tmp_path / "burgers1d_train.h5"
    with h5py.File(file_path, "w") as handle:
        handle.create_dataset("data", data=data.numpy())
        handle.create_dataset("beta", data=beta.numpy())
        handle.create_dataset("left_bc", data=left_bc.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": "burgers1d",
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
            "param_keys": ["beta"],
            "bc_keys": ["left_bc"],
        },
    }

    operator = train_script.make_operator(cfg)
    assert operator.conditioner is not None
    assert set(operator.conditioner.embedders.keys()) >= {
        "resolution",
        "spatial_dims",
        "equation_signature",
        "equation_nodes",
        "param_beta",
        "bc_left_bc",
        "param_presence",
        "parameter_signature",
        "parameter_nodes",
        "bc_presence",
        "boundary_signature",
        "boundary_nodes",
    }


def test_make_operator_auto_conditioning_infers_multitask_semantic_sources(tmp_path):
    for task_name in ("burgers1d", "advection1d"):
        data = torch.randn(2, 3, 4, dtype=torch.float32)
        file_path = tmp_path / f"{task_name}_train.h5"
        with h5py.File(file_path, "w") as handle:
            handle.create_dataset("data", data=data.numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": ["burgers1d", "advection1d"],
            "split": "train",
            "root": str(tmp_path),
            "patch_size": 1,
        },
    }

    operator = train_script.make_operator(cfg)
    assert operator.conditioner is not None
    assert set(operator.conditioner.embedders.keys()) >= {
        "task_id",
        "task_family",
        "equation_traits",
        "equation_signature",
        "equation_nodes",
        "resolution",
        "spatial_dims",
    }


def test_make_operator_resolves_task_roots_keys_and_union_parameter_vocab(tmp_path):
    roots = {"advection1d": tmp_path / "advection", "burgers1d": tmp_path / "burgers"}
    for task_name, key in (("advection1d", "beta"), ("burgers1d", "nu")):
        root = roots[task_name]
        root.mkdir()
        data = torch.randn(2, 3, 4, dtype=torch.float32)
        with h5py.File(root / f"{task_name}_train.h5", "w") as handle:
            handle.create_dataset("data", data=data.numpy())
            handle.create_dataset(key, data=torch.ones(2, 1).numpy())

    cfg = {
        "training": {"batch_size": 2, "dt": 0.1, "auto_conditioning": True},
        "latent": {"dim": 8, "tokens": 4},
        "data": {
            "task": ["advection1d", "burgers1d"],
            "split": "train",
            "root": str(tmp_path / "unused"),
            "task_roots": {task: str(root) for task, root in roots.items()},
            "task_param_keys": {"advection1d": ["beta"], "burgers1d": ["nu"]},
            "patch_size": 1,
        },
    }

    operator = train_script.make_operator(cfg)
    assert operator.conditioner is not None
    assert set(operator.conditioner.embedders.keys()) >= {
        "param_beta",
        "param_nu",
        "param_presence",
        "parameter_signature",
        "parameter_nodes",
    }
    assert operator.conditioner.embedders["param_presence"][0].in_features == 2


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

    batch = collate_latent_pairs_with_sequences([pair])
    assert set(batch["cond_seq"].keys()) == {"param_forcing", "bc_left", "bc_right"}
    for value in batch["cond_seq"].values():
        assert value.shape[:2] == (1, 3)


def test_infer_grid_shape_handles_channel_first_scalar_2d():
    sample_fields = torch.randn(4, 1, 4, 4)

    grid_shape = infer_grid_shape(sample_fields)
    channels = infer_channel_count(sample_fields, grid_shape)

    assert grid_shape == (4, 4)
    assert channels == 1


def test_grid_latent_pair_dataset_maps_darcy_coefficient_to_solution_once():
    tensor_data = {
        "fields": torch.zeros(1, 1, 4, 4, 1),
        "targets": torch.ones(1, 1, 4, 4, 1),
    }
    dataset = PDEBenchDataset(PDEBenchConfig(task="darcy2d"), tensor_data=tensor_data)

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

    assert pair.z0.shape == (1, 4, 6)
    assert pair.z1.shape == (1, 4, 6)
    assert pair.z_seq is not None
    assert pair.z_seq.shape == (2, 4, 6)


def test_collate_conditions_zero_fills_missing_keys():
    from ups.data.latent_pairs import LatentPair, collate_latent_pairs

    item_a = LatentPair(
        z0=torch.randn(2, 4, 3),
        z1=torch.randn(2, 4, 3),
        cond={"task_id": torch.tensor([[1.0, 0.0], [1.0, 0.0]])},
    )
    item_b = LatentPair(
        z0=torch.randn(1, 4, 3),
        z1=torch.randn(1, 4, 3),
        cond={},
    )

    _, _, cond = collate_latent_pairs([item_a, item_b])
    assert cond["task_id"].shape == (3, 2)
    assert torch.allclose(cond["task_id"][-1], torch.zeros(2))


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
