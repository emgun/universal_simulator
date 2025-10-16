#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from numcodecs import Blosc


def _make_coords(H: int, W: int) -> np.ndarray:
    ys = np.linspace(0.0, 1.0, H, dtype=np.float32)
    xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    coords = np.stack([X, Y], axis=-1).reshape(H * W, 2)
    return coords


def _diffusion2d(T: int, H: int, W: int, dt: float = 0.05, nu: float = 0.1, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((H, W)).astype(np.float32)

    kx = np.fft.fftfreq(W) * 2 * math.pi
    ky = np.fft.fftfreq(H) * 2 * math.pi
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    u_series = []
    u_hat = np.fft.fft2(u)
    for _ in range(T):
        # exact spectral step: u_hat(t+dt) = e^{-nu k^2 dt} u_hat(t)
        u_hat = np.exp(-nu * k2 * dt) * u_hat
        u = np.fft.ifft2(u_hat).real.astype(np.float32)
        u_series.append(u)
    arr = np.stack(u_series, axis=0)[..., None]  # (T, H, W, 1)
    return arr


def _burgers_like(T: int, H: int, W: int, dt: float = 0.05, seed: int = 17) -> Tuple[np.ndarray, np.ndarray]:
    # Synthetic divergence-free-ish velocity field via time-varying Fourier modes
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 2 * math.pi, W, dtype=np.float32)
    ys = np.linspace(0.0, 2 * math.pi, H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    amps = rng.uniform(0.5, 1.0, size=(3,)).astype(np.float32)
    kx = np.array([1, 2, 3], dtype=np.float32)
    ky = np.array([2, 1, 2], dtype=np.float32)
    w = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    U, V = [], []
    for t in range(T):
        phase = w * (t * dt)
        u = sum(
            float(amps[i]) * np.cos(kx[i] * X + ky[i] * Y + phase[i]) for i in range(3)
        ).astype(np.float32)
        v = sum(
            float(amps[i]) * np.sin(kx[i] * X - ky[i] * Y - phase[i]) for i in range(3)
        ).astype(np.float32)
        U.append(u)
        V.append(v)
    U = np.stack(U, axis=0)[..., None]
    V = np.stack(V, axis=0)[..., None]
    return U, V


def _kolmogorov_flow(T: int, H: int, W: int, dt: float = 0.05, k: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    ys = np.linspace(0.0, 2 * math.pi, H, dtype=np.float32)
    xs = np.linspace(0.0, 2 * math.pi, W, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    U, V = [], []
    for t in range(T):
        amp = 1.0 + 0.25 * math.sin(0.2 * t * dt)
        u = amp * np.sin(k * Y).astype(np.float32)
        v = (0.1 * amp * np.sin(k * X + 0.5 * t * dt)).astype(np.float32)
        U.append(u)
        V.append(v)
    U = np.stack(U, axis=0)[..., None]
    V = np.stack(V, axis=0)[..., None]
    return U, V


def _ensure_dir(p: str) -> None:
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _write_zarr(out_path: str, name: str, coords: np.ndarray, time: np.ndarray, fields: dict, dt: float) -> None:
    import zarr

    store = zarr.open(out_path, mode="a")
    g = store.create_group(name, overwrite=True)
    g.attrs["kind"] = "grid"
    g.attrs["dt"] = float(dt)
    H = fields[next(iter(fields))].shape[1]
    W = fields[next(iter(fields))].shape[2]
    g.attrs["H"] = int(H)
    g.attrs["W"] = int(W)

    g.create_dataset("coords", data=coords.astype("f4"), chunks=(min(len(coords), 65536), 2), dtype="f4")
    g.create_dataset("time", data=time.astype("f4"), chunks=(min(len(time), 256),), dtype="f4")

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    fg = g.create_group("fields")
    for k, arr in fields.items():
        # arr: (T, H, W, C)
        chunks = (1, min(H, 128), min(W, 128), arr.shape[-1])
        fg.create_dataset(
            k,
            data=arr.astype("f4"),
            chunks=chunks,
            dtype="f4",
            compressor=compressor,
        )


def _save_grid_preview_png(out_png: str, trio: dict) -> None:
    # trio: name -> np.ndarray (T,H,W,1)
    fig, axes = plt.subplots(1, len(trio), figsize=(4 * len(trio), 3), constrained_layout=True)
    if len(trio) == 1:
        axes = [axes]
    for ax, (name, arr) in zip(axes, trio.items()):
        img = arr[0, ..., 0]
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, shrink=0.7)
    _ensure_dir(out_png)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _sample_delaunay_mesh(num_nodes: int, seed: int):
    """Generate a 2D triangular mesh via Delaunay triangulation."""

    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    points = rng.random((num_nodes, 2), dtype=np.float64)
    tri = Delaunay(points)
    coords = tri.points.astype(np.float32)
    cells = tri.simplices.astype(np.int32)
    return coords, cells


def _cells_to_edges(cells: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        [cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]],
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def _cotangent(a: np.ndarray, b: np.ndarray) -> float:
    cross = a[0] * b[1] - a[1] * b[0]
    if abs(cross) < 1e-12:
        return 0.0
    return float(np.dot(a, b) / cross)


def _cotangent_laplacian(coords: np.ndarray, cells: np.ndarray):
    from collections import defaultdict

    import scipy.sparse as sp

    n = coords.shape[0]
    edge_weights = defaultdict(float)
    diag = np.zeros(n, dtype=np.float64)

    for tri in cells:
        i, j, k = tri
        pi, pj, pk = coords[i], coords[j], coords[k]

        cot_k = max(_cotangent(pi - pk, pj - pk), 0.0)
        cot_j = max(_cotangent(pk - pj, pi - pj), 0.0)
        cot_i = max(_cotangent(pj - pi, pk - pi), 0.0)

        def _accumulate(u, v, w):
            if w <= 0:
                return
            edge_weights[(u, v)] += w
            edge_weights[(v, u)] += w
            diag[u] += w
            diag[v] += w

        _accumulate(i, j, 0.5 * cot_k)
        _accumulate(j, k, 0.5 * cot_i)
        _accumulate(k, i, 0.5 * cot_j)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for (u, v), w in edge_weights.items():
        rows.append(u)
        cols.append(v)
        data.append(-w)

    for idx, val in enumerate(diag):
        rows.append(idx)
        cols.append(idx)
        data.append(val)

    lap = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return lap


def _write_mesh_group(group, coords: np.ndarray, cells: np.ndarray, edges: np.ndarray, laplacian) -> None:
    group.attrs["kind"] = "mesh"
    group.attrs["num_nodes"] = int(coords.shape[0])

    group.create_dataset(
        "coords",
        data=coords.astype("f4"),
        chunks=(min(len(coords), 4096), 2),
        dtype="f4",
    )
    group.create_dataset(
        "cells",
        data=cells.astype("i4"),
        chunks=(min(len(cells), 8192), 3),
        dtype="i4",
    )
    group.create_dataset(
        "edges",
        data=edges.astype("i4"),
        chunks=(min(len(edges), 8192), 2),
        dtype="i4",
    )

    lap = laplacian.tocsr()
    lap_grp = group.create_group("laplacian")
    lap_grp.create_dataset("data", data=lap.data.astype("f4"), dtype="f4")
    lap_grp.create_dataset("indices", data=lap.indices.astype("i4"), dtype="i4")
    lap_grp.create_dataset("indptr", data=lap.indptr.astype("i4"), dtype="i4")


def _velocity_field(points: np.ndarray, time: float) -> np.ndarray:
    """Analytical 3-D velocity field producing smooth swirling motion."""

    two_pi = 2.0 * math.pi
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    vx = np.sin(two_pi * x + 0.5 * time) * np.cos(two_pi * y)
    vy = np.sin(two_pi * y + 0.5 * time) * np.cos(two_pi * z)
    vz = np.sin(two_pi * z + 0.5 * time) * np.cos(two_pi * x)
    vel = np.stack([vx, vy, vz], axis=-1)
    return vel.astype(np.float32)


def _advect_particles(num_particles: int, steps: int, dt: float, seed: int):
    """Generate particle trajectories by integrating the synthetic velocity field."""

    rng = np.random.default_rng(seed)
    positions = np.zeros((steps, num_particles, 3), dtype=np.float32)
    velocities = np.zeros_like(positions)

    positions[0] = rng.random((num_particles, 3), dtype=np.float32)

    for t in range(steps):
        pos_t = positions[t]
        vel_t = _velocity_field(pos_t, time=t * dt)
        velocities[t] = vel_t
        if t + 1 < steps:
            nxt = pos_t + dt * vel_t
            positions[t + 1] = nxt - np.floor(nxt)

    return positions, velocities


def _radius_graph(points: np.ndarray, radius: float):
    """Build a symmetric radius-neighborhood graph using KD-tree acceleration."""

    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=radius)

    indices: List[int] = []
    indptr: List[int] = [0]
    edge_set: set[Tuple[int, int]] = set()

    for i, nbrs in enumerate(neighbor_lists):
        nbrs_sorted = sorted(j for j in nbrs if j != i)
        indices.extend(nbrs_sorted)
        indptr.append(len(indices))
        for j in nbrs_sorted:
            edge_set.add((min(i, j), max(i, j)))

    edges = np.array(sorted(edge_set), dtype=np.int32)
    return np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32), edges


def _write_particle_group(
    group,
    positions: np.ndarray,
    velocities: np.ndarray,
    radius: float,
    indices: np.ndarray,
    indptr: np.ndarray,
    edges: np.ndarray,
) -> None:
    group.attrs["kind"] = "particles"
    group.attrs["num_particles"] = int(positions.shape[1])
    group.attrs["steps"] = int(positions.shape[0])
    group.attrs["radius"] = float(radius)

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)

    group.create_dataset(
        "positions",
        data=positions.astype("f4"),
        chunks=(1, min(positions.shape[1], 65536), 3),
        dtype="f4",
        compressor=compressor,
    )
    group.create_dataset(
        "velocities",
        data=velocities.astype("f4"),
        chunks=(1, min(velocities.shape[1], 65536), 3),
        dtype="f4",
        compressor=compressor,
    )

    nbr_grp = group.create_group("neighbors")
    nbr_grp.attrs["radius"] = float(radius)
    nbr_grp.create_dataset("indices", data=indices.astype("i4"), dtype="i4")
    nbr_grp.create_dataset("indptr", data=indptr.astype("i4"), dtype="i4")
    nbr_grp.create_dataset("edges", data=edges.astype("i4"), dtype="i4")


def _assign_splits(num_records: int, seed: int = 17) -> List[str]:
    """Assign deterministic dataset splits close to the desired ratios."""

    if num_records <= 0:
        return []

    ratios = [SPLIT_RATIOS[s] for s in SPLITS]
    targets = [r * num_records for r in ratios]
    counts = [int(round(t)) for t in targets]
    total = sum(counts)

    if total != num_records:
        diff = num_records - total
        order = np.argsort([-abs(t - c) for t, c in zip(targets, counts)])
        for idx in order:
            if diff == 0:
                break
            counts[idx] += np.sign(diff)
            diff -= np.sign(diff)

    actual = [c / num_records for c in counts]
    tolerance = max(0.01, 1.0 / num_records)
    for split, ratio, current in zip(SPLITS, ratios, actual):
        if abs(current - ratio) > tolerance:
            raise ValueError(
                f"Unable to satisfy split ratio for '{split}': got {current:.3f} vs target {ratio:.3f}. "
                "Increase sample count (e.g., raise T)."
            )

    split_list: List[str] = []
    for split, count in zip(SPLITS, counts):
        split_list.extend([split] * count)

    rng = np.random.default_rng(seed)
    rng.shuffle(split_list)
    return split_list


def _write_metadata(path: str, records: Iterable[Dict[str, str]], seed: int) -> None:
    rows = list(records)
    if not rows:
        return

    splits = _assign_splits(len(rows), seed=seed)
    for row, split in zip(rows, splits):
        row["split"] = split

    df = pd.DataFrame.from_records(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


@dataclass
class Config:
    dataset: str = "poc_trio"  # options: poc_trio
    out: str = "data/poc_trio.zarr"
    H: int = 256
    W: int = 256
    T: int = 10
    dt: float = 0.05
    seed: int = 17
    preview_png: str = "reports/samples_grid.png"
    metadata_out: str = "data/metadata.parquet"
    write_metadata: bool = False
    mesh_samples: int = 10
    mesh_nodes_min: int = 2500
    mesh_nodes_max: int = 20000
    particle_samples: int = 8
    particle_num_min: int = 10000
    particle_num_max: int = 50000
    particle_steps: int = 16
    particle_dt: float = 0.02
    particle_radius: float = 0.04


def main(cfg: Config) -> None:
    np.random.seed(cfg.seed)

    if cfg.dataset == "poc_trio":
        coords = _make_coords(cfg.H, cfg.W)
        time = np.arange(cfg.T, dtype=np.float32) * cfg.dt
        diff = _diffusion2d(cfg.T, cfg.H, cfg.W, dt=cfg.dt, nu=0.1, seed=cfg.seed)
        bu, bv = _burgers_like(cfg.T, cfg.H, cfg.W, dt=cfg.dt, seed=cfg.seed)
        ku, kv = _kolmogorov_flow(cfg.T, cfg.H, cfg.W, dt=cfg.dt, k=4)

        _ensure_dir(cfg.out)
        _write_zarr(cfg.out, "diffusion2d", coords, time, {"u": diff}, cfg.dt)
        _write_zarr(cfg.out, "burgers2d", coords, time, {"u": bu, "v": bv}, cfg.dt)
        _write_zarr(cfg.out, "kolmogorov2d", coords, time, {"u": ku, "v": kv}, cfg.dt)

        _save_grid_preview_png(cfg.preview_png, {"diffusion2d": diff, "burgers2d(u)": bu, "kolmogorov2d(u)": ku})
        print(f"Wrote Zarr to {cfg.out} and preview {cfg.preview_png}")

        if cfg.write_metadata:
            units = {
                "diffusion2d": {"u": "dimensionless"},
                "burgers2d": {"u": "dimensionless", "v": "dimensionless"},
                "kolmogorov2d": {"u": "dimensionless", "v": "dimensionless"},
            }
            records = []
            for name, group_units in units.items():
                units_json = json.dumps(group_units, sort_keys=True)
                for idx in range(cfg.T):
                    records.append(
                        {
                            "id": f"{name}/{idx:05d}",
                            "pde": name,
                            "bc": "periodic",
                            "geom": "unit_square",
                            "units_json": units_json,
                        }
                    )
            _write_metadata(cfg.metadata_out, records, seed=cfg.seed)
            print(f"Wrote metadata table to {cfg.metadata_out}")
    elif cfg.dataset == "mesh_poisson":
        import zarr

        _ensure_dir(cfg.out)
        root = zarr.open(cfg.out, mode="a")
        mesh_group = root.create_group("mesh_poisson", overwrite=True)

        rng = np.random.default_rng(cfg.seed)
        records = []

        for sample_idx in range(cfg.mesh_samples):
            nodes = int(rng.integers(cfg.mesh_nodes_min, cfg.mesh_nodes_max + 1))
            sample_seed = int(rng.integers(0, 10_000_000))
            coords, cells = _sample_delaunay_mesh(nodes, seed=sample_seed)
            edges = _cells_to_edges(cells)
            lap = _cotangent_laplacian(coords, cells)

            sample_group = mesh_group.create_group(f"sample_{sample_idx:05d}", overwrite=True)
            _write_mesh_group(sample_group, coords, cells, edges, lap)

            if cfg.write_metadata:
                records.append(
                    {
                        "id": f"mesh_poisson/{sample_idx:05d}",
                        "pde": "poisson",
                        "bc": "dirichlet_zero",
                        "geom": "unit_square",
                        "units_json": json.dumps({"u": "dimensionless"}, sort_keys=True),
                    }
                )

        if cfg.write_metadata and records:
            _write_metadata(cfg.metadata_out, records, seed=cfg.seed)
            print(f"Wrote metadata table to {cfg.metadata_out}")

        print(
            "Wrote mesh Poisson dataset to"
            f" {cfg.out} with {cfg.mesh_samples} samples (nodes {cfg.mesh_nodes_min}-{cfg.mesh_nodes_max})."
        )
    elif cfg.dataset == "particles_advect":
        import zarr

        _ensure_dir(cfg.out)
        root = zarr.open(cfg.out, mode="a")
        particles_group = root.create_group("particles_advect", overwrite=True)

        rng = np.random.default_rng(cfg.seed)
        records = []

        for sample_idx in range(cfg.particle_samples):
            num_particles = int(rng.integers(cfg.particle_num_min, cfg.particle_num_max + 1))
            sample_seed = int(rng.integers(0, 10_000_000))
            positions, velocities = _advect_particles(num_particles, cfg.particle_steps, cfg.particle_dt, sample_seed)
            radius = cfg.particle_radius
            indices, indptr, edges = _radius_graph(positions[0], radius=radius)

            sample_group = particles_group.create_group(f"sample_{sample_idx:05d}", overwrite=True)
            _write_particle_group(sample_group, positions, velocities, radius, indices, indptr, edges)

            if cfg.write_metadata:
                records.append(
                    {
                        "id": f"particles_advect/{sample_idx:05d}",
                        "pde": "advection_particles",
                        "bc": "periodic",
                        "geom": "unit_cube",
                        "units_json": json.dumps({"position": "unit", "velocity": "unit/s"}, sort_keys=True),
                    }
                )

        if cfg.write_metadata and records:
            _write_metadata(cfg.metadata_out, records, seed=cfg.seed)
            print(f"Wrote metadata table to {cfg.metadata_out}")

        print(
            "Wrote particle advection dataset to"
            f" {cfg.out} with {cfg.particle_samples} samples (N {cfg.particle_num_min}-{cfg.particle_num_max})."
        )
    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")


if __name__ == "__main__":
    # Support Hydra-like overrides if available; otherwise simple argparse fallback
    try:
        import hydra
        from omegaconf import OmegaConf

        @hydra.main(version_base=None, config_name=None)
        def _hydra_entry(cfg):
            # Merge defaults from dataclass with overrides
            base = OmegaConf.structured(Config)
            merged = OmegaConf.merge(base, cfg)
            cfg_dc = Config(**OmegaConf.to_container(merged, resolve=True))
            main(cfg_dc)

        _hydra_entry()  # type: ignore[misc]
    except Exception:
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--dataset", default="poc_trio")
        p.add_argument("--out", default="data/poc_trio.zarr")
        p.add_argument("--H", type=int, default=256)
        p.add_argument("--W", type=int, default=256)
        p.add_argument("--T", type=int, default=10)
        p.add_argument("--dt", type=float, default=0.05)
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--preview_png", default="reports/samples_grid.png")
        p.add_argument("--metadata_out", default="data/metadata.parquet")
        p.add_argument("--write_metadata", action="store_true")
        p.add_argument("--mesh_samples", type=int, default=10)
        p.add_argument("--mesh_nodes_min", type=int, default=2500)
        p.add_argument("--mesh_nodes_max", type=int, default=20000)
        p.add_argument("--particle_samples", type=int, default=8)
        p.add_argument("--particle_num_min", type=int, default=10000)
        p.add_argument("--particle_num_max", type=int, default=50000)
        p.add_argument("--particle_steps", type=int, default=16)
        p.add_argument("--particle_dt", type=float, default=0.02)
        p.add_argument("--particle_radius", type=float, default=0.04)
        args = p.parse_args()
        main(Config(**vars(args)))
