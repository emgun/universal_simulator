#!/usr/bin/env python3
"""
Utility script for synthesising small mesh and particle datasets in Zarr format.

These datasets are used by unit tests and local experiments to verify that the
UPS data pipeline can ingest mesh and particle modalities without relying on
large upstream downloads.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard exercised in environments without deps
    import zarr  # type: ignore
except ImportError:  # pragma: no cover
    zarr = None  # type: ignore

try:  # pragma: no cover
    from scipy.sparse import csr_matrix  # type: ignore
    from scipy.spatial import Delaunay, cKDTree  # type: ignore
except ImportError:  # pragma: no cover
    csr_matrix = None  # type: ignore
    Delaunay = None  # type: ignore
    cKDTree = None  # type: ignore


SPLIT_RATIOS: Dict[str, float] = {"train": 0.7, "val": 0.2, "test": 0.1}


def _normalise_split_ratios(split_ratios: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(split_ratios.values()))
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return {k: v / total for k, v in split_ratios.items()}


def _iter_splits(count: int, split_ratios: Dict[str, float]) -> Iterator[str]:
    """Yield dataset splits according to the configured ratios."""

    normalised = _normalise_split_ratios(split_ratios)
    counts = {split: int(round(ratio * count)) for split, ratio in normalised.items()}
    shortfall = count - sum(counts.values())
    if shortfall != 0:
        # Adjust the largest bucket to compensate for rounding
        largest_split = max(normalised.items(), key=lambda item: item[1])[0]
        counts[largest_split] += shortfall

    per_split: Dict[str, int] = {split: 0 for split in normalised}
    for split, total_for_split in counts.items():
        for _ in range(total_for_split):
            per_split[split] += 1
            yield split


@dataclass
class Config:
    dataset: str
    out: str
    metadata_out: Optional[str] = None
    write_metadata: bool = False
    seed: int = 17

    # Mesh dataset parameters
    mesh_samples: int = 8
    mesh_nodes_min: int = 96
    mesh_nodes_max: int = 128

    # Particle dataset parameters
    particle_samples: int = 6
    particle_num_min: int = 256
    particle_num_max: int = 384
    particle_steps: int = 5
    particle_dt: float = 0.05
    particle_radius: float = 0.2

    split_ratios: Dict[str, float] = field(default_factory=lambda: dict(SPLIT_RATIOS))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _compute_mesh_laplacian(edges: np.ndarray, num_nodes: int) -> csr_matrix:
    """Construct an unnormalised graph Laplacian from an edge list."""

    if csr_matrix is None:
        raise RuntimeError("SciPy is required to generate mesh Laplacians.")
    src = edges[:, 0]
    dst = edges[:, 1]
    data = np.ones(len(edges), dtype=np.float32)
    adjacency = csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    adjacency = adjacency + adjacency.T  # ensure symmetry
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()

    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = csr_matrix(np.diag(degrees)) - adjacency
    return laplacian


def _triangulate_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cells, edges) for a 2D Delaunay triangulation."""

    if Delaunay is None:
        raise RuntimeError("SciPy is required to generate mesh triangulations.")
    delaunay = Delaunay(points)
    cells = delaunay.simplices.astype(np.int32)

    edge_set = set()
    for tri in cells:
        tri_edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for i, j in tri_edges:
            edge = tuple(sorted((int(i), int(j))))
            edge_set.add(edge)
    edges = np.array(sorted(edge_set), dtype=np.int32)
    return cells, edges


def _store_mesh_dataset(cfg: Config) -> pd.DataFrame:
    if zarr is None:
        raise RuntimeError("The 'zarr' package is required to write mesh datasets.")
    rng = np.random.default_rng(cfg.seed)
    out_path = Path(cfg.out)
    _ensure_parent(out_path)

    store = zarr.open(str(out_path), mode="w")
    group = store.create_group("mesh_poisson")
    group.attrs["kind"] = "mesh"
    group.attrs["dt"] = 0.0

    rows: List[Dict[str, object]] = []
    for idx, split in enumerate(_iter_splits(cfg.mesh_samples, cfg.split_ratios)):
        num_nodes = int(rng.integers(cfg.mesh_nodes_min, cfg.mesh_nodes_max + 1))
        coords = rng.random((num_nodes, 2), dtype=np.float32)
        cells, edges = _triangulate_points(coords)

        laplacian = _compute_mesh_laplacian(edges, num_nodes).tocsr()
        sample = group.create_group(f"sample_{idx:05d}")
        sample.create_dataset("coords", data=coords, dtype="f4")
        sample.create_dataset("edges", data=edges, dtype="i4")
        sample.create_dataset("cells", data=cells, dtype="i4")

        lap_group = sample.create_group("laplacian")
        lap_group.create_dataset("data", data=laplacian.data.astype(np.float32), dtype="f4")
        lap_group.create_dataset("indices", data=laplacian.indices.astype(np.int32), dtype="i4")
        lap_group.create_dataset("indptr", data=laplacian.indptr.astype(np.int32), dtype="i4")

        rows.append({"sample": f"sample_{idx:05d}", "split": split})

    return pd.DataFrame.from_records(rows)


def _csr_from_edges(edges: np.ndarray, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Construct CSR representation (indices, indptr) from an undirected edge list."""

    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    indices: List[int] = []
    indptr = [0]
    for neighbours in adjacency:
        neighbours_sorted = sorted(set(neighbours))
        indices.extend(neighbours_sorted)
        indptr.append(len(indices))
    return np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)


def _generate_particle_trajectory(
    rng: np.random.Generator,
    steps: int,
    num_particles: int,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a bounded random walk for particle positions and velocities."""

    velocities = rng.normal(loc=0.0, scale=0.1, size=(steps, num_particles, 3)).astype(np.float32)
    positions = np.empty((steps, num_particles, 3), dtype=np.float32)

    positions[0] = rng.random((num_particles, 3), dtype=np.float32)
    for t in range(1, steps):
        positions[t] = positions[t - 1] + velocities[t] * dt
        positions[t] %= 1.0  # keep particles inside unit cube
    return positions, velocities


def _build_radius_graph(points: np.ndarray, radius: float) -> np.ndarray:
    """Return undirected edges for radius graph on final step positions."""

    if cKDTree is None:
        raise RuntimeError("SciPy is required to construct particle neighbour graphs.")
    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    if pairs.size == 0:
        # Ensure graph is connected enough for downstream tests by relaxing radius
        k = min(10, points.shape[0] - 1)
        distances, indices = tree.query(points, k=k + 1)
        edges = []
        for i in range(points.shape[0]):
            for j in indices[i, 1:]:
                if i == j:
                    continue
                edge = tuple(sorted((int(i), int(j))))
                edges.append(edge)
        return np.array(sorted(set(edges)), dtype=np.int32)
    return pairs.astype(np.int32)


def _store_particle_dataset(cfg: Config) -> pd.DataFrame:
    if zarr is None:
        raise RuntimeError("The 'zarr' package is required to write particle datasets.")
    rng = np.random.default_rng(cfg.seed)
    out_path = Path(cfg.out)
    _ensure_parent(out_path)

    store = zarr.open(str(out_path), mode="w")
    group = store.create_group("particles_advect")
    group.attrs["kind"] = "particles"

    rows: List[Dict[str, object]] = []
    for idx, split in enumerate(_iter_splits(cfg.particle_samples, cfg.split_ratios)):
        num_particles = int(rng.integers(cfg.particle_num_min, cfg.particle_num_max + 1))
        positions, velocities = _generate_particle_trajectory(
            rng,
            steps=cfg.particle_steps,
            num_particles=num_particles,
            dt=cfg.particle_dt,
        )

        final_positions = positions[-1]
        edges = _build_radius_graph(final_positions, radius=cfg.particle_radius)
        indices, indptr = _csr_from_edges(edges, num_particles)

        sample = group.create_group(f"sample_{idx:05d}")
        sample.create_dataset("positions", data=positions, dtype="f4")
        sample.create_dataset("velocities", data=velocities, dtype="f4")

        neighbours = sample.create_group("neighbors")
        neighbours.create_dataset("indices", data=indices, dtype="i4")
        neighbours.create_dataset("indptr", data=indptr, dtype="i4")
        neighbours.create_dataset("edges", data=edges, dtype="i4")
        neighbours.attrs["radius"] = float(cfg.particle_radius)

        rows.append({"sample": f"sample_{idx:05d}", "split": split})

    return pd.DataFrame.from_records(rows)


def main(cfg: Config) -> None:
    """Entry point callable from both CLI and tests."""

    dataset = cfg.dataset.lower()
    if dataset not in {"mesh_poisson", "particles_advect"}:
        raise ValueError(f"Unsupported dataset '{cfg.dataset}'. Expected 'mesh_poisson' or 'particles_advect'.")

    if dataset == "mesh_poisson":
        metadata = _store_mesh_dataset(cfg)
    else:
        metadata = _store_particle_dataset(cfg)

    if cfg.write_metadata:
        metadata_path = Path(cfg.metadata_out or (Path(cfg.out).with_suffix(".parquet")))
        _ensure_parent(metadata_path)
        metadata.to_parquet(metadata_path, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic mesh/particle datasets for testing.")
    parser.add_argument("dataset", choices=["mesh_poisson", "particles_advect"], help="Dataset type to generate.")
    parser.add_argument("--out", required=True, help="Path to the output Zarr store.")
    parser.add_argument("--metadata-out", help="Optional Parquet metadata output path.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip writing metadata parquet file.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for reproducibility.")

    # Mesh options
    parser.add_argument("--mesh-samples", type=int, default=8)
    parser.add_argument("--mesh-nodes-min", type=int, default=96)
    parser.add_argument("--mesh-nodes-max", type=int, default=128)

    # Particle options
    parser.add_argument("--particle-samples", type=int, default=6)
    parser.add_argument("--particle-num-min", type=int, default=256)
    parser.add_argument("--particle-num-max", type=int, default=384)
    parser.add_argument("--particle-steps", type=int, default=5)
    parser.add_argument("--particle-dt", type=float, default=0.05)
    parser.add_argument("--particle-radius", type=float, default=0.2)

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    split_ratios = dict(SPLIT_RATIOS)
    return Config(
        dataset=args.dataset,
        out=args.out,
        metadata_out=args.metadata_out,
        write_metadata=not args.no_metadata,
        seed=args.seed,
        mesh_samples=args.mesh_samples,
        mesh_nodes_min=args.mesh_nodes_min,
        mesh_nodes_max=args.mesh_nodes_max,
        particle_samples=args.particle_samples,
        particle_num_min=args.particle_num_min,
        particle_num_max=args.particle_num_max,
        particle_steps=args.particle_steps,
        particle_dt=args.particle_dt,
        particle_radius=args.particle_radius,
        split_ratios=split_ratios,
    )


if __name__ == "__main__":
    parser = _build_parser()
    namespace = parser.parse_args()
    config = _config_from_args(namespace)
    main(config)
