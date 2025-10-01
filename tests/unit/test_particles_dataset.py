import numpy as np
import pandas as pd
import pytest

from scripts.prepare_data import Config, SPLIT_RATIOS, main


def _adjacency_from_csr(indices: np.ndarray, indptr: np.ndarray) -> dict[int, set[int]]:
    adj = {}
    for i in range(len(indptr) - 1):
        start, end = indptr[i], indptr[i + 1]
        adj[i] = set(indices[start:end])
    return adj


def test_particle_advection_dataset(tmp_path):
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("scipy.spatial")

    out = tmp_path / "particles.zarr"
    metadata_out = tmp_path / "particles.parquet"

    cfg = Config(
        dataset="particles_advect",
        out=str(out),
        metadata_out=str(metadata_out),
        write_metadata=True,
        particle_samples=6,
        particle_num_min=256,
        particle_num_max=384,
        particle_steps=5,
        particle_dt=0.05,
        particle_radius=0.2,
    )

    main(cfg)

    from ups.data.datasets import ParticleZarrDataset

    ds = ParticleZarrDataset(str(out))
    assert len(ds) == cfg.particle_samples

    for sample in ds:
        positions = sample["fields"]["positions"].numpy()
        velocities = sample["fields"]["velocities"].numpy()

        assert positions.shape[0] == cfg.particle_steps
        assert positions.shape[2] == 3
        assert velocities.shape == positions.shape
        assert np.all((positions >= 0.0) & (positions < 1.0))
        assert np.isfinite(velocities).all()

        meta = sample["meta"]
        indices = meta["indices_csr"].numpy()
        indptr = meta["indptr_csr"].numpy()
        edges = meta["edges"].numpy()

        adj = _adjacency_from_csr(indices, indptr)
        for i, j in edges:
            assert j in adj[i]
            assert i in adj[j]

        degrees = [len(neigh) for neigh in adj.values()]
        assert np.mean(degrees) > 4

    df = pd.read_parquet(metadata_out)
    assert len(df) == cfg.particle_samples
    counts = df["split"].value_counts(normalize=True).to_dict()
    tol = max(0.05, 1.0 / cfg.particle_samples)
    for split, ratio in SPLIT_RATIOS.items():
        assert counts.get(split, 0.0) == pytest.approx(ratio, abs=tol)
