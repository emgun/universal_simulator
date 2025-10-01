import numpy as np
import pandas as pd
import pytest

from scripts.prepare_data import Config, SPLIT_RATIOS, main


def _load_laplacian(sample_group):
    import scipy.sparse as sp

    coords = sample_group["coords"]
    n = coords.shape[0]
    data = sample_group["laplacian"]["data"][:]
    indices = sample_group["laplacian"]["indices"][:]
    indptr = sample_group["laplacian"]["indptr"][:]
    lap = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    return lap


def test_mesh_poisson_dataset(tmp_path):
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("scipy.sparse")

    out = tmp_path / "mesh.zarr"
    metadata_out = tmp_path / "mesh.parquet"

    cfg = Config(
        dataset="mesh_poisson",
        out=str(out),
        metadata_out=str(metadata_out),
        write_metadata=True,
        mesh_samples=10,
        mesh_nodes_min=90,
        mesh_nodes_max=110,
    )

    main(cfg)

    from ups.data.datasets import MeshZarrDataset

    ds = MeshZarrDataset(str(out))
    assert len(ds) == cfg.mesh_samples

    for sample in ds:
        coords = sample["coords"].numpy()
        geom = sample["geom"]
        cells = geom["cells"].numpy()
        edges = sample["connect"].numpy()

        assert coords.shape[1] == 2
        assert cells.shape[1] == 3
        assert edges.shape[1] == 2

        num_nodes = coords.shape[0]
        assert cfg.mesh_nodes_min <= num_nodes <= cfg.mesh_nodes_max

        lap = sample["meta"]["laplacian"].toarray()
        assert np.allclose(lap, lap.T, atol=1e-5)

        eigvals = np.linalg.eigvalsh(lap)
        assert eigvals[0] >= -1e-5
        assert abs(eigvals[0]) < 1e-4

    df = pd.read_parquet(metadata_out)
    assert len(df) == cfg.mesh_samples
    counts = df["split"].value_counts(normalize=True).to_dict()
    for split, ratio in SPLIT_RATIOS.items():
        assert counts.get(split, 0.0) == pytest.approx(ratio, abs=0.01)
