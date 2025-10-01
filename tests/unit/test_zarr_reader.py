import os
import tempfile
import time

import numpy as np
import pytest


zarr = pytest.importorskip("zarr")


def _write_minimal_zarr(path: str):
    import zarr as _z

    root = _z.open(path, mode="a")
    g = root.create_group("diffusion2d", overwrite=True)
    H, W, T = 32, 32, 3
    coords = np.stack(np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H), indexing="xy"), axis=-1).reshape(H * W, 2).astype(
        "f4"
    )
    time_arr = (np.arange(T, dtype="f4") * 0.1).astype("f4")
    g.attrs["kind"] = "grid"
    g.attrs["dt"] = 0.1
    g.attrs["H"] = H
    g.attrs["W"] = W
    g.create_dataset("coords", data=coords, chunks=(coords.shape[0], 2), dtype="f4")
    g.create_dataset("time", data=time_arr, chunks=(T,), dtype="f4")
    u = np.random.randn(T, H, W, 1).astype("f4")
    fg = g.create_group("fields")
    fg.create_dataset("u", data=u, chunks=(1, H, W, 1), dtype="f4")


def test_grid_zarr_dataset_fast_load():
    from ups.data.datasets import GridZarrDataset

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "poc.zarr")
        _write_minimal_zarr(path)
        t0 = time.perf_counter()
        ds = GridZarrDataset(path, group="diffusion2d")
        sample = ds[0]
        dt = time.perf_counter() - t0
        assert sample["kind"] == "grid"
        assert "u" in sample["fields"]
        assert sample["coords"].shape[1] == 2
        assert dt < 1.0
