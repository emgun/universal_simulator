from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ups.data.convert_pdebench import convert_files


def _write_raw_h5(path: Path, data: np.ndarray, ds_name: str = "tensor") -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset(ds_name, data=data)


def test_convert_1d_stack(tmp_path: Path):
    f1 = tmp_path / "burgers_a.hdf5"
    f2 = tmp_path / "burgers_b.hdf5"
    _write_raw_h5(f1, np.random.randn(100, 4, 6).astype(np.float32))
    _write_raw_h5(f2, np.random.randn(100, 4, 6).astype(np.float32))

    out = tmp_path / "burgers1d_train.h5"
    written = convert_files([f1, f2], out, limit=2, sample_size=10)

    with h5py.File(out, "r") as h5:
        arr = h5["data"][...]
    assert written == 20
    assert arr.shape == (20, 4, 6, 1)
