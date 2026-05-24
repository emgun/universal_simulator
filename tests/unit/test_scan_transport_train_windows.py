from __future__ import annotations

from argparse import Namespace

import h5py
import numpy as np

from scripts.scan_transport_train_windows import scan_windows


def test_scan_train_windows_reports_shift_histogram(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    width = 8
    rows = []
    for shift in (1, 1, 2, 2):
        base = np.zeros(width, dtype=np.float32)
        base[1] = 1.0
        rows.append([np.roll(base, shift * step) for step in range(5)])
    with h5py.File(root / "advection1d_train.h5", "w") as handle:
        handle.create_dataset("data", data=np.asarray(rows, dtype=np.float32)[..., None])

    args = Namespace(
        data_root=root,
        task="advection1d",
        split="train",
        window_size=2,
        min_samples=1,
        stride=2,
        start_index=0,
        max_windows=None,
        rollout_steps=3,
        shift=[0, 1, 2],
        metric="nrmse",
        top_k=2,
    )

    record = scan_windows(args)

    assert record["windows_scanned"] == 2
    assert record["best_shift_histogram"] == {"1": 1, "2": 1}
    assert [row["best"]["shift"] for row in record["windows"]] == [1, 2]
