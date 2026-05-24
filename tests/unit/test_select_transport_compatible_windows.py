from __future__ import annotations

import json
from argparse import Namespace

from scripts.select_transport_compatible_windows import select_compatible


def _write_scan(path, split: str, shifts: list[int]) -> None:
    windows = [
        {
            "start_index": idx * 32,
            "sample_count": 32,
            "best": {"shift": shift, "nrmse": 0.1 + idx, "mse": 0.01 + idx},
            "top": [],
        }
        for idx, shift in enumerate(shifts)
    ]
    hist: dict[str, int] = {}
    for shift in shifts:
        hist[str(shift)] = hist.get(str(shift), 0) + 1
    path.write_text(
        json.dumps(
            {
                "split": split,
                "best_shift_histogram": hist,
                "windows": windows,
            }
        ),
        encoding="utf-8",
    )


def test_select_compatible_windows_requires_common_shift(tmp_path):
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    test = tmp_path / "test.json"
    _write_scan(train, "train", [0, 8, 24])
    _write_scan(val, "val", [24, 40])
    _write_scan(test, "test", [24, 72])

    record = select_compatible(
        Namespace(
            train_scan=train,
            val_scan=val,
            test_scan=test,
            require_test=True,
            metric="nrmse",
            top_k=5,
        )
    )

    assert record["compatible"] is True
    assert record["common_shifts"] == [24]
    assert record["selected"]["shift"] == 24
    assert record["selected"]["windows"]["train"]["start_index"] == 64


def test_select_compatible_windows_reports_no_match(tmp_path):
    train = tmp_path / "train.json"
    val = tmp_path / "val.json"
    _write_scan(train, "train", [0, 8])
    _write_scan(val, "val", [40])

    record = select_compatible(
        Namespace(
            train_scan=train,
            val_scan=val,
            test_scan=None,
            require_test=False,
            metric="nrmse",
            top_k=5,
        )
    )

    assert record["compatible"] is False
    assert record["common_shifts"] == []
    assert record["selected"] is None
