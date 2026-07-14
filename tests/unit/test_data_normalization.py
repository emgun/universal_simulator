from __future__ import annotations

import json

import pytest
import torch

from ups.data.normalization import NormalizationStats, fit_normalization_stats


def test_fit_normalization_is_incremental_and_channel_aware(tmp_path):
    fields = [
        torch.tensor([[[1.0, 10.0], [3.0, 14.0]]]),
        torch.tensor([[[5.0, 18.0], [7.0, 22.0]]]),
    ]

    stats = fit_normalization_stats(
        fields,
        data_lock_sha256="a" * 64,
        selection_sha256="b" * 64,
    )
    path = tmp_path / "stats.json"
    stats.save(path)
    loaded = NormalizationStats.load(
        path,
        expected_data_lock_sha256="a" * 64,
        expected_selection_sha256="b" * 64,
    )

    normalized = torch.cat([loaded.apply(item) for item in fields], dim=1)
    assert torch.allclose(normalized.mean(dim=(0, 1)), torch.zeros(2), atol=1e-6)
    assert torch.allclose(normalized.square().mean(dim=(0, 1)), torch.ones(2), atol=1e-6)


def test_normalization_rejects_non_train_fit_role(tmp_path):
    path = tmp_path / "stats.json"
    payload = {
        "version": 1,
        "method": "zscore",
        "fit_role": "test",
        "channel_axis": -1,
        "count": 2,
        "mean": [0.0],
        "std": [1.0],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="training data only"):
        NormalizationStats.load(path)


def test_normalization_rejects_lock_mismatch(tmp_path):
    stats = NormalizationStats(
        mean=(0.0,),
        std=(1.0,),
        count=1,
        data_lock_sha256="a" * 64,
    )
    path = tmp_path / "stats.json"
    stats.save(path)

    with pytest.raises(ValueError, match="resolved data lock"):
        NormalizationStats.load(path, expected_data_lock_sha256="c" * 64)


def test_normalization_detects_tampering(tmp_path):
    stats = NormalizationStats(mean=(0.0,), std=(1.0,), count=1)
    path = tmp_path / "stats.json"
    stats.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mean"] = [5.0]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        NormalizationStats.load(path)
