import math
import os

import pandas as pd
import pytest

from scripts.prepare_data import Config, SPLIT_RATIOS, main


def test_metadata_written_with_expected_schema(tmp_path):
    zarr = pytest.importorskip("zarr")
    out = tmp_path / "poc.zarr"
    preview = tmp_path / "preview.png"
    metadata_path = tmp_path / "metadata.parquet"

    cfg = Config(
        out=str(out),
        preview_png=str(preview),
        metadata_out=str(metadata_path),
        H=16,
        W=16,
        T=10,
        write_metadata=True,
    )

    main(cfg)

    assert metadata_path.exists()

    df = pd.read_parquet(metadata_path)
    expected_columns = {"id", "split", "pde", "bc", "geom", "units_json"}
    assert set(df.columns) == expected_columns
    assert df["id"].is_unique
    assert len(df) == 3 * cfg.T

    counts = df["split"].value_counts(normalize=True).to_dict()
    for split, ratio in SPLIT_RATIOS.items():
        assert math.isclose(counts[split], ratio, rel_tol=0, abs_tol=0.01)

    store = zarr.open(str(out), mode="r")
    grp = store["diffusion2d"]
    ds = grp["fields"]["u"]
    assert ds.chunks == (1, min(cfg.H, 128), min(cfg.W, 128), 1)
    compressor = ds.compressor
    assert compressor is not None
    assert compressor.cname == "zstd"
