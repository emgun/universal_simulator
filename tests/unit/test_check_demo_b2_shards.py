from __future__ import annotations

import yaml

from scripts.check_demo_b2_shards import expected_keys_from_manifest


def test_expected_keys_from_manifest_placeholder_schema(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": "light-v1",
                "remote_prefix": "light-v1",
                "tasks": ["burgers1d", "darcy2d"],
                "splits": {
                    "train": {"samples": 2},
                    "val": {"samples": 1},
                    "test": {"samples": 1},
                },
                "records": [],
            }
        ),
        encoding="utf-8",
    )

    assert expected_keys_from_manifest(manifest) == [
        "light-v1/burgers1d/burgers1d_test.h5",
        "light-v1/burgers1d/burgers1d_train.h5",
        "light-v1/burgers1d/burgers1d_val.h5",
        "light-v1/darcy2d/darcy2d_test.h5",
        "light-v1/darcy2d/darcy2d_train.h5",
        "light-v1/darcy2d/darcy2d_val.h5",
    ]


def test_expected_keys_from_manifest_records_take_precedence(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "remote_prefix": "light-v1",
                "tasks": ["burgers1d"],
                "splits": {"train": {"samples": 2}},
                "records": [
                    {"remote_key": "custom/a.h5"},
                    {"remote_key": "custom/a.h5"},
                    {"remote_key": "custom/b.h5"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert expected_keys_from_manifest(manifest) == ["custom/a.h5", "custom/b.h5"]

