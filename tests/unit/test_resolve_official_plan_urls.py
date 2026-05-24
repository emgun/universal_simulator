from __future__ import annotations

import json
from argparse import Namespace

from scripts.resolve_official_plan_urls import resolve_plan_urls


def test_resolve_plan_urls_persists_source_urls(monkeypatch, tmp_path):
    input_plan = tmp_path / "plan.json"
    output_plan = tmp_path / "resolved.json"
    input_plan.write_text(
        json.dumps(
            {
                "commands": {
                    "download_official_train_files": [
                        "python scripts/download_pdebench_file.py '1D/Advection/Train/a.hdf5' --out data/raw"
                    ]
                },
                "remote_entries": [
                    {
                        "path": "1D/Advection/Train/a.hdf5",
                        "file_id": 1,
                        "size_bytes": 3,
                        "checksum": "abc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def fake_resolve(url, *, timeout, retries, retry_backoff):
        assert url == "https://darus.uni-stuttgart.de/api/access/datafile/1?format=original"
        return "https://s3.example/a.hdf5?signature=1"

    monkeypatch.setattr(
        "scripts.resolve_official_plan_urls._resolve_redirect_url_with_curl", fake_resolve
    )

    record = resolve_plan_urls(
        Namespace(
            plan_json=str(input_plan),
            output_json=str(output_plan),
            timeout=5,
            retries=2,
            retry_backoff=0,
            continue_on_error=False,
        )
    )

    resolved = json.loads(output_plan.read_text(encoding="utf-8"))
    assert record["status"] == "resolved"
    assert record["resolved_count"] == 1
    assert resolved["remote_entries"][0]["source_url"] == "https://s3.example/a.hdf5?signature=1"
    assert resolved["remote_entries"][0]["resolved_from_url"] == (
        "https://darus.uni-stuttgart.de/api/access/datafile/1?format=original"
    )
    assert resolved["commands"]["download_official_train_files"][0].startswith(
        "PDEBENCH_DATAFILE_URL_TEMPLATE="
    )


def test_resolve_plan_urls_blocks_on_unresolved_entry(monkeypatch, tmp_path):
    input_plan = tmp_path / "plan.json"
    output_plan = tmp_path / "resolved.json"
    input_plan.write_text(
        json.dumps({"remote_entries": [{"path": "a.hdf5", "file_id": 1, "size_bytes": 3}]}),
        encoding="utf-8",
    )

    def fake_resolve(url, *, timeout, retries, retry_backoff):
        raise RuntimeError("dns failed")

    monkeypatch.setattr(
        "scripts.resolve_official_plan_urls._resolve_redirect_url_with_curl", fake_resolve
    )

    record = resolve_plan_urls(
        Namespace(
            plan_json=str(input_plan),
            output_json=str(output_plan),
            timeout=5,
            retries=1,
            retry_backoff=0,
            continue_on_error=False,
        )
    )

    assert record["status"] == "blocked"
    assert record["resolved_count"] == 0
    assert "dns failed" in record["entries"][0]["error"]
    assert not output_plan.exists()
