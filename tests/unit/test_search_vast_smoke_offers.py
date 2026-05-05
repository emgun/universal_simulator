from __future__ import annotations

import json

from scripts.search_vast_smoke_offers import summarize_offer, write_tsv


def test_summarize_offer_keeps_cost_and_reliability_fields():
    row = summarize_offer(
        {
            "id": 123,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "dph_total": 0.33,
            "disk_space": 64,
            "reliability": 0.99,
            "geolocation": "Texas, US",
        }
    )

    assert row["id"] == 123
    assert row["gpu_name"] == "RTX 4090"
    assert row["dph_total"] == 0.33
    assert row["reliability"] == 0.99


def test_write_tsv_outputs_offer_table(tmp_path):
    path = tmp_path / "offers.tsv"
    rows = [{"id": 1, "gpu_name": "RTX 4090", "dph_total": 0.4, "reliability": 0.9}]

    write_tsv(rows, path)

    text = path.read_text(encoding="utf-8")
    assert "id\tgpu_name" in text
    assert "RTX 4090" in text
    assert json.dumps(rows[0]["dph_total"]) in text
