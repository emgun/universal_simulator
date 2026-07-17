from __future__ import annotations

import json
import subprocess
import sys

import pytest

from scripts.select_vast_offer import projected_dph, select_offer


def _offer(offer_id: int, *, gpu_dph: float, storage_cost: float) -> dict:
    return {
        "id": offer_id,
        "dph_base": gpu_dph,
        "dph_total": gpu_dph + 0.001,
        "storage_cost": storage_cost,
        "disk_space": 200,
        "rentable": True,
        "verification": "verified",
        "search": {"slaPremiumPerHour": 0.0},
    }


def test_projected_dph_includes_requested_disk_allocation() -> None:
    row = _offer(1, gpu_dph=0.2533333333333333, storage_cost=0.8666666666666666)
    assert projected_dph(row, disk_gb=96) == pytest.approx(0.3688888888888888)


def test_select_offer_ranks_projected_not_search_price() -> None:
    high_storage = _offer(1, gpu_dph=0.25, storage_cost=0.90)
    low_storage = _offer(2, gpu_dph=0.29, storage_cost=0.10)
    offer_id, price = select_offer(
        [high_storage, low_storage],
        disk_gb=96,
        max_dph=0.45,
        max_runtime_minutes=600,
        max_total_cost=4.50,
    )
    assert offer_id == "2"
    assert price == pytest.approx(0.3033333333333333)


def test_select_offer_rejects_projected_cap_violation() -> None:
    row = _offer(1, gpu_dph=0.34, storage_cost=0.90)
    with pytest.raises(ValueError, match="no verified offer"):
        select_offer(
            [row],
            disk_gb=96,
            max_dph=0.45,
            max_runtime_minutes=600,
            max_total_cost=4.50,
        )


def test_select_offer_honors_requested_offer() -> None:
    rows = [
        _offer(1, gpu_dph=0.25, storage_cost=0.10),
        _offer(2, gpu_dph=0.30, storage_cost=0.10),
    ]
    offer_id, _ = select_offer(
        rows,
        disk_gb=96,
        max_dph=0.45,
        max_runtime_minutes=600,
        max_total_cost=4.50,
        requested_offer="2",
    )
    assert offer_id == "2"


def test_cli_prints_offer_id_then_projected_price() -> None:
    row = _offer(7, gpu_dph=0.29, storage_cost=0.10)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/select_vast_offer.py",
            "--disk-gb",
            "96",
            "--max-dph",
            "0.45",
            "--max-runtime-minutes",
            "600",
            "--max-total-cost",
            "4.50",
        ],
        input=json.dumps([row]),
        text=True,
        capture_output=True,
        check=True,
    )
    offer_id, price = result.stdout.split()
    assert offer_id == "7"
    assert float(price) == pytest.approx(0.3033333333333333)
