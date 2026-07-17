#!/usr/bin/env python
"""Select a verified Vast offer using projected GPU plus requested-disk cost."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


def projected_dph(row: dict[str, Any], *, disk_gb: float) -> float:
    if disk_gb <= 0:
        raise ValueError("disk_gb must be positive")
    gpu_dph = float(row.get("dph_base", row.get("dph_total", math.inf)))
    storage_monthly_per_gb = float(row.get("storage_cost", math.inf))
    sla_dph = float((row.get("search") or {}).get("slaPremiumPerHour", 0.0))
    projected = gpu_dph + storage_monthly_per_gb * disk_gb / (30.0 * 24.0) + sla_dph
    if not math.isfinite(projected) or projected <= 0:
        raise ValueError("offer has invalid projected hourly cost")
    return projected


def select_offer(
    rows: list[dict[str, Any]],
    *,
    disk_gb: float,
    max_dph: float,
    max_runtime_minutes: float,
    max_total_cost: float,
    requested_offer: str = "",
) -> tuple[str, float]:
    candidates: list[tuple[float, str]] = []
    for row in rows:
        offer_id = str(row.get("id") or row.get("ask_contract_id") or "")
        if not offer_id or (requested_offer and offer_id != requested_offer):
            continue
        if row.get("rentable") is not True:
            continue
        if row.get("verification") != "verified" and row.get("vericode") != 1:
            continue
        if float(row.get("disk_space", 0.0)) < disk_gb:
            continue
        price = projected_dph(row, disk_gb=disk_gb)
        if (
            price <= max_dph + 1e-12
            and price * max_runtime_minutes / 60.0 <= max_total_cost + 1e-12
        ):
            candidates.append((price, offer_id))
    if not candidates:
        raise ValueError("no verified offer satisfies projected hourly and total cost caps")
    price, offer_id = min(candidates)
    return offer_id, price


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disk-gb", type=float, required=True)
    parser.add_argument("--max-dph", type=float, required=True)
    parser.add_argument("--max-runtime-minutes", type=float, required=True)
    parser.add_argument("--max-total-cost", type=float, required=True)
    parser.add_argument("--offer-id", default="")
    args = parser.parse_args()
    rows = json.load(sys.stdin)
    price, offer_id = select_offer(
        rows,
        disk_gb=args.disk_gb,
        max_dph=args.max_dph,
        max_runtime_minutes=args.max_runtime_minutes,
        max_total_cost=args.max_total_cost,
        requested_offer=args.offer_id,
    )
    print(offer_id, f"{price:.12f}")


if __name__ == "__main__":
    main()
