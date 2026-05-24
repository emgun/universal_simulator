#!/usr/bin/env python
from __future__ import annotations

"""Search and summarize cheap Vast.ai offers for the UPS smoke pipeline."""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_FILTERS = [
    "gpu_name=RTX_4090",
    "num_gpus=1",
    "disk_space>=32",
    "dph_total<=0.5",
    "verified=true",
]


def summarize_offer(offer: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": offer.get("id") or offer.get("ask_contract_id"),
        "gpu_name": offer.get("gpu_name"),
        "num_gpus": offer.get("num_gpus"),
        "gpu_ram": offer.get("gpu_ram"),
        "dph_total": offer.get("dph_total"),
        "discounted_dph_total": offer.get("discounted_dph_total"),
        "disk_space": offer.get("disk_space"),
        "disk_bw": offer.get("disk_bw"),
        "inet_down": offer.get("inet_down"),
        "inet_up": offer.get("inet_up"),
        "reliability": offer.get("reliability"),
        "geolocation": offer.get("geolocation"),
        "driver_version": offer.get("driver_version"),
        "cuda_max_good": offer.get("cuda_max_good"),
    }


def sort_key(offer: dict[str, Any]) -> tuple[float, float]:
    price = offer.get("dph_total")
    reliability = offer.get("reliability")
    return (float(price) if price is not None else 999.0, -float(reliability or 0.0))


def search_offers(filters: list[str], *, limit: int) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["vastai", "search", "offers", *filters, "--raw"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout or "[]")
    if not isinstance(payload, list):
        raise ValueError("Expected Vast --raw output to be a JSON list")
    return [summarize_offer(offer) for offer in sorted(payload, key=sort_key)[:limit]]


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "gpu_name",
        "num_gpus",
        "gpu_ram",
        "dph_total",
        "discounted_dph_total",
        "disk_space",
        "disk_bw",
        "inet_down",
        "inet_up",
        "reliability",
        "geolocation",
        "driver_version",
        "cuda_max_good",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append(
            "\t".join("" if row.get(field) is None else str(row.get(field)) for field in fields)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize cheap Vast offers for smoke pipeline runs"
    )
    parser.add_argument("filters", nargs="*", default=DEFAULT_FILTERS, help="Vast search filters")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output-json", default="reports/demo/vast_smoke_offers.json")
    parser.add_argument("--output-tsv", default="reports/demo/vast_smoke_offers.tsv")
    args = parser.parse_args()

    filters = args.filters or DEFAULT_FILTERS
    rows = search_offers(filters, limit=args.limit)
    output_json = Path(args.output_json)
    output_tsv = Path(args.output_tsv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"filters": filters, "offers": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(rows, output_tsv)
    print(output_json)
    print(output_tsv)
    for row in rows:
        print(
            f"{row['id']}\t{row['gpu_name']}\t${row['dph_total']}/hr\t"
            f"disk={row['disk_space']}GB\treliability={row['reliability']}\t{row['geolocation']}"
        )


if __name__ == "__main__":
    main()
