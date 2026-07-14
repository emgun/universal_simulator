#!/usr/bin/env python3
"""Create exact The Well source/protocol YAML from Hub metadata only."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ups.data.hf_inventory import build_well_manifests, fetch_hub_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Hugging Face dataset repository")
    parser.add_argument("--revision", required=True, help="Exact 40-character Hub commit")
    parser.add_argument("--package-version", required=True, help="Exact The Well release")
    parser.add_argument("--package-commit", required=True, help="Exact The Well Git commit")
    parser.add_argument("--pilot-parameter", default="0.03", help="Exact tcool pilot value")
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--protocol-output", type=Path, required=True)
    args = parser.parse_args()

    metadata = fetch_hub_inventory(args.repo, args.revision)
    source, protocol = build_well_manifests(
        metadata,
        repo_id=args.repo,
        revision=args.revision,
        package_version=args.package_version,
        package_commit=args.package_commit,
        pilot_parameter=args.pilot_parameter,
    )
    for path, document in (
        (args.source_output, source),
        (args.protocol_output, protocol),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
