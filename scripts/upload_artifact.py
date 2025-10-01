#!/usr/bin/env python
from __future__ import annotations

"""Small helper to push local files/directories to Weights & Biases artifacts."""

import argparse
from pathlib import Path
from typing import Iterable

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload files to a W&B artifact")
    parser.add_argument("name", help="Artifact name, e.g. burgers1d-subset")
    parser.add_argument("type", help="Artifact type, e.g. dataset")
    parser.add_argument("paths", nargs="+", help="Files or directories to include")
    parser.add_argument("--project", default="universal-simulator", help="W&B project")
    parser.add_argument("--entity", default=None, help="Optional W&B entity")
    parser.add_argument("--run-name", default=None, help="Override run name")
    parser.add_argument("--metadata", default=None, help="Optional JSON string metadata")
    return parser.parse_args()


def ensure_paths(paths: Iterable[str]) -> list[Path]:
    resolved = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        resolved.append(path)
    return resolved


def main() -> None:
    args = parse_args()
    files = ensure_paths(args.paths)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name or f"artifact-upload-{args.name}",
        job_type="artifact-upload",
    )
    metadata = None
    if args.metadata:
        import json

        metadata = json.loads(args.metadata)

    artifact = wandb.Artifact(args.name, type=args.type, metadata=metadata)
    for path in files:
        artifact.add_file(str(path)) if path.is_file() else artifact.add_dir(str(path))
    wandb.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
