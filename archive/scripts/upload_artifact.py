#!/usr/bin/env python
from __future__ import annotations

"""Minimal helper to push local files/directories to Weights & Biases artifacts.

Pattern used:
  import wandb
  wandb.init(entity=..., project=...)
  art = wandb.Artifact(name, type='dataset')
  art.add_file(...) or art.add_dir(...)
  wandb.log_artifact(art)

Adds --cache-dir to steer W&B staging to a local directory.
"""

import argparse
from pathlib import Path
import os
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
    parser.add_argument("--cache-dir", default=None, help="Optional W&B cache/staging directory (sets WANDB_CACHE_DIR)")
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

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_CACHE_DIR"] = str(cache_dir)

    run = wandb.init(project=args.project, entity=args.entity, name=args.run_name or f"artifact-upload-{args.name}")
    metadata = None
    if args.metadata:
        import json

        metadata = json.loads(args.metadata)

    artifact = wandb.Artifact(args.name, type=args.type, metadata=metadata)
    for path in files:
        if path.is_file():
            artifact.add_file(str(path))
        else:
            artifact.add_dir(str(path))
    wandb.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
