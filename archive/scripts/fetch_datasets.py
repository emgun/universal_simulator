#!/usr/bin/env python
from __future__ import annotations

"""Fetch datasets from the registry and hydrate a local DATA_ROOT."""

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import yaml
try:
    import wandb
except ImportError as exc:  # pragma: no cover - optional dependency for local dev
    raise SystemExit("wandb is required for scripts/fetch_datasets.py. Install wandb or run via W&B-enabled environment") from exc

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs" / "dataset_registry.yaml"


def load_registry(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    datasets = data.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid registry format: missing 'datasets' mapping in {path}")
    return datasets


def ensure_output_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)


def fetch_artifact(
    artifact_id: str,
    *,
    cache: Path | None = None,
    project: str | None = None,
    entity: str | None = None,
) -> Path:
    """Download artifact via wandb; optionally re-use a cache directory."""
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        existing = sorted(cache.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in existing:
            manifest = candidate / "wandb_manifest.json"
            if manifest.exists() and artifact_id.replace(":", "_") in candidate.name:
                return candidate
    run = wandb.init(project=project or "dataset-fetch", entity=entity, job_type="dataset-fetch", reinit=True)
    artifact = run.use_artifact(artifact_id)
    output = artifact.download(root=None)
    run.finish()
    path = Path(output)
    if cache is not None:
        target = cache / f"{artifact_id.replace('/', '_').replace(':', '_')}"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(path, target)
        return target
    return path


def copy_splits(source: Path, target_root: Path, entries: dict) -> None:
    import tarfile
    
    # Check if source contains a tarball to extract first
    tarballs = list(source.glob("*.tar.gz")) + list(source.glob("*.tar"))
    if tarballs:
        print(f"Found tarball: {tarballs[0]}, extracting...")
        with tarfile.open(tarballs[0], "r:*") as tar:
            tar.extractall(target_root)
        print(f"Extracted to {target_root}")
        return
    
    # Otherwise copy individual files
    for split_key in ("train_split", "val_split", "test_split"):
        split_file = entries.get(split_key)
        if not split_file:
            continue
        src = source / split_file
        dst = target_root / split_file
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch datasets listed in the registry")
    parser.add_argument("datasets", nargs="+", help="Dataset keys to fetch (see docs/dataset_registry.yaml)")
    parser.add_argument("--root", default="data/pdebench", help="Destination root directory")
    parser.add_argument("--cache", default=None, help="Optional cache directory for artifact downloads")
    parser.add_argument("--registry", default=str(REGISTRY_PATH), help="Custom registry path")
    parser.add_argument("--entity", default=None, help="W&B entity override")
    parser.add_argument("--project", default=None, help="W&B project override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = load_registry(Path(args.registry))
    missing: List[str] = []
    selected = {}
    for name in args.datasets:
        entry = datasets.get(name)
        if entry is None:
            missing.append(name)
        else:
            selected[name] = entry
    if missing:
        raise SystemExit(f"Unknown dataset keys: {', '.join(missing)}")

    if args.entity:
        wandb.env.set_entity(args.entity)
    if args.project:
        wandb.env.set_project(args.project)

    root = Path(args.root).expanduser().resolve()
    ensure_output_dir(root)
    cache = Path(args.cache).expanduser().resolve() if args.cache else None
    project = args.project
    entity = args.entity

    for name, entry in selected.items():
        artifact_id = entry["artifact"]
        print(f"Fetching {name} from {artifact_id}…")
        download_dir = fetch_artifact(artifact_id, cache=cache, project=project, entity=entity)
        copy_splits(download_dir, root, entry)
        print(f"Copied splits to {root}")


if __name__ == "__main__":
    main()
