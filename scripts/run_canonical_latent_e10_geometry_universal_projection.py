#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.run_canonical_latent_e9_geometry_universal_projection import (
        GeometryProjectionConfig,
        run_geometry_projection,
    )
else:
    from run_canonical_latent_e9_geometry_universal_projection import (  # type: ignore[no-redef]
        GeometryProjectionConfig,
        run_geometry_projection,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
E9_CONTRACT = (
    REPO_ROOT
    / "docs/research/2026-07-23-canonical-latent-e9-geometry-universal-projection-contract.md"
)
E10_CONTRACT = (
    REPO_ROOT
    / "docs/research/2026-07-23-canonical-latent-e10-geometry-universal-projection-contract.md"
)
SHARED_RUNNER = REPO_ROOT / "scripts/run_canonical_latent_e9_geometry_universal_projection.py"
E10_ENTRYPOINT = Path(__file__).resolve()


def frozen_e10_config() -> GeometryProjectionConfig:
    return GeometryProjectionConfig(
        seed=23,
        geometry_seed_start=30_000,
        max_condition_number=100.0,
        max_weighted_design_condition_number=10.0,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _committed_sha256(path: Path) -> str | None:
    relative_path = path.relative_to(REPO_ROOT).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None
    return hashlib.sha256(completed.stdout).hexdigest()


def _worktree_clean() -> bool:
    return not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _provenance() -> dict[str, Any]:
    source_files = {
        "e9_contract": E9_CONTRACT,
        "e10_contract": E10_CONTRACT,
        "shared_runner": SHARED_RUNNER,
        "e10_entrypoint": E10_ENTRYPOINT,
    }
    source_hashes = {name: _sha256(path) for name, path in source_files.items()}
    committed_hashes = {name: _committed_sha256(path) for name, path in source_files.items()}
    source_files_match_git_head = all(
        committed_hashes[name] == source_hash for name, source_hash in source_hashes.items()
    )
    git_head = _git_head()
    return {
        "source_sha256": source_hashes,
        "committed_source_sha256": committed_hashes,
        "git_head": git_head,
        "source_files_match_git_head": source_files_match_git_head,
        "worktree_clean": _worktree_clean(),
        "git_head_present": len(git_head) == 40,
    }


def run_e10(cfg: GeometryProjectionConfig, *, run_dir: Path) -> dict[str, Any]:
    if asdict(cfg) != asdict(frozen_e10_config()):
        raise ValueError("E10 requires the exact frozen configuration")

    provenance = _provenance()
    provenance_gate = bool(
        provenance["source_files_match_git_head"]
        and provenance["worktree_clean"]
        and provenance["git_head_present"]
    )
    if not provenance_gate:
        raise RuntimeError(
            "E10 provenance must match a clean committed Git HEAD before state access"
        )

    result = run_geometry_projection(cfg, run_dir=run_dir)
    result["experiment"] = "canonical_latent_e10_geometry_universal_projection"
    result["provenance"] = provenance
    result["causal_decision"]["gates"]["provenance"] = provenance_gate

    result_path = run_dir / "result.json"
    artifact = dict(result)
    artifact.pop("result_path", None)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["result_path"] = str(result_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen E10 geometry-universal projection repair"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = frozen_e10_config()
    result = run_e10(cfg, run_dir=args.run_dir)
    summary = {
        "causal_decision": result["causal_decision"],
        "result_path": result["result_path"],
    }
    if args.print_json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(
            f"classification={summary['causal_decision']['classification']} "
            f"result={summary['result_path']}"
        )


if __name__ == "__main__":
    main()
