#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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


def frozen_e10_config(
    *,
    validation_states: int = GeometryProjectionConfig.validation_states,
    calibration_resolution: int = GeometryProjectionConfig.calibration_resolution,
    geometry_realizations: int = GeometryProjectionConfig.geometry_realizations,
) -> GeometryProjectionConfig:
    return GeometryProjectionConfig(
        seed=23,
        validation_states=validation_states,
        calibration_resolution=calibration_resolution,
        geometry_realizations=geometry_realizations,
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


def _provenance() -> dict[str, Any]:
    source_hashes = {
        "e9_contract_sha256": _sha256(E9_CONTRACT),
        "e10_contract_sha256": _sha256(E10_CONTRACT),
        "shared_runner_sha256": _sha256(SHARED_RUNNER),
        "e10_entrypoint_sha256": _sha256(E10_ENTRYPOINT),
    }
    git_head = _git_head()
    return {
        **source_hashes,
        "git_head": git_head,
        "all_source_hashes_present": all(len(value) == 64 for value in source_hashes.values()),
        "git_head_present": len(git_head) == 40,
    }


def run_e10(cfg: GeometryProjectionConfig, *, run_dir: Path) -> dict[str, Any]:
    if (
        cfg.seed != 23
        or cfg.geometry_seed_start != 30_000
        or cfg.max_condition_number != 100.0
        or cfg.max_weighted_design_condition_number != 10.0
    ):
        raise ValueError("E10 seed and conditioning invariants are frozen")

    result = run_geometry_projection(cfg, run_dir=run_dir)
    result["experiment"] = "canonical_latent_e10_geometry_universal_projection"
    result["provenance"] = _provenance()
    provenance_gate = bool(
        result["provenance"]["all_source_hashes_present"]
        and result["provenance"]["git_head_present"]
    )
    result["causal_decision"]["gates"]["provenance"] = provenance_gate
    if not provenance_gate:
        result["causal_decision"]["classification"] = "geometry_universal_projection_not_qualified"
        result["causal_decision"]["next_move"] = "repair provenance before scientific promotion"

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
    parser.add_argument(
        "--validation-states", type=int, default=GeometryProjectionConfig.validation_states
    )
    parser.add_argument(
        "--geometry-realizations",
        type=int,
        default=GeometryProjectionConfig.geometry_realizations,
    )
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = frozen_e10_config(
        validation_states=args.validation_states,
        geometry_realizations=args.geometry_realizations,
    )
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
