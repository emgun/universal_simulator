#!/usr/bin/env python3
"""Audit whether a UPS experiment actually tests a universal latent encoder.

This audit is deliberately fail-closed.  Equal latent tensor shapes are not
treated as evidence that grids, meshes, and particles share a representation.
The report distinguishes training-contract facts from measurements that would
require paired physical states represented in more than one discretization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ups.utils.config_loader import load_config_with_includes

SCHEMA_VERSION = "ups.universal-latent-contract-audit.v1"
UNIVERSAL_MODALITIES = ("grid", "mesh", "particle")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    value = cfg.get("stages", {}).get(name, {})
    return value if isinstance(value, dict) else {}


def _epochs(cfg: dict[str, Any], name: str) -> int:
    return int(_stage(cfg, name).get("epochs", 0) or 0)


def audit_contract(config_path: Path, *, repo_root: Path) -> dict[str, Any]:
    cfg = load_config_with_includes(config_path)
    data = cfg.get("data", {})
    kind = str(data.get("kind", "unknown"))
    raw_tasks = data.get("task", [])
    tasks = [str(task) for task in raw_tasks] if isinstance(raw_tasks, list) else [str(raw_tasks)]
    tasks = [task for task in tasks if task]
    routed = cfg.get("operator", {}).get("routed_adapters", {})
    joint = _stage(cfg, "joint_codec_operator")

    source_paths = {
        "grid_encoder": repo_root / "src/ups/io/enc_grid.py",
        "mesh_particle_encoder": repo_root / "src/ups/io/enc_mesh_particle.py",
        "training_pipeline": repo_root / "scripts/train.py",
    }
    source_hashes = {
        name: {"path": str(path.relative_to(repo_root)), "sha256": _sha256(path)}
        for name, path in source_paths.items()
    }

    observed_modalities = [kind] if kind in UNIVERSAL_MODALITIES else []
    missing_modalities = [item for item in UNIVERSAL_MODALITIES if item not in observed_modalities]
    operator_epochs = _epochs(cfg, "operator")
    decoder_epochs = _epochs(cfg, "decoder")
    joint_epochs = _epochs(cfg, "joint_codec_operator")

    measurements = {
        "codec_only_reconstruction_by_task": {
            "status": "unmeasured",
            "reason": "the experiment reports decoded rollouts, not decode(encode(x)) by task",
        },
        "cross_task_latent_geometry": {
            "status": "unmeasured",
            "reason": "no effective-rank, covariance, scale, or task-probe artifact is registered",
        },
        "paired_cross_discretization_alignment": {
            "status": "unmeasured",
            "reason": "no physical state is paired across grid, mesh, or particle discretizations",
        },
        "cross_encoding_and_cross_decoding": {
            "status": "unmeasured",
            "reason": "the experiment has no cross-modality encode/decode matrix",
        },
        "resolution_or_sampling_invariance": {
            "status": "unmeasured",
            "reason": "the experiment does not resample the same state at multiple resolutions",
        },
    }

    try:
        display_config_path = str(config_path.relative_to(repo_root))
    except ValueError:
        display_config_path = str(config_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "path": display_config_path,
            "sha256": _sha256(config_path),
            "tasks": tasks,
            "data_kind": kind,
            "latent_shape": {
                "tokens": int(cfg.get("latent", {}).get("tokens", 0) or 0),
                "dim": int(cfg.get("latent", {}).get("dim", 0) or 0),
            },
        },
        "source_bindings": source_hashes,
        "architecture_facts": {
            "observed_modalities": observed_modalities,
            "missing_universal_modalities": missing_modalities,
            "separate_encoder_implementations": ["GridEncoder", "MeshParticleEncoder"],
            "shared_encoder_parameters_across_modalities": False,
            "equal_output_shape_is_alignment_evidence": False,
            "operator_stage": {
                "epochs": operator_epochs,
                "encoder_is_optimizer_owned": False,
                "encoder_role": "materialized fixed feature map used to build latent training pairs",
            },
            "decoder_stage": {
                "epochs": decoder_epochs,
                "encoder_is_optimizer_owned": False,
                "encoder_role": "loaded, evaluated, and frozen",
            },
            "joint_codec_operator_stage": {
                "epochs": joint_epochs,
                "encoder_is_optimizer_owned": joint_epochs > 0,
                "rollout_steps": int(joint.get("rollout_steps", 1) or 1),
                "lambda_rollout": float(joint.get("lambda_rollout", 1.0) or 1.0),
                "lambda_reconstruction": float(joint.get("lambda_reconstruction", 0.0) or 0.0),
            },
            "task_router": {
                "enabled": bool(routed.get("enabled", False)),
                "route_source": routed.get("route_source"),
                "route_vocab": list(routed.get("route_vocab", [])),
                "location": "operator",
            },
        },
        "required_measurements": measurements,
        "classification": {
            "end_to_end_shared_candidate": "negative",
            "universal_encoder_claim": "not_tested",
            "common_latent_space_claim": "not_tested",
            "codec_vs_dynamics_causality": "unresolved",
            "family_router_authorized": False,
            "reason": (
                "The run used one grid encoder path and did not measure codec-only quality or "
                "paired cross-discretization alignment. Its joint-versus-ablation gap can arise "
                "in the encoder, decoder, latent operator, or their joint optimization."
            ),
        },
        "next_gate": {
            "name": "universal_latent_encoder_audit",
            "provider_required": False,
            "heldout_access_allowed": False,
            "minimum_evidence": [
                "codec-only reconstruction by task and representation",
                "latent effective rank, covariance, scale, and task leakage",
                "paired grid-mesh-particle alignment for the same physical states",
                "cross-encoding/cross-decoding and resolution invariance",
                "only then a frozen shared-operator versus codec ablation",
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_contract(args.config.resolve(), repo_root=args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
