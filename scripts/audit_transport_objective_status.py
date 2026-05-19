#!/usr/bin/env python
from __future__ import annotations

"""Aggregate transport-shift evidence against the original thread objective."""

import argparse
import json
from pathlib import Path
from typing import Any


PASS_STATUSES_BY_MODE = {
    "report": None,
    "literal-achieved": {"literal_achieved"},
    "observed-accepted": {"literal_achieved", "observed_context_achieved", "context_transport_achieved"},
    "context-accepted": {"literal_achieved", "context_transport_achieved"},
}


def _load_json(path: str | None) -> tuple[dict[str, Any] | None, str | None]:
    if not path:
        return None, None
    json_path = Path(path)
    if not json_path.exists():
        return None, f"missing: {json_path}"
    return json.loads(json_path.read_text(encoding="utf-8")), str(json_path)


def _status(record: dict[str, Any] | None) -> str | None:
    return str(record.get("status")) if record and record.get("status") is not None else None


def audit_objective(args: argparse.Namespace) -> dict[str, Any]:
    constant_audit, constant_path = _load_json(args.constant_audit_json)
    observed_audit, observed_path = _load_json(args.observed_audit_json)
    context_audit, context_path = _load_json(args.context_audit_json)
    feature_diag, feature_path = _load_json(args.train_feature_diagnostic_json)

    constant_status = _status(constant_audit)
    observed_status = _status(observed_audit)
    context_status = _status(context_audit)
    feature_conclusion = str(feature_diag.get("conclusion")) if feature_diag else None
    observed_accepted = bool(getattr(args, "accept_observed_context", False))
    context_accepted = bool(getattr(args, "accept_context_transport", False))

    blockers: list[str] = []
    if constant_status == "achieved":
        status = "literal_achieved"
    elif context_accepted and context_status == "achieved":
        status = "context_transport_achieved"
        blockers.append(
            "literal train-only shift objective remains unproven; status depends on accepting two-frame context transport"
        )
    elif observed_accepted and observed_status == "achieved":
        status = "observed_context_achieved"
        blockers.append(
            "literal train-only shift objective remains unproven; status depends on accepting two-frame observed context"
        )
    else:
        status = "literal_blocked"
        if constant_status:
            blockers.append(f"constant train-only audit status is {constant_status}")
        else:
            blockers.append("constant train-only audit evidence is missing")
        if feature_conclusion:
            blockers.append(f"train-only feature diagnostic conclusion is {feature_conclusion}")
        if context_status == "achieved":
            blockers.append("two-frame context transport result is achieved but not accepted for literal objective")
        if observed_status == "achieved":
            blockers.append("observed-context result is achieved but not accepted for literal objective")

    requirements = [
        {
            "name": "real_light_v1_train_val_accessed",
            "status": "satisfied"
            if (constant_audit or observed_audit or context_audit or feature_diag)
            else "missing",
            "evidence": ", ".join(path for path in (constant_path, observed_path, context_path, feature_path) if path)
            or "no evidence artifacts found",
        },
        {
            "name": "fit_transport_shift_only_on_train",
            "status": "satisfied"
            if constant_status == "achieved"
            else "blocked",
            "evidence": f"constant_audit_status={constant_status}; feature_conclusion={feature_conclusion}",
        },
        {
            "name": "validate_on_val_against_sota_guard",
            "status": "satisfied"
            if (
                constant_status == "achieved"
                or (context_accepted and context_status == "achieved")
                or (observed_accepted and observed_status == "achieved")
            )
            else "blocked",
            "evidence": (
                f"constant_audit_status={constant_status}; "
                f"context_audit_status={context_status}; observed_audit_status={observed_status}"
            ),
        },
        {
            "name": "exactly_one_held_out_test_only_after_validation",
            "status": "satisfied"
            if (
                constant_status == "achieved"
                or (context_accepted and context_status == "achieved")
                or (observed_accepted and observed_status == "achieved")
            )
            else "blocked",
            "evidence": (
                f"constant_policy={(constant_audit or {}).get('held_out_test_policy')}; "
                f"context_policy={(context_audit or {}).get('held_out_test_policy')}; "
                f"observed_policy={(observed_audit or {}).get('held_out_test_policy')}"
            ),
        },
        {
            "name": "results_recorded",
            "status": "satisfied"
            if (
                ((constant_audit or {}).get("result_record_policy") or {}).get("passed")
                or ((context_audit or {}).get("result_record_policy") or {}).get("passed")
                or ((observed_audit or {}).get("result_record_policy") or {}).get("passed")
            )
            else "missing",
            "evidence": (
                f"constant_record_policy={(constant_audit or {}).get('result_record_policy')}; "
                f"context_record_policy={(context_audit or {}).get('result_record_policy')}; "
                f"observed_record_policy={(observed_audit or {}).get('result_record_policy')}"
            ),
        },
    ]

    return {
        "status": status,
        "accept_observed_context": observed_accepted,
        "accept_context_transport": context_accepted,
        "blockers": blockers,
        "requirements": requirements,
        "evidence": {
            "constant_audit_json": constant_path,
            "constant_status": constant_status,
            "context_audit_json": context_path,
            "context_status": context_status,
            "observed_audit_json": observed_path,
            "observed_status": observed_status,
            "train_feature_diagnostic_json": feature_path,
            "train_feature_conclusion": feature_conclusion,
        },
        "recommendation": (
            "If two-frame context is benchmark-accepted, prefer the context transport result; "
            "if per-step observed context is accepted, promote observed-context result; "
            "otherwise rebuild split-compatible shards or pursue a richer train-only causal mechanism."
        ),
    }


def exit_code_for_status(status: str, mode: str) -> int:
    allowed = PASS_STATUSES_BY_MODE[mode]
    if allowed is None:
        return 0
    return 0 if status in allowed else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate transport objective status")
    parser.add_argument(
        "--constant-audit-json",
        default="reports/research/sota_loop/transport_shift_goal_audit.json",
    )
    parser.add_argument(
        "--observed-audit-json",
        default="reports/research/sota_loop/observed_transport_shift_goal_audit.json",
    )
    parser.add_argument(
        "--context-audit-json",
        default="reports/research/sota_loop/context_transport_shift_goal_audit.json",
    )
    parser.add_argument(
        "--train-feature-diagnostic-json",
        default="reports/research/sota_loop/train_only_transport_feature_diagnostic_full.json",
    )
    parser.add_argument("--accept-observed-context", action="store_true")
    parser.add_argument("--accept-context-transport", action="store_true")
    parser.add_argument("--require-status", choices=tuple(PASS_STATUSES_BY_MODE), default="report")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    record = audit_objective(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    raise SystemExit(exit_code_for_status(str(record["status"]), args.require_status))


if __name__ == "__main__":
    main()
