from __future__ import annotations

import json
import math

import pytest
import torch

from ups.data.latent_pairs import prepare_conditioning
from ups.data.parameter_conditioning import resolve_parameter_conditioning
from ups.utils.config_loader import load_config_with_includes

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_resolver_builds_sorted_union_and_task_specific_contract() -> None:
    contract = resolve_parameter_conditioning(
        {
            "task": ["advection1d", "burgers1d"],
            "root": "shared",
            "task_roots": {"burgers1d": "burgers"},
            "task_param_keys": {"advection1d": ["beta"], "burgers1d": ["nu"]},
            "parameter_transforms": {
                "advection1d": {
                    "beta": {
                        "kind": "log10_zscore",
                        "mean": 0.5,
                        "std": 2.0,
                        "count": 8,
                        "source_sha256": SHA_A,
                    }
                }
            },
        }
    )

    assert contract.param_vocab == ("beta", "nu")
    assert contract.task_vocab == ("advection1d", "burgers1d")
    assert contract.param_keys_for("burgers1d") == ("nu",)
    assert contract.root_for("advection1d") == "shared"
    assert contract.root_for("burgers1d") == "burgers"
    transformed = prepare_conditioning(
        {"beta": torch.tensor(10.0)},
        None,
        1,
        param_vocab=contract.param_vocab,
        parameter_transforms=contract.transforms_for("advection1d"),
    )
    assert transformed["param_beta"].item() == pytest.approx((1.0 - 0.5) / 2.0)
    assert transformed["param_presence"].tolist() == [[1.0, 0.0]]


def test_resolver_preserves_legacy_param_keys_and_raw_values() -> None:
    contract = resolve_parameter_conditioning(
        {"task": "burgers1d", "root": "legacy", "param_keys": ["nu"]}
    )
    assert contract.param_vocab == ("nu",)
    assert contract.task_vocab == ("burgers1d",)
    assert contract.param_keys_for("burgers1d") == ("nu",)
    cond = prepare_conditioning(
        {"nu": torch.tensor(0.25)}, None, 1, param_vocab=contract.param_vocab
    )
    assert cond["param_nu"].item() == pytest.approx(0.25)


def test_explicit_conditioning_schema_is_fixed_across_selected_task_subsets() -> None:
    data_cfg = {
        "task": ["advection1d", "burgers1d", "darcy2d"],
        "conditioning_schema": {
            "task_vocab": ["advection1d", "burgers1d", "darcy2d"],
            "param_vocab": ["nu", "beta", "forcing"],
        },
        "task_param_keys": {
            "advection1d": ["beta"],
            "burgers1d": ["nu"],
            "darcy2d": ["beta"],
        },
    }

    shared = resolve_parameter_conditioning(data_cfg)
    specialist = resolve_parameter_conditioning(data_cfg, task_names=("darcy2d",))

    assert shared.task_names == ("advection1d", "burgers1d", "darcy2d")
    assert specialist.task_names == ("darcy2d",)
    assert (
        shared.task_vocab
        == specialist.task_vocab
        == (
            "advection1d",
            "burgers1d",
            "darcy2d",
        )
    )
    assert shared.param_vocab == specialist.param_vocab == ("nu", "beta", "forcing")
    assert specialist.param_keys_for("darcy2d") == ("beta",)


@pytest.mark.parametrize(
    "config,task_names,match",
    [
        (
            {
                "task": ["a"],
                "conditioning_schema": {"task_vocab": ["b"], "param_vocab": ["beta"]},
                "task_param_keys": {"b": ["beta"]},
            },
            None,
            "Selected tasks are not covered",
        ),
        (
            {
                "task": ["a", "b"],
                "conditioning_schema": {"task_vocab": ["a", "b"], "param_vocab": ["nu"]},
                "task_param_keys": {"a": ["beta"], "b": ["nu"]},
            },
            ("a",),
            "Selected parameter keys are not covered",
        ),
        (
            {
                "task": ["a"],
                "conditioning_schema": {"task_vocab": ["a"]},
                "task_param_keys": {"a": ["beta"]},
            },
            None,
            "requires explicit task_vocab and param_vocab",
        ),
    ],
)
def test_explicit_conditioning_schema_fails_closed_on_missing_coverage(
    config: dict, task_names: tuple[str, ...] | None, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_parameter_conditioning(config, task_names=task_names)


@pytest.mark.parametrize(
    "config,match",
    [
        (
            {"task": ["a"], "task_param_keys": {"other": ["beta"]}},
            "unknown task entries",
        ),
        (
            {"task": ["a", "b"], "task_param_keys": {"a": ["beta"]}},
            "missing selected tasks",
        ),
        (
            {
                "task": ["a"],
                "task_param_keys": {"a": ["beta"]},
                "parameter_transforms": {"a": {"nu": "log10"}},
            },
            "outside the task schema",
        ),
        (
            {
                "task": ["a"],
                "task_param_keys": {"a": ["beta"]},
                "parameter_transforms": {"a": {"beta": {"kind": "log10_zscore"}}},
            },
            "requires frozen mean, std, count, and source_sha256",
        ),
    ],
)
def test_resolver_fails_closed_on_schema_mismatch(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_parameter_conditioning(config)


def test_log_transform_domain_fails_closed() -> None:
    contract = resolve_parameter_conditioning(
        {
            "task": "a",
            "param_keys": ["beta"],
            "parameter_transforms": {"a": {"beta": "log10"}},
        }
    )
    with pytest.raises(ValueError, match="must be positive"):
        prepare_conditioning(
            {"beta": torch.tensor(0.0)},
            None,
            1,
            parameter_transforms=contract.transforms_for("a"),
        )


def test_frozen_transform_source_must_match_training_lock(tmp_path) -> None:
    lock_path = tmp_path / "training.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": "a-train",
                        "path": "a_train.h5",
                        "role": "train",
                        "checksums": {"sha256": "b" * 64},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not match the task's training object"):
        resolve_parameter_conditioning(
            {
                "task": "a",
                "param_keys": ["beta"],
                "data_lock_path": str(lock_path),
                "parameter_transforms": {
                    "a": {
                        "beta": {
                            "kind": "log10_zscore",
                            "mean": 0.0,
                            "std": 1.0,
                            "count": 1,
                            "source_sha256": "a" * 64,
                        }
                    }
                },
            }
        )


def test_active_a4_config_uses_auto_sources_and_frozen_transforms() -> None:
    cfg = load_config_with_includes("configs/a4_strat_v1_baselines.yaml")
    assert cfg["operator"]["conditioning"]["sources"] is None
    contract = resolve_parameter_conditioning(cfg["data"])
    assert contract.param_vocab == ("beta", "nu")
    for task, key in (("advection1d", "beta"), ("burgers1d", "nu"), ("darcy2d", "beta")):
        spec = contract.transforms_for(task)[key]
        assert spec.kind == "log10_zscore"
        assert spec.count is not None and spec.count > 0
        assert spec.source_sha256 is not None and len(spec.source_sha256) == 64
        assert spec.mean is not None and math.isfinite(spec.mean)
        assert spec.std is not None and spec.std > 0.0
