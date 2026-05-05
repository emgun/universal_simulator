from __future__ import annotations

from scripts.plan_demo_experiments import build_rows, write_shell


def test_build_rows_applies_tier_caps_and_variant_overrides():
    rows = build_rows(
        tier="light",
        variants=["current_best", "joint48"],
        train_config="configs/train_multitask_heterogeneous_light_best.yaml",
        tasks="burgers1d,advection1d,darcy2d",
        output_root="reports/light_experiments_remote",
        eval_split="test",
        stages="operator,decoder,operator_decoded,joint_codec_operator",
        run_prefix="ups",
        remote_b2_prefix=None,
        required_gb=None,
    )

    assert [row["variant"] for row in rows] == ["current_best", "joint48"]
    assert rows[0]["remote_b2_prefix"] == "light-v1"
    assert rows[0]["train_max_samples"] == 128
    assert rows[0]["eval_max_samples"] == 32
    assert "--decoded-rollout-steps 16" in rows[0]["light_extra_args"]
    assert "stages.joint_codec_operator.epochs=48" in rows[1]["light_extra_args"]


def test_write_shell_defaults_to_dry_run_commands(tmp_path):
    rows = build_rows(
        tier="smoke",
        variants=["task_signature_only"],
        train_config="configs/train_multitask_heterogeneous_light_best.yaml",
        tasks="burgers1d,darcy2d",
        output_root="reports/light_experiments_remote",
        eval_split="test",
        stages="operator,decoder",
        run_prefix="ups",
        remote_b2_prefix="smoke-v1",
        required_gb=4,
    )
    output = tmp_path / "queue.sh"

    write_shell(rows, output, wrapper="scripts/run_remote_light_promotion.sh", env_file="/workspace/.env", dry_run=1)
    text = output.read_text(encoding="utf-8")

    assert "DRY_RUN=1" in text
    assert "RUN_NAME=ups_smoke_task_signature_only" in text
    assert "operator.conditioning.sources=" in text
    assert "bash scripts/run_remote_light_promotion.sh" in text


def test_task_signature_focused_variants_compose_overrides():
    rows = build_rows(
        tier="smoke",
        variants=["task_signature_semigroup0", "task_signature_joint48_rollout4"],
        train_config="configs/train_multitask_heterogeneous_light_best.yaml",
        tasks="burgers1d,advection1d,darcy2d",
        output_root="reports/light_experiments_remote",
        eval_split="test",
        stages="operator,decoder,operator_decoded,joint_codec_operator",
        run_prefix="ups",
        remote_b2_prefix=None,
        required_gb=None,
    )

    assert [row["variant"] for row in rows] == [
        "task_signature_semigroup0",
        "task_signature_joint48_rollout4",
    ]
    assert 'operator.conditioning.sources={"task_id":3,"equation_signature":15}' in rows[0]["light_extra_args"]
    assert "training.lambda_semigroup=0.0" in rows[0]["light_extra_args"]
    assert "stages.joint_codec_operator.epochs=48" in rows[1]["light_extra_args"]
    assert "stages.joint_codec_operator.rollout_steps=4" in rows[1]["light_extra_args"]


def test_task_signature_decoded_and_reconstruction_variants():
    rows = build_rows(
        tier="smoke",
        variants=["task_signature_joint16", "task_signature_opdecoded4_joint16", "task_signature_recon0"],
        train_config="configs/train_multitask_heterogeneous_light_best.yaml",
        tasks="burgers1d,advection1d,darcy2d",
        output_root="reports/light_experiments_remote",
        eval_split="test",
        stages="operator,decoder,operator_decoded,joint_codec_operator",
        run_prefix="ups",
        remote_b2_prefix=None,
        required_gb=None,
    )

    assert [row["variant"] for row in rows] == [
        "task_signature_joint16",
        "task_signature_opdecoded4_joint16",
        "task_signature_recon0",
    ]
    assert "stages.joint_codec_operator.epochs=16" in rows[0]["light_extra_args"]
    assert "stages.operator_decoded.epochs=4" in rows[1]["light_extra_args"]
    assert "stages.joint_codec_operator.epochs=16" in rows[1]["light_extra_args"]
    assert "stages.joint_codec_operator.lambda_reconstruction=0.0" in rows[2]["light_extra_args"]
