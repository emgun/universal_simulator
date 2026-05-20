from __future__ import annotations

import json
import os
import subprocess
import sys


def test_remote_official_hydration_wrapper_rejects_positional_args():
    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh", "not-an-assignment"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 2
    assert "Pass KEY=VALUE assignments" in proc.stderr


def test_remote_official_hydration_wrapper_generates_missing_plan(tmp_path):
    env = os.environ.copy()
    env["EXECUTE"] = "0"
    env["EXECUTE_DOWNLOADS"] = "0"
    env["PLAN_JSON"] = str(tmp_path / "generated_plan.json")
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert (tmp_path / "generated_plan.json").exists()
    assert '"status": "dry_run"' in proc.stdout


def test_remote_official_hydration_wrapper_can_chain_guarded_post_validation_test(tmp_path):
    env = os.environ.copy()
    env["EXECUTE"] = "0"
    env["EXECUTE_DOWNLOADS"] = "0"
    env["RUN_POST_VALIDATION_TEST"] = "1"
    env["EXECUTE_TEST"] = "0"
    env["PLAN_JSON"] = "reports/research/sota_loop/official_advection_hydration_plan.json"
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    env["OBJECTIVE_STATUS_JSON"] = str(tmp_path / "objective.json")
    env["POST_VALIDATION_TEST_JSON"] = str(tmp_path / "post_validation_test.json")
    (tmp_path / "objective.json").write_text('{"status":"literal_blocked"}', encoding="utf-8")

    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert "expected literal_test_ready" in proc.stdout
    post_validation = json.loads((tmp_path / "post_validation_test.json").read_text(encoding="utf-8"))
    assert post_validation["objective_status"] == "literal_blocked"
    assert post_validation["held_out_test_policy"]["requires_literal_test_ready"] is True


def test_remote_official_hydration_exports_custom_objective_status_to_audit(tmp_path):
    plan = {
        "status": "ready_for_explicit_hydration",
        "selected_official_advection_train_files": [
            "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"
        ],
        "estimated_download_bytes": 1,
        "held_out_test_policy": {
            "test_split_downloaded": False,
            "test_split_sharded": False,
            "test_may_run_only_after_validation_guard": True,
        },
        "commands": {
            "download_official_train_files": [
                "python scripts/download_pdebench_file.py '1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5' --help"
            ],
            "build_train_val_source": "true",
            "build_light_train_val_shards": "true --test-count 0",
            "validate_without_test": "true",
            "objective_audit_after_validation": (
                "REQUIRE_STATUS=literal-test-ready python -c "
                "'import json, os, pathlib; "
                "pathlib.Path(os.environ[\"OBJECTIVE_STATUS_JSON\"]).write_text("
                "json.dumps({\"status\":\"literal_blocked\"}), encoding=\"utf-8\")'"
            ),
        },
        "notes": ["The current workspace has not performed these downloads."],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    objective_path = tmp_path / "custom_objective.json"
    env = os.environ.copy()
    env["EXECUTE"] = "1"
    env["EXECUTE_DOWNLOADS"] = "1"
    env["RUN_POST_VALIDATION_TEST"] = "0"
    env["EXECUTE_TEST"] = "0"
    env["MIN_DOWNLOAD_BYTES"] = "1"
    env["PLAN_JSON"] = str(plan_path)
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    env["OBJECTIVE_STATUS_JSON"] = str(objective_path)
    env["POST_VALIDATION_TEST_JSON"] = str(tmp_path / "post_validation_test.json")

    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert json.loads(objective_path.read_text(encoding="utf-8"))["status"] == "literal_blocked"
    assert not (tmp_path / "post_validation_test.json").exists()


def test_remote_official_hydration_can_publish_report_artifacts(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    rclone_log = tmp_path / "rclone_args.txt"
    rclone = fake_bin / "rclone"
    rclone.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" > {rclone_log}\n",
        encoding="utf-8",
    )
    rclone.chmod(0o755)

    plan = {
        "status": "ready_for_explicit_hydration",
        "selected_official_advection_train_files": [
            "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"
        ],
        "estimated_download_bytes": 1,
        "held_out_test_policy": {
            "test_split_downloaded": False,
            "test_split_sharded": False,
            "test_may_run_only_after_validation_guard": True,
        },
        "commands": {
            "download_official_train_files": [
                "python scripts/download_pdebench_file.py '1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5' --help"
            ],
            "build_train_val_source": "true",
            "build_light_train_val_shards": "true --test-count 0",
            "validate_without_test": "true",
            "objective_audit_after_validation": (
                "REQUIRE_STATUS=literal-test-ready python -c 'import json, os, pathlib; "
                "pathlib.Path(os.environ[\"OBJECTIVE_STATUS_JSON\"]).write_text("
                "json.dumps({\"status\":\"literal_blocked\"}), encoding=\"utf-8\")'"
            ),
        },
        "notes": ["The current workspace has not performed these downloads."],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["EXECUTE"] = "1"
    env["EXECUTE_DOWNLOADS"] = "1"
    env["RUN_POST_VALIDATION_TEST"] = "0"
    env["PUBLISH_ARTIFACTS"] = "1"
    env["B2_KEY_ID"] = "key"
    env["B2_APP_KEY"] = "secret"
    env["B2_BUCKET"] = "bucket"
    env["ARTIFACT_NAME"] = "official_test.tar.gz"
    env["ARTIFACT_PREFIX"] = "remote-runs/test"
    env["MIN_DOWNLOAD_BYTES"] = "1"
    env["PLAN_JSON"] = str(plan_path)
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    env["OBJECTIVE_STATUS_JSON"] = str(tmp_path / "objective.json")
    env["POST_VALIDATION_TEST_JSON"] = str(tmp_path / "post_validation_test.json")

    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert "Published official hydration artifacts" in proc.stdout
    assert "copyto /tmp/official_test.tar.gz UPSB2:bucket/remote-runs/test/official_test.tar.gz" in rclone_log.read_text(
        encoding="utf-8"
    )


def test_remote_official_hydration_can_install_rclone_before_publishing(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python").symlink_to(sys.executable)
    apt_log = tmp_path / "apt_args.txt"
    rclone_log = tmp_path / "rclone_args.txt"
    apt_get = fake_bin / "apt-get"
    apt_get.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {apt_log}\n"
        "if [ \"$1\" = install ]; then\n"
        f"  cat > {fake_bin / 'rclone'} <<'EOF'\n"
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" > {rclone_log}\n"
        "EOF\n"
        f"  chmod +x {fake_bin / 'rclone'}\n"
        "fi\n",
        encoding="utf-8",
    )
    apt_get.chmod(0o755)

    plan = {
        "status": "ready_for_explicit_hydration",
        "selected_official_advection_train_files": [
            "1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5"
        ],
        "estimated_download_bytes": 1,
        "held_out_test_policy": {
            "test_split_downloaded": False,
            "test_split_sharded": False,
            "test_may_run_only_after_validation_guard": True,
        },
        "commands": {
            "download_official_train_files": [
                "python scripts/download_pdebench_file.py '1D/Advection/Train/1D_Advection_Sols_beta0.1.hdf5' --help"
            ],
            "build_train_val_source": "true",
            "build_light_train_val_shards": "true --test-count 0",
            "validate_without_test": "true",
            "objective_audit_after_validation": (
                "REQUIRE_STATUS=literal-test-ready python -c 'import json, os, pathlib; "
                "pathlib.Path(os.environ[\"OBJECTIVE_STATUS_JSON\"]).write_text("
                "json.dumps({\"status\":\"literal_blocked\"}), encoding=\"utf-8\")'"
            ),
        },
        "notes": ["The current workspace has not performed these downloads."],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}/bin{os.pathsep}/usr/bin"
    env["EXECUTE"] = "1"
    env["EXECUTE_DOWNLOADS"] = "1"
    env["RUN_POST_VALIDATION_TEST"] = "0"
    env["PUBLISH_ARTIFACTS"] = "1"
    env["B2_KEY_ID"] = "key"
    env["B2_APP_KEY"] = "secret"
    env["B2_BUCKET"] = "bucket"
    env["ARTIFACT_NAME"] = "official_installed.tar.gz"
    env["ARTIFACT_PREFIX"] = "remote-runs/test"
    env["MIN_DOWNLOAD_BYTES"] = "1"
    env["PLAN_JSON"] = str(plan_path)
    env["VALIDATION_JSON"] = str(tmp_path / "validation.json")
    env["RUN_JSON"] = str(tmp_path / "run.json")
    env["OBJECTIVE_STATUS_JSON"] = str(tmp_path / "objective.json")
    env["POST_VALIDATION_TEST_JSON"] = str(tmp_path / "post_validation_test.json")

    proc = subprocess.run(
        ["bash", "scripts/run_remote_official_hydration.sh"],
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 0
    assert "update" in apt_log.read_text(encoding="utf-8")
    assert "install -y rclone" in apt_log.read_text(encoding="utf-8")
    assert "copyto /tmp/official_installed.tar.gz UPSB2:bucket/remote-runs/test/official_installed.tar.gz" in rclone_log.read_text(
        encoding="utf-8"
    )
