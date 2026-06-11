from __future__ import annotations

import os
import stat
import subprocess
import tarfile


def test_remote_data_budget_sweep_generates_summary_before_publishing(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    rclone_log = tmp_path / "rclone.log"
    fake_rclone = fake_bin / "rclone"
    fake_rclone.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'echo "$@" >> "$RCLONE_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_rclone.chmod(fake_rclone.stat().st_mode | stat.S_IXUSR)

    python_log = tmp_path / "python.log"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
echo "$@" >> "$PYTHON_LOG"
script="$1"
shift

if [ "$script" = "scripts/run_light_experiment.py" ]; then
  output_root=""
  run_name=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --output-root) output_root="$2"; shift 2 ;;
      --name) run_name="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$output_root/$run_name"
  cat > "$output_root/$run_name/summary.json" <<JSON
{"run_name":"$run_name","duration_sec":1.0,"extra":{"decoded_split":"val","decoded_task_roots":{},"decoded_decoded_context_roll_shift_estimator":{},"decoded_decoded_data_conditioned_roll_shift_estimator":{},"decoded_decoded_observed_roll_shift_estimator":{},"decoded_decoded_prediction_roll_shift_estimator":{}},"extra_evaluations":{},"metrics":{"decoded_rollout_nrmse":0.5,"decoded_h16_nrmse":0.6}}
JSON
  exit 0
fi

if [ "$script" = "scripts/summarize_data_budget_sweep.py" ]; then
  output_json=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --output-json) output_json="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  mkdir -p "$(dirname "$output_json")"
  cat > "$output_json" <<'JSON'
{"measurement_type":"p1_data_budget_sweep_results","held_out_test_data_read":false}
JSON
  exit 0
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)

    artifact_name = f"data_budget_sweep_{tmp_path.name}.tar.gz"
    artifact_path = f"/tmp/{artifact_name}"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "PYTHON_LOG": str(python_log),
            "RCLONE_LOG": str(rclone_log),
            "B2_KEY_ID": "key-id",
            "B2_APP_KEY": "app-key",
            "B2_BUCKET": "bucket",
            "DRY_RUN": "0",
            "FETCH_DATA": "0",
            "RUN_SWEEP": "1",
            "TRAIN_BUDGETS": "128,256",
            "PUBLISH_SWEEP_ARTIFACTS": "1",
            "SWEEP_ARTIFACT_NAME": artifact_name,
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "OUTPUT_ROOT": str(tmp_path / "data_budget_runs"),
            "DATA_ROOT": str(tmp_path / "data"),
            "ALLOW_WANDB": "0",
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_data_budget_sweep.sh"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert "Published sweep artifacts: b2://bucket/remote-runs/data-budget-sweep/" in proc.stdout
    python_calls = python_log.read_text(encoding="utf-8")
    assert "scripts/run_light_experiment.py" in python_calls
    assert "scripts/summarize_data_budget_sweep.py" in python_calls
    assert "training.batch_size=4" in python_calls
    assert "data.max_samples=128" in python_calls
    assert "data.max_samples=256" in python_calls
    assert "copyto /tmp/" in rclone_log.read_text(encoding="utf-8")
    with tarfile.open(artifact_path, "r:gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("pipeline/data_budget_sweep_summary.json") for name in names)


def test_remote_data_budget_sweep_refuses_test_split(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "FETCH_DATA": "0",
            "RUN_SWEEP": "0",
            "EVAL_SPLIT": "test",
            "PIPELINE_ROOT": str(tmp_path / "pipeline"),
            "OUTPUT_ROOT": str(tmp_path / "data_budget_runs"),
            "DATA_ROOT": str(tmp_path / "data"),
        }
    )

    proc = subprocess.run(
        ["bash", "scripts/run_remote_data_budget_sweep.sh"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert proc.returncode == 1
    assert "validation-only" in proc.stderr
