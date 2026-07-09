from __future__ import annotations

from scripts.run_model_side_transport_head_smoke import run_smoke
from scripts.validate_model_side_transport_head_summary import validate_summary


def test_model_side_transport_head_smoke_writes_validator_clean_summary(tmp_path):
    summary = run_smoke(output_dir=tmp_path / "smoke", rollout_steps=16, width=8)

    assert (tmp_path / "smoke" / "summary.json").is_file()
    assert summary["held_out_test_used"] is False
    assert summary["held_out_test_data_read"] is False
    assert summary["extra"]["model_side_transport_head_metrics"]["beta_missing_count"] == 0
    assert summary["extra"]["model_side_transport_head_metrics"]["applied_sample_count"] == 16
    assert summary["metrics"]["task_advection1d_decoded_rollout_nrmse"] < 1e-6
    assert validate_summary(summary) == []
