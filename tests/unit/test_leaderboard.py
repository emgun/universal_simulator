from __future__ import annotations

import json
from pathlib import Path

from ups.utils.leaderboard import update_leaderboard


def test_update_leaderboard_creates_csv_and_html(tmp_path: Path) -> None:
    metrics = {"metrics": {"mse": 0.1, "nrmse": 0.2}, "extra": {"ttc": True}}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    csv_path = tmp_path / "leaderboard.csv"
    html_path = tmp_path / "leaderboard.html"

    row = update_leaderboard(
        metrics_path=metrics_path,
        run_id="test_run",
        leaderboard_csv=csv_path,
        leaderboard_html=html_path,
        label="small_eval",
        config="configs/small_eval_burgers.yaml",
        notes="smoke",
        tags={"stage": "operator"},
    )

    assert csv_path.exists()
    assert html_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "metric:mse" in csv_text
    assert "small_eval" in csv_text
    assert "stage" in csv_text

    assert "leaderboard" in row.data["run_id"] or row.data["run_id"] == "test_run"


def test_update_leaderboard_appends_rows(tmp_path: Path) -> None:
    metrics1 = {"metrics": {"mse": 0.1}, "extra": {}}
    metrics2 = {"metrics": {"mse": 0.05}, "extra": {"ttc": False}}
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    m1.write_text(json.dumps(metrics1), encoding="utf-8")
    m2.write_text(json.dumps(metrics2), encoding="utf-8")

    csv_path = tmp_path / "leaderboard.csv"
    html_path = tmp_path / "leaderboard.html"

    update_leaderboard(
        metrics_path=m1,
        run_id="run1",
        leaderboard_csv=csv_path,
        leaderboard_html=html_path,
    )
    update_leaderboard(
        metrics_path=m2,
        run_id="run2",
        leaderboard_csv=csv_path,
        leaderboard_html=html_path,
        label="full_eval",
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    assert csv_text.count("run1") == 1
    assert csv_text.count("run2") == 1
    assert "full_eval" in csv_text
