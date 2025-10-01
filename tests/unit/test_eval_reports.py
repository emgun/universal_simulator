import json
from pathlib import Path

from ups.eval.reports import MetricReport


def test_metric_report_writes_json(tmp_path: Path):
    report = MetricReport(metrics={"nrmse": 0.1}, extra={"notes": "placeholder"})
    out = tmp_path / "report.json"
    report.to_json(out)
    loaded = json.loads(out.read_text())
    assert loaded["metrics"]["nrmse"] == 0.1
    assert loaded["extra"]["notes"] == "placeholder"
