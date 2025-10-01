from __future__ import annotations

import json
from pathlib import Path

from ups.utils.monitoring import init_monitoring_session


def test_monitoring_session_writes_json(tmp_path):
    cfg = {
        "logging": {
            "backend": "stdout",
            "level": "INFO",
        }
    }
    path = tmp_path / "log.jsonl"
    session = init_monitoring_session(cfg, component="unit-test", file_path=str(path))
    session.log({"stage": "unit-test", "epoch": 0, "loss": 1.23})
    session.finish()
    data = path.read_text(encoding="utf-8").strip().splitlines()
    assert data
    entry = json.loads(data[0])
    assert entry["stage"] == "unit-test"
    assert entry["loss"] == 1.23
