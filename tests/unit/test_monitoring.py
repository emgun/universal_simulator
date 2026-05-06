from __future__ import annotations

import json
from pathlib import Path

from ups.utils import monitoring
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


def test_monitoring_session_records_wandb_metadata(tmp_path, monkeypatch):
    class FakeRun:
        id = "abc123"
        name = "training-operator-demo"
        project = "universal-simulator"
        entity = "physics-team"
        group = "light-v1"
        job_type = "light-experiment"
        url = "https://wandb.ai/physics-team/universal-simulator/runs/abc123"

        def log(self, data):
            pass

        def finish(self):
            pass

    class FakeWandb:
        def init(self, **kwargs):
            self.kwargs = kwargs
            return FakeRun()

    fake = FakeWandb()
    monkeypatch.setattr(monitoring, "wandb", fake)

    cfg = {
        "logging": {
            "wandb": {
                "enabled": True,
                "project": "universal-simulator",
                "entity": "physics-team",
                "run_name": "demo",
                "group": "light-v1",
                "tags": ["light-experiment"],
                "job_type": "light-experiment",
            }
        }
    }
    path = tmp_path / "logs" / "training.jsonl"
    session = init_monitoring_session(cfg, component="training-operator", file_path=str(path))
    session.finish()

    metadata_path = path.parent / "wandb_runs.jsonl"
    payload = json.loads(metadata_path.read_text(encoding="utf-8").strip())

    assert fake.kwargs["entity"] == "physics-team"
    assert payload["id"] == "abc123"
    assert payload["component"] == "training-operator"
    assert payload["url"].endswith("/abc123")
