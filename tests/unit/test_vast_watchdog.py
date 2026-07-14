from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def load_module():
    spec = importlib.util.spec_from_file_location("vast_watchdog", Path("scripts/vast_watchdog.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def receipt(tmp_path: Path, *, deadline: float = 100.0) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": 42,
                "deadline_unix": deadline,
                "success_marker": "PUBLISHED_OK",
                "status": "pending",
            }
        )
    )
    return path


def test_terminal_requires_success_marker_and_zero_exit():
    module = load_module()
    assert (
        module.terminal_reason("PUBLISHED_OK\nREMOTE_BOOTSTRAP_EXIT_STATUS=0", "PUBLISHED_OK")[0]
        == "succeeded"
    )
    assert (
        module.terminal_reason("REMOTE_BOOTSTRAP_EXIT_STATUS=7", "PUBLISHED_OK")[0]
        == "remote_failed"
    )
    assert (
        module.terminal_reason("REMOTE_BOOTSTRAP_EXIT_STATUS=0", "PUBLISHED_OK")[0]
        == "remote_failed"
    )


def test_monitor_destroys_after_success(tmp_path, monkeypatch):
    module = load_module()
    path = receipt(tmp_path)
    monkeypatch.setattr(
        module, "remote_logs", lambda _id: "PUBLISHED_OK\nREMOTE_BOOTSTRAP_EXIT_STATUS=0"
    )
    destroyed = []
    monkeypatch.setattr(
        module, "destroy", lambda instance_id: destroyed.append(instance_id) or True
    )
    assert module.monitor(path, clock=lambda: 0.0, sleeper=lambda _seconds: None) == 0
    payload = json.loads(path.read_text())
    assert destroyed == [42]
    assert payload["status"] == "succeeded"
    assert payload["destroyed"] is True


def test_monitor_destroys_at_deadline_without_polling_logs(tmp_path, monkeypatch):
    module = load_module()
    path = receipt(tmp_path, deadline=5.0)
    monkeypatch.setattr(module, "remote_logs", lambda _id: (_ for _ in ()).throw(AssertionError()))
    destroyed = []
    monkeypatch.setattr(
        module, "destroy", lambda instance_id: destroyed.append(instance_id) or True
    )
    assert module.monitor(path, clock=lambda: 5.0, sleeper=lambda _seconds: None) == 0
    payload = json.loads(path.read_text())
    assert destroyed == [42]
    assert payload["status"] == "timed_out"


def test_destroy_is_idempotent_when_instance_is_absent(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "vast",
        lambda _args: SimpleNamespace(returncode=1, stdout="instance not found", stderr=""),
    )
    assert module.destroy(42)


def test_destroy_retries_until_reconciled(monkeypatch):
    module = load_module()
    calls = []

    def fake_vast(_args):
        calls.append(1)
        if len(calls) < 4:
            return SimpleNamespace(returncode=1, stdout="", stderr="temporary outage")
        return SimpleNamespace(returncode=0, stdout="destroyed", stderr="")

    monkeypatch.setattr(module, "vast", fake_vast)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    assert module.destroy(42)
    assert len(calls) == 4
