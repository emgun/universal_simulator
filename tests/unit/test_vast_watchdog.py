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


def receipt(tmp_path: Path, *, deadline: float = 100.0, startup_deadline: float = 10.0) -> Path:
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {
                "instance_id": 42,
                "deadline_unix": deadline,
                "startup_deadline_unix": startup_deadline,
                "success_marker": "PUBLISHED_OK",
                "status": "pending",
                "bootstrap_started": False,
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
        module,
        "remote_logs",
        lambda _id: "REMOTE_BOOTSTRAP_STARTED=1\nPUBLISHED_OK\nREMOTE_BOOTSTRAP_EXIT_STATUS=0",
    )
    monkeypatch.setattr(module, "instance_state", lambda _id: {"actual_status": "running"})
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


def test_monitor_destroys_when_bootstrap_never_starts(tmp_path, monkeypatch):
    module = load_module()
    path = receipt(tmp_path, deadline=100.0, startup_deadline=5.0)
    monkeypatch.setattr(module, "remote_logs", lambda _id: "waiting on logs")
    monkeypatch.setattr(
        module,
        "instance_state",
        lambda _id: {"actual_status": "loading", "status_msg": "could not resolve host"},
    )
    destroyed = []
    monkeypatch.setattr(
        module, "destroy", lambda instance_id: destroyed.append(instance_id) or True
    )

    assert module.monitor(path, clock=lambda: 5.0, sleeper=lambda _seconds: None) == 0
    payload = json.loads(path.read_text())
    assert destroyed == [42]
    assert payload["status"] == "startup_failed"
    assert payload["bootstrap_started"] is False
    assert payload["last_instance_status"] == "loading"


def test_monitor_marks_bootstrap_before_startup_deadline(tmp_path, monkeypatch):
    module = load_module()
    path = receipt(tmp_path, deadline=100.0, startup_deadline=5.0)
    logs = iter(
        [
            "REMOTE_BOOTSTRAP_STARTED=1",
            "REMOTE_BOOTSTRAP_STARTED=1\nPUBLISHED_OK\nREMOTE_BOOTSTRAP_EXIT_STATUS=0",
        ]
    )
    monkeypatch.setattr(module, "remote_logs", lambda _id: next(logs))
    monkeypatch.setattr(module, "instance_state", lambda _id: {"actual_status": "running"})
    monkeypatch.setattr(module, "destroy", lambda _id: True)
    clocks = iter([0.0, 0.0, 6.0, 6.0])

    assert module.monitor(path, clock=lambda: next(clocks), sleeper=lambda _seconds: None) == 0
    payload = json.loads(path.read_text())
    assert payload["status"] == "succeeded"
    assert payload["bootstrap_started"] is True


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


def test_instance_exists_uses_collection_and_detects_absence(monkeypatch):
    module = load_module()
    calls = []

    def fake_vast(args):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='[{"id": 7}]', stderr="")

    monkeypatch.setattr(module, "vast", fake_vast)
    assert module.instance_exists(42) is False
    assert calls == [["show", "instances", "--raw"]]


def test_instance_exists_fails_safe_on_control_plane_errors(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "vast",
        lambda _args: SimpleNamespace(returncode=1, stdout="", stderr="temporary outage"),
    )
    assert module.instance_exists(42) is True
