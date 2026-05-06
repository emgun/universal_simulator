from __future__ import annotations

"""Lightweight monitoring helpers with optional Weights & Biases hooks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore


@dataclass
class MonitoringSession:
    file_path: Optional[Path]
    run: Any = None
    component: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def log(self, data: Dict[str, Any]) -> None:
        if self.file_path is not None:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.file_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(data) + "\n")
        if self.run is not None:
            try:  # pragma: no cover - depends on external service
                self.run.log(data)
            except Exception:  # pragma: no cover
                pass

    def log_image(self, key: str, image_path: Path) -> None:
        if self.run is not None and wandb is not None:  # pragma: no cover - optional
            try:
                self.run.log({key: wandb.Image(str(image_path))}, commit=False)
            except Exception:
                pass

    def finish(self) -> None:
        if self.run is not None:
            try:  # pragma: no cover - depends on external service
                self.run.finish()
            except Exception:
                pass


def init_monitoring_session(
    cfg: Dict[str, Any],
    *,
    component: str,
    file_path: Optional[str] = None,
) -> MonitoringSession:
    logging_cfg = cfg.get("logging", {})
    run = None

    wandb_cfg = logging_cfg.get("wandb", {})
    if wandb_cfg.get("enabled") and wandb is None:
        raise RuntimeError(
            "W&B logging is enabled but the 'wandb' package is not installed. "
            "Install wandb or disable logging.wandb.enabled."
        )
    if wandb_cfg.get("enabled") and wandb is not None:  # pragma: no cover - optional
        run_name_cfg = wandb_cfg.get("run_name")
        run_name = f"{component}-{run_name_cfg}" if run_name_cfg else component
        tags = wandb_cfg.get("tags")
        run = wandb.init(
            project=wandb_cfg.get("project", "universal-simulator"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            config=cfg,
            reinit=True,
            tags=tags,
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type"),
        )

    path = Path(file_path) if file_path else None
    metadata = _wandb_run_metadata(run, component=component, wandb_cfg=wandb_cfg)
    if path is not None and metadata is not None:
        _append_wandb_metadata(path.parent / "wandb_runs.jsonl", metadata)
    return MonitoringSession(path, run, component, metadata)


def _wandb_run_metadata(run: Any, *, component: str, wandb_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if run is None:
        return None
    return {
        "component": component,
        "id": str(getattr(run, "id", "") or ""),
        "name": str(getattr(run, "name", "") or ""),
        "project": str(getattr(run, "project", "") or wandb_cfg.get("project", "") or ""),
        "entity": str(getattr(run, "entity", "") or wandb_cfg.get("entity", "") or ""),
        "group": str(getattr(run, "group", "") or wandb_cfg.get("group", "") or ""),
        "job_type": str(getattr(run, "job_type", "") or wandb_cfg.get("job_type", "") or ""),
        "url": str(getattr(run, "url", "") or ""),
    }


def _append_wandb_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
