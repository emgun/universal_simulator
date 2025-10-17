from __future__ import annotations

"""Leaderboard aggregation helpers shared between CLI tools."""

from dataclasses import dataclass
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class LeaderboardRow:
    data: Dict[str, Any]


def _load_metrics_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        raise ValueError(f"metrics entry in {path} must be a mapping")
    extra = payload.get("extra") or {}
    if extra and not isinstance(extra, Mapping):
        raise ValueError(f"extra entry in {path} must be a mapping")
    flattened: Dict[str, Any] = {f"metric:{k}": v for k, v in metrics.items()}
    flattened.update({f"extra:{k}": v for k, v in extra.items()})
    return flattened


def _read_existing(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _write_csv(csv_path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _render_html(html_path: Path, fieldnames: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    header_cells = "".join(f"<th>{name}</th>" for name in fieldnames)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{row.get(name, '')}</td>" for name in fieldnames)
        body_rows.append(f"      <tr>{cells}</tr>")
    body = "\n".join(body_rows)
    html = f"""
<html>
  <head>
    <title>UPS Leaderboard</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
      table {{ border-collapse: collapse; width: 100%; max-width: 1080px; }}
      th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
      th {{ background-color: #f3f4f6; }}
      tr:nth-child(odd) {{ background-color: #fafafa; }}
    </style>
  </head>
  <body>
    <h1>UPS Evaluation Leaderboard</h1>
    <table>
      <thead>
        <tr>{header_cells}</tr>
      </thead>
      <tbody>
{body or '        <tr><td colspan="%d">(empty)</td></tr>' % len(fieldnames)}
      </tbody>
    </table>
  </body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def _log_to_wandb(row: Dict[str, Any], *, project: Optional[str], entity: Optional[str], run_name: Optional[str]) -> None:
    try:  # pragma: no cover - optional dependency
        import wandb  # type: ignore
    except Exception as exc:
        print(f"wandb logging skipped: {exc}")
        return

    run = wandb.run
    created_run = False
    if run is None:
        run = wandb.init(project=project, entity=entity, name=run_name, reinit=True)
        created_run = True

    payload = {f"leaderboard/{k}": v for k, v in row.items() if k != "notes"}
    run.log(payload)
    if created_run:
        run.finish()


def update_leaderboard(
    *,
    metrics_path: Path,
    run_id: str,
    leaderboard_csv: Path,
    leaderboard_html: Path,
    label: Optional[str] = None,
    config: Optional[str] = None,
    notes: Optional[str] = None,
    tags: Optional[Mapping[str, str]] = None,
    wandb_log: bool = False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> LeaderboardRow:
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    flattened = _load_metrics_json(metrics_path)
    row: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
    }
    if label:
        row["label"] = label
    if config:
        row["config"] = config
    if notes:
        row["notes"] = notes
    if tags:
        for key, value in tags.items():
            row[key] = value
    row.update(flattened)

    existing = _read_existing(leaderboard_csv)
    existing.append({k: str(v) for k, v in row.items()})

    fieldnames: List[str] = []
    for entry in existing:
        for key in entry.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    _write_csv(leaderboard_csv, existing, fieldnames)
    _render_html(leaderboard_html, fieldnames, existing)

    if wandb_log:
        _log_to_wandb(row, project=wandb_project, entity=wandb_entity, run_name=wandb_run_name)

    return LeaderboardRow(row)


__all__ = ["LeaderboardRow", "update_leaderboard"]
