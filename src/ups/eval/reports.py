from __future__ import annotations

"""Simple logging/reporting utilities for UPS."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MetricReport:
    metrics: dict[str, float]
    extra: dict[str, Any] | None = None

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
