from __future__ import annotations

"""Simple logging/reporting utilities for UPS."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


@dataclass
class MetricReport:
    metrics: Dict[str, float]
    extra: Dict[str, Any] | None = None

    def to_json(self, path: str | Path) -> None:
        data = asdict(self)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

