"""Lightweight CSV experiment tracking utilities."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EXPERIMENT_LOG_HEADERS = [
    "timestamp",
    "dataset_name",
    "model_name",
    "key_metrics",
    "hyperparameters",
]


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for experiment events."""
    return datetime.now(timezone.utc).isoformat()


def to_compact_json(payload: dict[str, Any]) -> str:
    """Serialize dictionaries as compact JSON for CSV storage."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def append_experiment_log_row(log_path: str | Path, row: dict[str, Any]) -> None:
    """Append one experiment row to CSV, creating file + headers if needed."""
    resolved = Path(log_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)

    file_exists = resolved.exists()
    should_write_header = (not file_exists) or resolved.stat().st_size == 0

    normalized = {header: row.get(header, "") for header in EXPERIMENT_LOG_HEADERS}
    with resolved.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EXPERIMENT_LOG_HEADERS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(normalized)