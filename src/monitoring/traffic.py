"""Append-only record of the traffic the API receives, for monitoring.

Each line of a JSON Lines file is one ``/classify`` or ``/labels`` request:

    <root>/<mode>/predictions-YYYY-MM-DD.jsonl
    <root>/<mode>/labels-YYYY-MM-DD.jsonl

``mode`` is ``live``, ``sandbox`` or ``unknown`` (anything else, including
requests without the field), so sandbox and manual traffic never mix with the
live data used for drift and performance monitoring. Dates are UTC.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

MODES = ("live", "sandbox")
UNKNOWN_MODE = "unknown"


def normalize_mode(mode: Optional[str]) -> str:
    """The folder a request belongs to; never a path supplied by the client."""
    mode = (mode or "").strip().lower()
    return mode if mode in MODES else UNKNOWN_MODE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TrafficLog:
    """Thread-safe appender; the API serves requests from a thread pool."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._lock = threading.Lock()

    def path(self, kind: str, mode: str, received_at: str) -> Path:
        return self.root / normalize_mode(mode) / f"{kind}-{received_at[:10]}.jsonl"

    def append(self, kind: str, entry: dict[str, Any]) -> Path:
        """Write one entry, which must contain ``mode`` and ``received_at``."""
        path = self.path(kind, entry["mode"], entry["received_at"])
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line)
        return path
