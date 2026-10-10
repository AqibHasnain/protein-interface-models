from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import IO, Any

import numpy as np

_LOGGER_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def json_default(obj: Any) -> Any:
    """Fallback encoder for values json cannot serialize natively.

    Numpy scalars are common in metric dicts, and run configs carry Paths.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, PurePath):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class JsonlLogger:
    """Append-only structured event log for a single run.

    Every call is flushed to disk so the log survives a killed run.
    """

    def __init__(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "events.jsonl"
        self._fh: IO[str] | None = open(self._path, "a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        # ts and event are written last so caller fields cannot override them.
        record = {**fields, "ts": datetime.now(UTC).isoformat(), "event": event}
        if self._fh is None:
            raise ValueError("JsonlLogger is closed")
        self._fh.write(json.dumps(record, default=json_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    # Guard against adding a second handler when the same name is requested again.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOGGER_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
