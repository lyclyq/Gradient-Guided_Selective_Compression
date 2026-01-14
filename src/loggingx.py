from __future__ import annotations

import csv
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class CSVLogger:
    """
    Simple CSV logger:
      - Always writes to disk (flush per row)
      - Allows new keys to appear over time (extends header dynamically)
    """
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None

    def log(self, step: int, data: Dict[str, Any]) -> None:
        row = {"step": step}
        row.update(data)

        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        else:
            # If new keys appear, extend header (cheap + practical).
            for k in row.keys():
                if k not in self._writer.fieldnames:
                    self._writer.fieldnames = list(self._writer.fieldnames) + [k]

        # Fill missing keys so DictWriter won't complain
        for k in self._writer.fieldnames:
            row.setdefault(k, "")

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


class SwanLabLogger:
    """
    Fail-open SwanLab logger:
      - logs through a background thread
      - if swanlab import/logging fails, training continues
      - drops logs if queue is full
    """
    def __init__(self, enabled: bool, project: str, run_name: str, config: Dict[str, Any]):
        self.enabled = bool(enabled)
        self.failed = False

        self._q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=1024)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sl = None

        if not self.enabled:
            return

        try:
            import swanlab  # type: ignore
            self._sl = swanlab
            self._sl.init(project=project, experiment_name=run_name, config=config)

            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
        except Exception:
            # fail-open
            self.enabled = False
            self.failed = True

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue

            if item is None:
                return

            try:
                if self._sl is not None:
                    self._sl.log(item)
            except Exception:
                # fail-open permanently
                self.failed = True
                self.enabled = False
                return

    def log(self, data: Dict[str, Any]) -> None:
        if not self.enabled or self.failed:
            return
        try:
            self._q.put_nowait(dict(data))
        except Exception:
            # drop if full / any error
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except Exception:
            pass


@dataclass
class RunLogger:
    """
    Unified logger used by trainer:
      logger.log(step, {"train/loss":..., "val/acc":...})
    """
    csv: CSVLogger
    swan: SwanLabLogger

    def log(self, step: int, data: Dict[str, Any]) -> None:
        # CSV expects flat columns
        self.csv.log(step, data)
        # SwanLab: add step field
        self.swan.log({"step": step, **data})

    def close(self) -> None:
        self.csv.close()
        self.swan.close()
