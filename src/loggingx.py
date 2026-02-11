# src/loggingx.py
from __future__ import annotations

import csv
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_scalar(v: Any) -> Any:
    """
    Convert common numeric-like objects to python scalars for CSV writing.
    Keep strings as-is. For dict/list, stringify safely.
    """
    try:
        # torch.Tensor / numpy scalar
        if hasattr(v, "item") and callable(getattr(v, "item")):
            return v.item()
    except Exception:
        pass

    if isinstance(v, (int, float, str, bool)) or v is None:
        return v

    try:
        return str(v)
    except Exception:
        return repr(v)


# ---------------------------------------------------------------------
# Internal: stable CSV table logger
# ---------------------------------------------------------------------
@dataclass
class _CSVTableLogger:
    """
    Minimal CSV logger with stable schema:

    - Always writes a 'step' column (int).
    - Keeps metric keys EXACTLY as provided (e.g., 'val/acc').
    - Dynamically expands columns as new metric keys appear.
    - Backfills previous rows with empty cells when schema expands.
    """

    run_dir: Path
    enable_csv: bool = True
    csv_name: str = "metrics.csv"

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        _ensure_dir(self.run_dir)

        self.csv_path = self.run_dir / self.csv_name
        self._rows: List[Dict[str, Any]] = []
        self._fieldnames: List[str] = ["step"]
        self._last_flush_t = 0.0

        # Enforce overwrite semantics for metrics.csv (no cross-run append)
        if self.enable_csv and self.csv_path.exists():
            try:
                self.csv_path.write_text("", encoding="utf-8")
            except Exception:
                pass

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enable_csv:
            return

        row: Dict[str, Any] = {"step": int(step)}
        for k, v in (metrics or {}).items():
            row[str(k)] = _as_scalar(v)

        new_keys = [k for k in row.keys() if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)

        self._rows.append(row)

        now = time.time()
        if (now - self._last_flush_t) > 0.5 or len(self._rows) >= 50:
            self.flush()

    def flush(self) -> None:
        if not self.enable_csv or not self._rows:
            return

        all_rows: List[Dict[str, Any]] = []
        if self.csv_path.exists():
            try:
                with self.csv_path.open("r", encoding="utf-8") as f:
                    dr = csv.DictReader(f)
                    if dr.fieldnames:
                        existing = list(dr.fieldnames)
                        if "step" not in existing:
                            existing = ["step"] + existing

                        # merge schema
                        for k in existing:
                            if k not in self._fieldnames:
                                self._fieldnames.append(k)
                        for k in self._fieldnames:
                            if k not in existing:
                                existing.append(k)

                        for r in dr:
                            all_rows.append(dict(r))
            except Exception:
                all_rows = []

        for r in self._rows:
            all_rows.append({k: r.get(k, "") for k in self._fieldnames})

        tmp_path = self.csv_path.with_suffix(self.csv_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fieldnames)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in self._fieldnames})

        os.replace(str(tmp_path), str(self.csv_path))
        self._rows.clear()
        self._last_flush_t = time.time()

    def close(self) -> None:
        self.flush()


# ---------------------------------------------------------------------
# SwanLab: hard-timeout via thread + join(timeout)
# ---------------------------------------------------------------------
class SwanLabLogger:
    """
    Optional SwanLab logger, fail-open and never blocks training.

    - Uses thread + join(timeout) to hard-limit time spent on swanlab calls.
    - Timeout or exception => skip and count as failure.
    - After max_failures, auto-disable.
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        run_name: str,
        config: Dict[str, Any],
        timeout_s: float = 20.0,
        max_failures: int = 3,
    ):
        self.enabled = bool(enabled)
        self._run = None

        self.timeout_s = float(timeout_s)
        self.max_failures = int(max(1, max_failures))
        self._failures = 0

        if not self.enabled:
            return

        try:
            import swanlab  # type: ignore
        except Exception:
            self.enabled = False
            return

        # init also guarded by timeout
        ok, run = self._call_with_timeout(lambda: swanlab.init(project=project, name=run_name, config=config))
        if not ok:
            self.enabled = False
            self._run = None
            return
        self._run = run

    def _mark_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.max_failures:
            self.enabled = False
            self._run = None

    def _call_with_timeout(self, fn):
        """
        Run fn() in a daemon thread, wait timeout_s.
        Return (ok, result).
        - ok=False if timeout or exception.
        """
        if not self.enabled:
            return False, None

        box = {"ok": False, "res": None, "err": None}

        def _target():
            try:
                box["res"] = fn()
                box["ok"] = True
            except Exception as e:
                box["err"] = e
                box["ok"] = False

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=self.timeout_s)

        if t.is_alive():
            # hard timeout: give up; daemon thread won't block process exit
            self._mark_failure()
            return False, None

        if not box["ok"]:
            self._mark_failure()
            return False, None

        return True, box["res"]

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled or self._run is None:
            return

        def _do():
            # common swanlab pattern
            return self._run.log(metrics, step=int(step))

        ok, _ = self._call_with_timeout(_do)
        if not ok:
            return

    def close(self) -> None:
        if not self.enabled or self._run is None:
            return

        # finish/close best-effort
        if hasattr(self._run, "finish"):
            ok, _ = self._call_with_timeout(lambda: self._run.finish())
            if not ok:
                return
        elif hasattr(self._run, "close"):
            ok, _ = self._call_with_timeout(lambda: self._run.close())
            if not ok:
                return


# ---------------------------------------------------------------------
# Public API expected by runner.py
# ---------------------------------------------------------------------
class CSVLogger:
    """
    Runner expects: CSVLogger(metrics_csv_path)
    """
    def __init__(self, csv_path: Path, enabled: bool = True):
        csv_path = Path(csv_path)
        self._impl = _CSVTableLogger(run_dir=csv_path.parent, enable_csv=enabled, csv_name=csv_path.name)

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        self._impl.log(step, metrics)

    def flush(self) -> None:
        self._impl.flush()

    def close(self) -> None:
        self._impl.close()


class RunLogger:
    """
    Aggregator used everywhere else (trainer.py expects .log/.close).
    Runner builds: RunLogger(csv=CSVLogger(...), swan=SwanLabLogger(...))
    """
    def __init__(self, csv: Optional[CSVLogger] = None, swan: Optional[SwanLabLogger] = None):
        self.csv = csv
        self.swan = swan

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if self.csv is not None:
            self.csv.log(step, metrics)
        if self.swan is not None:
            self.swan.log(step, metrics)

    def close(self) -> None:
        if self.csv is not None:
            self.csv.close()
        if self.swan is not None:
            self.swan.close()
