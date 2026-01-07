# utils/exp_logger.py
import csv, os, time
from typing import Dict, List

class CSVLogger:
    def __init__(self, path: str, fieldnames: List[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if write_header:
            self.w.writeheader()

    def log(self, row: Dict):
        row = {k: row.get(k, "") for k in self.fieldnames}
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
