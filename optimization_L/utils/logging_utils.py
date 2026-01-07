
import csv, os, time

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        self._w.writeheader()
        self._f.flush()

    def log(self, row: dict):
        row2 = {k: row.get(k, None) for k in self.fieldnames}
        self._w.writerow(row2)
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

class SwanLogger:
    def __init__(self, project="optimization", run_name=None, config=None):
        self.ok = False
        self.run = None
        try:
            import swanlab
            self.run = swanlab.init(project=project, experiment_name=run_name or "run", config=config or {})
            self.ok = True
        except Exception as e:
            self.err = str(e)
            self.ok = False

    def log(self, metrics: dict, step=None):
        if not self.ok: return
        try:
            self.run.log(metrics, step=step)
        except Exception:
            pass

    def finish(self):
        if not self.ok: return
        try:
            self.run.finish()
        except Exception:
            pass
