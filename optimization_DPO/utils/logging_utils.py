# logging_utils.py (drop-in patch)

import csv, os, time

class CSVLogger:
    def __init__(self, log_dir: str, run_name: str, fieldnames=None, flush_every:int=10):
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, run_name)
        if os.path.isdir(path):
            path = os.path.join(path, "log.csv")
        elif not os.path.splitext(path)[1]:
            path = path + ".csv"

        self.path = path
        self.fieldnames = fieldnames or ["step", "epoch", "split", "loss", "acc"]
        self.flush_every = max(1, int(flush_every))
        self._count = 0

        file_exists = os.path.exists(self.path)
        # 追加写；仅当新文件或空文件时写表头
        self.f = open(self.path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames, extrasaction="ignore")
        if not file_exists or os.path.getsize(self.path) == 0:
            self.w.writeheader()
        print(f"[CSVLogger] Writing logs to: {self.path} (append mode)")

    def log(self, **row):
        # 缺失字段补空，避免 KeyError
        for k in self.fieldnames:
            row.setdefault(k, "")
        self.w.writerow(row)
        self._count += 1
        if self._count % self.flush_every == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


class SwanLogger:
    def __init__(self, project="meta_mask", run_name=None, config=None):
        self.enabled = False
        self.run = None
        self.err = None
        try:
            import swanlab
            self.swanlab = swanlab
            self.run = swanlab.init(
                project=project,
                name=run_name or f"run_{int(time.time())}",
                config=config or {}
            )
            self.enabled = True
            print(f"[SwanLogger] Online logging enabled (project={project}, run={run_name})")
        except Exception as e:
            self.err = str(e)
            print(f"[SwanLogger] Disabled: {self.err}")

    def log(self, data: dict, step: int = None):
        if not self.enabled:
            return
        try:
            if step is not None:
                self.swanlab.log(data, step=step)
            else:
                self.swanlab.log(data)
        except Exception:
            pass

    def finish(self):
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
