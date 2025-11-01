import csv, os, time

class CSVLogger:
    def __init__(self, path, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.w.writeheader()
        self.flush_every = 1
        self._count = 0

    def log(self, row: dict):
        self.w.writerow(row)
        self._count += 1
        if self._count % self.flush_every == 0:
            self.f.flush()

    def close(self):
        self.f.flush()
        self.f.close()
