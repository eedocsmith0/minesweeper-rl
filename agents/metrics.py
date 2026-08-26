"""Minimal metric/board logging for training pipelines (JSONL files)."""
import json
import os
import time


class MetricLogger:
    def __init__(self, run_name="default", root="runs"):
        self.dir = os.path.join(root, run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self.boards_path = os.path.join(self.dir, "boards.jsonl")

    def log(self, **record):
        record["ts"] = time.time()
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_board(self, env, result, moves, extra=None):
        h, w = env.height, env.width
        cells = []
        for r in range(h):
            row = []
            for c in range(w):
                if env.revealed[r, c]:
                    row.append(11 if env.mines[r, c]
                               else int(env.counts[r, c]) + 1)
                elif env.flagged[r, c]:
                    row.append(10)
                else:
                    row.append(0)
            cells.append(row)
        rec = {"cells": cells, "result": result, "moves": moves,
               "width": w, "height": h, "ts": time.time()}
        if extra:
            rec.update(extra)
        with open(self.boards_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
