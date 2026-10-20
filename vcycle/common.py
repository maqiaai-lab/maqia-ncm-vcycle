import json
import os
import time

import torch

from vcycle.loading import REPO_ROOT

ARTIFACTS = os.environ.get("NCM_ARTIFACTS", os.path.join(REPO_ROOT, "artifacts"))
DATA = os.environ.get("NCM_DATA", os.path.join(REPO_ROOT, "data"))
RESULTS = os.path.join(REPO_ROOT, "results")


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def batches(n, bs):
    for s in range(0, n, bs):
        yield slice(s, min(n, s + bs))


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.s = time.perf_counter() - self.t0
