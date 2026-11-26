"""Validate data and print three-seed memory-load results in manuscript order."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev
import numpy as np
from recall_load import LOADS, examples

p = argparse.ArgumentParser()
p.add_argument("results", nargs="?", type=Path, default=Path("recall-load-results.json"))
d = json.loads(p.parse_args().results.read_text())
cfg = d["config"]
assert len(d["runs"]) == 6
for load in LOADS:
    x, y = examples(np.random.default_rng(cfg["test_seed_base"] + load),
                    np.full(cfg["test_per_load"], load))
    keys, values = x[:, :2*load:2], x[:, 1:2*load:2]
    match = keys == x[:, -1, None]
    assert (match.sum(axis=1) == 1).all()
    assert np.array_equal(values[np.arange(len(x)), match.argmax(axis=1)], y)
    freq = (values[..., None] == np.arange(32, 48)).sum(axis=1)
    baseline = 100 * (freq.argmax(axis=1) + 32 == y).mean()
    row = [str(load), f"{baseline:.2f}"]
    for step in (1000, 3000):
        for model in ("pooler", "transformer"):
            runs = [r for r in d["runs"] if r["model"] == model]
            assert {r["seed"] for r in runs} == {0, 1, 2}
            a = [100*r["points"][str(step)][str(load)]["accuracy"] for r in runs]
            row.append(f"\\pmv{{{mean(a):.2f}}}{{{stdev(a):.2f}}}")
    print(" & ".join(row) + " " + chr(92)*2)
