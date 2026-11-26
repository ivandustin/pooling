"""Evaluate saved equal-time checkpoints on all possible held-out windows."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from experiments.benchmark import batch, count
from experiments.depth_100k_test import Model as Transformer
from experiments.longer_training_100k import TokenProductModel, evaluation_data
from experiments.language_depth_test import evaluate
from experiments.readout_experiments import score
from experiments.validate_layer_depth import load_weights

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--checkpoints", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
if a.output.exists():
    p.error("refusing to replace an existing result")
original = json.loads((a.checkpoints / "results.json").read_text())
raw = (a.repo / "data/shakespeare.txt").read_bytes()
assert hashlib.sha256(raw).hexdigest() == original["config"]["dataset_sha256"]
text = raw.decode()
encode = {c: i for i, c in enumerate(sorted(set(text)))}
tokens = np.array([encode[c] for c in text], np.int32)
_, val, test = np.split(tokens, [int(.8*len(tokens)), int(.9*len(tokens))])
_, sample = evaluation_data("language", val, test)
x, y = batch(test, np.arange(len(test)-32), 32)
rows = []
for row in original["runs"]:
    name, seed = row["model"], row["seed"]
    if name == "learned_norm_pooler":
        model = TokenProductModel("language", seed, variant="token_product_norm_query_learned",
                                  width=256, hidden=230)
    elif name == "sinusoidal_transformer":
        model = Transformer("language", seed, "transformer", width=64, ff_hidden=223)
    else:
        raise ValueError(name)
    assert count(model) == row["parameters"]
    load_weights(model, a.checkpoints / f"{name}-seed-{seed}.npz")
    sampled = evaluate(model, *sample)
    assert math.isclose(sampled["loss"], row["loss"], abs_tol=1e-5)
    assert math.isclose(sampled["accuracy"], row["accuracy"], abs_tol=1e-6)
    loss_sum = accuracy_sum = 0.0
    for start in range(0, len(y), 128):
        xb, yb = x[start:start+128], y[start:start+128]
        loss, accuracy = score(model, xb, yb)
        loss_sum += float(loss) * len(yb)
        accuracy_sum += float(accuracy) * len(yb)
    loss = loss_sum / len(y)
    point = dict(model=name, seed=seed, loss=loss, accuracy=accuracy_sum/len(y),
                 perplexity=math.exp(loss))
    rows.append(point)
    print(json.dumps(point), flush=True)
a.output.write_text(json.dumps(dict(test_windows=len(y), runs=rows), indent=2) + "\n")
