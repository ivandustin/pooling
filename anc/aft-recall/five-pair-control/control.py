"""Matched-time Transformer control: train on five pairs, preserve evaluation layout."""
import argparse
import json
import os
import platform
import time
from pathlib import Path

import jax
import numpy as np
import optax
from flax import nnx

from experiments.benchmark import count
from experiments.speed_benchmark import compiled_training
from experiments.language_depth_test import evaluate
from language_comparison import AFTBlock, sanity
from recall_data import CONTEXT, VOCAB, LOADS, all_keys, examples, make as original_make

MODELS = ("transformer",)
BATCH, CHUNK, LR = 64, 20, .0003


def make(name, seed):
    model, hidden = original_make("pooler" if name == "pooler" else "transformer", seed)
    if name == "aft_simple":
        rng = nnx.Rngs(seed)
        model.token = nnx.Embed(VOCAB, 64, rngs=rng)
        model.blocks = nnx.List([AFTBlock(64, hidden, rng) for _ in range(2)])
        model.norm = nnx.LayerNorm(64, rngs=rng)
        model.head = nnx.Linear(64, VOCAB, rngs=rng)
    return model, hidden


def bank(rng):
    loads = rng.choice(LOADS, BATCH * CHUNK)
    loads.fill(5)  # Keep the same RNG draw; change only the number of pairs taught.
    x, y = examples(rng, loads)
    return jax.device_put(x.reshape(CHUNK, BATCH, CONTEXT)), jax.device_put(y.reshape(CHUNK, BATCH))


def check_examples(tests):
    for load, (x, target) in tests.items():
        keys, values = x[:, :2*load:2], x[:, 1:2*load:2]
        assert ((np.diff(np.sort(keys, axis=1), axis=1)) > 0).all()
        assert ((keys >= 0) & (keys < 32)).all()
        assert ((values >= 32) & (values < 48)).all()
        assert (x[:, 2*load:40] == 95).all()
        assert np.array_equal(values[keys == x[:, -1, None]], target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.out.exists() or a.seconds <= 0:
        p.error("refusing to overwrite results or use a nonpositive budget")
    assert all(d.platform == "cpu" for d in jax.devices())
    sanity()
    tests = {load: examples(np.random.default_rng(2026112600 + load), np.full(16384, load))
             for load in LOADS}
    check_examples(tests)
    if a.smoke:
        tests = {load: (x[:256], y[:256]) for load, (x, y) in tests.items()}
    cfg = {"seconds": a.seconds, "batch": BATCH, "chunk": CHUNK, "learning_rate": LR,
           "training_loads": [5], "seeds": a.seeds, "context": CONTEXT, "vocab": VOCAB, "loads": LOADS,
           "test_memories_per_load": len(tests[5][0]), "all_key_memories_per_load": min(4096, len(tests[5][0])),
           "training_seed_base": 26000, "test_seed_base": 2026112600,
           "padding_is_visible": True, "jit": "warm, compile excluded",
           "cpu_affinity": sorted(os.sched_getaffinity(0)), "platform": platform.platform(),
           "cpu": next(s.split(": ", 1)[1] for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),
           "jax": jax.__version__, "smoke": a.smoke}
    report = {"config": cfg, "runs": []}
    print(json.dumps(cfg), flush=True)
    for seed in a.seeds:
        for name in MODELS:
            model, hidden = make(name, seed)
            params = count(model)
            assert 99000 <= params <= 100000, (name, params)
            state, runner = compiled_training(model, optax.adam(LR))
            rng = np.random.default_rng(26000 + seed)
            warm = bank(np.random.default_rng(26000 + seed))
            run = runner.lower(state, *warm).compile()
            jax.block_until_ready(run(state, *warm))
            steps = 0
            started = time.perf_counter()
            while True:
                state, loss = run(state, *bank(rng))
                jax.block_until_ready((state, loss))
                steps += CHUNK
                elapsed = time.perf_counter() - started
                if elapsed >= a.seconds:
                    break
            nnx.update(model, state[0])
            row = {"model": name, "seed": seed, "parameters": params, "hidden": hidden,
                   "elapsed": elapsed, "steps": steps, "train_loss": float(loss[-1]),
                   "one_query": {load: evaluate(model, *data) for load, data in tests.items()},
                   "all_keys": {load: all_keys(model, data[0], load) for load, data in tests.items()}}
            assert np.isfinite(row["train_loss"])
            report["runs"].append(row)
            temp = a.out.with_suffix(".tmp")
            temp.write_text(json.dumps(report, indent=2) + "\n")
            temp.replace(a.out)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
