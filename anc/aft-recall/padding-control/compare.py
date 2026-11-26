"""Three-way recall with a 60-second CPU training budget per model and seed."""
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


from masked_model import ProbeTransformer
from experiments.positional_encoding import SinusoidalPositions
import jax.numpy as jnp


def make(name, seed):
    assert name == "transformer"
    ref, h = original_make(name, seed)
    model = ProbeTransformer(VOCAB, CONTEXT, 64, nnx.Rngs(seed), depth=2, ff_hidden=h)
    model.position = SinusoidalPositions(64)
    assert count(ref) == count(model) == 99838
    for a, b in zip(jax.tree.leaves(nnx.state(ref)), jax.tree.leaves(nnx.state(model)), strict=True):
        np.testing.assert_array_equal(a, b)
    x, _ = examples(np.random.default_rng(4291), np.full(32, 20))
    np.testing.assert_allclose(ref(x), model(x), atol=2e-6, rtol=2e-6)
    x, _ = examples(np.random.default_rng(4292), np.full(32, 5))
    altered = jnp.asarray(x).at[:, 10:40].set(94)
    mask = x != 95
    np.testing.assert_allclose(model(x), model(altered, mask), atol=2e-6, rtol=2e-6)
    return model, h


def bank(rng):
    loads = rng.choice(LOADS, BATCH * CHUNK)
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
    p.add_argument("--seconds", type=float, default=60)
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
           "seeds": a.seeds, "context": CONTEXT, "vocab": VOCAB, "loads": LOADS,
           "test_memories_per_load": len(tests[5][0]), "all_key_memories_per_load": min(4096, len(tests[5][0])),
           "training_seed_base": 26000, "test_seed_base": 2026112600,
           "padding_is_visible": False, "jit": "warm, compile excluded",
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
