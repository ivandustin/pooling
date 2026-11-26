"""Fixed-context associative recall with mixed memory loads."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import optax
from flax import nnx

from experiments.benchmark import count
from experiments.budget_sweep import hidden_for_budget
from experiments.capabilities import ProbeTransformer
from experiments.last_token_probe import TokenInteraction
from experiments.positional_encoding import SinusoidalPositions
from experiments.speed_benchmark import compiled_training
from experiments.language_depth_test import evaluate

CONTEXT, VOCAB, LOADS = 41, 96, (5, 10, 20)


def examples(rng, loads):
    """Unique keys from 0..31, values 32..47, padding 95, query at index 40."""
    n = len(loads)
    keys = rng.random((n, 32)).argsort(axis=1)[:, :20].astype(np.int32)
    vals = rng.integers(32, 48, size=(n, 20), dtype=np.int32)
    q = rng.integers(0, loads)
    mask = np.arange(20)[None, :] < loads[:, None]
    x = np.full((n, CONTEXT), 95, dtype=np.int32)
    x[:, :40:2] = np.where(mask, keys, 95)
    x[:, 1:40:2] = np.where(mask, vals, 95)
    x[:, -1] = keys[np.arange(n), q]
    return x, vals[np.arange(n), q]


@nnx.jit
def predict(model, x):
    return model(x).argmax(axis=-1)


def all_keys(model, x, load):
    """Ask every stored key while holding the memory and its order fixed."""
    x = x[:4096]
    repeated = np.repeat(x, load, axis=0)
    repeated[:, -1] = x[:, :2*load:2].reshape(-1)
    pred = np.concatenate([np.asarray(predict(model, repeated[i:i+256]))
                           for i in range(0, len(repeated), 256)]).reshape(-1, load)
    values = x[:, 1:2*load:2]
    correct = pred == values
    counts = (values[..., None] == np.arange(32, 48)).sum(axis=1)
    return {"memories": len(x), "accuracy": float(correct.mean()),
            "query_blind_ceiling": float((counts.max(axis=1) / load).mean()),
            "all_correct": float(correct.all(axis=1).mean()),
            "query_blind_all_correct_ceiling": float((counts.max(axis=1) == load).mean()),
            "changes_answer": float((pred != pred[:, :1]).any(axis=1).mean())}


def make(name, seed):
    if name == "pooler":
        width = 256
        h = hidden_for_budget(CONTEXT, VOCAB, width, 100000 - 4 * width, 1)
        return TokenInteraction(seed, "token_product_norm_query_learned", h, 1,
                                CONTEXT, VOCAB, width=width), h
    def transformer(h):
        m = ProbeTransformer(VOCAB, CONTEXT, 64, nnx.Rngs(seed), depth=2, ff_hidden=h)
        m.position = SinusoidalPositions(64)
        return m
    h = 1 + (100000 - count(transformer(1))) // (2 * (2 * 64 + 1))
    return transformer(h), h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-keys", action="store_true")
    p.add_argument("--steps", type=int, default=3000, choices=[3000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    if args.output.exists():
        p.error("refusing to replace an existing result")
    tests = {
        load: examples(np.random.default_rng(2026112600 + load), np.full(16384, load))
        for load in LOADS
    }
    report = {"config": {"steps": args.steps, "seeds": args.seeds, "batch": 64,
        "context": CONTEXT, "vocab": VOCAB, "loads": LOADS, "test_per_load": 16384,
        "lr": .0003, "training_seed_base": 26000, "test_seed_base": 2026112600,
        "devices": [str(d) for d in jax.devices()]}, "runs": []}
    for seed in args.seeds:
        for name in (("pooler", "transformer") if seed % 2 == 0 else ("transformer", "pooler")):
            model, hidden = make(name, seed)
            params = count(model)
            assert 99000 <= params <= 100000, (name, params)
            state, run = compiled_training(model, optax.adam(.0003))
            rng = np.random.default_rng(26000 + seed)
            points = {}
            t0 = time.perf_counter()
            for done in range(100, args.steps + 1, 100):
                loads = rng.choice(LOADS, 6400)
                x, y = examples(rng, loads)
                state, losses = run(state, x.reshape(100, 64, CONTEXT), y.reshape(100, 64))
                jax.block_until_ready((state, losses))
                if done in (1000, 3000):
                    nnx.update(model, state[0])
                    points[done] = {load: evaluate(model, *batch) for load, batch in tests.items()}
            row = {"model": name, "seed": seed, "parameters": params, "ff_hidden": hidden,
                   "points": points, "elapsed_seconds_not_comparable": time.perf_counter() - t0}
            if args.all_keys:
                row["all_keys"] = {load: all_keys(model, batch[0], load)
                                   for load, batch in tests.items()}
            report["runs"].append(row)
            tmp = args.output.with_suffix(".tmp")
            tmp.write_text(json.dumps(report, indent=2) + "\n")
            tmp.replace(args.output)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
