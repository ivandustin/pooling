"""Equal warmed CPU wall-clock budget per architecture for Tiny Shakespeare."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time

import jax
import numpy as np
import optax

from experiments.benchmark import count, nnx
from experiments.depth_100k_test import Model as Transformer
from experiments.longer_training_100k import TokenProductModel, training_batch, evaluation_data
from experiments.language_depth_test import evaluate
from experiments.readout_experiments import save_weights
from experiments.speed_benchmark import compiled_training


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--chunk", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--save-checkpoints", action="store_true")
    args = p.parse_args()
    if min(args.seconds, args.chunk) <= 0:
        p.error("seconds and chunk must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = (args.repo / "data/shakespeare.txt").read_bytes()
    text = raw.decode()
    encode = {c: i for i, c in enumerate(sorted(set(text)))}
    assert len(encode) == 65
    ids = np.array([encode[c] for c in text], dtype=np.int32)
    train, val, test = np.split(ids, [int(len(ids)*.8), int(len(ids)*.9)])
    _, heldout = evaluation_data("language", val, test)
    config = {
        "task": "Tiny Shakespeare next character", "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "budget_seconds_per_model_seed": args.seconds, "chunk_updates": args.chunk,
        "seeds": args.seeds, "batch": 64, "context": 32, "lr": 0.0003,
        "test_windows": len(heldout[0]), "devices": [str(d) for d in jax.devices()],
        "cpu": platform.processor(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "timing": "includes batch creation, transfer and synchronized Adam updates; excludes compile, warmup and held-out eval; overshoot bounded by final chunk",
    }
    print(json.dumps(config), flush=True)

    def bank(rng):
        pairs = [training_batch("language", rng, train) for _ in range(args.chunk)]
        return tuple(jax.device_put(np.stack(part)) for part in zip(*pairs))

    all_rows = []
    for i, seed in enumerate(args.seeds):
        model_factories = [
            ("learned_norm_pooler", lambda: TokenProductModel("language", seed,
                variant="token_product_norm_query_learned", width=256, hidden=230)),
            ("sinusoidal_transformer", lambda: Transformer("language", seed, "transformer",
                positions="sinusoidal", width=64, ff_hidden=223)),
        ]
        if i % 2:
            model_factories.reverse()
        for name, factory in model_factories:
            model = factory()
            params = count(model)
            assert 99000 <= params <= 100000, (name, params)
            state, run = compiled_training(model, optax.adam(config["lr"]))
            sample = bank(np.random.default_rng(seed))
            t0 = time.perf_counter()
            executable = run.lower(state, *sample).compile()
            jax.block_until_ready(executable(state, *sample))
            init_seconds = time.perf_counter() - t0
            rng = np.random.default_rng(seed)
            steps = 0
            t0 = time.perf_counter()
            while True:
                state, losses = executable(state, *bank(rng))
                jax.block_until_ready((state, losses))
                steps += args.chunk
                elapsed = time.perf_counter() - t0
                if elapsed >= args.seconds:
                    break
            nnx.update(model, state[0])
            row = dict(model=name, seed=seed, parameters=params, updates=steps,
                       training_seconds=elapsed, compile_and_warmup_seconds=init_seconds,
                       last_train_loss=float(losses[-1]), **evaluate(model, *heldout))
            if args.save_checkpoints:
                save_weights(model, args.output / f"{name}-seed-{seed}.npz")
            all_rows.append(row)
            tmp = args.output / "results.tmp"
            tmp.write_text(json.dumps(dict(config=config, runs=all_rows), indent=2) + "\n")
            tmp.replace(args.output / "results.json")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
