"""Match the mixed-run five-pair examples and update counts, with other losses zeroed."""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from compare_recall import BATCH, CHUNK, CONTEXT, LOADS, LR, bank, check_examples, make
from experiments.benchmark import count
from experiments.speed_benchmark import compiled_training
from recall_data import all_keys, examples


def weighted_training(model):
    optimizer = nnx.Optimizer(model, optax.adam(LR), wrt=nnx.Param)
    graph, state = nnx.split((model, optimizer))

    def body(carry, batch):
        model, opt = nnx.merge(graph, carry)
        x, y, weight = batch

        def loss_fn(mod):
            losses = optax.softmax_cross_entropy_with_integer_labels(mod(x), y)
            return jnp.where(weight, losses, 0.0).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return nnx.state((model, opt)), loss

    @jax.jit
    def train(state, xs, ys, weights):
        return jax.lax.scan(body, state, (xs, ys, weights))

    return state, train


def replay_bank(rng):
    loads = rng.choice(LOADS, BATCH * CHUNK)
    x, y = examples(rng, loads)
    w = (loads == 5).reshape(CHUNK, BATCH)
    return x.reshape(CHUNK, BATCH, CONTEXT), y.reshape(CHUNK, BATCH), w


def verify(seed, training_seed, state, train, warm):
    for v, recorded in zip(warm[:2], bank(np.random.default_rng(training_seed)), strict=True):
        np.testing.assert_array_equal(v, recorded)
    ref_model, _ = make("transformer", seed)
    ref_state, ref_train = compiled_training(ref_model, optax.adam(LR))
    x, y, w = warm
    first = (x[:1], y[:1], w[:1])
    a = ref_train(ref_state, *first[:2])
    b = train(state, *first[:2], np.ones_like(first[2]))
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_allclose(left, right, atol=2e-6, rtol=2e-6)
    a = train(state, *first)
    b = train(state, first[0], np.where(first[2], first[1], (first[1] + 1) % 96), first[2])
    for left, right in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(left, right)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mixed", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite results")
    assert all(d.platform == "cpu" for d in jax.devices())
    baseline = json.loads(args.mixed.read_text())
    cfg = baseline["config"]
    assert (cfg["batch"], cfg["chunk"], cfg["context"], cfg["learning_rate"]) == (BATCH, CHUNK, CONTEXT, LR)
    assert cfg["padding_is_visible"] and tuple(cfg["loads"]) == LOADS
    tests = {load: examples(np.random.default_rng(cfg["test_seed_base"] + load),
                            np.full(cfg["test_memories_per_load"], load))
             for load in LOADS}
    check_examples(tests)
    result = {"config": cfg | {"stopping_rule": "mixed run updates by seed",
                               "loss_loads": [5], "loss_divisor": BATCH,
                               "smoke": args.smoke}, "runs": []}
    for row in baseline["runs"]:
        if row["model"] != "transformer":
            continue
        seed = row["seed"]
        train_seed = cfg["training_seed_base"] + seed
        steps = CHUNK if args.smoke else row["steps"]
        assert steps % CHUNK == 0
        model, hidden = make("transformer", seed)
        assert (count(model), hidden) == (row["parameters"], row["hidden"])
        state, train = weighted_training(model)
        warm = replay_bank(np.random.default_rng(train_seed))
        verify(seed, train_seed, state, train, warm)
        rng = np.random.default_rng(train_seed)
        train = train.lower(state, *warm).compile()
        jax.block_until_ready(train(state, *warm))
        active = 0
        started = time.perf_counter()
        for _ in range(steps // CHUNK):
            x, y, w = replay_bank(rng)
            active += int(w.sum())
            state, loss = train(state, x, y, w)
            jax.block_until_ready((state, loss))
        elapsed = time.perf_counter() - started
        assert np.isfinite(np.asarray(loss)).all()
        nnx.update(model, state[0])
        kept = 256 if args.smoke else cfg["all_key_memories_per_load"]
        assert kept <= 4096
        out = {"model": "transformer", "seed": seed, "steps": steps, "elapsed": elapsed,
               "parameters": count(model), "five_pair_examples": active,
               "all_keys": {load: all_keys(model, x[:kept], load) for load, (x, _) in tests.items()}}
        result["runs"].append(out)
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(args.out)
        print(json.dumps(out), flush=True)
        if args.smoke:
            return


if __name__ == "__main__":
    main()
