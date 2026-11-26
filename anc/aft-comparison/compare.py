"""Run matched next-character experiments with causal AFT-simple as a baseline."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from experiments.benchmark import batch, count
from experiments.capabilities import ProbeTransformer
from experiments.last_token_probe import TokenInteraction
from experiments.longer_training_100k import evaluation_data, training_batch
from experiments.positional_encoding import SinusoidalPositions
from experiments.readout_experiments import metrics, score
from experiments.speed_benchmark import compiled_training

MODELS = ("pooler", "aft_simple", "transformer")
CONTEXT, WIDTH, BUDGET = 32, 64, 100000
ALICE_BLOB = "0f25e9af379233267a09623e0c650472f9e463ea"


def aft_mix(q, k, v):
    """Equation 5 of Zhai et al. (arXiv:2105.14103), causal prefixes."""
    weights = jnp.exp(k)
    return jax.nn.sigmoid(q) * (
        jnp.cumsum(weights * v, axis=1) / jnp.cumsum(weights, axis=1)
    )


class AFTBlock(nnx.Module):
    """AFT-simple mixer in the same pre-norm residual scaffold as the Transformer."""
    def __init__(self, width, hidden, rng):
        self.norm1 = nnx.LayerNorm(width, rngs=rng)
        self.qkv = nnx.Linear(width, 3 * width, rngs=rng)
        self.proj = nnx.Linear(width, width, rngs=rng)
        self.norm2 = nnx.LayerNorm(width, rngs=rng)
        self.up = nnx.Linear(width, hidden, rngs=rng)
        self.down = nnx.Linear(hidden, width, rngs=rng)

    def __call__(self, x):
        q, k, v = jnp.split(self.qkv(self.norm1(x)), 3, axis=-1)
        x = x + self.proj(aft_mix(q, k, v))
        return x + self.down(jax.nn.gelu(self.up(self.norm2(x))))


def make(name, vocab, seed):
    def build(hidden):
        if name == "pooler":
            return TokenInteraction(seed, "token_product_norm_query_learned",
                                    hidden, 1, CONTEXT, vocab, width=256)
        model = ProbeTransformer(vocab, CONTEXT, WIDTH, nnx.Rngs(seed),
                                 depth=2, ff_hidden=hidden)
        model.position = SinusoidalPositions(WIDTH)
        if name == "aft_simple":
            rng = nnx.Rngs(seed)
            # Consume the token table initialization first, as in the softmax reference
            # with a fixed positional encoding installed before allocating the blocks.
            model.token = nnx.Embed(vocab, WIDTH, rngs=rng)
            model.blocks = nnx.List([AFTBlock(WIDTH, hidden, rng) for _ in range(2)])
            model.norm = nnx.LayerNorm(WIDTH, rngs=rng)
            model.head = nnx.Linear(WIDTH, vocab, rngs=rng)
        return model

    n1, n2 = count(build(1)), count(build(2))
    h = 1 + (BUDGET - n1) // (n2 - n1)
    model = build(h)
    assert 99500 <= count(model) <= BUDGET, (name, count(model))
    return model, h


def load_data(label, path):
    raw = path.read_bytes()
    if label == "alice":
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != ALICE_BLOB:
            raise ValueError("Alice source bytes differ from pinned Gutenberg mirror blob")
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        text = text[text.index("CHAPTER I. Down the Rabbit-Hole"):
                    text.index("End of Project Gutenberg")].rstrip() + "\n"
    else:
        text = raw.decode("utf-8")
    b1, b2 = int(len(text) * .8), int(len(text) * .9)
    # Preserve original Shakespeare vocabulary to reproduce the draft protocol;
    # use train-only vocabulary and one UNK token on the independent corpus.
    vocab = sorted(set(text if label == "shakespeare" else text[:b1]))
    table = {ch: i for i, ch in enumerate(vocab)}
    ids = np.array([table.get(ch, len(vocab)) for ch in text], np.int32)
    return np.split(ids, [b1, b2]), (len(vocab) + (label == "alice")), {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "clean_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "characters": len(text), "splits": [b1, b2-b1, len(text)-b2],
        "unknown_test_chars": int(np.count_nonzero(ids[b2:] == len(vocab)))
    }


def full_score(model, test):
    total_loss = total_accuracy = 0.
    n = len(test) - CONTEXT
    for start in range(0, n, 128):
        x, y = batch(test, np.arange(start, min(n, start+128)), CONTEXT)
        loss, acc = score(model, x, y)
        total_loss += float(loss) * len(y)
        total_accuracy += float(acc) * len(y)
    loss = total_loss / n
    return dict(windows=n, loss=loss, perplexity=math.exp(loss), accuracy=total_accuracy/n)


def sanity():
    rng = np.random.default_rng(17)
    q, k, v = [jnp.asarray(rng.normal(size=(2, 7, 5)), dtype=jnp.float32) for _ in range(3)]
    z = aft_mix(q, k, v)
    ref = jnp.stack([jax.nn.sigmoid(q[:, t]) * jnp.sum(
        jax.nn.softmax(k[:, :t+1], axis=1) * v[:, :t+1], axis=1)
        for t in range(q.shape[1])], axis=1)
    np.testing.assert_allclose(z, ref, rtol=1e-5, atol=2e-6)
    changed = aft_mix(q, k.at[:, 4:].add(50.), v.at[:, 4:].set(0.))
    np.testing.assert_array_equal(z[:, :4], changed[:, :4])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shakespeare", type=Path, required=True)
    p.add_argument("--alice", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.out.exists() or a.seconds <= 0:
        p.error("refusing to overwrite results or use a nonpositive budget")
    sanity()
    corpora = {k: load_data(k, getattr(a, k)) for k in ("shakespeare", "alice")}
    out = {"config": {
      "seconds": a.seconds, "batch": 64, "chunk": 20, "context": CONTEXT,
      "seeds": a.seeds, "learning_rate": .0003, "cpu_affinity": sorted(os.sched_getaffinity(0)),
      "cpu": next(s.split(": ", 1)[1] for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")),
      "platform": platform.platform(), "jax": jax.__version__,
      "datasets": {k: (meta | {"vocab": vocab}) for k, (_, vocab, meta) in corpora.items()},
      "smoke": a.smoke
    }, "runs": []}
    print(json.dumps(out["config"]), flush=True)
    for i, seed in enumerate(a.seeds):
        order = MODELS[i % 3:] + MODELS[:i % 3]
        names = tuple(corpora) if i % 2 == 0 else tuple(reversed(corpora))
        for corpus in names:
            (train, val, test), vocab, _ = corpora[corpus]
            _, heldout = evaluation_data("language", val, test)
            for name in order:
                model, h = make(name, vocab, seed)
                state, train_step = compiled_training(model, optax.adam(.0003))
                def bank(rng):
                    xy = [training_batch("language", rng, train) for _ in range(20)]
                    return tuple(jax.device_put(np.stack(x)) for x in zip(*xy))
                rng = np.random.default_rng(seed)
                warm = bank(np.random.default_rng(seed))
                c0 = time.perf_counter()
                run = train_step.lower(state, *warm).compile()
                jax.block_until_ready(run(state, *warm))
                compile_seconds = time.perf_counter()-c0
                steps = 0
                t0 = time.perf_counter()
                while True:
                    state, loss = run(state, *bank(rng))
                    jax.block_until_ready((state, loss))
                    steps += 20
                    elapsed = time.perf_counter()-t0
                    if elapsed >= a.seconds:
                        break
                nnx.update(model, state[0])
                sampled = metrics(model, *[v[:256] if a.smoke else v for v in heldout])
                sampled["perplexity"] = math.exp(sampled["loss"])
                row = dict(corpus=corpus, model=name, seed=seed, parameters=count(model),
                           hidden=h, elapsed=elapsed, steps=steps, compile_seconds=compile_seconds,
                           train_loss=float(loss[-1]), sampled=sampled,
                           full=None if a.smoke else full_score(model, test))
                if not np.isfinite([row["train_loss"], row["sampled"]["loss"]]).all():
                    raise ArithmeticError("nonfinite loss")
                out["runs"].append(row)
                tmp = a.out.with_suffix(".tmp")
                tmp.write_text(json.dumps(out, indent=2) + "\n")
                tmp.replace(a.out)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
