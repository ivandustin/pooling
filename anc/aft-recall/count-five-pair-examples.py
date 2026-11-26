"""Count five-pair examples by replaying the original training RNG, without training."""
import argparse
import json
from pathlib import Path

import numpy as np

from recall_data import examples

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mixed", type=Path)
parser.add_argument("fixed", type=Path)
args = parser.parse_args()
mixed, fixed = (json.loads(path.read_text()) for path in (args.mixed, args.fixed))
cfg = mixed["config"]
assert fixed["config"]["training_loads"] == [5]
fixed_steps = {r["seed"]: r["steps"] for r in fixed["runs"] if r["model"] == "transformer"}
for run in mixed["runs"]:
    if run["model"] != "transformer":
        continue
    seed, steps = run["seed"], run["steps"]
    rng = np.random.default_rng(cfg["training_seed_base"] + seed)
    assert steps % cfg["chunk"] == 0
    five = 0
    for _ in range(steps // cfg["chunk"]):
        loads = rng.choice(cfg["loads"], cfg["batch"] * cfg["chunk"])
        five += int(np.count_nonzero(loads == 5))
        # Generate the rest of each bank to advance the RNG just as training did.
        examples(rng, loads)
    print(f"seed {seed}: mixed {five:,}; fixed-five {fixed_steps[seed] * cfg['batch']:,}")
