"""Summarize paired results; requires a complete 18-run experiment."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
a = p.parse_args()
data = json.loads(a.results.read_text())
runs = data["runs"]
assert not data["config"]["smoke"] and data["config"]["seconds"] == 60
assert len(runs) == 18
for corpus in ("shakespeare", "alice"):
    print(corpus)
    subsets = {}
    for model in ("pooler", "aft_simple", "transformer"):
        subset = sorted([r for r in runs if r["corpus"] == corpus and r["model"] == model],
                        key=lambda r: r["seed"])
        assert [r["seed"] for r in subset] == [0,1,2]
        assert all(r["elapsed"] >= 60 and math.isfinite(r["full"]["loss"]) for r in subset)
        subsets[model] = subset
        ppl = [r["full"]["perplexity"] for r in subset]
        acc = [100*r["full"]["accuracy"] for r in subset]
        print(f"  {model}: {subset[0]['parameters']:,} params; {subset[0]['full']['windows']:,} windows; "
              f"PPL {mean(ppl):.3f} ± {stdev(ppl):.3f}; "
              f"acc. {mean(acc):.2f}% ± {stdev(acc):.2f} pp; "
              f"{mean(r['steps'] for r in subset):,.0f} steps; "
              f"{mean(r['elapsed'] for r in subset):.2f} s")
    diffs = [l["full"]["perplexity"]-r["full"]["perplexity"]
             for l,r in zip(subsets["pooler"],subsets["aft_simple"])]
    print("  paired test PPL differences (pooler minus AFT-simple):", ", ".join(f"{x:+.3f}" for x in diffs))
