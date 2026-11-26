"""Check completed runs and summarize per-key and complete-memory accuracy."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
a = p.parse_args()
d = json.loads(a.results.read_text())
c = d["config"]
assert not c["smoke"] and c["seconds"] == 300 and c["seeds"] == [0,1,2]
runs = d["runs"]
models = ("pooler", "aft_simple", "transformer")
assert len(runs) == 9 and {(r["model"], r["seed"]) for r in runs} == {(m, s) for m in models for s in c["seeds"]}

def avg(values):
    values = list(values)
    return f"{mean(values):.2f} ± {stdev(values):.2f}"

for name in models:
    rr = [r for r in runs if r["model"] == name]
    assert all(r["elapsed"] >= 300 and r["steps"] > 0 for r in rr)
    print(f"{name}: {rr[0]['parameters']:,} parameters, {mean(r['steps'] for r in rr):,.0f} steps in {mean(r['elapsed'] for r in rr):.2f} s")
for load in map(str, c["loads"]):
    b = {r["all_keys"][load]["query_blind_ceiling"] for r in runs}
    assert len(b) == 1
    print(f"\n{load} pairs: no-query baseline {100*b.pop():.2f}% on the all-key set")
    for name in models:
        rr = [r for r in runs if r["model"] == name]
        assert all(r["all_keys"][load]["memories"] == 4096 for r in rr)
        one = avg(100*r["one_query"][load]["accuracy"] for r in rr)
        each = avg(100*r["all_keys"][load]["accuracy"] for r in rr)
        complete = avg(100*r["all_keys"][load]["all_correct"] for r in rr)
        print(f"  {name}: per-key {each}%; all keys right {complete}%; one-query {one}%")
