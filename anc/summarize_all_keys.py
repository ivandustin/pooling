"""Verify the repeated single-query results and summarize the all-key check."""
import argparse
import json
from pathlib import Path
from statistics import mean

p = argparse.ArgumentParser()
p.add_argument("diagnostic", nargs="?", type=Path, default=Path("all-keys-results.json"))
p.add_argument("--original", type=Path, default=Path("recall-load-results.json"))
a = p.parse_args()
original, repeated = [json.loads(path.read_text())["runs"] for path in (a.original, a.diagnostic)]
key = lambda r: (r["model"], r["seed"])
orig_by_key = {key(r): r for r in original}
rep_by_key = {key(r): r for r in repeated}
assert orig_by_key.keys() == rep_by_key.keys() and len(rep_by_key) == 6
for k, old in orig_by_key.items():
    new = rep_by_key[k]
    assert old["points"] == new["points"], ("ordinary test results differ", k)
print("Ordinary single-query results match on all six runs.")
columns = ("accuracy", "all_correct", "changes_answer")
for load in ("5", "10", "20"):
    scores = {model: [r["all_keys"][load] for r in repeated if r["model"] == model]
              for model in ("pooler", "transformer")}
    ceilings = {r["query_blind_ceiling"] for rr in scores.values() for r in rr}
    assert len(ceilings) == 1
    print(f"{load} pairs (query-blind ceiling {100 * ceilings.pop():.4f}%)")
    for model, rows in scores.items():
        assert {r["memories"] for r in rows} == {4096}
        stats = " ".join(f"{k}={100*mean(r[k] for r in rows):.4f}%" for k in columns)
        print(f"  {model}: {stats}")
