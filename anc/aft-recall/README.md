# Three-way recall under one- and five-minute CPU budgets

This test compares the width-256 learned-normalization pooler, two-block AFT-simple, and two-block softmax Transformer. Each model trains from scratch for 60 or 300 seconds per seed with a matching parameter budget near 100,000. AFT-simple uses the causal mixer from [Zhai et al.](https://arxiv.org/abs/2105.14103) inside the same Transformer scaffold; it is not a reproduction of the original larger models. Three seeds are run in rotating model order at each budget. The 300-second runs do not resume the 60-second runs.

Training windows have length 41. Each example stores 5, 10, or 20 pairs sampled uniformly by load with unique keys from 32 symbols and independent values from 16 symbols. Padding is visible and the query is always the last token. Within each seed, models see prefixes of the same generated stream. Each uses Adam at 0.0003, batch size 64, and synchronized chunks of 20 updates. The timer includes example generation and input transfer but excludes compilation, discarded warm-up, and held-out scoring. The number of updates is an outcome, not a stopping criterion.

For every load, the one-query test has 16,384 held-out memories. The all-key test asks every key on the first 4,096 of those memories without changing the stored pairs. It reports per-key accuracy, complete-memory accuracy (every answer right), and the best mean per-key accuracy available to a rule that can see the stored values but ignores the query. That rule always predicts the most frequent stored value. Accuracy with more pairs is not a test of longer input contexts.

## Reproduce on Linux

From the extracted benchmark directory:

```sh
git clone https://github.com/ivandustin/llm.git
git -C llm checkout 002ccabcb90cef25ab25678a90fea81e36876651
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
export PYTHONPATH="$PWD/llm" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python compare_recall.py --out reproduction.json --seconds 60 --seeds 0 1 2
python summarize.py reproduction.json
python compare_recall.py --out reproduction-300s.json --seconds 300 --seeds 0 1 2
python summarize-300s.py reproduction-300s.json
```

Use an otherwise idle CPU for reproducible timing. The exact recorded settings and all per-run scores are in `results.json` (one minute) and `results-300s.json` (five minutes).

## Follow-up controls

The fixed-five controls train the pooler and Transformer on five pairs only. Both use the same five-pair test memories, visible padding, and five-minute budget as the mixed-load comparison. The padding control keeps the mixed loads and ignores padding keys in both Transformer attention blocks, without moving the query or changing position indices. It checks that initial model states match, that the no-padding forward pass matches, and that changing masked token contents leaves predictions unchanged.

From the same directory and environment:

```sh
export PYTHONPATH="$PWD:$PYTHONPATH"
python five-pair-control/control.py --out fixed-transformer.json --seconds 300 --seeds 0 1 2
python five-pair-control/pooler-control.py --out fixed-pooler.json --seconds 300 --seeds 0 1 2
python padding-control/compare.py --out masked-transformer.json --seconds 300 --seeds 0 1 2
python count-five-pair-examples.py results-300s.json five-pair-control/results.json
```

Recorded results are in `five-pair-control/results.json` (Transformer), `five-pair-control/pooler-results.json`, and `padding-control/results.json`. The last command counts training examples from the saved run lengths and original random seeds, without retraining.
