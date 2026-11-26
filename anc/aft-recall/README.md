# Three-way recall under a one-minute CPU budget

This test compares the width-256 learned-normalization pooler, two-block AFT-simple, and two-block softmax Transformer. Each model trains for 60 seconds per seed with a matching parameter budget near 100,000. AFT-simple uses the causal mixer from [Zhai et al.](https://arxiv.org/abs/2105.14103) inside the same Transformer scaffold; it is not a reproduction of the original larger models. Three seeds are run in rotating model order.

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
```

Use an otherwise idle CPU for reproducible timing. The exact recorded settings and all per-run scores are in `results.json`.
