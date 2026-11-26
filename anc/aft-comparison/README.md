# One-minute CPU comparison with AFT-simple

This follow-up compares the learned-normalization pooler, a two-block AFT-simple baseline, and the two-block softmax Transformer. There are 18 runs: three models, two corpora, and three seeds. The AFT-simple mixer implements equation 5 of [Zhai et al.](https://arxiv.org/abs/2105.14103) inside the same width-64, two-block pre-normalized residual scaffold as the softmax baseline, with the same sinusoidal position features and prediction head. This is a matched comparison, not a reproduction of the original paper's larger models.

Each run uses approximately 100,000 parameters, 32-character inputs, batch size 64, and Adam at 0.0003. The training timer includes batch creation, transfer, and synchronized updates in chunks of 20; it excludes compilation, discarded warm-up, and evaluation. Runs are sequential and model order rotates by seed. Each corpus uses a contiguous 80/10/10 split; every valid held-out window is evaluated. Results are specific to this CPU, these small models, and this budget.

## Reproduce on Linux

From the extracted archive:

```sh
git clone https://github.com/ivandustin/llm.git
git -C llm checkout 002ccabcb90cef25ab25678a90fea81e36876651
curl -fL https://raw.githubusercontent.com/GITenberg/Alice-s-Adventures-in-Wonderland_11/master/11-0.txt -o alice-raw.txt
git hash-object alice-raw.txt
# Expected blob: 0f25e9af379233267a09623e0c650472f9e463ea
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
export PYTHONPATH="$PWD/llm" JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python compare.py --shakespeare llm/data/shakespeare.txt --alice alice-raw.txt \
  --out reproduction.json --seconds 60 --seeds 0 1 2
python summarize.py reproduction.json
```

The Shakespeare source is the original repository's Tiny Shakespeare file. The second source is *Alice's Adventures in Wonderland*, [Project Gutenberg ebook #11](https://www.gutenberg.org/ebooks/11), from the GITenberg repository named above. The script checks the raw Alice blob before stripping the license material and normalizing newlines. Hashes, corpus sizes, vocabulary sizes, per-seed scores, and measured training times are saved in `results.json`. The values in `summary.txt` are means and sample standard deviations across seeds. Equal-time update counts can vary with the machine and its load.
