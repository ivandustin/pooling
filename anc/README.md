# One Vector, No Attention: Language Prediction on a One-Minute CPU Budget

Manuscript draft and reproducibility files. Nothing in this bundle has been submitted to arXiv.

## Build the manuscript

Run twice from the top-level directory of the extracted archive:

```sh
pdflatex -interaction=nonstopmode -halt-on-error last-token-pooling.tex
pdflatex -interaction=nonstopmode -halt-on-error last-token-pooling.tex
```

The three TeX files, both at the archive root, are enough to build the manuscript. The `anc/` folder holds the code and results that arXiv can make available as ancillary files with the TeX submission. This archive is prepared for author review; nothing has been uploaded to arXiv.

## Reproduce the new experiments

Clone the public implementation without modifying it. From the extracted archive, enter `anc/`. Run the following commands from that directory (when downloading the ancillary files individually, place them together first):

```sh
git clone https://github.com/ivandustin/llm.git
git -C llm checkout 002ccabcb90cef25ab25678a90fea81e36876651
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-tested.txt
export PYTHONPATH="$PWD/llm"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python equal_time.py --repo "$PWD/llm" --output equal-time-reproduction \
  --seconds 60 --chunk 20 --seeds 0 1 2 --save-checkpoints
python full_test.py --repo "$PWD/llm" --checkpoints equal-time-reproduction \
  --output full-test-reproduction.json
python recall_load.py --all-keys --output recall-load-reproduction.json --seeds 0 1 2
python summarize_recall.py recall-load-reproduction.json
python summarize_all_keys.py recall-load-reproduction.json
```

Run the CPU comparisons with no competing jobs. Equal-time update counts and performance are hardware- and software-dependent; the reported reference runs used Linux on an AMD EPYC 9V74 VM with nine available logical CPUs. Equal-time results are recorded in `equal-time-results.json`; the full-split results are in `full-test-results.json`; recall-load results are in `recall-load-results.json`. All means are arithmetic means over seeds 0–2; error bars are sample standard deviations (denominator $n-1$). The fixed-update comparison comes from the pinned repo's `experiments/norm-query-learned-d256-50k` and `experiments/current64-lr0.0003-50k` directories. The scripts in this bundle do not alter the repository.

To re-score the exact original six checkpoints without rerunning timed training:

```sh
python full_test.py --repo "$PWD/llm" --checkpoints one-minute-checkpoints \
  --output original-checkpoints-full-test.json
```

The recall summary script regenerates the fixed test inputs, checks every answer against exact lookup, and computes the query-blind frequency baseline with deterministic tie-breaking.

The optional `--all-keys` flag also scores every key for the first 4,096 held-out memories at each load. The run used for this diagnostic is stored in `all-keys-results.json`. It repeats the original training recipe and saves the ordinary one-query scores as a parity check. The all-key scores use the same set of memories for both models and all seeds.

The [AFT-simple comparison](aft-comparison/README.md) contains the scripts, exact settings, and results for the two-corpus one-minute tests.
