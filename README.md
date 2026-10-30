# Beyond Similarity

Static, interactive learning guide to embedding geometry, hubness, score
correction, frozen intent assignment, and the evidence behind each method.

Production URL: https://tech.anujsadani.in/understanding-debias/

## Publishing

This repository is the GitHub Pages project site for the existing
`tech.anujsadani.in` host. Pages publishes the `main` branch root. The repository
intentionally has no `CNAME`; the parent account site owns the host and this
project is served under its repository path.

The publishable files are `index.html` and `assets/`. Keep asset paths relative
so styles, scripts, and the measured-result JSON resolve beneath the project
path. The website does not need a build step.

`assets/benchmark-results.json` records the frozen-encoder experiment and its
dataset revision, split counts, methods, metrics, and limitations. The benchmark
runner is included in `research/benchmark.py`.

## Reproduce the benchmark

Use Python 3.12 and install `research/requirements.txt`, then run:

```sh
python research/benchmark.py
```

The script downloads the pinned PolyAI CSVs and Qwen3-Embedding-0.6B, keeps the
official test split out of model and correction selection, and writes the result
to `assets/benchmark-results.json`. Set `BANKING77_DATA_DIR` to a directory
containing `banking77-train.csv` and `banking77-test.csv` to use local copies.
The dataset's source, revision, and CC BY 4.0 license are recorded in the result.
