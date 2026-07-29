# Paper benchmark results

This directory contains the curated, Git-safe evidence produced for the paper
benchmark run. Compression ratio is always `source bytes / output bytes`, so
larger values are better.

## Main results

- `paper-dsl-refactor-defaults-hot/summary-corpus-v2/` contains the verified
  12-checkpoint, six-method compression matrix and pairwise paper analysis.
- `throughput-llama70-paper-table/` contains the Llama-3.1-70B BF16
  peak-throughput comparison in CSV, Markdown, and LaTeX.
- `paper-extra-analysis/` contains paper-ready Pareto, worker-scaling,
  ablation, attribution, and information-theory tables.
- `information/llama-3.1-8b-bf16/` contains the full-checkpoint empirical BF16
  entropy scan.
- `attribution/llama-3.1-8b-bf16-shard1-budget32/` contains static archive
  accounting for the representative budget-32 shard.

The corresponding append-only JSONL measurement records and environment
metadata are included for the main corpus, the representative-shard studies,
and the Llama-70B throughput runs.

## Specialized baselines

- The DFloat11 Llama-3.1-8B result is measured and independently bit-validated.
- The Qwen3-32B and Llama-3.1-70B DFloat11 files are calibrated estimates, not
  native conversion measurements.
- ECF8 completed official encoding but failed its official CUDA validation in
  this environment; no successful ECF8 result is claimed.

## Scope

Pareto, worker scaling, and ablation are hot-cache, single-run pilot
measurements on the first Llama-3.1-8B shard. The information scan covers the
full Llama-3.1-8B checkpoint. The Llama-70B throughput table is an `n=1`
peak-throughput estimate under each listed practical process configuration.

The approximately 53 GB of generated archives, decompressed files, and verbose
runtime logs are intentionally excluded from Git. Checkpoints are also not
included.
