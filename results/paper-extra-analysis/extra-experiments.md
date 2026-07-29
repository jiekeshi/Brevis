# Brevis extra experiments

All benchmark points below are measured hot-cache runs on one representative Llama-3.1-8B BF16 shard, exact-verified, with n=1 per configuration. They are pilot measurements, not full-model timings or multi-run uncertainty estimates.

Attribution is measured static accounting of the representative budget-32 archive. It does not execute the stored programs. The information-theory section is a separate full-model BF16 scan (n=1 checkpoint); empirical entropy values are descriptive references, not arbitrary-structure lower bounds.

## Search-budget Pareto sweep

| Budget | Ratio | Time (s) | MiB/s | Saved vs B=0 (MiB) | Saved vs B=0 (%) | Peak RSS (GiB) | Status |
|---|---|---|---|---|---|---|---|
| 0 | 1.517093× | 2.559 | 1854.876 | 0.000 | 0.0000 | 6.244 | complete |
| 1 | 1.517366× | 9.044 | 524.809 | 0.562 | 0.0180 | 7.825 | complete |
| 2 | 1.517093× | 7.131 | 665.521 | 0.000 | 0.0000 | 6.627 | complete |
| 4 | 1.517093× | 19.181 | 247.439 | 0.000 | 0.0000 | 6.968 | complete |
| 8 | 1.517093× | 34.402 | 137.960 | 0.000 | 0.0000 | 6.984 | complete |
| 16 | 1.517093× | 58.308 | 81.398 | 0.000 | 0.0000 | 7.677 | complete |
| 32 | 1.518540× | 135.041 | 35.146 | 2.981 | 0.0953 | 14.969 | complete |
| 64 | 1.518582× | 361.752 | 13.120 | 3.068 | 0.0981 | 16.141 | complete |
| 128 | 1.519513× | 826.295 | 5.744 | 4.982 | 0.1593 | 15.231 | complete |
| 256 | 1.519631× | 1399.890 | 3.390 | 5.225 | 0.1670 | 15.126 | complete |

Compression ratio is source bytes / archive bytes; larger is better. Savings are relative to the measured budget-0 archive.

## Worker scaling

| Workers | Time (s) | MiB/s | Speedup vs 1 | Efficiency (%) | Peak RSS (GiB) | Ratio | Status |
|---|---|---|---|---|---|---|---|
| 1 | 37.728 | 125.800 | 1.000 | 100.000 | 4.806 | 1.517366× | complete |
| 4 | 18.792 | 252.566 | 2.008 | 50.192 | 5.279 | 1.517366× | complete |
| 8 | 13.820 | 343.418 | 2.730 | 34.124 | 5.455 | 1.517366× | complete |
| 16 | 10.761 | 441.068 | 3.506 | 21.913 | 5.906 | 1.517366× | complete |
| 32 | 9.026 | 525.831 | 4.180 | 13.062 | 8.037 | 1.517366× | complete |

## Ablation

The paper-facing table selects search budget 32. Budget 32 is preferred when its four expected variants are complete; otherwise the largest complete budget is selected. `ablation.csv` retains every measured budget.

| Budget | Variant | Ratio | Time (s) | MiB/s | Extra bytes vs full | Speedup vs full |
|---|---|---|---|---|---|---|
| 32 | full | 1.518540× | 140.240 | 33.843 | 0 | 1.000 |
| 32 | no-astar | 1.517210× | 118.086 | 40.192 | 2,873,336 | 1.188 |
| 32 | no-phog | 1.518540× | 88.815 | 53.439 | 389 | 1.579 |
| 32 | no-phog-no-astar | 1.517094× | 70.690 | 67.140 | 3,123,197 | 1.984 |

## Archive attribution by tensor role

| Role | Tensors | Source bytes | Record bytes | Saved bytes | Ratio | Literal fallbacks |
|---|---|---|---|---|---|---|
| attention | 36 | 754,974,720 | 502,137,370 | 252,837,350 | 1.503522× | 0 |
| embedding | 1 | 1,050,673,152 | 691,938,594 | 358,734,558 | 1.518449× | 1 |
| mlp | 27 | 3,170,893,824 | 2,083,144,099 | 1,087,749,725 | 1.522167× | 0 |
| norm | 18 | 147,456 | 61,712 | 85,744 | 2.389422× | 0 |

## Full-model BF16 information analysis

| Group | Weights | Share (%) | BF16 H0 (bpw) | Exponent H0 (bpw) | 8 + exponent H0 | 8 + adjacent exponent ref. |
|---|---|---|---|---|---|---|
| overall | 8,030,261,248 | 100.000 | 10.549 | 2.617 | 10.617 | 10.609 |
| embedding | 525,336,576 | 6.542 | 10.514 | 2.591 | 10.591 | 10.567 |
| attention | 1,342,177,280 | 16.714 | 10.672 | 2.731 | 10.731 | 10.707 |
| mlp | 5,637,144,576 | 70.199 | 10.509 | 2.581 | 10.581 | 10.580 |
| norm | 266,240 | 0.003 | 7.911 | 1.365 | 9.365 | 8.745 |
| other | 525,336,576 | 6.542 | 10.515 | 2.587 | 10.587 | 10.586 |

## Full-model method BPW

| Method | Ratio | Output BPW | Δ vs BF16 H0 | Δ vs 8+exp H0 | Δ vs adjacent ref. |
|---|---|---|---|---|---|
| Brevis | 1.518121× | 10.539 | -0.009 | -0.077 | -0.070 |
| ZipNN | 1.506628× | 10.620 | 0.071 | 0.003 | 0.011 |
| DFloat11 | 1.474054× | 10.854 | 0.306 | 0.238 | 0.245 |
| Zstd-9 | 1.297201× | 12.334 | 1.786 | 1.717 | 1.725 |
| libdeflate-1 | 1.284863× | 12.453 | 1.904 | 1.836 | 1.844 |
| LZ4-HC-9 | 1.007785× | 15.876 | 5.328 | 5.260 | 5.267 |
| Snappy | 0.999974× | 16.000 | 5.452 | 5.384 | 5.391 |

Method BPW uses each explicit whole-output byte count recorded by the information analyzer and the measured full-model BF16 weight count; this summarizer does not re-run the codecs.
