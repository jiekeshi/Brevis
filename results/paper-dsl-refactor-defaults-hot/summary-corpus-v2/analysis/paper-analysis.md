# Paper-oriented analysis of verified compression results

Input: `results/paper-dsl-refactor-defaults-hot/summary-corpus-v2/verified-ratios.json`

All statistics use complete, verification-gated checkpoint cells. A win means strictly fewer compressed output bytes for Brevis. Win rate is wins divided by all models (ties remain in the denominator).
Bracketed 95% CIs use 10,000 checkpoint-resampling percentile bootstrap replicates with fixed seed 20260729.

## Overall pairwise results

Compression ratios below are always `source bytes / compressed output bytes` and are displayed with `×`. “GM” is the model-macro geometric mean. “Total-byte” sums source and output bytes before taking their ratio.

| Baseline | W–T–L | Win rate [95% CI] | Brevis GM CR | Baseline GM CR | GM advantage [95% CI] | Brevis total-byte CR | Baseline total-byte CR | Total-byte advantage [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zstd-9 | 12–0–0 | 100.00% [100.00%, 100.00%] | 1.4004× | 1.2321× | 1.1366× [1.1046×, 1.1635×] | 1.5114× | 1.2951× | 1.1670× [1.1354×, 1.1730×] |
| zipnn | 10–0–2 | 83.33% [58.33%, 100.00%] | 1.4004× | 1.3945× | 1.0042× [0.9993×, 1.0075×] | 1.5114× | 1.5007× | 1.0071× [1.0063×, 1.0079×] |
| lz4-hc-9 | 12–0–0 | 100.00% [100.00%, 100.00%] | 1.4004× | 1.0055× | 1.3928× [1.3034×, 1.4775×] | 1.5114× | 1.0080× | 1.4995× [1.4278×, 1.5070×] |
| libdeflate-1 | 12–0–0 | 100.00% [100.00%, 100.00%] | 1.4004× | 1.2238× | 1.1443× [1.1088×, 1.1749×] | 1.5114× | 1.2819× | 1.1790× [1.1438×, 1.1859×] |
| snappy | 12–0–0 | 100.00% [100.00%, 100.00%] | 1.4004× | 1.0000× | 1.4004× [1.3119×, 1.4892×] | 1.5114× | 0.9999× | 1.5116× [1.4365×, 1.5193×] |

| Baseline | Macro archive saving [95% CI] | Pooled archive saving [95% CI] | Total bytes saved |
|---|---:|---:|---:|
| zstd-9 | 11.92% [9.35%, 14.03%] | 14.31% [11.93%, 14.75%] | 236,517,836,007 |
| zipnn | 0.41% [-0.08%, 0.74%] | 0.71% [0.63%, 0.78%] | 10,095,020,238 |
| lz4-hc-9 | 27.75% [22.74%, 32.18%] | 33.31% [29.96%, 33.64%] | 707,297,238,646 |
| libdeflate-1 | 12.49% [9.64%, 14.85%] | 15.18% [12.57%, 15.67%] | 253,496,593,843 |
| snappy | 28.11% [23.20%, 32.70%] | 33.84% [30.39%, 34.18%] | 724,442,711,993 |

## By model domain

| Group | Baseline | N | W–T–L | GM advantage [95% CI] | Total-byte advantage [95% CI] | Macro archive saving [95% CI] | Pooled archive saving [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| audio | zstd-9 | 2 | 2–0–0 | 1.1224× [1.0800×, 1.1664×] | 1.1406× [1.0800×, 1.1664×] | 10.84% [7.41%, 14.26%] | 12.33% [7.41%, 14.26%] |
| audio | zipnn | 2 | 1–0–1 | 1.0021× [0.9978×, 1.0065×] | 1.0039× [0.9978×, 1.0065×] | 0.21% [-0.23%, 0.65%] | 0.39% [-0.23%, 0.65%] |
| audio | lz4-hc-9 | 2 | 2–0–0 | 1.3256× [1.1726×, 1.4985×] | 1.4013× [1.1726×, 1.4985×] | 23.99% [14.72%, 33.27%] | 28.64% [14.72%, 33.27%] |
| audio | libdeflate-1 | 2 | 2–0–0 | 1.1294× [1.0813×, 1.1796×] | 1.1503× [1.0813×, 1.1796×] | 11.38% [7.52%, 15.23%] | 13.07% [7.52%, 15.23%] |
| audio | snappy | 2 | 2–0–0 | 1.3322× [1.1736×, 1.5123×] | 1.4112× [1.1736×, 1.5123×] | 24.33% [14.79%, 33.87%] | 29.14% [14.79%, 33.87%] |
| image-generation | zstd-9 | 2 | 2–0–0 | 1.1171× [1.0663×, 1.1703×] | 1.1562× [1.0663×, 1.1703×] | 10.39% [6.22%, 14.56%] | 13.51% [6.22%, 14.56%] |
| image-generation | zipnn | 2 | 1–0–1 | 0.9942× [0.9813×, 1.0072×] | 1.0037× [0.9813×, 1.0072×] | -0.59% [-1.91%, 0.72%] | 0.37% [-1.91%, 0.72%] |
| image-generation | lz4-hc-9 | 2 | 2–0–0 | 1.3195× [1.1601×, 1.5008×] | 1.4547× [1.1601×, 1.5008×] | 23.59% [13.80%, 33.37%] | 31.26% [13.80%, 33.37%] |
| image-generation | libdeflate-1 | 2 | 2–0–0 | 1.1227× [1.0668×, 1.1816×] | 1.1660× [1.0668×, 1.1816×] | 10.82% [6.26%, 15.37%] | 14.24% [6.26%, 15.37%] |
| image-generation | snappy | 2 | 2–0–0 | 1.3249× [1.1603×, 1.5127×] | 1.4650× [1.1603×, 1.5127×] | 23.86% [13.82%, 33.89%] | 31.74% [13.82%, 33.89%] |
| language | zstd-9 | 8 | 8–0–0 | 1.1452× [1.1055×, 1.1733×] | 1.1675× [1.1325×, 1.1750×] | 12.58% [9.35%, 14.77%] | 14.35% [11.70%, 14.89%] |
| language | zipnn | 8 | 8–0–0 | 1.0072× [1.0060×, 1.0081×] | 1.0073× [1.0071×, 1.0083×] | 0.72% [0.59%, 0.80%] | 0.72% [0.70%, 0.82%] |
| language | lz4-hc-9 | 8 | 8–0–0 | 1.4292× [1.3477×, 1.5078×] | 1.5016× [1.4323×, 1.5092×] | 29.72% [25.34%, 33.68%] | 33.40% [30.18%, 33.74%] |
| language | libdeflate-1 | 8 | 8–0–0 | 1.1536× [1.1093×, 1.1857×] | 1.1796× [1.1410×, 1.1882×] | 13.19% [9.63%, 15.66%] | 15.23% [12.35%, 15.84%] |
| language | snappy | 8 | 8–0–0 | 1.4378× [1.3295×, 1.5198×] | 1.5137× [1.4453×, 1.5209×] | 30.12% [24.39%, 34.20%] | 33.94% [30.81%, 34.25%] |

## By numeric format

| Group | Baseline | N | W–T–L | GM advantage [95% CI] | Total-byte advantage [95% CI] | Macro archive saving [95% CI] | Pooled archive saving [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| BF16 | zstd-9 | 8 | 8–0–0 | 1.1716× [1.1694×, 1.1738×] | 1.1706× [1.1695×, 1.1752×] | 14.65% [14.49%, 14.80%] | 14.57% [14.49%, 14.91%] |
| BF16 | zipnn | 8 | 8–0–0 | 1.0075× [1.0071×, 1.0080×] | 1.0072× [1.0071×, 1.0081×] | 0.75% [0.71%, 0.79%] | 0.72% [0.70%, 0.81%] |
| BF16 | lz4-hc-9 | 8 | 8–0–0 | 1.5053× [1.5026×, 1.5079×] | 1.5071× [1.5042×, 1.5093×] | 33.57% [33.45%, 33.68%] | 33.65% [33.52%, 33.75%] |
| BF16 | libdeflate-1 | 8 | 8–0–0 | 1.1839× [1.1816×, 1.1863×] | 1.1830× [1.1817×, 1.1886×] | 15.53% [15.37%, 15.71%] | 15.47% [15.37%, 15.87%] |
| BF16 | snappy | 8 | 8–0–0 | 1.5174× [1.5148×, 1.5198×] | 1.5195× [1.5163×, 1.5211×] | 34.10% [33.99%, 34.20%] | 34.19% [34.05%, 34.26%] |
| FP16 | zstd-9 | 2 | 2–0–0 | 1.0731× [1.0663×, 1.0800×] | 1.0705× [1.0663×, 1.0800×] | 6.81% [6.22%, 7.41%] | 6.58% [6.22%, 7.41%] |
| FP16 | zipnn | 2 | 0–0–2 | 0.9895× [0.9813×, 0.9978×] | 0.9863× [0.9813×, 0.9978×] | -1.07% [-1.91%, -0.23%] | -1.39% [-1.91%, -0.23%] |
| FP16 | lz4-hc-9 | 2 | 2–0–0 | 1.1663× [1.1601×, 1.1726×] | 1.1639× [1.1601×, 1.1726×] | 14.26% [13.80%, 14.72%] | 14.08% [13.80%, 14.72%] |
| FP16 | libdeflate-1 | 2 | 2–0–0 | 1.0741× [1.0668×, 1.0813×] | 1.0713× [1.0668×, 1.0813×] | 6.89% [6.26%, 7.52%] | 6.65% [6.26%, 7.52%] |
| FP16 | snappy | 2 | 2–0–0 | 1.1669× [1.1603×, 1.1736×] | 1.1644× [1.1603×, 1.1736×] | 14.30% [13.82%, 14.79%] | 14.12% [13.82%, 14.79%] |
| FP32 | zstd-9 | 1 | 1–0–0 | 1.1148× [1.1148×, 1.1148×] | 1.1148× [1.1148×, 1.1148×] | 10.30% [10.30%, 10.30%] | 10.30% [10.30%, 10.30%] |
| FP32 | zipnn | 1 | 1–0–0 | 1.0032× [1.0032×, 1.0032×] | 1.0032× [1.0032×, 1.0032×] | 0.32% [0.32%, 0.32%] | 0.32% [0.32%, 0.32%] |
| FP32 | lz4-hc-9 | 1 | 1–0–0 | 1.2063× [1.2063×, 1.2063×] | 1.2063× [1.2063×, 1.2063×] | 17.10% [17.10%, 17.10%] | 17.10% [17.10%, 17.10%] |
| FP32 | libdeflate-1 | 1 | 1–0–0 | 1.1151× [1.1151×, 1.1151×] | 1.1151× [1.1151×, 1.1151×] | 10.32% [10.32%, 10.32%] | 10.32% [10.32%, 10.32%] |
| FP32 | snappy | 1 | 1–0–0 | 1.2064× [1.2064×, 1.2064×] | 1.2064× [1.2064×, 1.2064×] | 17.11% [17.11%, 17.11%] | 17.11% [17.11%, 17.11%] |
| FP8 | zstd-9 | 1 | 1–0–0 | 1.0202× [1.0202×, 1.0202×] | 1.0202× [1.0202×, 1.0202×] | 1.98% [1.98%, 1.98%] | 1.98% [1.98%, 1.98%] |
| FP8 | zipnn | 1 | 1–0–0 | 1.0082× [1.0082×, 1.0082×] | 1.0082× [1.0082×, 1.0082×] | 0.81% [0.81%, 0.81%] | 0.81% [0.81%, 0.81%] |
| FP8 | lz4-hc-9 | 1 | 1–0–0 | 1.2313× [1.2313×, 1.2313×] | 1.2313× [1.2313×, 1.2313×] | 18.79% [18.79%, 18.79%] | 18.79% [18.79%, 18.79%] |
| FP8 | libdeflate-1 | 1 | 1–0–0 | 1.0158× [1.0158×, 1.0158×] | 1.0158× [1.0158×, 1.0158×] | 1.55% [1.55%, 1.55%] | 1.55% [1.55%, 1.55%] |
| FP8 | snappy | 1 | 1–0–0 | 1.2324× [1.2324×, 1.2324×] | 1.2324× [1.2324×, 1.2324×] | 18.86% [18.86%, 18.86%] | 18.86% [18.86%, 18.86%] |

## Counterexamples

| Checkpoint | Baseline winner | Brevis ratio | Baseline ratio | Brevis archive saving |
|---|---|---:|---:|---:|
| whisper-large-v3-f16 | zipnn | 1.1736× | 1.1762× | -0.23% |
| sdxl-base-1.0-f16 | zipnn | 1.1603× | 1.1825× | -1.91% |

## Interpretation constraints

- The geometric-mean advantage is the geometric mean of `Brevis compression ratio / baseline compression ratio`; values above 1 favor Brevis.
- Total-byte compression ratio is `sum(source bytes) / sum(compressed output bytes)`. It is size-weighted and differs from the equal-model geometric mean.
- Macro archive saving gives every checkpoint equal weight. Pooled archive saving sums bytes first and is dominated by large checkpoints.
- Bracketed 95% intervals are percentile bootstrap intervals with checkpoints as the resampling unit. They quantify corpus composition sensitivity, not repeated-run measurement noise.
- ZipNN is tensor-exact while Brevis and the other listed baselines are byte-exact. ZipNN comparisons therefore do not have identical container-layout preservation requirements.
- Domain groups with two models and numeric-format groups with one or two models are descriptive slices, not evidence of broad generalization.
- These are deterministic archive-size observations from one fixed corpus and configuration. The bootstrap describes sensitivity to checkpoint composition, not repeated-run or population uncertainty; the results do not support throughput, memory, or significance claims.
