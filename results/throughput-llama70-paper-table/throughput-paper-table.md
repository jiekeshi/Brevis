# Llama-3.1-70B BF16 throughput

| Method | Process/thread configuration | Compress (s) | Decompress (s) | Compress (GB/s) | Decompress (GB/s) | Compress (GiB/s) | Decompress (GiB/s) | Ratio | Round-trip check | Evidence | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Brevis | 30 shard processes × 1 worker | 39.25 | 21.36 | 3.60 | 6.61 | 3.35 | 6.15 | 1.522× | exact | n=1; single-run peak-throughput estimate | complete |
| Zstd-9 | 30 shard processes × 1 codec thread | 141.52 | 17.22 | 1.00 | 8.20 | 0.93 | 7.63 | 1.300× | exact | n=1; single-run peak-throughput estimate | complete |
| ZipNN | 16 shard processes × 1 codec thread | 42.08 | 23.23 | 3.35 | 6.07 | 3.12 | 5.66 | 1.509× | not_checked | n=1; single-run peak-throughput estimate | complete |
| LZ4-HC-9 | 30 shard processes × 1 codec thread | 103.70 | 15.50 | 1.36 | 9.11 | 1.27 | 8.48 | 1.008× | not_checked | n=1; single-run peak-throughput estimate | complete |
| libdeflate-1 | 30 shard jobs; 60 processes (adapter + codec); 1 codec thread/job | 47.32 | 37.97 | 2.98 | 3.72 | 2.78 | 3.46 | 1.287× | not_checked | n=1; single-run peak-throughput estimate | complete |
| Snappy | 30 shard processes × 1 codec thread | 22.22 | 28.17 | 6.35 | 5.01 | 5.91 | 4.66 | 1.000× | not_checked | n=1; single-run peak-throughput estimate | complete |

GB/s uses decimal GB (10^9 bytes); GiB/s uses 2^30 bytes. Throughput uses original checkpoint bytes divided by whole-checkpoint phase makespan.

`single-run peak-throughput estimate` means n=1 under the listed practical concurrency; it is not a multi-run uncertainty estimate.

## Missing or incomplete

- None.

## Sources

- `/workspace/Brevis/results/throughput-llama70-brevis-max30`
- `/workspace/Brevis/results/throughput-llama70-max30`
- `/workspace/Brevis/results/throughput-llama70-fast-baselines`
