# Preliminary Results Ledger

This ledger preserves pilot measurements that are useful for debugging but do
not meet the final timing protocol. A pilot result remains in the repository
even when it is unfavorable to Brevis. It must not be copied into the paper as a
formal result unless a later protocol-compliant run reproduces it.

## 2026-07-21: mixed I8/BF16 SmolLM pilot

Raw record: `results-preliminary-i8-schema2.json`

Record SHA-256:
`d111db190ee14136c9af187c5ee33a48a5f98c8ac04ae992ae9c5d0adb03c2ec`

Command:

```bash
python3 eval/run_eval.py \
  --models eval/models.json \
  --tag i8 \
  --results eval/results-preliminary-i8-schema2.json \
  --jobs 8 \
  --tensors 32
```

The result reports Brevis commit
`cbdd5641dfff0818f87259a3c43f55d5b1fc171f` with `git_dirty=false`. The
219,850,592-byte input is the complete
`RedHatAI/SmolLM-135M-Instruct-quantized.w8a8` safetensors file at revision
`0407d4a4dcc9601d11cf028866ee2272c5233557`. Its measured SHA-256,
`1f6ee9463560573bcccb6b48dfd89827ad9d49a4f50b6123a35b6f4efafb3baf`,
matches the independently preregistered tiered manifest.

| Configuration | Complete bytes | Ratio | Storage saving | Exact |
| --- | ---: | ---: | ---: | :---: |
| Brevis fixed DSL | 171,953,324 | 1.279 | 21.786% | yes |
| Brevis uniform search | 171,846,669 | 1.279 | 21.835% | yes |
| Brevis PHOG search | 171,838,570 | 1.279 | 21.838% | yes |
| gzip `-9` | 186,889,559 | 1.176 | 14.992% | yes |
| Zstandard `-19 -T0` | 183,862,733 | 1.196 | 16.369% | yes |
| xz `-9` | 137,655,300 | 1.597 | 37.387% | yes |

This pilot is a mixed result. All executed methods reconstructed the file
exactly. Brevis produced smaller archives than the tested gzip and Zstandard
configurations, but xz produced a substantially smaller archive than Brevis.
Fixed, uniform, and PHOG-guided Brevis differed by less than 0.1% in archive
size. These observations motivate, but do not replace, the planned terminal,
operator, budget, depth, and guidance ablations.

The record uses evaluator schema 2 and contains one timing observation per
operation. It has no peak-memory measurement, rotated method order, timing
variance, generic speed/default/ratio matrix, or external weight-specific
baseline. The legacy evaluator also times file-streaming paths differently from
the repeated baseline harness. Therefore its elapsed times are diagnostic only,
and neither its timing nor its compression values are a final paper result.
