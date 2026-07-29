# BF16 empirical information references

> **Interpretation warning:** empirical H0 is an iid coding reference for the observed checkpoint. It is **not** an absolute information-theoretic lower bound for arbitrary compressors that exploit order, repetition, tensor roles, program structure, or side information. Signed gaps may therefore be negative and must not be called impossible savings.

- Source files: 4; bytes: 16,060,556,376.
- BF16 tensors: 291; parameters: 8,030,261,248; payload bytes: 16,060,522,496.
- Skipped non-BF16 tensors: 0; bytes: 0.

The `8 + Hexp` column assumes all sign and mantissa bits cost exactly eight raw bits per weight, exponent symbols use an ideal iid entropy code, and model/framing/alignment costs are zero.
When adjacent statistics are enabled, the finite-sequence exponent reference is `[K H0(E_first) + P H(E_i|E_{i-1})] / N`, where K is the number of non-empty tensors, P is the number of within-tensor transitions, and N is the BF16 parameter count.

## Overall and role groups

| group | tensors (non-empty) | parameters | share | nominal BPW | BF16-symbol H0 | exponent H0 | idealized 8+Hexp | first exponent H0 | adjacent exponent H1 | idealized 8+finite-adj |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 291 (291) | 8,030,261,248 | 100.000% | 16.0000 | 10.5487 | 2.6169 | 10.6169 | 3.0664 | 2.6090 | 10.6090 |
| embedding | 1 (1) | 525,336,576 | 6.542% | 16.0000 | 10.5135 | 2.5910 | 10.5910 | 0.0000 | 2.5673 | 10.5673 |
| attention | 128 (128) | 1,342,177,280 | 16.714% | 16.0000 | 10.6721 | 2.7311 | 10.7311 | 2.6671 | 2.7075 | 10.7075 |
| mlp | 96 (96) | 5,637,144,576 | 70.199% | 16.0000 | 10.5091 | 2.5810 | 10.5810 | 2.5730 | 2.5796 | 10.5796 |
| norm | 65 (65) | 266,240 | 0.003% | 16.0000 | 7.9111 | 1.3646 | 9.3646 | 1.1886 | 0.7444 | 8.7446 |
| other | 1 (1) | 525,336,576 | 6.542% | 16.0000 | 10.5153 | 2.5866 | 10.5866 | 0.0000 | 2.5860 | 10.5860 |

## Supplied compressor results

Amortized whole-output BPW divides the supplied whole compressed checkpoint size (including framing and any non-BF16 content) by the number of analyzed BF16 parameters. A size input is exact; a ratio input only implies a possibly fractional output size from the discovered whole safetensors source-file bytes. Prefer exact `--method-size` inputs for publication.

| method | input | output-size basis | ratio | whole output bytes | amortized BPW | BPW−symbol H0 | BPW−(8+Hexp) | BPW−finite-adj |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Brevis | size_bytes=10579234560 | explicit-size | 1.518121× | 10,579,234,560.000 | 10.5394 | -0.0094 | -0.0775 | -0.0696 |
| ZipNN | size_bytes=10659934698 | explicit-size | 1.506628× | 10,659,934,698.000 | 10.6198 | 0.0710 | 0.0029 | 0.0108 |
| DFloat11 | size_bytes=10895502042 | explicit-size | 1.474054× | 10,895,502,042.000 | 10.8544 | 0.3057 | 0.2376 | 0.2455 |
| Zstd-9 | size_bytes=12380929936 | explicit-size | 1.297201× | 12,380,929,936.000 | 12.3343 | 1.7855 | 1.7174 | 1.7253 |
| libdeflate-1 | size_bytes=12499823800 | explicit-size | 1.284863× | 12,499,823,800.000 | 12.4527 | 1.9040 | 1.8359 | 1.8437 |
| LZ4-HC-9 | size_bytes=15936490262 | explicit-size | 1.007785× | 15,936,490,262.000 | 15.8764 | 5.3277 | 5.2596 | 5.2675 |
| Snappy | size_bytes=16060980257 | explicit-size | 0.999974× | 16,060,980,257.000 | 16.0005 | 5.4517 | 5.3836 | 5.3915 |

## Method notes

- BF16-symbol H0 uses the empirical distribution of complete physical 16-bit words and ignores their order.
- Exponent H0 uses the empirical 8-bit BF16 exponent distribution and also ignores order.
- Optional adjacent H1 is `H(E_i | E_{i-1})` in flattened physical order within each tensor. Tensor boundaries are never joined. The reported finite reference separately accounts for each tensor's first exponent before normalizing by all BF16 parameters.
- No finite-block coding, codebook, metadata, alignment, random-access, or decoder cost is included in an entropy reference.
- Compressor differences are signed descriptive gaps to these references, not proofs of remaining universally achievable compression.
