# Evaluation status

The scripts, protocols, and result files in this directory describe the
superseded block/template implementation and are retained as historical
artifacts. They invoke removed CLI modes and report per-block concepts that do
not exist in the paper-aligned `BRTA` version 1 implementation.

Do not use these artifacts as evidence for the current compressor. In
particular, legacy `fixed`/`uniform`/`phog` comparisons, block reports, schema
5/6 archive sizes, and recorded program inventories are not comparable to the
whole-tensor semantic DSL.

A replacement evaluation must bind results to:

- the exact source revision and Zig toolchain;
- full input paths, sizes, and SHA-256 digests;
- the canonical `BRTA` archive digest and size;
- every synthesis and grammar limit;
- the optional canonical `BRGP` prior digest;
- complete byte-for-byte decompression verification;
- separately measured calibration, compression, and decompression resources.

Until that harness is implemented and rerun, the current repository makes
correctness claims only from the Zig acceptance and adversarial tests.
