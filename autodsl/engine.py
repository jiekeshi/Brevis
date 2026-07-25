"""Thin wrapper over the `brevis` binary.

Everything the outer loop learns about a library, it learns by running the
engine. Nothing here reimplements search, encoding, or legality.

These are *development* measurements. `eval/` is the audited apparatus and
enforces gates this module does not: clean tree, harness hashing, repetition
fingerprints, unset codec environment variables. Numbers produced here are for
steering the loop and must never be reported as formal results.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
BINARY = REPO / "zig-out" / "bin" / "brevis"

# Set by the eval protocol's own gate; recorded here only so a surprising
# measurement can be explained after the fact.
CODEC_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS", "GZIP", "BZIP2", "XZ_OPT", "XZ_DEFAULTS",
    "ZSTD_CLEVEL", "ZSTD_NBTHREADS", "LZ4_CLEVEL", "BROTLI_PARAM_QUALITY",
)


class EngineError(RuntimeError):
    pass


def label(model: pathlib.Path) -> str:
    """A readable name for a cached checkpoint.

    Every cache entry is `<repo>/<revision>/model.safetensors`, so the file
    name alone identifies nothing.
    """
    repository = model.resolve().parent.parent.name
    return f"{repository}/{model.name}" if repository else model.name


@dataclasses.dataclass(frozen=True)
class Budget:
    """The search configuration every measurement in one loop shares.

    `rerank_candidates` and `rerank_blocks` are deliberately more generous than
    the engine defaults (8 and 4). With the default shortlist, adding *any*
    production reshuffles which candidates reach the full-block rerank, and the
    resulting swing is larger than the effect being measured — a gate built on
    it would reward luck. Holding a wide shortlist fixed across the A/B is what
    makes the comparison about structure.
    """

    max_depth: int = 2
    max_expansions: int = 256
    max_nodes: int = 12
    rerank_candidates: int = 64
    rerank_blocks: int = 8
    jobs: int = 12

    def flags(self) -> list[str]:
        return [
            "--max-depth", str(self.max_depth),
            "--max-expansions", str(self.max_expansions),
            "--max-nodes", str(self.max_nodes),
            "--rerank-candidates", str(self.rerank_candidates),
            "--rerank-blocks", str(self.rerank_blocks),
            "--jobs", str(self.jobs),
        ]

    def describe(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One bench run reduced to what the gate and the miner need."""

    model: str
    archive_bytes: int
    raw_bytes: int
    bytecode_bytes: int
    payload_bytes: int
    planning_wall_ms: int
    report: dict

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.archive_bytes if self.archive_bytes else 0.0


def _run(argv: list[str], *, capture_json: bool) -> dict | str:
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        raise EngineError(
            f"{' '.join(argv[:3])} ... exited {result.returncode}\n{result.stderr[-4000:]}"
        )
    if not capture_json:
        return result.stdout
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise EngineError(f"{argv[1]} did not emit JSON: {exc}") from exc


def environment_notes() -> dict:
    return {
        "codec_environment_variables_set": {
            name: os.environ[name]
            for name in CODEC_ENVIRONMENT_VARIABLES
            if name in os.environ
        },
        "measurement_status": "development_only_not_formal",
    }


def config(binary: pathlib.Path = BINARY, macros: pathlib.Path | None = None) -> dict:
    argv = [str(binary), "config"]
    if macros is not None:
        argv += ["--macros", str(macros)]
    return _run(argv, capture_json=True)


def bench(
    model: pathlib.Path,
    budget: Budget,
    macros: pathlib.Path | None = None,
    *,
    binary: pathlib.Path = BINARY,
    plan: str = "search",
    prior: pathlib.Path | None = None,
) -> Measurement:
    argv = [str(binary), "bench", str(model), "--plan", plan, "--format", "json"]
    argv += budget.flags()
    if macros is not None:
        argv += ["--macros", str(macros)]
    if prior is not None:
        argv += ["--prior", str(prior)]
    report = _run(argv, capture_json=True)
    blocks = report["blocks"]
    return Measurement(
        model=str(model),
        archive_bytes=report["projected_archive_bytes"],
        raw_bytes=report["raw_bytes"],
        bytecode_bytes=sum(b["program_bytecode_bytes"] for b in blocks),
        payload_bytes=sum(b["packed_terminal_payload_bytes"] for b in blocks),
        planning_wall_ms=report["planning_wall_ms"],
        report=report,
    )


def compress_and_verify(
    model: pathlib.Path,
    archive: pathlib.Path,
    budget: Budget,
    macros: pathlib.Path | None = None,
    *,
    binary: pathlib.Path = BINARY,
) -> int:
    """Write a real archive and prove it decodes bit-exact. Returns its size.

    `bench` projects an archive without writing one, which is what the loop
    uses for speed. This is the confirmation step: only a real `.brv` that
    verifies is evidence that a library is safe to keep.
    """
    argv = [str(binary), "compress", str(model), str(archive)] + budget.flags()
    if macros is not None:
        argv += ["--macros", str(macros)]
    _run(argv, capture_json=False)
    output = _run([str(binary), "verify", str(archive), str(model)], capture_json=False)
    if "bit-exact" not in output:
        raise EngineError(f"verify did not confirm bit-exactness:\n{output}")
    return archive.stat().st_size
