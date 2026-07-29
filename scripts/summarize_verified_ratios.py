#!/usr/bin/env python3
"""Build deterministic, verification-gated checkpoint compression summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark_corpus import CORPUS_PRESETS

DEFAULT_METHODS = (
    "brevis",
    "zstd-9",
    "zipnn",
    "lz4-hc-9",
    "libdeflate-1",
    "snappy",
)
EXACTNESS_SCOPE = {
    method: "tensor-exact" if method == "zipnn" else "byte-exact"
    for method in DEFAULT_METHODS
}
CSV_COLUMNS = (
    "checkpoint",
    "revision",
    "method",
    "exactness",
    "status",
    "verified_shards",
    "expected_shards",
    "source_bytes",
    "expected_source_bytes",
    "output_bytes",
    "compression_ratio",
    "archive_percent",
    "missing_shards",
    "unexpected_shards",
    "validation_errors",
)


class SummaryError(RuntimeError):
    """Raised when benchmark records or corpus metadata are inconsistent."""


@dataclass(frozen=True)
class ExpectedCheckpoint:
    name: str
    revision: str
    shard_sizes: dict[str, int]

    @property
    def source_bytes(self) -> int:
        return sum(self.shard_sizes.values())


@dataclass(frozen=True)
class VerifiedCompression:
    compression: dict[str, Any]
    verification: dict[str, Any]
    verification_line: int


def read_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    records = []
    for line_number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SummaryError(f"{path}:{line_number}: invalid JSONL") from exc
    return records, hashlib.sha256(data).hexdigest()


def load_expected_checkpoints(
    models_root: Path,
    preset: str,
) -> list[ExpectedCheckpoint]:
    checkpoints = []
    for spec in CORPUS_PRESETS[preset]:
        manifest_path = models_root / spec.name / "download-manifest.json"
        if not manifest_path.is_file():
            raise SummaryError(f"missing checkpoint manifest: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise SummaryError(f"invalid checkpoint manifest: {manifest_path}") from exc
        if manifest.get("name") != spec.name:
            raise SummaryError(
                f"{manifest_path}: expected name {spec.name!r}, "
                f"found {manifest.get('name')!r}"
            )
        if manifest.get("revision") != spec.revision:
            raise SummaryError(
                f"{manifest_path}: expected revision {spec.revision}, "
                f"found {manifest.get('revision')}"
            )

        shard_sizes: dict[str, int] = {}
        for item in manifest.get("weights", ()):
            shard = Path(item["path"]).name
            if shard in shard_sizes:
                raise SummaryError(
                    f"{manifest_path}: duplicate shard basename {shard!r}"
                )
            size = item.get("size")
            if not isinstance(size, int) or size <= 0:
                raise SummaryError(
                    f"{manifest_path}: invalid size for shard {shard!r}"
                )
            shard_sizes[shard] = size
        if not shard_sizes:
            raise SummaryError(f"{manifest_path}: no weight shards")
        if manifest.get("source_bytes") != sum(shard_sizes.values()):
            raise SummaryError(f"{manifest_path}: source byte total is inconsistent")

        checkpoints.append(
            ExpectedCheckpoint(spec.name, spec.revision, shard_sizes)
        )
    return checkpoints


def select_verified_compressions(
    records: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, str, str], VerifiedCompression],
    list[str],
]:
    """Select the last successful exact verification for each corpus shard."""

    attempts: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        run_id = record.get("run_id")
        attempt_id = record.get("attempt_id")
        if isinstance(run_id, str) and isinstance(attempt_id, str):
            attempts[run_id, attempt_id] = record

    selected: dict[tuple[str, str, str], VerifiedCompression] = {}
    diagnostics = []
    for line_number, verification in enumerate(records, 1):
        if not (
            verification.get("stage") == "corpus"
            and verification.get("operation") == "verify"
            and verification.get("status") == "ok"
            and verification.get("exact") is True
        ):
            continue

        checkpoint = verification.get("checkpoint")
        method = verification.get("method")
        shard = verification.get("shard")
        if not all(isinstance(value, str) for value in (checkpoint, method, shard)):
            diagnostics.append(
                f"record {line_number}: exact verification has an invalid identity"
            )
            continue

        compressions = []
        for pair in verification.get("verified_attempts", ()):
            if not (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(value, str) for value in pair)
            ):
                continue
            candidate = attempts.get((pair[0], pair[1]))
            if (
                candidate
                and candidate.get("stage") == "corpus"
                and candidate.get("operation") == "compress"
                and candidate.get("status") == "ok"
            ):
                compressions.append(candidate)

        if len(compressions) != 1:
            diagnostics.append(
                f"record {line_number}: {checkpoint}/{method}/{shard} "
                f"references {len(compressions)} successful corpus compressions"
            )
            continue
        compression = compressions[0]
        identity = (checkpoint, method, shard)
        if any(
            compression.get(field) != verification.get(field)
            for field in ("checkpoint", "method", "shard")
        ):
            diagnostics.append(
                f"record {line_number}: verification identity does not match "
                "its compression attempt"
            )
            continue

        selected[identity] = VerifiedCompression(
            compression,
            verification,
            line_number,
        )
    return selected, diagnostics


def summarize_cells(
    expected: list[ExpectedCheckpoint],
    methods: tuple[str, ...],
    selected: dict[tuple[str, str, str], VerifiedCompression],
) -> list[dict[str, Any]]:
    cells = []
    for checkpoint in expected:
        expected_shards = set(checkpoint.shard_sizes)
        for method in methods:
            matches = {
                shard: verified
                for (name, candidate_method, shard), verified in selected.items()
                if name == checkpoint.name and candidate_method == method
            }
            actual_shards = set(matches)
            missing = sorted(expected_shards - actual_shards)
            unexpected = sorted(actual_shards - expected_shards)
            validation_errors = []
            source_bytes = 0
            output_bytes = 0
            for shard in sorted(expected_shards & actual_shards):
                record = matches[shard].compression
                expected_size = checkpoint.shard_sizes[shard]
                if record.get("revision") != checkpoint.revision:
                    validation_errors.append(f"{shard}: revision mismatch")
                if record.get("source_bytes") != expected_size:
                    validation_errors.append(f"{shard}: source size mismatch")
                output_size = record.get("output_bytes")
                if not isinstance(output_size, int) or output_size <= 0:
                    validation_errors.append(f"{shard}: invalid output size")
                    continue
                source_bytes += expected_size
                output_bytes += output_size

            complete = not missing and not unexpected and not validation_errors
            ratio = source_bytes / output_bytes if output_bytes else None
            cells.append(
                {
                    "checkpoint": checkpoint.name,
                    "revision": checkpoint.revision,
                    "method": method,
                    "exactness": EXACTNESS_SCOPE[method],
                    "status": "complete" if complete else "incomplete",
                    "verified_shards": len(actual_shards & expected_shards),
                    "expected_shards": len(expected_shards),
                    "source_bytes": source_bytes,
                    "expected_source_bytes": checkpoint.source_bytes,
                    "output_bytes": output_bytes or None,
                    "compression_ratio": ratio,
                    "archive_percent": 100.0 / ratio if ratio else None,
                    "missing_shards": ";".join(missing),
                    "unexpected_shards": ";".join(unexpected),
                    "validation_errors": ";".join(validation_errors),
                }
            )
    return cells


def write_csv(path: Path, cells: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(cells)


def markdown_cell(cell: dict[str, Any]) -> str:
    if cell["status"] != "complete":
        return (
            f"incomplete "
            f"({cell['verified_shards']}/{cell['expected_shards']} shards)"
        )
    return f"{cell['compression_ratio']:.6f}× ({cell['output_bytes']:,} B)"


def write_markdown(
    path: Path,
    expected: list[ExpectedCheckpoint],
    methods: tuple[str, ...],
    cells: list[dict[str, Any]],
) -> None:
    lookup = {
        (cell["checkpoint"], cell["method"]): cell
        for cell in cells
    }
    complete = sum(cell["status"] == "complete" for cell in cells)
    lines = [
        "# Verified compression ratios",
        "",
        f"- Complete paper cells: {complete}/{len(cells)}",
        "- Ratio is source bytes / compressed output bytes.",
        "- ZipNN is tensor-exact: tensor names, dtype, shape, metadata, and "
        "payload bits are verified; safetensors layout may differ.",
        "- Brevis, zstd-9, lz4-hc-9, libdeflate-1, and Snappy are byte-exact.",
        "",
        "| Checkpoint | Source bytes | "
        + " | ".join(methods)
        + " |",
        "|---|---:|" + "|".join("---:" for _ in methods) + "|",
    ]
    for checkpoint in expected:
        values = [
            markdown_cell(lookup[checkpoint.name, method])
            for method in methods
        ]
        lines.append(
            f"| {checkpoint.name} | {checkpoint.source_bytes:,} | "
            + " | ".join(values)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_json(
    path: Path,
    *,
    runs: Path,
    runs_sha256: str,
    record_count: int,
    preset: str,
    methods: tuple[str, ...],
    diagnostics: list[str],
    cells: list[dict[str, Any]],
) -> None:
    complete = sum(cell["status"] == "complete" for cell in cells)
    payload = {
        "schema_version": 1,
        "runs": str(runs),
        "runs_sha256": runs_sha256,
        "record_count": record_count,
        "preset": preset,
        "methods": list(methods),
        "exactness_policy": {
            "zipnn": "tensor-exact",
            "other_methods": "byte-exact",
        },
        "complete_cells": complete,
        "expected_cells": len(cells),
        "all_complete": complete == len(cells),
        "selection_diagnostics": diagnostics,
        "cells": cells,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize only corpus compressions referenced by successful "
            "exact-verification records."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="benchmark result directory containing raw/runs.jsonl",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path("/workspace/checkpoints"),
    )
    parser.add_argument(
        "--preset",
        choices=tuple(CORPUS_PRESETS),
        default="paper-v1",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to RESULTS/summary",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write partial tables and exit successfully",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = tuple(dict.fromkeys(args.methods))
    runs = args.results / "raw" / "runs.jsonl"
    output_dir = args.output_dir or args.results / "summary"

    records, runs_sha256 = read_records(runs)
    expected = load_expected_checkpoints(args.models_root, args.preset)
    selected, diagnostics = select_verified_compressions(records)
    cells = summarize_cells(expected, methods, selected)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "verified-ratios.csv", cells)
    write_markdown(
        output_dir / "verified-ratios.md",
        expected,
        methods,
        cells,
    )
    write_json(
        output_dir / "verified-ratios.json",
        runs=runs,
        runs_sha256=runs_sha256,
        record_count=len(records),
        preset=args.preset,
        methods=methods,
        diagnostics=diagnostics,
        cells=cells,
    )

    complete = sum(cell["status"] == "complete" for cell in cells)
    print(f"verified paper cells: {complete}/{len(cells)}")
    print(f"wrote verified summaries under {output_dir}")
    if diagnostics:
        print(f"ignored malformed exact verifications: {len(diagnostics)}")
    if complete != len(cells) and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
