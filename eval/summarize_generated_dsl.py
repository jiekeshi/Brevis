#!/usr/bin/env python3
"""Aggregate auditable Generated-DSL records without re-reading model weights.

The input contract is deliberately narrow: every input must be either a formal
result directory containing ``dsl/manifest.json`` or an analyzer output
directory containing ``manifest.json``.  Only
``brevis.generated-dsl-analysis`` schema 1 is accepted.  Logical JSONL bytes
are checked against the analyzer manifest before any result is returned;
tracked ``.jsonl.zst`` files are decompressed with the system ``zstd`` binary
and checked against the *uncompressed* manifest identity.

The output keeps three different weighting views separate.  Tensor-plan
statistics use tensor-count and tensor-raw-byte weights.  Realized block
statistics additionally use block-count and tensor-equal weights (each tensor
contributes total weight one across its blocks).  Operator/terminal presence
is not a byte partition, and schema 1 cannot attribute packed payload bytes to
individual terminal nodes.  Search counters are algorithmic planning-work
proxies; actual planning wall time is not present in the analyzer records and
is reported as unsupported rather than inferred.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, BinaryIO


INPUT_SCHEMA = {"id": "brevis.generated-dsl-analysis", "version": 1}
OUTPUT_SCHEMA = {"id": "brevis.generated-dsl-aggregate", "version": 1}
REQUIRED_LOGICAL_FILES = (
    "reports.jsonl", "tensors.jsonl", "blocks.jsonl", "nodes.jsonl",
    "errors.jsonl",
)
SUMMARY_COUNT_FIELDS = {
    "reports.jsonl": "accepted_semantic_reports",
    "tensors.jsonl": "tensor_records",
    "blocks.jsonl": "block_records",
    "nodes.jsonl": "node_records",
    "errors.jsonl": "excluded_records",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_OPERATORS = frozenset(("raw", "bitpack", "huffman", "rans"))
KNOWN_OPERATORS = TERMINAL_OPERATORS | frozenset((
    "xor_const", "add_const_mod", "xor_prev", "diff_mod", "zigzag", "gray",
    "rotate_bits", "bit_reverse", "split_field", "topk_codebook", "rle",
    "deinterleave", "split_float", "bit_plane", "byte_plane",
))
SEARCH_COUNTERS = (
    "search_expansions", "candidates_realized", "candidates_reranked",
    "probe_blocks_used", "selected_sample_rank_zero_based",
)
ROLE_LABELS = (
    "component_role", "parameter_kind", "projection_role", "routing_scope",
)

REPORT_KEYS = frozenset((
    "analysis_schema", "record_type", "report_id", "input_label",
    "source_document_sha256", "source", "outer_configuration_id",
    "outer_configuration_spec", "bench_mode", "raw_terminal_identity",
    "semantic_fingerprint_sha256", "semantic_variant_index",
    "semantic_variant_count", "canonical_detail_source",
    "technical_repetitions", "technical_repetition_indices",
    "technical_repetition_count", "warmup_repetition_count",
    "measured_repetition_count",
    "technical_repetitions_are_independent_tensor_samples",
    "semantic_variant_weight", "run_class", "formal_eligible",
    "paper_metrics_eligible", "role_rules", "counts", "accounting", "search",
    "program_bytecode_evidence", "terminal_payload_semantics",
))
TENSOR_KEYS = frozenset((
    "analysis_schema", "record_type", "report_id", "tensor_id", "tensor_index",
    "name", "role", "dtype", "dtype_family", "shape", "shape_class", "rank",
    "numel", "file_data_start_byte", "file_data_end_byte_exclusive",
    "block_start", "block_count", "weights", "accounting", "search",
    "raw_root_blocks", "planned_raw_root_blocks", "fallback_raw_root_blocks",
    "tensor_plan", "realized_block_programs", "terminal_presence_is_payload_share",
    "paper_metrics_eligible",
))
BLOCK_KEYS = frozenset((
    "analysis_schema", "record_type", "report_id", "block_id", "block_index",
    "tensor_id", "tensor_index", "element_offset", "element_count", "weights",
    "accounting", "raw_classification", "program", "program_bytecode_sha256",
    "terminal_presence_is_payload_share", "paper_metrics_eligible",
))
NODE_KEYS = frozenset((
    "analysis_schema", "record_type", "report_id", "owner_id", "tree_scope",
    "node_id", "op", "params_u32", "is_terminal", "child_count", "path",
    "preorder_index", "depth_zero_based", "subtree_structure_signature",
    "subtree_parameter_signature", "subtree_structure_sha256",
    "subtree_parameter_sha256", "operator_kind", "program_structure_signature",
    "program_parameter_signature", "program_structure_sha256",
    "program_parameter_sha256", "first_occurrence_of_operator_in_program",
    "weights", "terminal_payload_attribution",
))
PROGRAM_KEYS = frozenset((
    "structure_signature", "parameter_signature", "structure_sha256",
    "parameter_sha256", "node_count", "node_depth", "transform_depth",
    "terminal_count", "operators_preorder", "terminal_codecs_preorder",
))
BLOCK_PROGRAM_EXTRA_KEYS = frozenset(("operators_present", "terminal_codecs_present"))


class AggregateError(ValueError):
    """Fail-closed input, schema, or accounting error with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AggregateError("noncanonical_json", str(exc)) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregateError("missing_or_invalid_field", f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AggregateError("missing_or_invalid_field", f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AggregateError("missing_or_invalid_field", f"{label} must be nonempty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AggregateError(
            "missing_or_invalid_field", f"{label} must be an integer >= {minimum}",
        )
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregateError("missing_or_invalid_field", f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        bound = "finite" if minimum is None else f"finite and >= {minimum}"
        raise AggregateError("missing_or_invalid_field", f"{label} must be {bound}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise AggregateError("missing_or_invalid_field", f"{label} must be boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AggregateError("invalid_sha256", f"{label} must be a lowercase SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    actual = set(value)
    expected_set = set(expected)
    if actual != expected_set:
        missing = sorted(expected_set - actual)
        unsupported = sorted(actual - expected_set)
        raise AggregateError(
            "unsupported_record_shape",
            f"{label} keys differ from schema 1; missing={missing}, unsupported={unsupported}",
        )


def _analysis_record(record: Mapping[str, Any], kind: str, keys: Iterable[str], label: str) -> None:
    _exact_keys(record, keys, label)
    if record.get("analysis_schema") != INPUT_SCHEMA or record.get("record_type") != kind:
        raise AggregateError(
            "unsupported_record_schema",
            f"{label} is not a {kind!r} record from generated-DSL analysis schema 1",
        )


def _close(actual: Any, expected: float | None, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise AggregateError("derived_metric_mismatch", f"{label} must be null")
        return
    observed = _number(actual, label)
    if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AggregateError(
            "derived_metric_mismatch", f"{label}={observed!r}, expected {expected!r}",
        )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _saving(raw_bytes: int, stored_bytes: int) -> float | None:
    return (raw_bytes - stored_bytes) / raw_bytes if raw_bytes else None


def _file_identity(path: pathlib.Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise AggregateError("unreadable_input", f"cannot read {path}: {exc}") from exc
    return {"bytes": size, "sha256": digest.hexdigest()}


@dataclasses.dataclass(frozen=True)
class ResolvedInput:
    argument: str
    analysis_dir: pathlib.Path
    manifest_path: pathlib.Path
    result_dir: pathlib.Path | None


def _resolve_input(raw_path: os.PathLike[str] | str) -> ResolvedInput:
    argument = str(raw_path)
    path = pathlib.Path(raw_path).resolve()
    if path.is_file():
        candidates = [path] if path.name == "manifest.json" else []
    elif path.is_dir():
        candidates = [candidate for candidate in (
            path / "manifest.json", path / "dsl" / "manifest.json",
        ) if candidate.is_file()]
    else:
        raise AggregateError("missing_input", f"input does not exist: {path}")
    if len(candidates) != 1:
        raise AggregateError(
            "ambiguous_input",
            f"{path} must resolve to exactly one manifest.json or dsl/manifest.json; "
            f"found {len(candidates)}",
        )
    manifest = candidates[0].resolve()
    analysis_dir = manifest.parent
    result_dir = analysis_dir.parent if analysis_dir.name == "dsl" else None
    return ResolvedInput(argument, analysis_dir, manifest, result_dir)


def _load_manifest(resolved: ResolvedInput) -> tuple[Mapping[str, Any], dict[str, Any]]:
    identity = _file_identity(resolved.manifest_path)
    try:
        raw = resolved.manifest_path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AggregateError(
            "invalid_manifest", f"cannot parse {resolved.manifest_path}: {exc}",
        ) from exc
    manifest = _mapping(document, "analysis manifest")
    _exact_keys(
        manifest,
        ("analysis_schema", "files", "paper_metrics_policy", "role_rules", "summary"),
        "analysis manifest",
    )
    if manifest.get("analysis_schema") != INPUT_SCHEMA:
        raise AggregateError(
            "unsupported_analysis_schema",
            f"{resolved.manifest_path} is not generated-DSL analysis schema 1",
        )
    files = _mapping(manifest.get("files"), "manifest.files")
    _exact_keys(files, REQUIRED_LOGICAL_FILES, "manifest.files")
    for logical_name in REQUIRED_LOGICAL_FILES:
        metadata = _mapping(files[logical_name], f"manifest.files.{logical_name}")
        _exact_keys(metadata, ("bytes", "sha256"), f"manifest.files.{logical_name}")
        _integer(metadata.get("bytes"), f"manifest.files.{logical_name}.bytes")
        _sha256(metadata.get("sha256"), f"manifest.files.{logical_name}.sha256")
    summary = _mapping(manifest.get("summary"), "manifest.summary")
    _exact_keys(
        summary,
        (*SUMMARY_COUNT_FIELDS.values(), "paper_metrics_eligible_reports"),
        "manifest.summary",
    )
    for name in (*SUMMARY_COUNT_FIELDS.values(), "paper_metrics_eligible_reports"):
        _integer(summary.get(name), f"manifest.summary.{name}")
    role_rules = _mapping(manifest.get("role_rules"), "manifest.role_rules")
    _exact_keys(role_rules, ("schema", "file_sha256", "scope"), "manifest.role_rules")
    schema = _mapping(role_rules.get("schema"), "manifest.role_rules.schema")
    if schema != {"id": "brevis.tensor-role-rules", "version": 1}:
        raise AggregateError("unsupported_role_taxonomy", "role taxonomy schema is unsupported")
    _sha256(role_rules.get("file_sha256"), "manifest.role_rules.file_sha256")
    if not isinstance(role_rules.get("scope"), str):
        raise AggregateError("missing_or_invalid_field", "manifest.role_rules.scope must be text")
    if not isinstance(manifest.get("paper_metrics_policy"), str):
        raise AggregateError(
            "missing_or_invalid_field", "manifest.paper_metrics_policy must be text",
        )
    return manifest, identity


def _stored_jsonl_path(directory: pathlib.Path, logical_name: str) -> tuple[pathlib.Path, str]:
    plain = directory / logical_name
    compressed = directory / f"{logical_name}.zst"
    found = [(plain, "plain"), (compressed, "zstd")] if plain.is_file() and compressed.is_file() else (
        [(plain, "plain")] if plain.is_file() else
        [(compressed, "zstd")] if compressed.is_file() else []
    )
    if len(found) != 1:
        raise AggregateError(
            "ambiguous_logical_file",
            f"{directory}/{logical_name} must have exactly one plain or .zst representation",
        )
    return found[0]


def _consume_jsonl(
    directory: pathlib.Path,
    logical_name: str,
    expected: Mapping[str, Any],
    callback: Callable[[Mapping[str, Any], int], None],
) -> dict[str, Any]:
    stored_path, storage = _stored_jsonl_path(directory, logical_name)
    stored_identity = _file_identity(stored_path)
    process: subprocess.Popen[bytes] | None = None
    stream: BinaryIO
    try:
        if storage == "plain":
            stream = stored_path.open("rb")
        else:
            try:
                process = subprocess.Popen(
                    ["zstd", "-q", "-d", "-c", "--", str(stored_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                raise AggregateError(
                    "zstd_unavailable", "system zstd is required for .jsonl.zst inputs",
                ) from exc
            if process.stdout is None:  # defensive: stdout=PIPE above
                raise AggregateError("zstd_decode_failed", "zstd stdout pipe was not created")
            stream = process.stdout

        digest = hashlib.sha256()
        logical_bytes = 0
        records = 0
        last_had_newline = True
        try:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                logical_bytes += len(line)
                last_had_newline = line.endswith(b"\n")
                if line in (b"\n", b"\r\n"):
                    raise AggregateError(
                        "invalid_jsonl", f"{stored_path}:{line_number} is blank",
                    )
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AggregateError(
                        "invalid_jsonl", f"cannot parse {stored_path}:{line_number}: {exc}",
                    ) from exc
                mapped = _mapping(record, f"{stored_path}:{line_number}")
                canonical_line = (_canonical_json(mapped) + "\n").encode("utf-8")
                if canonical_line != line:
                    raise AggregateError(
                        "noncanonical_jsonl",
                        f"{stored_path}:{line_number} is not canonical analyzer JSONL",
                    )
                callback(mapped, line_number)
                records += 1
        except Exception:
            if process is not None:
                process.kill()
                process.wait()
            raise
        finally:
            stream.close()

        if records and not last_had_newline:
            raise AggregateError("invalid_jsonl", f"{stored_path} lacks a final newline")
        if process is not None:
            if process.stderr is None:  # defensive: stderr=PIPE above
                raise AggregateError("zstd_decode_failed", "zstd stderr pipe was not created")
            stderr = process.stderr.read().decode("utf-8", "replace").strip()
            process.stderr.close()
            return_code = process.wait()
            if return_code != 0:
                raise AggregateError(
                    "zstd_decode_failed",
                    f"zstd failed for {stored_path} with code {return_code}: {stderr}",
                )
        logical_sha = digest.hexdigest()
        expected_bytes = _integer(expected.get("bytes"), f"manifest {logical_name}.bytes")
        expected_sha = _sha256(expected.get("sha256"), f"manifest {logical_name}.sha256")
        if logical_bytes != expected_bytes or logical_sha != expected_sha:
            raise AggregateError(
                "logical_file_identity_mismatch",
                f"{stored_path} decompresses to bytes={logical_bytes}, sha256={logical_sha}; "
                f"manifest requires bytes={expected_bytes}, sha256={expected_sha}",
            )
        return {
            "logical_name": logical_name,
            "stored_path": str(stored_path),
            "storage": storage,
            "stored_bytes": stored_identity["bytes"],
            "stored_sha256": stored_identity["sha256"],
            "logical_bytes": logical_bytes,
            "logical_sha256": logical_sha,
            "record_count": records,
            "identity_verified_against_manifest": True,
        }
    except OSError as exc:
        raise AggregateError("unreadable_input", f"cannot stream {stored_path}: {exc}") from exc


def _weighted_quantile(histogram: Mapping[int, float], quantile: float) -> int | None:
    total = sum(histogram.values())
    if total <= 0:
        return None
    threshold = quantile * total
    cumulative = 0.0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative >= threshold:
            return value
    return max(histogram)


@dataclasses.dataclass
class WeightedHistogram:
    values: dict[int, float] = dataclasses.field(default_factory=lambda: defaultdict(float))

    def add(self, value: int, weight: float) -> None:
        if weight < 0 or not math.isfinite(weight):
            raise AggregateError("invalid_weight", f"histogram weight is invalid: {weight}")
        self.values[value] += weight

    def render(self) -> dict[str, Any]:
        total = sum(self.values.values())
        entries = [
            {"value": value, "weight": self.values[value],
             "fraction": self.values[value] / total if total else None}
            for value in sorted(self.values)
        ]
        return {
            "weight_total": total,
            "support_size": len(entries),
            "min": min(self.values) if self.values else None,
            "max": max(self.values) if self.values else None,
            "weighted_mean": (
                sum(value * weight for value, weight in self.values.items()) / total
                if total else None
            ),
            "weighted_quantiles": {
                "p50": _weighted_quantile(self.values, 0.50),
                "p90": _weighted_quantile(self.values, 0.90),
                "p95": _weighted_quantile(self.values, 0.95),
                "p99": _weighted_quantile(self.values, 0.99),
            },
            "histogram": entries,
        }


@dataclasses.dataclass
class WeightedCategories:
    fields: tuple[str, ...]
    values: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)

    def add(self, key: str, weights: Mapping[str, float]) -> None:
        if set(weights) != set(self.fields):
            raise AggregateError("internal_error", "category weight fields disagree")
        row = self.values.setdefault(key, {field: 0.0 for field in self.fields})
        for field in self.fields:
            weight = float(weights[field])
            if weight < 0 or not math.isfinite(weight):
                raise AggregateError("invalid_weight", f"category weight is invalid: {weight}")
            row[field] += weight

    def totals(self) -> dict[str, float]:
        return {
            field: sum(row[field] for row in self.values.values()) for field in self.fields
        }

    def render(
        self, *, key_name: str = "key",
        fraction_denominators: Mapping[str, float] | None = None,
        fraction_semantics: str = "share_of_category_partition_weight",
    ) -> dict[str, Any]:
        totals = {
            field: sum(row[field] for row in self.values.values()) for field in self.fields
        }
        denominators = totals if fraction_denominators is None else {
            field: float(fraction_denominators[field]) for field in self.fields
        }
        ordered = sorted(
            self.values.items(),
            key=lambda item: (-item[1][self.fields[0]], item[0]),
        )
        entries = []
        for key, row in ordered:
            entry: dict[str, Any] = {key_name: key}
            for field in self.fields:
                entry[field] = row[field]
                entry[f"{field}_fraction"] = (
                    row[field] / denominators[field] if denominators[field] else None
                )
            entries.append(entry)
        return {
            "weight_totals": totals,
            "fraction_denominators": denominators,
            "fraction_semantics": fraction_semantics,
            "distinct_values": len(entries),
            "entries": entries,
        }


PROGRAM_WEIGHTS = ("program_count_weight", "raw_byte_weight", "tensor_equal_weight")


@dataclasses.dataclass
class ProgramAccumulator:
    serialized_bytecode_applicable: bool = False
    histograms: dict[str, dict[str, WeightedHistogram]] = dataclasses.field(
        default_factory=lambda: {
            metric: {weight: WeightedHistogram() for weight in PROGRAM_WEIGHTS}
            for metric in ("node_count", "node_depth", "transform_depth", "terminal_count")
        }
    )
    structures: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    parameterized_programs: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    operator_combinations: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    terminal_combinations: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    terminal_presence: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    bytecode_histograms: dict[str, WeightedHistogram] = dataclasses.field(
        default_factory=lambda: {weight: WeightedHistogram() for weight in PROGRAM_WEIGHTS}
    )
    bytecode_bytes_total: float = 0.0
    encoded_bytes_total: float = 0.0
    framed_bytes_total: float = 0.0
    raw_bytes_total: float = 0.0

    def add(
        self, program: Mapping[str, Any], weights: Mapping[str, float],
        *, bytecode_accounting: Mapping[str, int] | None = None,
    ) -> None:
        for metric in self.histograms:
            value = int(program[metric])
            for weight_name, histogram in self.histograms[metric].items():
                histogram.add(value, float(weights[weight_name]))
        self.structures.add(str(program["structure_signature"]), weights)
        self.parameterized_programs.add(str(program["parameter_signature"]), weights)
        operators = tuple(sorted(set(str(op) for op in program["operators_preorder"])))
        terminals = tuple(sorted(set(str(op) for op in program["terminal_codecs_preorder"])))
        self.operator_combinations.add(" + ".join(operators), weights)
        self.terminal_combinations.add(" + ".join(terminals), weights)
        for terminal in terminals:
            self.terminal_presence.add(terminal, weights)
        if self.serialized_bytecode_applicable:
            if bytecode_accounting is None:
                raise AggregateError(
                    "internal_error", "realized program lacks serialized bytecode accounting",
                )
            bytecode = int(bytecode_accounting["program_bytecode_bytes"])
            for weight_name, histogram in self.bytecode_histograms.items():
                histogram.add(bytecode, float(weights[weight_name]))
            semantic_weight = float(weights["program_count_weight"])
            self.bytecode_bytes_total += bytecode * semantic_weight
            self.encoded_bytes_total += int(
                bytecode_accounting["encoded_bytes_without_frame_headers"]
            ) * semantic_weight
            self.framed_bytes_total += int(bytecode_accounting["framed_bytes"]) * semantic_weight
            self.raw_bytes_total += int(bytecode_accounting["raw_bytes"]) * semantic_weight
        elif bytecode_accounting is not None:
            raise AggregateError(
                "internal_error", "tensor-plan template unexpectedly has bytecode accounting",
            )

    def render(self) -> dict[str, Any]:
        program_denominators = {
            weight: sum(self.histograms["node_count"][weight].values.values())
            for weight in PROGRAM_WEIGHTS
        }
        if self.serialized_bytecode_applicable:
            bytecode = {
                "status": "available_for_realized_block_programs",
                "distribution": {
                    weight: histogram.render()
                    for weight, histogram in self.bytecode_histograms.items()
                },
                "accounting": {
                    "program_bytecode_bytes_total": self.bytecode_bytes_total,
                    "encoded_bytes_without_frame_headers_total": self.encoded_bytes_total,
                    "framed_bytes_total": self.framed_bytes_total,
                    "raw_bytes_total": self.raw_bytes_total,
                    "fraction_of_encoded_bytes_without_frame_headers": (
                        self.bytecode_bytes_total / self.encoded_bytes_total
                        if self.encoded_bytes_total else None
                    ),
                    "fraction_of_framed_bytes": (
                        self.bytecode_bytes_total / self.framed_bytes_total
                        if self.framed_bytes_total else None
                    ),
                    "fraction_of_raw_bytes": (
                        self.bytecode_bytes_total / self.raw_bytes_total
                        if self.raw_bytes_total else None
                    ),
                },
                "scope": (
                    "serialized bytecode for each realized block program; payload and the "
                    "12-byte non-bytecode frame overhead are excluded from the numerator"
                ),
            }
        else:
            bytecode = {
                "status": "not_applicable_to_tensor_plan_templates",
                "distribution": None,
                "accounting": None,
                "reason": (
                    "schema 1 reports serialized bytecode only for realized block programs; "
                    "a tensor plan is a planning template and may differ from realized blocks"
                ),
            }
        return {
            "length_and_depth": {
                metric: {
                    weight: histogram.render() for weight, histogram in views.items()
                }
                for metric, views in self.histograms.items()
            },
            "serialized_program_bytecode_length": bytecode,
            "structure_frequency": self.structures.render(key_name="structure_signature"),
            "parameterized_program_frequency": self.parameterized_programs.render(
                key_name="parameter_signature"
            ),
            "operator_combination_frequency": self.operator_combinations.render(
                key_name="operator_combination"
            ),
            "terminal_combination_frequency": self.terminal_combinations.render(
                key_name="terminal_combination"
            ),
            "terminal_codec_presence_frequency": self.terminal_presence.render(
                key_name="terminal_codec",
                fraction_denominators=program_denominators,
                fraction_semantics=(
                    "fraction_of_program_weight_containing_the_terminal; fractions across "
                    "terminals may sum above one"
                ),
            ),
        }


def _pearson(xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]) -> float | None:
    total = sum(weights)
    if len(xs) < 2 or total <= 0:
        return None
    mean_x = sum(x * weight for x, weight in zip(xs, weights, strict=True)) / total
    mean_y = sum(y * weight for y, weight in zip(ys, weights, strict=True)) / total
    covariance = sum(
        weight * (x - mean_x) * (y - mean_y)
        for x, y, weight in zip(xs, ys, weights, strict=True)
    )
    variance_x = sum(weight * (x - mean_x) ** 2 for x, weight in zip(xs, weights, strict=True))
    variance_y = sum(weight * (y - mean_y) ** 2 for y, weight in zip(ys, weights, strict=True))
    if variance_x <= 0 or variance_y <= 0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


@dataclasses.dataclass
class RelationshipAccumulator:
    xs: list[float] = dataclasses.field(default_factory=list)
    ys: list[float] = dataclasses.field(default_factory=list)
    raw_weights: list[float] = dataclasses.field(default_factory=list)
    saved_bytes: list[float] = dataclasses.field(default_factory=list)
    inapplicable: int = 0

    def observe(self, cost: float, saving_fraction: float, raw_bytes: int) -> None:
        self.xs.append(cost)
        self.ys.append(saving_fraction)
        self.raw_weights.append(float(raw_bytes))
        self.saved_bytes.append(saving_fraction * raw_bytes)

    def mark_inapplicable(self) -> None:
        self.inapplicable += 1

    def render(self) -> dict[str, Any]:
        n = len(self.xs)
        unweighted = _pearson(self.xs, self.ys, [1.0] * n)
        weighted = _pearson(self.xs, self.ys, self.raw_weights)
        spearman = (
            _pearson(_ranks(self.xs), _ranks(self.ys), [1.0] * n) if n >= 2 else None
        )
        if n < 2:
            status = "insufficient_observations"
        elif len(set(self.xs)) < 2:
            status = "constant_cost"
        elif len(set(self.ys)) < 2:
            status = "constant_saving_fraction"
        else:
            status = "computed"
        total_cost = sum(self.xs)
        return {
            "status": status,
            "observed_tensor_records": n,
            "inapplicable_tensor_records": self.inapplicable,
            "missing_required_tensor_records": 0,
            "cost": {
                "sum": total_cost,
                "mean": statistics.fmean(self.xs) if self.xs else None,
                "median": statistics.median(self.xs) if self.xs else None,
                "min": min(self.xs) if self.xs else None,
                "max": max(self.xs) if self.xs else None,
            },
            "compression_outcome": {
                "mean_tensor_saving_fraction": statistics.fmean(self.ys) if self.ys else None,
                "aggregate_raw_byte_weighted_saving_fraction": (
                    sum(self.saved_bytes) / sum(self.raw_weights)
                    if sum(self.raw_weights) else None
                ),
                "total_saved_bytes": sum(self.saved_bytes),
                "saved_bytes_per_cost_unit": (
                    sum(self.saved_bytes) / total_cost if total_cost else None
                ),
            },
            "correlations": {
                "pearson_tensor_count_weighted": unweighted,
                "pearson_raw_byte_weighted": weighted,
                "spearman_tensor_count_weighted": spearman,
            },
        }


@dataclasses.dataclass
class PreferenceAccumulator:
    tensor_records: float = 0.0
    raw_bytes: float = 0.0
    encoded_bytes: float = 0.0
    framed_bytes: float = 0.0
    tensor_plan_structures: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(("tensor_count_weight", "raw_byte_weight"))
    )
    realized_structures: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    operator_combinations: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    terminal_presence: WeightedCategories = dataclasses.field(
        default_factory=lambda: WeightedCategories(PROGRAM_WEIGHTS)
    )
    search_sums: dict[str, float] = dataclasses.field(
        default_factory=lambda: defaultdict(float)
    )
    search_observed: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    search_inapplicable: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )

    def add_tensor(
        self, tensor: Mapping[str, Any], program: Mapping[str, Any] | None,
        *, search_expected: bool,
    ) -> None:
        variant = float(_mapping(tensor["weights"], "tensor weights")["semantic_variant_weight"])
        accounting = _mapping(tensor["accounting"], "tensor accounting")
        raw = int(accounting["raw_bytes"])
        self.tensor_records += variant
        self.raw_bytes += raw * variant
        self.encoded_bytes += int(accounting["encoded_bytes_without_frame_headers"]) * variant
        self.framed_bytes += int(accounting["block_frame_bytes_excluding_container_header_footer"]) * variant
        if program is not None:
            self.tensor_plan_structures.add(str(program["structure_signature"]), {
                "tensor_count_weight": variant,
                "raw_byte_weight": raw * variant,
            })
        search = _mapping(tensor["search"], "tensor search")
        for field in SEARCH_COUNTERS:
            value = search[field]
            applicable = program is not None and (field == "search_expansions" or search_expected)
            if applicable:
                if value is None:
                    raise AggregateError("internal_error", f"validated {field} became null")
                self.search_sums[field] += float(value) * variant
                self.search_observed[field] += 1
            else:
                self.search_inapplicable[field] += 1

    def add_block(self, program: Mapping[str, Any], weights: Mapping[str, float]) -> None:
        self.realized_structures.add(str(program["structure_signature"]), weights)
        operators = " + ".join(sorted(set(str(op) for op in program["operators_preorder"])))
        self.operator_combinations.add(operators, weights)
        for terminal in sorted(set(str(op) for op in program["terminal_codecs_preorder"])):
            self.terminal_presence.add(terminal, weights)

    def render(self) -> dict[str, Any]:
        realized_program_denominators = self.realized_structures.totals()
        return {
            "tensor_observations": {
                "semantic_variant_weighted_count": self.tensor_records,
                "raw_bytes": self.raw_bytes,
                "encoded_bytes_without_frame_headers": self.encoded_bytes,
                "block_framed_bytes": self.framed_bytes,
                "storage_saved_bytes_against_block_framed": self.raw_bytes - self.framed_bytes,
                "aggregate_saving_fraction_with_block_frames": (
                    (self.raw_bytes - self.framed_bytes) / self.raw_bytes
                    if self.raw_bytes else None
                ),
            },
            "tensor_plan_structure_frequency": self.tensor_plan_structures.render(
                key_name="structure_signature"
            ),
            "realized_block_structure_frequency": self.realized_structures.render(
                key_name="structure_signature"
            ),
            "realized_block_operator_combination_frequency": self.operator_combinations.render(
                key_name="operator_combination"
            ),
            "realized_block_terminal_presence_frequency": self.terminal_presence.render(
                key_name="terminal_codec",
                fraction_denominators=realized_program_denominators,
                fraction_semantics=(
                    "fraction_of_realized_program_weight_containing_the_terminal; fractions "
                    "across terminals may sum above one"
                ),
            ),
            "search_work": {
                field: {
                    "observed_tensor_records": self.search_observed[field],
                    "inapplicable_tensor_records": self.search_inapplicable[field],
                    "missing_required_tensor_records": 0,
                    "sum": self.search_sums[field],
                    "mean_over_observed": (
                        self.search_sums[field] / self.search_observed[field]
                        if self.search_observed[field] else None
                    ),
                }
                for field in SEARCH_COUNTERS
            },
        }


@dataclasses.dataclass
class ReportState:
    record: Mapping[str, Any]
    input_index: int
    line_number: int
    configuration_key: str
    tensor_count: int = 0
    declared_block_cursor: int = 0
    block_count: int = 0
    tensor_plan_nodes: int = 0
    realized_block_nodes: int = 0
    tensor_raw_bytes: int = 0
    tensor_encoded_bytes: int = 0
    tensor_framed_bytes: int = 0
    block_raw_bytes: int = 0
    block_encoded_bytes: int = 0
    block_framed_bytes: int = 0


@dataclasses.dataclass
class TensorState:
    report_id: str
    tensor_index: int
    configuration_key: str
    block_count_expected: int
    raw_bytes: int
    encoded_bytes: int
    framed_bytes: int
    dimensions: tuple[tuple[str, str], ...]
    expected_structure_counts: Mapping[str, int]
    expected_parameter_counts: Mapping[str, int]
    expected_terminals_present: tuple[str, ...]
    block_count: int = 0
    block_raw_bytes: int = 0
    block_encoded_bytes: int = 0
    block_framed_bytes: int = 0
    actual_structure_counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    actual_parameter_counts: dict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    actual_terminals_present: set[str] = dataclasses.field(default_factory=set)


@dataclasses.dataclass(frozen=True)
class OwnerState:
    report_id: str
    configuration_key: str
    tree_scope: str
    node_count: int
    operators: tuple[str, ...]
    structure_signature: str
    parameter_signature: str
    structure_sha256: str
    parameter_sha256: str
    raw_bytes: int
    tensor_weight: float


def _tensor_size_bucket(raw_bytes: int) -> str:
    if raw_bytes == 0:
        return "zero_bytes"
    for upper, label in (
        (4 * 1024, "1B_to_4KiB"),
        (64 * 1024, "over_4KiB_to_64KiB"),
        (1024 * 1024, "over_64KiB_to_1MiB"),
        (16 * 1024 * 1024, "over_1MiB_to_16MiB"),
    ):
        if raw_bytes <= upper:
            return label
    return "over_16MiB"


def _program_summary(value: Any, label: str, *, block: bool) -> Mapping[str, Any]:
    program = _mapping(value, label)
    expected = PROGRAM_KEYS | (BLOCK_PROGRAM_EXTRA_KEYS if block else frozenset())
    _exact_keys(program, expected, label)
    for field in ("structure_signature", "parameter_signature"):
        _string(program.get(field), f"{label}.{field}")
    for field in ("structure_sha256", "parameter_sha256"):
        _sha256(program.get(field), f"{label}.{field}")
    node_count = _integer(program.get("node_count"), f"{label}.node_count", minimum=1)
    node_depth = _integer(program.get("node_depth"), f"{label}.node_depth", minimum=1)
    transform_depth = _integer(program.get("transform_depth"), f"{label}.transform_depth")
    terminal_count = _integer(program.get("terminal_count"), f"{label}.terminal_count", minimum=1)
    if transform_depth >= node_depth or terminal_count > node_count:
        raise AggregateError("invalid_program", f"{label} length/depth fields are inconsistent")
    operators = _array(program.get("operators_preorder"), f"{label}.operators_preorder")
    terminals = _array(program.get("terminal_codecs_preorder"), f"{label}.terminal_codecs_preorder")
    if len(operators) != node_count or len(terminals) != terminal_count:
        raise AggregateError("invalid_program", f"{label} operator counts disagree")
    if not all(isinstance(op, str) and op in KNOWN_OPERATORS for op in operators):
        raise AggregateError("unsupported_operator", f"{label} contains an unsupported operator")
    expected_terminals = [op for op in operators if op in TERMINAL_OPERATORS]
    if terminals != expected_terminals:
        raise AggregateError("invalid_program", f"{label} terminal preorder disagrees")
    if block:
        if program.get("operators_present") != list(dict.fromkeys(operators)):
            raise AggregateError("invalid_program", f"{label}.operators_present disagrees")
        if program.get("terminal_codecs_present") != list(dict.fromkeys(terminals)):
            raise AggregateError("invalid_program", f"{label}.terminal_codecs_present disagrees")
    return program


class Aggregator:
    def __init__(self) -> None:
        self.reports: dict[str, ReportState] = {}
        self.tensors: dict[str, TensorState] = {}
        self.owners: dict[str, OwnerState] = {}
        self.node_masks: dict[str, int] = defaultdict(int)
        self.record_counts: dict[str, int] = defaultdict(int)
        self.configuration_definitions: dict[str, dict[str, Any]] = {}
        self.configuration_report_ids: dict[str, list[str]] = defaultdict(list)
        self.programs: dict[str, dict[str, ProgramAccumulator]] = defaultdict(
            lambda: {"tensor_plan_template": ProgramAccumulator(False),
                     "realized_block": ProgramAccumulator(True)}
        )
        node_fields = (
            "node_occurrence_weight", "operator_presence_program_weight",
            "operator_presence_tensor_weight", "operator_presence_raw_byte_weight",
        )
        self.node_operators: dict[tuple[str, str], WeightedCategories] = defaultdict(
            lambda: WeightedCategories(node_fields)
        )
        self.preferences: dict[tuple[str, str, str], PreferenceAccumulator] = defaultdict(
            PreferenceAccumulator
        )
        self.relationships: dict[tuple[str, str], RelationshipAccumulator] = defaultdict(
            RelationshipAccumulator
        )
        self.role_rules_sha256: str | None = None
        self.input_provenance: list[dict[str, Any]] = []
        self.current_input_index = -1

    def _register_owner(
        self, owner_id: str, report_id: str, scope: str, program: Mapping[str, Any],
        *, raw_bytes: int, tensor_weight: float,
    ) -> None:
        if owner_id in self.owners:
            raise AggregateError("duplicate_owner_id", f"duplicate program owner {owner_id!r}")
        state = self.reports[report_id]
        self.owners[owner_id] = OwnerState(
            report_id, state.configuration_key, scope, int(program["node_count"]),
            tuple(str(op) for op in program["operators_preorder"]),
            str(program["structure_signature"]), str(program["parameter_signature"]),
            str(program["structure_sha256"]), str(program["parameter_sha256"]),
            raw_bytes, tensor_weight,
        )

    def report(self, record: Mapping[str, Any], line_number: int) -> None:
        _analysis_record(record, "report", REPORT_KEYS, "report record")
        report_id = _string(record.get("report_id"), "report.report_id")
        if report_id in self.reports:
            raise AggregateError("duplicate_report_id", f"duplicate report_id {report_id!r}")
        if (
            record.get("run_class") != "formal"
            or record.get("formal_eligible") is not True
            or record.get("paper_metrics_eligible") is not True
        ):
            raise AggregateError("ineligible_record", f"report {report_id!r} is not formal eligible")
        if record.get("technical_repetitions_are_independent_tensor_samples") is not False:
            raise AggregateError("invalid_sampling_semantics", "technical repetitions cannot be tensor samples")
        if (
            record.get("semantic_variant_index") != 0
            or record.get("semantic_variant_count") != 1
            or _number(record.get("semantic_variant_weight"), "report semantic weight", minimum=0.0) != 1.0
        ):
            raise AggregateError(
                "unsupported_semantic_variants",
                "schema-1 aggregation accepts exactly one unit-weight semantic variant per report",
            )
        source = _mapping(record.get("source"), "report.source")
        _exact_keys(source, ("size_bytes", "sha256"), "report.source")
        _integer(source.get("size_bytes"), "report.source.size_bytes", minimum=1)
        _sha256(source.get("sha256"), "report.source.sha256")
        _sha256(record.get("source_document_sha256"), "report.source_document_sha256")
        _sha256(record.get("semantic_fingerprint_sha256"), "report.semantic_fingerprint_sha256")
        config_id = _string(record.get("outer_configuration_id"), "report configuration id")
        spec = _mapping(record.get("outer_configuration_spec"), "report configuration spec")
        plan = spec.get("plan")
        if plan not in ("fixed", "search"):
            raise AggregateError("unsupported_configuration", f"report {report_id!r} plan is unsupported")
        search = _mapping(record.get("search"), "report.search")
        if not search:
            raise AggregateError("missing_or_invalid_field", "report.search must not be empty")
        configuration_projection = {
            "outer_configuration_id": config_id,
            "outer_configuration_spec": spec,
            "bench_mode": record.get("bench_mode"),
            "raw_terminal_identity": record.get("raw_terminal_identity"),
            "effective_search_configuration": search,
        }
        configuration_key = "configuration-" + _canonical_sha256(configuration_projection)
        existing = self.configuration_definitions.get(configuration_key)
        if existing is not None and existing != configuration_projection:
            raise AggregateError("configuration_identity_collision", "configuration key collision")
        self.configuration_definitions[configuration_key] = configuration_projection
        counts = _mapping(record.get("counts"), "report.counts")
        _exact_keys(
            counts, ("tensors", "blocks", "tensor_plan_nodes", "realized_block_nodes"),
            "report.counts",
        )
        for field in counts:
            _integer(counts[field], f"report.counts.{field}")
        accounting = _mapping(record.get("accounting"), "report.accounting")
        expected_accounting = (
            "input_size_bytes", "raw_bytes", "tensor_data_bytes", "safetensors_prefix_bytes",
            "encoded_bytes_without_frame_headers",
            "block_frame_bytes_excluding_container_header_footer", "container_header_bytes",
            "container_footer_bytes", "projected_archive_bytes",
        )
        _exact_keys(accounting, expected_accounting, "report.accounting")
        for field in expected_accounting:
            _integer(accounting[field], f"report.accounting.{field}")
        if accounting["input_size_bytes"] != source["size_bytes"]:
            raise AggregateError("accounting_mismatch", "report input/source sizes disagree")
        if accounting["raw_bytes"] != accounting["tensor_data_bytes"]:
            raise AggregateError("accounting_mismatch", "report raw/tensor bytes disagree")
        if accounting["input_size_bytes"] != accounting["raw_bytes"] + accounting["safetensors_prefix_bytes"]:
            raise AggregateError("accounting_mismatch", "report input accounting does not close")
        if accounting["projected_archive_bytes"] != (
            accounting["container_header_bytes"]
            + accounting["block_frame_bytes_excluding_container_header_footer"]
            + accounting["container_footer_bytes"]
        ):
            raise AggregateError("accounting_mismatch", "report archive accounting does not close")
        role_rules = _mapping(record.get("role_rules"), "report.role_rules")
        _exact_keys(role_rules, ("schema", "file_sha256"), "report.role_rules")
        if role_rules.get("schema") != {"id": "brevis.tensor-role-rules", "version": 1}:
            raise AggregateError("unsupported_role_taxonomy", "report role taxonomy is unsupported")
        taxonomy_sha = _sha256(role_rules.get("file_sha256"), "report role taxonomy sha")
        if self.role_rules_sha256 is not None and taxonomy_sha != self.role_rules_sha256:
            raise AggregateError("mixed_role_taxonomies", "report role taxonomy differs from manifests")
        if record.get("terminal_payload_semantics") != (
            "packed_terminal_payload_bytes is available only per complete block program; "
            "terminal-node presence is not a terminal-specific payload share"
        ):
            raise AggregateError(
                "invalid_terminal_semantics", "report terminal payload semantics disagree",
            )
        _mapping(record.get("program_bytecode_evidence"), "report program bytecode evidence")
        self.reports[report_id] = ReportState(
            record, self.current_input_index, line_number, configuration_key,
        )
        self.configuration_report_ids[configuration_key].append(report_id)
        self.record_counts["reports"] += 1

    def tensor(self, record: Mapping[str, Any], line_number: int) -> None:
        del line_number
        _analysis_record(record, "tensor", TENSOR_KEYS, "tensor record")
        tensor_id = _string(record.get("tensor_id"), "tensor.tensor_id")
        if tensor_id in self.tensors:
            raise AggregateError("duplicate_tensor_id", f"duplicate tensor_id {tensor_id!r}")
        report_id = _string(record.get("report_id"), "tensor.report_id")
        if report_id not in self.reports:
            raise AggregateError("dangling_reference", f"tensor references unknown report {report_id!r}")
        if record.get("paper_metrics_eligible") is not True:
            raise AggregateError("ineligible_record", f"tensor {tensor_id!r} is not paper eligible")
        report = self.reports[report_id]
        tensor_index = _integer(record.get("tensor_index"), "tensor.tensor_index")
        if tensor_id != f"{report_id}:tensor:{tensor_index}":
            raise AggregateError("invalid_record_id", f"tensor_id {tensor_id!r} is not canonical")
        if tensor_index != report.tensor_count:
            raise AggregateError(
                "noncanonical_record_order",
                f"report {report_id!r} tensor index {tensor_index} is not the next index "
                f"{report.tensor_count}",
            )
        _string(record.get("name"), "tensor.name")
        dtype = _string(record.get("dtype"), "tensor.dtype")
        dtype_family = _string(record.get("dtype_family"), "tensor.dtype_family")
        shape_class = _string(record.get("shape_class"), "tensor.shape_class")
        shape = _array(record.get("shape"), "tensor.shape")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape):
            raise AggregateError("missing_or_invalid_field", "tensor.shape entries must be non-negative integers")
        if _integer(record.get("rank"), "tensor.rank") != len(shape):
            raise AggregateError("tensor_shape_mismatch", "tensor rank disagrees with shape")
        numel = _integer(record.get("numel"), "tensor.numel")
        block_count = _integer(record.get("block_count"), "tensor.block_count")
        block_start = _integer(record.get("block_start"), "tensor.block_start")
        if block_start != report.declared_block_cursor:
            raise AggregateError(
                "block_tensor_mismatch",
                f"tensor {tensor_id!r} block_start={block_start}, expected "
                f"{report.declared_block_cursor}",
            )
        start = _integer(record.get("file_data_start_byte"), "tensor file start")
        end = _integer(record.get("file_data_end_byte_exclusive"), "tensor file end")
        if end < start:
            raise AggregateError("tensor_range_mismatch", "tensor file range is reversed")
        weights = _mapping(record.get("weights"), "tensor.weights")
        _exact_keys(weights, ("semantic_variant_weight", "tensor_count_weight", "raw_byte_weight"), "tensor.weights")
        variant = _number(weights.get("semantic_variant_weight"), "tensor semantic weight", minimum=0.0)
        if variant != 1.0 or _number(weights.get("tensor_count_weight"), "tensor count weight") != variant:
            raise AggregateError("invalid_weight", "schema-1 tensor weights must have unit semantic/count weight")
        accounting = _mapping(record.get("accounting"), "tensor.accounting")
        accounting_fields = (
            "raw_bytes", "encoded_bytes_without_frame_headers",
            "block_frame_bytes_excluding_container_header_footer",
            "compression_ratio_without_frame_headers", "compression_ratio_with_block_frames",
            "saving_fraction_with_block_frames", "packed_terminal_payload_bytes_all_terminals",
            "packed_payload_fraction_of_encoded_program",
        )
        _exact_keys(accounting, accounting_fields, "tensor.accounting")
        raw = _integer(accounting.get("raw_bytes"), "tensor raw_bytes")
        encoded = _integer(accounting.get("encoded_bytes_without_frame_headers"), "tensor encoded bytes")
        framed = _integer(accounting.get("block_frame_bytes_excluding_container_header_footer"), "tensor framed bytes")
        payload = _integer(accounting.get("packed_terminal_payload_bytes_all_terminals"), "tensor payload bytes")
        if end - start != raw or _number(weights.get("raw_byte_weight"), "tensor raw weight") != raw:
            raise AggregateError("accounting_mismatch", "tensor range/raw-byte weight disagrees")
        _close(
            accounting.get("compression_ratio_without_frame_headers"),
            _ratio(raw, encoded), "tensor ratio without frames",
        )
        _close(accounting.get("compression_ratio_with_block_frames"), _ratio(raw, framed), "tensor ratio with frames")
        _close(accounting.get("saving_fraction_with_block_frames"), _saving(raw, framed), "tensor saving fraction")
        _close(
            accounting.get("packed_payload_fraction_of_encoded_program"),
            _ratio(payload, encoded), "tensor payload fraction",
        )
        if record.get("terminal_presence_is_payload_share") is not False:
            raise AggregateError("invalid_terminal_semantics", "tensor terminal presence cannot be payload share")
        role = _mapping(record.get("role"), "tensor.role")
        _exact_keys(
            role,
            ("interpretation", "labels", "matched_rule_ids", "normalized_name",
             "orthogonal_role_key", "taxonomy_file_sha256", "taxonomy_schema"),
            "tensor.role",
        )
        labels = _mapping(role.get("labels"), "tensor.role.labels")
        _exact_keys(labels, ROLE_LABELS, "tensor.role.labels")
        if not all(isinstance(labels[name], str) and labels[name] for name in ROLE_LABELS):
            raise AggregateError("missing_or_invalid_field", "tensor role labels must be nonempty text")
        if (
            role.get("taxonomy_schema") != {"id": "brevis.tensor-role-rules", "version": 1}
            or role.get("taxonomy_file_sha256") != self.role_rules_sha256
        ):
            raise AggregateError("mixed_role_taxonomies", "tensor role taxonomy disagrees")
        _string(role.get("orthogonal_role_key"), "tensor role key")
        _string(role.get("normalized_name"), "tensor normalized name")
        if role.get("interpretation") != "deterministic_name_heuristic_not_architecture_ground_truth":
            raise AggregateError("unsupported_role_taxonomy", "tensor role interpretation is unsupported")
        matched_rules = _mapping(role.get("matched_rule_ids"), "tensor.role.matched_rule_ids")
        _exact_keys(matched_rules, ROLE_LABELS, "tensor.role.matched_rule_ids")
        if not all(
            matched_rules[name] is None
            or (isinstance(matched_rules[name], str) and matched_rules[name])
            for name in ROLE_LABELS
        ):
            raise AggregateError(
                "missing_or_invalid_field", "matched role rule ids must be nonempty text or null",
            )
        program_value = record.get("tensor_plan")
        program = None if program_value is None else _program_summary(
            program_value, "tensor.tensor_plan", block=False,
        )
        if (program is None) != (numel == 0 and block_count == 0):
            raise AggregateError("invalid_program", "only empty tensors may have a null tensor plan")
        search = _mapping(record.get("search"), "tensor.search")
        _exact_keys(search, SEARCH_COUNTERS, "tensor.search")
        plan = _mapping(report.record["outer_configuration_spec"], "report spec")["plan"]
        search_expected = plan == "search"
        for field in SEARCH_COUNTERS:
            value = search[field]
            applicable = program is not None and (field == "search_expansions" or search_expected)
            if applicable:
                _integer(value, f"tensor.search.{field}")
            elif value is not None:
                raise AggregateError(
                    "unexpected_search_counter", f"tensor.search.{field} must be null when inapplicable",
                )
        raw_roots = _integer(record.get("raw_root_blocks"), "tensor.raw_root_blocks")
        planned_raw = _integer(record.get("planned_raw_root_blocks"), "tensor.planned_raw_root_blocks")
        fallback_raw = _integer(record.get("fallback_raw_root_blocks"), "tensor.fallback_raw_root_blocks")
        if raw_roots != planned_raw + fallback_raw or raw_roots > block_count:
            raise AggregateError("raw_classification_mismatch", "tensor raw block counts disagree")

        realized = _mapping(
            record.get("realized_block_programs"), "tensor.realized_block_programs",
        )
        _exact_keys(
            realized,
            ("structure_signature_counts", "parameter_signature_counts", "terminal_codecs_present"),
            "tensor.realized_block_programs",
        )
        expected_structure_counts = _mapping(
            realized.get("structure_signature_counts"),
            "tensor.realized_block_programs.structure_signature_counts",
        )
        expected_parameter_counts = _mapping(
            realized.get("parameter_signature_counts"),
            "tensor.realized_block_programs.parameter_signature_counts",
        )
        for label, counter in (
            ("structure_signature_counts", expected_structure_counts),
            ("parameter_signature_counts", expected_parameter_counts),
        ):
            if not all(
                isinstance(key, str) and key
                and isinstance(value, int) and not isinstance(value, bool) and value > 0
                for key, value in counter.items()
            ):
                raise AggregateError("missing_or_invalid_field", f"tensor {label} is invalid")
            if sum(counter.values()) != block_count:
                raise AggregateError(
                    "record_count_mismatch", f"tensor {label} does not sum to block_count",
                )
        terminals_present = _array(
            realized.get("terminal_codecs_present"),
            "tensor.realized_block_programs.terminal_codecs_present",
        )
        if (
            not all(isinstance(value, str) and value in TERMINAL_OPERATORS for value in terminals_present)
            or len(terminals_present) != len(set(terminals_present))
        ):
            raise AggregateError("unsupported_operator", "tensor terminal codec list is invalid")

        dimensions = [
            ("dtype", dtype), ("dtype_family", dtype_family),
            ("shape_class", shape_class), ("tensor_size_bucket", _tensor_size_bucket(raw)),
            ("orthogonal_role_key", str(role["orthogonal_role_key"])),
        ] + [(f"role.{name}", str(labels[name])) for name in ROLE_LABELS]
        tensor_state = TensorState(
            report_id, tensor_index, report.configuration_key, block_count, raw,
            encoded, framed, tuple(dimensions), dict(expected_structure_counts),
            dict(expected_parameter_counts), tuple(terminals_present),
        )
        self.tensors[tensor_id] = tensor_state
        report.tensor_count += 1
        report.declared_block_cursor += block_count
        report.tensor_raw_bytes += raw
        report.tensor_encoded_bytes += encoded
        report.tensor_framed_bytes += framed
        if program is not None:
            report.tensor_plan_nodes += int(program["node_count"])
            self._register_owner(
                tensor_id, report_id, "tensor_plan_template", program,
                raw_bytes=raw, tensor_weight=1.0,
            )
            plan_weights = {
                "program_count_weight": variant,
                "raw_byte_weight": raw * variant,
                "tensor_equal_weight": variant,
            }
            for scope in ("all_configurations", report.configuration_key):
                self.programs[scope]["tensor_plan_template"].add(program, plan_weights)
        for dimension, value in dimensions:
            self.preferences[(report.configuration_key, dimension, value)].add_tensor(
                record, program, search_expected=search_expected,
            )
        saving_fraction = _saving(raw, framed)
        if program is not None:
            if saving_fraction is None:
                raise AggregateError("internal_error", "nonempty tensor has no saving fraction")
        for field in SEARCH_COUNTERS:
            value = search[field]
            applicable = program is not None and (field == "search_expansions" or search_expected)
            for scope in ("all_configurations", report.configuration_key):
                relationship = self.relationships[(scope, field)]
                if applicable:
                    if value is None or saving_fraction is None:
                        raise AggregateError("internal_error", f"validated {field} became null")
                    relationship.observe(float(value), saving_fraction, raw)
                else:
                    relationship.mark_inapplicable()
        self.record_counts["tensors"] += 1

    def block(self, record: Mapping[str, Any], line_number: int) -> None:
        del line_number
        _analysis_record(record, "block", BLOCK_KEYS, "block record")
        block_id = _string(record.get("block_id"), "block.block_id")
        if block_id in self.owners:
            raise AggregateError("duplicate_block_id", f"duplicate block_id {block_id!r}")
        report_id = _string(record.get("report_id"), "block.report_id")
        tensor_id = _string(record.get("tensor_id"), "block.tensor_id")
        if report_id not in self.reports or tensor_id not in self.tensors:
            raise AggregateError("dangling_reference", f"block {block_id!r} has an unknown owner")
        tensor = self.tensors[tensor_id]
        report = self.reports[report_id]
        if tensor.report_id != report_id:
            raise AggregateError("dangling_reference", "block report/tensor references disagree")
        if record.get("paper_metrics_eligible") is not True:
            raise AggregateError("ineligible_record", f"block {block_id!r} is not paper eligible")
        block_index = _integer(record.get("block_index"), "block.block_index")
        if block_id != f"{report_id}:block:{block_index}":
            raise AggregateError("invalid_record_id", f"block_id {block_id!r} is not canonical")
        if block_index != report.block_count:
            raise AggregateError(
                "noncanonical_record_order",
                f"report {report_id!r} block index {block_index} is not the next index "
                f"{report.block_count}",
            )
        if _integer(record.get("tensor_index"), "block.tensor_index") != tensor.tensor_index:
            raise AggregateError("block_tensor_mismatch", "block tensor index disagrees")
        _integer(record.get("element_offset"), "block.element_offset")
        _integer(record.get("element_count"), "block.element_count", minimum=1)
        accounting = _mapping(record.get("accounting"), "block.accounting")
        accounting_fields = (
            "raw_bytes", "encoded_bytes_without_frame_headers", "program_bytecode_bytes",
            "packed_terminal_payload_bytes_all_terminals", "frame_header_bytes",
            "framed_bytes", "compression_ratio_without_frame_headers",
            "compression_ratio_with_block_frame", "saving_fraction_with_block_frame",
            "packed_payload_fraction_of_encoded_program",
        )
        _exact_keys(accounting, accounting_fields, "block.accounting")
        raw = _integer(accounting.get("raw_bytes"), "block raw bytes", minimum=1)
        encoded = _integer(accounting.get("encoded_bytes_without_frame_headers"), "block encoded bytes", minimum=1)
        bytecode = _integer(accounting.get("program_bytecode_bytes"), "block bytecode bytes", minimum=1)
        payload = _integer(accounting.get("packed_terminal_payload_bytes_all_terminals"), "block payload bytes")
        frame_header = _integer(accounting.get("frame_header_bytes"), "block frame header bytes")
        framed = _integer(accounting.get("framed_bytes"), "block framed bytes", minimum=1)
        if encoded != bytecode + payload or frame_header != bytecode + 12 or framed != encoded + 12:
            raise AggregateError("accounting_mismatch", "block bytecode/payload/frame accounting does not close")
        _close(accounting.get("compression_ratio_without_frame_headers"), raw / encoded, "block ratio without frames")
        _close(accounting.get("compression_ratio_with_block_frame"), raw / framed, "block ratio with frame")
        _close(accounting.get("saving_fraction_with_block_frame"), (raw - framed) / raw, "block saving fraction")
        _close(
            accounting.get("packed_payload_fraction_of_encoded_program"),
            payload / encoded, "block payload fraction",
        )
        weights = _mapping(record.get("weights"), "block.weights")
        _exact_keys(
            weights,
            ("semantic_variant_weight", "block_count_weight", "tensor_equal_weight", "raw_byte_weight"),
            "block.weights",
        )
        variant = _number(weights.get("semantic_variant_weight"), "block semantic weight")
        if (
            variant != 1.0
            or _number(weights.get("block_count_weight"), "block count weight") != variant
            or _number(weights.get("raw_byte_weight"), "block raw weight") != raw
        ):
            raise AggregateError("invalid_weight", "block count/raw weights disagree")
        tensor_equal = _number(weights.get("tensor_equal_weight"), "block tensor-equal weight")
        expected_tensor_equal = 1.0 / tensor.block_count_expected
        if not math.isclose(tensor_equal, expected_tensor_equal, rel_tol=1e-12, abs_tol=1e-12):
            raise AggregateError("invalid_weight", "block tensor-equal weight disagrees")
        program = _program_summary(record.get("program"), "block.program", block=True)
        _sha256(record.get("program_bytecode_sha256"), "block.program_bytecode_sha256")
        if record.get("terminal_presence_is_payload_share") is not False:
            raise AggregateError("invalid_terminal_semantics", "block terminal presence cannot be payload share")
        if record.get("raw_classification") not in (None, "planned_raw", "fallback_raw"):
            raise AggregateError("raw_classification_mismatch", "block raw classification is unsupported")
        is_raw_root = program["operators_preorder"][0] == "raw"
        if is_raw_root != (record.get("raw_classification") is not None):
            raise AggregateError("raw_classification_mismatch", "block program/raw classification disagree")
        self._register_owner(
            block_id, report_id, "realized_block", program,
            raw_bytes=raw, tensor_weight=tensor_equal,
        )
        report.block_count += 1
        report.realized_block_nodes += int(program["node_count"])
        report.block_raw_bytes += raw
        report.block_encoded_bytes += encoded
        report.block_framed_bytes += framed
        tensor.block_count += 1
        tensor.block_raw_bytes += raw
        tensor.block_encoded_bytes += encoded
        tensor.block_framed_bytes += framed
        tensor.actual_structure_counts[str(program["structure_signature"])] += 1
        tensor.actual_parameter_counts[str(program["parameter_signature"])] += 1
        tensor.actual_terminals_present.update(
            str(value) for value in program["terminal_codecs_preorder"]
        )
        program_weights = {
            "program_count_weight": variant,
            "raw_byte_weight": raw * variant,
            "tensor_equal_weight": tensor_equal,
        }
        for scope in ("all_configurations", report.configuration_key):
            self.programs[scope]["realized_block"].add(
                program, program_weights,
                bytecode_accounting={
                    "program_bytecode_bytes": bytecode,
                    "encoded_bytes_without_frame_headers": encoded,
                    "framed_bytes": framed,
                    "raw_bytes": raw,
                },
            )
        for dimension, value in tensor.dimensions:
            self.preferences[(report.configuration_key, dimension, value)].add_block(
                program, program_weights,
            )
        self.record_counts["blocks"] += 1

    def node(self, record: Mapping[str, Any], line_number: int) -> None:
        del line_number
        _analysis_record(record, "node", NODE_KEYS, "node record")
        owner_id = _string(record.get("owner_id"), "node.owner_id")
        owner = self.owners.get(owner_id)
        if owner is None:
            raise AggregateError("dangling_reference", f"node references unknown owner {owner_id!r}")
        if record.get("report_id") != owner.report_id or record.get("tree_scope") != owner.tree_scope:
            raise AggregateError("node_owner_mismatch", f"node owner metadata disagrees for {owner_id!r}")
        index = _integer(record.get("preorder_index"), "node.preorder_index")
        if index >= owner.node_count:
            raise AggregateError("node_owner_mismatch", "node preorder index exceeds program length")
        node_id = _string(record.get("node_id"), "node.node_id")
        if node_id != f"{owner_id}:node:{index}":
            raise AggregateError("invalid_record_id", f"node_id {node_id!r} is not canonical")
        bit = 1 << index
        if self.node_masks[owner_id] & bit:
            raise AggregateError("duplicate_node_id", f"duplicate node {node_id!r}")
        self.node_masks[owner_id] |= bit
        operator = _string(record.get("op"), "node.op")
        if operator not in KNOWN_OPERATORS or operator != owner.operators[index]:
            raise AggregateError("node_owner_mismatch", "node operator disagrees with owner program")
        terminal = _boolean(record.get("is_terminal"), "node.is_terminal")
        if terminal != (operator in TERMINAL_OPERATORS):
            raise AggregateError("invalid_program", "node terminal flag disagrees with operator")
        if record.get("operator_kind") != ("terminal" if terminal else "transform"):
            raise AggregateError("invalid_program", "node operator kind disagrees")
        _integer(record.get("params_u32"), "node.params_u32")
        child_count = _integer(record.get("child_count"), "node.child_count")
        if terminal != (child_count == 0):
            raise AggregateError("invalid_program", "node child count disagrees with terminal flag")
        path = _array(record.get("path"), "node.path")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in path):
            raise AggregateError("invalid_program", "node.path must contain non-negative integers")
        if _integer(record.get("depth_zero_based"), "node depth") != len(path):
            raise AggregateError("invalid_program", "node path/depth disagree")
        expected_first = operator not in owner.operators[:index]
        if record.get("first_occurrence_of_operator_in_program") is not expected_first:
            raise AggregateError("node_owner_mismatch", "node first-occurrence flag disagrees")
        for field, expected in (
            ("program_structure_signature", owner.structure_signature),
            ("program_parameter_signature", owner.parameter_signature),
            ("program_structure_sha256", owner.structure_sha256),
            ("program_parameter_sha256", owner.parameter_sha256),
        ):
            if record.get(field) != expected:
                raise AggregateError("node_owner_mismatch", f"node {field} disagrees")
        _string(record.get("subtree_structure_signature"), "node subtree structure")
        _string(record.get("subtree_parameter_signature"), "node subtree parameters")
        _sha256(record.get("subtree_structure_sha256"), "node subtree structure sha")
        _sha256(record.get("subtree_parameter_sha256"), "node subtree parameter sha")
        attribution = _mapping(record.get("terminal_payload_attribution"), "node terminal attribution")
        _exact_keys(
            attribution,
            ("terminal_present", "terminal_specific_payload_bytes",
             "terminal_specific_payload_share", "status", "terminal_presence_is_payload_share"),
            "node terminal attribution",
        )
        if (
            attribution.get("terminal_present") is not terminal
            or attribution.get("terminal_specific_payload_bytes") is not None
            or attribution.get("terminal_specific_payload_share") is not None
            or attribution.get("terminal_presence_is_payload_share") is not False
            or attribution.get("status") != "unavailable_in_bench_schema_4_aggregate_payload_only"
        ):
            raise AggregateError("invalid_terminal_semantics", "node terminal payload attribution disagrees")
        weights = _mapping(record.get("weights"), "node.weights")
        _exact_keys(
            weights,
            ("semantic_variant_weight", "node_occurrence_weight",
             "operator_presence_program_weight", "operator_presence_tensor_weight",
             "operator_presence_raw_byte_weight", "weight_semantics"),
            "node.weights",
        )
        if _number(weights.get("semantic_variant_weight"), "node semantic weight") != 1.0:
            raise AggregateError("invalid_weight", "node semantic weight must be one")
        expected_weights = {
            "node_occurrence_weight": 1.0,
            "operator_presence_program_weight": 1.0 if expected_first else 0.0,
            "operator_presence_tensor_weight": owner.tensor_weight if expected_first else 0.0,
            "operator_presence_raw_byte_weight": float(owner.raw_bytes) if expected_first else 0.0,
        }
        observed_weights = {}
        for field, expected in expected_weights.items():
            observed = _number(weights.get(field), f"node.weights.{field}")
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise AggregateError("invalid_weight", f"node.weights.{field} disagrees")
            observed_weights[field] = observed
        if weights.get("weight_semantics") != (
            "presence weights apply once per distinct operator per program; they are not "
            "a partition of program bytes and must not be summed across operators"
        ):
            raise AggregateError("invalid_weight", "node weight semantics disagree")
        for scope in ("all_configurations", owner.configuration_key):
            self.node_operators[(scope, owner.tree_scope)].add(operator, observed_weights)
        self.record_counts["nodes"] += 1

    def error(self, record: Mapping[str, Any], line_number: int) -> None:
        del record
        raise AggregateError(
            "analysis_exclusions_present",
            f"errors.jsonl contains an exclusion record at line {line_number}; formal aggregation stops",
        )

    def _validate_closed_accounting(self) -> None:
        for tensor_id, tensor in self.tensors.items():
            if tensor.block_count != tensor.block_count_expected:
                raise AggregateError(
                    "record_count_mismatch",
                    f"tensor {tensor_id!r} has {tensor.block_count} blocks, expected {tensor.block_count_expected}",
                )
            report = self.reports[tensor.report_id]
            tensor_record_raw = tensor.raw_bytes
            if (
                tensor.block_raw_bytes != tensor_record_raw
                or tensor.block_encoded_bytes != tensor.encoded_bytes
                or tensor.block_framed_bytes != tensor.framed_bytes
            ):
                raise AggregateError(
                    "accounting_mismatch", f"tensor {tensor_id!r} block accounting disagrees",
                )
            if (
                dict(tensor.actual_structure_counts) != dict(tensor.expected_structure_counts)
                or dict(tensor.actual_parameter_counts) != dict(tensor.expected_parameter_counts)
                or tuple(sorted(tensor.actual_terminals_present))
                != tuple(sorted(tensor.expected_terminals_present))
            ):
                raise AggregateError(
                    "program_summary_mismatch",
                    f"tensor {tensor_id!r} realized-program summary disagrees with blocks",
                )
        for report_id, state in self.reports.items():
            declared = _mapping(state.record["counts"], "report counts")
            actual_counts = {
                "tensors": state.tensor_count,
                "blocks": state.block_count,
                "tensor_plan_nodes": state.tensor_plan_nodes,
                "realized_block_nodes": state.realized_block_nodes,
            }
            if any(declared[field] != actual for field, actual in actual_counts.items()):
                raise AggregateError(
                    "record_count_mismatch",
                    f"report {report_id!r} declared counts {dict(declared)!r}, observed {actual_counts!r}",
                )
            if state.declared_block_cursor != state.block_count:
                raise AggregateError(
                    "record_count_mismatch",
                    f"report {report_id!r} tensor block ranges do not cover all blocks",
                )
            accounting = _mapping(state.record["accounting"], "report accounting")
            expected = {
                "raw_bytes": state.tensor_raw_bytes,
                "tensor_data_bytes": state.tensor_raw_bytes,
                "encoded_bytes_without_frame_headers": state.block_encoded_bytes,
                "block_frame_bytes_excluding_container_header_footer": state.block_framed_bytes,
            }
            if any(accounting[field] != value for field, value in expected.items()):
                raise AggregateError(
                    "accounting_mismatch",
                    f"report {report_id!r} accounting differs from tensor/block sums",
                )
            if (
                state.tensor_raw_bytes != state.block_raw_bytes
                or state.tensor_encoded_bytes != state.block_encoded_bytes
                or state.tensor_framed_bytes != state.block_framed_bytes
            ):
                raise AggregateError(
                    "accounting_mismatch", f"report {report_id!r} tensor/block accounting differs",
                )
        for owner_id, owner in self.owners.items():
            expected_mask = (1 << owner.node_count) - 1
            if self.node_masks.get(owner_id, 0) != expected_mask:
                raise AggregateError(
                    "record_count_mismatch",
                    f"program owner {owner_id!r} lacks exactly one node record per preorder index",
                )

    def _render_operators(self, configuration_scope: str, tree_scope: str) -> dict[str, Any]:
        categories = self.node_operators[(configuration_scope, tree_scope)]
        programs = self.programs[configuration_scope][tree_scope]
        program_totals = {
            weight: sum(programs.histograms["node_count"][weight].values.values())
            for weight in PROGRAM_WEIGHTS
        }
        occurrence_total = categories.totals()["node_occurrence_weight"]
        return categories.render(
            key_name="operator",
            fraction_denominators={
                "node_occurrence_weight": occurrence_total,
                "operator_presence_program_weight": program_totals["program_count_weight"],
                "operator_presence_tensor_weight": program_totals["tensor_equal_weight"],
                "operator_presence_raw_byte_weight": program_totals["raw_byte_weight"],
            },
            fraction_semantics=(
                "node_occurrence fraction partitions node occurrences; each operator-presence "
                "fraction is coverage of the corresponding program weight and presence "
                "fractions across operators may sum above one"
            ),
        )

    def consume_input(self, resolved: ResolvedInput) -> None:
        manifest, manifest_identity = _load_manifest(resolved)
        taxonomy_sha = str(_mapping(manifest["role_rules"], "role rules")["file_sha256"])
        if self.role_rules_sha256 is None:
            self.role_rules_sha256 = taxonomy_sha
        elif self.role_rules_sha256 != taxonomy_sha:
            raise AggregateError(
                "mixed_role_taxonomies", "input manifests use different role taxonomy files",
            )
        self.current_input_index = len(self.input_provenance)
        manifest_files = _mapping(manifest["files"], "manifest files")
        file_provenance: dict[str, Any] = {}
        observed: dict[str, int] = {}
        callbacks = {
            "errors.jsonl": self.error,
            "reports.jsonl": self.report,
            "tensors.jsonl": self.tensor,
            "blocks.jsonl": self.block,
            "nodes.jsonl": self.node,
        }
        for logical_name in (
            "errors.jsonl", "reports.jsonl", "tensors.jsonl", "blocks.jsonl", "nodes.jsonl",
        ):
            provenance = _consume_jsonl(
                resolved.analysis_dir, logical_name,
                _mapping(manifest_files[logical_name], f"manifest {logical_name}"),
                callbacks[logical_name],
            )
            file_provenance[logical_name] = provenance
            observed[logical_name] = int(provenance["record_count"])
        summary = _mapping(manifest["summary"], "manifest summary")
        for logical_name, summary_field in SUMMARY_COUNT_FIELDS.items():
            if observed[logical_name] != summary[summary_field]:
                raise AggregateError(
                    "manifest_count_mismatch",
                    f"{logical_name} has {observed[logical_name]} records but manifest "
                    f"{summary_field}={summary[summary_field]}",
                )
        if summary["paper_metrics_eligible_reports"] != observed["reports.jsonl"]:
            raise AggregateError(
                "ineligible_record", "not every accepted report is paper-metrics eligible",
            )
        adjacent_files = {}
        if resolved.result_dir is not None:
            for name in ("result.json", "summary.json", "artifact-manifest.json"):
                path = resolved.result_dir / name
                if path.is_file():
                    adjacent_files[name] = {"path": str(path), **_file_identity(path)}
        self.input_provenance.append({
            "input_index": self.current_input_index,
            "argument": resolved.argument,
            "resolved_analysis_directory": str(resolved.analysis_dir),
            "resolved_result_directory": (
                str(resolved.result_dir) if resolved.result_dir is not None else None
            ),
            "manifest": {
                "path": str(resolved.manifest_path),
                "bytes": manifest_identity["bytes"],
                "sha256": manifest_identity["sha256"],
                "analysis_schema": manifest["analysis_schema"],
                "role_rules": manifest["role_rules"],
                "declared_summary": manifest["summary"],
            },
            "logical_files": file_provenance,
            "adjacent_formal_result_files": adjacent_files,
        })

    def render(self) -> dict[str, Any]:
        self._validate_closed_accounting()
        report_rows = []
        for report_id, state in sorted(self.reports.items()):
            record = state.record
            accounting = _mapping(record["accounting"], "report accounting")
            raw = int(accounting["raw_bytes"])
            framed = int(accounting["block_frame_bytes_excluding_container_header_footer"])
            report_rows.append({
                "report_id": report_id,
                "configuration_key": state.configuration_key,
                "outer_configuration_id": record["outer_configuration_id"],
                "source": record["source"],
                "source_document_sha256": record["source_document_sha256"],
                "semantic_fingerprint_sha256": record["semantic_fingerprint_sha256"],
                "paper_metrics_eligible": True,
                "accounting": accounting,
                "storage_saved_bytes_against_block_framed": raw - framed,
                "saving_fraction_with_block_frames": _saving(raw, framed),
                "trace": {
                    "input_index": state.input_index,
                    "logical_file": "reports.jsonl",
                    "line_number": state.line_number,
                },
            })
        config_rows = []
        for key, definition in sorted(self.configuration_definitions.items()):
            report_ids = sorted(self.configuration_report_ids[key])
            config_rows.append({
                "configuration_key": key,
                **definition,
                "report_count": len(report_ids),
                "report_ids": report_ids,
                "tensor_plan_programs": self.programs[key]["tensor_plan_template"].render(),
                "realized_block_programs": self.programs[key]["realized_block"].render(),
                "operators": {
                    scope: self._render_operators(key, scope)
                    for scope in ("tensor_plan_template", "realized_block")
                },
                "search_cost_relationships": {
                    field: self.relationships[(key, field)].render()
                    for field in SEARCH_COUNTERS
                },
            })
        preference_rows = [
            {
                "configuration_key": configuration_key,
                "dimension": dimension,
                "value": value,
                **accumulator.render(),
            }
            for (configuration_key, dimension, value), accumulator
            in sorted(self.preferences.items())
        ]
        aggregate_identity_projection = {
            "schema": OUTPUT_SCHEMA,
            "input_manifests": [entry["manifest"]["sha256"] for entry in self.input_provenance],
            "logical_file_sha256": [
                {
                    name: entry["logical_files"][name]["logical_sha256"]
                    for name in REQUIRED_LOGICAL_FILES
                }
                for entry in self.input_provenance
            ],
            "formal_only": True,
        }
        tool_identity = _file_identity(pathlib.Path(__file__).resolve())
        return {
            "schema": OUTPUT_SCHEMA,
            "aggregate_input_fingerprint_sha256": _canonical_sha256(
                aggregate_identity_projection
            ),
            "tool": {
                "path": str(pathlib.Path(__file__).resolve()),
                "sha256": tool_identity["sha256"],
                "hash_scope": "exact script bytes",
            },
            "scope": {
                "formal_paper_metrics_only": True,
                "technical_repetitions_count_as_independent_tensor_samples": False,
                "input_count": len(self.input_provenance),
                "report_configuration_observations": len(self.reports),
                "cross_configuration_tensor_records_are_independent_models": False,
            },
            "traceability": {
                "inputs": self.input_provenance,
                "record_counts": {
                    "reports": self.record_counts["reports"],
                    "tensors": self.record_counts["tensors"],
                    "blocks": self.record_counts["blocks"],
                    "nodes": self.record_counts["nodes"],
                    "errors": 0,
                },
                "role_rules_sha256": self.role_rules_sha256,
            },
            "weighting_policy": {
                "semantic_variant_weight": (
                    "schema 1 contributes one unit-weight canonical semantic variant per report"
                ),
                "tensor_count_weight": "each non-repeated tensor/configuration record contributes one",
                "raw_byte_weight": "each program is weighted by its owning raw tensor or block bytes",
                "block_count_weight": "each realized block program contributes one",
                "tensor_equal_weight": (
                    "each block contributes 1/(blocks in its tensor), so each nonempty tensor "
                    "contributes total realized-program weight one"
                ),
                "configuration_pooling_warning": (
                    "the pooled view describes program observations across configurations; the "
                    "same source tensors appear once per configuration and are not independent models"
                ),
                "operator_presence_warning": (
                    "presence weights apply once per operator per program and do not partition bytes"
                ),
            },
            "unsupported_or_unavailable": {
                "terminal_specific_payload_bytes": {
                    "status": "unsupported_by_generated_dsl_analysis_schema_1",
                    "reason": (
                        "bench schema 4 exposes one packed payload total per complete program, "
                        "not bytes attributable to individual terminal nodes"
                    ),
                },
                "planning_wall_time": {
                    "status": "unsupported_by_generated_dsl_analysis_schema_1",
                    "reason": (
                        "planning_wall_ms is intentionally absent from canonical Generated-DSL "
                        "records; use the system benchmark summary for diagnostic replay time"
                    ),
                },
                "missing_required_values": {
                    "status": "none",
                    "policy": "required missing values abort aggregation; inapplicable search fields are counted",
                },
            },
            "tensor_size_bucket_definition": [
                {"label": "zero_bytes", "lower_exclusive": None, "upper_inclusive": 0},
                {"label": "1B_to_4KiB", "lower_exclusive": 0, "upper_inclusive": 4096},
                {"label": "over_4KiB_to_64KiB", "lower_exclusive": 4096, "upper_inclusive": 65536},
                {"label": "over_64KiB_to_1MiB", "lower_exclusive": 65536, "upper_inclusive": 1048576},
                {"label": "over_1MiB_to_16MiB", "lower_exclusive": 1048576, "upper_inclusive": 16777216},
                {"label": "over_16MiB", "lower_exclusive": 16777216, "upper_inclusive": None},
            ],
            "reports": report_rows,
            "pooled_program_observations": {
                "scope_warning": (
                    "descriptive pooling across configuration observations; consult by_configuration "
                    "for method comparisons"
                ),
                "tensor_plan_programs": self.programs["all_configurations"]["tensor_plan_template"].render(),
                "realized_block_programs": self.programs["all_configurations"]["realized_block"].render(),
                "operators": {
                    scope: self._render_operators("all_configurations", scope)
                    for scope in ("tensor_plan_template", "realized_block")
                },
                "search_cost_relationships": {
                    field: self.relationships[("all_configurations", field)].render()
                    for field in SEARCH_COUNTERS
                },
            },
            "by_configuration": config_rows,
            "stratified_preferences": preference_rows,
            "relationship_interpretation": {
                "unit": "one tensor/configuration record",
                "outcome": "saving_fraction_with_block_frames",
                "costs": list(SEARCH_COUNTERS),
                "search_expansions_semantics": "bounded-search/planning expansion counter",
                "other_counter_semantics": "search candidate/probe workload or selected candidate rank",
                "causal_claim_supported": False,
                "warning": (
                    "correlations are descriptive and can be confounded by model, dtype, tensor size, "
                    "role, and configuration; null denotes insufficient or constant support"
                ),
            },
        }


def aggregate_directories(paths: Sequence[os.PathLike[str] | str]) -> dict[str, Any]:
    if not paths:
        raise AggregateError("missing_input", "at least one input directory is required")
    resolved = [_resolve_input(path) for path in paths]
    identities = [str(item.manifest_path) for item in resolved]
    if len(identities) != len(set(identities)):
        raise AggregateError("duplicate_input", "the same analysis manifest was supplied more than once")
    aggregator = Aggregator()
    for item in resolved:
        aggregator.consume_input(item)
    return aggregator.render()


def _atomic_write(path: pathlib.Path, payload: bytes, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if force:
            os.replace(temporary, path)
        else:
            try:
                # Linking a fully written same-filesystem temporary into place is
                # an atomic no-clobber publication: unlike exists()+replace(),
                # concurrent writers cannot both succeed.
                os.link(temporary, path)
            except FileExistsError as exc:
                raise AggregateError("output_exists", f"refusing to overwrite {path}") from exc
            temporary.unlink()
        try:
            directory_descriptor = os.open(
                path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            # The completed file is already published. Some filesystems do not
            # permit directory fsync; this affects crash durability, not the
            # no-clobber or atomic-visibility contract.
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 eval/summarize_generated_dsl.py eval/results/formal/.../model.safetensors\n"
            "  python3 eval/summarize_generated_dsl.py result-a result-b --output dsl-summary.json\n\n"
            "Inputs are never modified. Existing output files are preserved unless --force is used."
        ),
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="formal result directories, DSL analysis directories, or analysis manifest.json files",
    )
    parser.add_argument(
        "--output", help="write pretty JSON atomically to this path (default: stdout)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing --output file",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        document = aggregate_directories(args.inputs)
        payload = (json.dumps(
            document, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True,
        ) + "\n").encode("utf-8")
        if args.output:
            _atomic_write(pathlib.Path(args.output), payload, force=args.force)
        else:
            sys.stdout.buffer.write(payload)
    except (AggregateError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, AggregateError) else "unexpected_failure"
        print(
            _canonical_json({"status": "failed", "error_code": code, "message": str(exc)}),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
