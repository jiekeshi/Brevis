#!/usr/bin/env python3
"""Extract auditable Generated-DSL evidence from Brevis system benchmarks.

The analyzer deliberately has a narrow input contract.  It accepts only a
successful ``brevis.system-benchmark`` schema-2 document whose technical runs
contain ``brevis.bench-report`` schema 4.  Warmups and measured repetitions are
technical consistency evidence rather than independent tensor samples. Repeated
technical measurements are resolved through the checkpoint-local canonical
report and validated against its semantic fingerprint.  The canonical report
is analyzed exactly once per outer configuration, so repeated timing runs are
not presented as independent tensors.

Schema 4 reports only a program-level packed terminal payload.  It does not
attribute bytes to individual terminal nodes.  The node output therefore
records terminal *presence* and leaves terminal-specific payload bytes/shares
null.  Downstream analyses must not interpret terminal presence as payload
share.
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
import struct
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


ANALYSIS_SCHEMA_ID = "brevis.generated-dsl-analysis"
ANALYSIS_SCHEMA_VERSION = 1
SYSTEM_SCHEMA_ID = "brevis.system-benchmark"
SYSTEM_SCHEMA_VERSION = 2
BENCH_KIND = "brevis.bench-report"
BENCH_SCHEMA_VERSION = 4
PROGRAM_SEQUENCE_SPEC_ID = "brevis.program-bytecode-sequence.v1"
BENCH_DETAIL_FINGERPRINT_SPEC_ID = "brevis.bench-detail-semantics.v1"
BENCH_DETAIL_FIELDS = ("tensors", "blocks")
BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS = (
    "/input", "/planning_wall_ms", "/encoding_wall_ms", "/prior/path",
)
ROLE_RULES_SCHEMA_ID = "brevis.tensor-role-rules"
ROLE_RULES_SCHEMA_VERSION = 1
DEFAULT_ROLE_RULES = pathlib.Path(__file__).with_name("tensor_role_rules_v1.json")

TERMINAL_OPERATORS = frozenset(("raw", "bitpack", "huffman", "rans"))
UNARY_OPERATORS = frozenset((
    "xor_const", "add_const_mod", "xor_prev", "diff_mod", "zigzag", "gray",
    "rotate_bits", "bit_reverse",
))
BINARY_OPERATORS = frozenset(("split_field", "topk_codebook", "rle", "deinterleave"))
TERNARY_OPERATORS = frozenset(("split_float",))
VARIABLE_OPERATORS = frozenset(("bit_plane", "byte_plane"))
KNOWN_OPERATORS = (
    TERMINAL_OPERATORS | UNARY_OPERATORS | BINARY_OPERATORS
    | TERNARY_OPERATORS | VARIABLE_OPERATORS
)

DTYPE_BYTES = {
    "F16": 2, "BF16": 2, "F32": 4,
    "U8": 1, "U16": 2, "U32": 4,
    "I8": 1, "I16": 2, "I32": 4,
    "F8_E4M3": 1, "F8_E5M2": 1,
}
FLOAT_DTYPES = frozenset(("F16", "BF16", "F32", "F8_E4M3", "F8_E5M2"))
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AnalysisError(ValueError):
    """A fail-closed schema or evidence validation error."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclasses.dataclass(frozen=True)
class RoleRules:
    document: Mapping[str, Any]
    file_sha256: str
    compiled: Mapping[str, tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...]]
    defaults: Mapping[str, str]


@dataclasses.dataclass
class AnalysisBundle:
    reports: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    tensors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    blocks: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    nodes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    role_rules: dict[str, Any] = dataclasses.field(default_factory=dict)

    def records(self) -> Mapping[str, list[dict[str, Any]]]:
        return {
            "reports": self.reports,
            "tensors": self.tensors,
            "blocks": self.blocks,
            "nodes": self.nodes,
            "errors": self.errors,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "accepted_semantic_reports": len(self.reports),
            "tensor_records": len(self.tensors),
            "block_records": len(self.blocks),
            "node_records": len(self.nodes),
            "excluded_records": len(self.errors),
            "paper_metrics_eligible_reports": sum(
                record["paper_metrics_eligible"] for record in self.reports
            ),
        }


@dataclasses.dataclass(frozen=True)
class ProgramInfo:
    canonical_tree: Mapping[str, Any]
    structure_signature: str
    parameter_signature: str
    structure_sha256: str
    parameter_sha256: str
    rendered_program: str
    node_count: int
    node_depth: int
    transform_depth: int
    terminal_count: int
    operators_preorder: tuple[str, ...]
    terminal_codecs_preorder: tuple[str, ...]
    nodes: tuple[Mapping[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class ValidatedBench:
    report: Mapping[str, Any]
    tensor_programs: tuple[ProgramInfo | None, ...]
    block_programs: tuple[ProgramInfo, ...]


@dataclasses.dataclass(frozen=True)
class ResolvedTechnicalReport:
    run: Mapping[str, Any]
    phase: str
    index: int
    materialized_report: Mapping[str, Any]
    semantic_sha256: str
    is_canonical: bool


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _bench_detail_semantic_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror the schema-2 harness fingerprint contract independently.

    Only machine-local paths and the two internal wall-clock observations are
    excluded.  Unknown schema-4 fields deliberately remain semantic.
    """

    projection = dict(report)
    projection.pop("input", None)
    projection.pop("planning_wall_ms", None)
    projection.pop("encoding_wall_ms", None)
    prior = projection.get("prior")
    if isinstance(prior, Mapping):
        projected_prior = dict(prior)
        projected_prior.pop("path", None)
        projection["prior"] = projected_prior
    return projection


def _bench_detail_semantic_sha256(report: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            _bench_detail_semantic_projection(report),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnalysisError(
            "invalid_semantic_fingerprint",
            f"bench report cannot be semantically fingerprinted: {exc}",
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _analysis_header(record_type: str) -> dict[str, Any]:
    return {
        "analysis_schema": {
            "id": ANALYSIS_SCHEMA_ID,
            "version": ANALYSIS_SCHEMA_VERSION,
        },
        "record_type": record_type,
    }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonnegative_int(value: Any, label: str) -> int:
    if not _is_int(value) or value < 0:
        raise AnalysisError("invalid_integer", f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise AnalysisError("invalid_integer", f"{label} must be positive")
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise AnalysisError("invalid_sha256", f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError("invalid_object", f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise AnalysisError("invalid_array", f"{label} must be an array")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _saving_fraction(raw_bytes: int, encoded_bytes: int) -> float | None:
    return (raw_bytes - encoded_bytes) / raw_bytes if raw_bytes else None


def _unique_preorder(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_role_rules(path: os.PathLike[str] | str = DEFAULT_ROLE_RULES) -> RoleRules:
    rules_path = pathlib.Path(path)
    payload = rules_path.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("invalid_role_rules", f"cannot parse role rules: {exc}") from exc
    root = _mapping(document, "role rules")
    if root.get("schema") != {
        "id": ROLE_RULES_SCHEMA_ID, "version": ROLE_RULES_SCHEMA_VERSION,
    }:
        raise AnalysisError("unsupported_role_rules", "role rules schema must be exactly version 1")
    if root.get("frozen") is not True:
        raise AnalysisError("invalid_role_rules", "role rules must declare frozen=true")
    taxonomies = _mapping(root.get("taxonomies"), "role rules taxonomies")
    if not taxonomies:
        raise AnalysisError("invalid_role_rules", "role rules contain no taxonomy axes")
    compiled: dict[str, tuple[tuple[str, str, tuple[re.Pattern[str], ...]], ...]] = {}
    defaults: dict[str, str] = {}
    for axis, raw_axis in taxonomies.items():
        if not isinstance(axis, str) or not axis:
            raise AnalysisError("invalid_role_rules", "taxonomy axis names must be nonempty strings")
        axis_document = _mapping(raw_axis, f"taxonomy {axis}")
        default = axis_document.get("default")
        if not isinstance(default, str) or not default:
            raise AnalysisError("invalid_role_rules", f"taxonomy {axis} has no string default")
        defaults[axis] = default
        identifiers: set[str] = set()
        rules: list[tuple[str, str, tuple[re.Pattern[str], ...]]] = []
        for position, raw_rule in enumerate(_list(axis_document.get("rules"), f"taxonomy {axis}.rules")):
            rule = _mapping(raw_rule, f"taxonomy {axis} rule {position}")
            identifier = rule.get("id")
            value = rule.get("value")
            patterns = rule.get("patterns")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise AnalysisError("invalid_role_rules", f"taxonomy {axis} has an invalid/duplicate rule id")
            if not isinstance(value, str) or not value:
                raise AnalysisError("invalid_role_rules", f"taxonomy {axis}.{identifier} has no value")
            raw_patterns = _list(patterns, f"taxonomy {axis}.{identifier}.patterns")
            if not raw_patterns or not all(isinstance(pattern, str) for pattern in raw_patterns):
                raise AnalysisError("invalid_role_rules", f"taxonomy {axis}.{identifier} has invalid patterns")
            try:
                regexes = tuple(re.compile(pattern) for pattern in raw_patterns)
            except re.error as exc:
                raise AnalysisError(
                    "invalid_role_rules", f"taxonomy {axis}.{identifier} regex is invalid: {exc}",
                ) from exc
            identifiers.add(identifier)
            rules.append((identifier, value, regexes))
        compiled[axis] = tuple(rules)
    return RoleRules(
        document=root,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        compiled=compiled,
        defaults=defaults,
    )


def normalize_tensor_name(name: str) -> str:
    normalized = name.casefold().replace("/", ".").replace(":", ".")
    normalized = re.sub(r"\.+", ".", normalized)
    return normalized.strip(".")


def classify_tensor_role(name: str, rules: RoleRules) -> dict[str, Any]:
    normalized = normalize_tensor_name(name)
    labels: dict[str, str] = {}
    matched_rule_ids: dict[str, str | None] = {}
    for axis, axis_rules in rules.compiled.items():
        labels[axis] = rules.defaults[axis]
        matched_rule_ids[axis] = None
        for identifier, value, patterns in axis_rules:
            if any(pattern.search(normalized) is not None for pattern in patterns):
                labels[axis] = value
                matched_rule_ids[axis] = identifier
                break
    return {
        "normalized_name": normalized,
        "labels": labels,
        "matched_rule_ids": matched_rule_ids,
        "orthogonal_role_key": "|".join(f"{axis}={labels[axis]}" for axis in sorted(labels)),
        "taxonomy_schema": {
            "id": ROLE_RULES_SCHEMA_ID,
            "version": ROLE_RULES_SCHEMA_VERSION,
        },
        "taxonomy_file_sha256": rules.file_sha256,
        "interpretation": "deterministic_name_heuristic_not_architecture_ground_truth",
    }


def _operator_arity_is_valid(operator: str, child_count: int) -> bool:
    if operator in TERMINAL_OPERATORS:
        return child_count == 0
    if operator in UNARY_OPERATORS:
        return child_count == 1
    if operator in BINARY_OPERATORS:
        return child_count == 2
    if operator in TERNARY_OPERATORS:
        return child_count == 3
    if operator == "bit_plane":
        return 1 <= child_count <= 32
    if operator == "byte_plane":
        return 1 <= child_count <= 4
    return False


def inspect_program_tree(tree: Any, label: str = "program_tree") -> ProgramInfo:
    preorder: list[dict[str, Any]] = []

    def visit(raw_node: Any, path: tuple[int, ...], depth: int) -> tuple[
        dict[str, Any], str, str, int, int, int, list[str], list[str]
    ]:
        node = _mapping(raw_node, f"{label}{list(path)}")
        operator = node.get("op")
        if not isinstance(operator, str) or operator not in KNOWN_OPERATORS:
            raise AnalysisError(
                "unsupported_operator", f"{label}{list(path)} has unsupported operator {operator!r}",
            )
        params = node.get("params_u32")
        if not _is_int(params) or not 0 <= params <= 0xFFFF_FFFF:
            raise AnalysisError(
                "invalid_program_tree", f"{label}{list(path)} params_u32 must be one u32",
            )
        terminal = node.get("terminal")
        if not isinstance(terminal, bool) or terminal is not (operator in TERMINAL_OPERATORS):
            raise AnalysisError(
                "invalid_program_tree", f"{label}{list(path)} terminal flag disagrees with operator",
            )
        children = _list(node.get("children"), f"{label}{list(path)}.children")
        if not _operator_arity_is_valid(operator, len(children)):
            raise AnalysisError(
                "invalid_program_tree",
                f"{label}{list(path)} operator {operator} has invalid arity {len(children)}",
            )
        preorder_index = len(preorder)
        placeholder: dict[str, Any] = {
            "preorder_index": preorder_index,
            "path": list(path),
            "depth_zero_based": depth,
            "op": operator,
            "params_u32": params,
            "is_terminal": terminal,
            "child_count": len(children),
        }
        preorder.append(placeholder)
        canonical_children: list[dict[str, Any]] = []
        structure_children: list[str] = []
        parameter_children: list[str] = []
        node_count = 1
        node_depth = 1
        transform_depth = 0 if terminal else 1
        operators = [operator]
        terminals = [operator] if terminal else []
        for child_index, child in enumerate(children):
            (canonical_child, structure_child, parameter_child, child_nodes,
             child_depth, child_transform_depth, child_ops, child_terms) = visit(
                child, (*path, child_index), depth + 1,
            )
            canonical_children.append(canonical_child)
            structure_children.append(structure_child)
            parameter_children.append(parameter_child)
            node_count += child_nodes
            node_depth = max(node_depth, child_depth + 1)
            if not terminal:
                transform_depth = max(transform_depth, child_transform_depth + 1)
            operators.extend(child_ops)
            terminals.extend(child_terms)
        structure = operator + ("(" + ",".join(structure_children) + ")" if children else "")
        parameter = (
            f"{operator}[{params}]"
            + ("(" + ",".join(parameter_children) + ")" if children else "")
        )
        canonical = {
            "op": operator,
            "params_u32": params,
            "terminal": terminal,
            "children": canonical_children,
        }
        placeholder.update({
            "subtree_structure_signature": structure,
            "subtree_parameter_signature": parameter,
        })
        # Hash the complete canonical subtree.  This assignment follows child
        # traversal, while preorder_index/path remain fixed at entry.
        placeholder["subtree_structure_sha256"] = _canonical_sha256(
            {"signature": structure}
        )
        placeholder["subtree_parameter_sha256"] = _canonical_sha256(
            {"signature": parameter}
        )
        return (
            canonical, structure, parameter, node_count, node_depth,
            transform_depth, operators, terminals,
        )

    (canonical, structure, parameter, node_count, node_depth, transform_depth,
     operators, terminals) = visit(tree, (), 0)
    return ProgramInfo(
        canonical_tree=canonical,
        structure_signature=structure,
        parameter_signature=parameter,
        structure_sha256=_canonical_sha256({"signature": structure}),
        parameter_sha256=_canonical_sha256({"signature": parameter}),
        rendered_program=structure,
        node_count=node_count,
        node_depth=node_depth,
        transform_depth=transform_depth,
        terminal_count=len(terminals),
        operators_preorder=tuple(operators),
        terminal_codecs_preorder=tuple(terminals),
        nodes=tuple(preorder),
    )


def _validate_program_fields(record: Mapping[str, Any], info: ProgramInfo, label: str) -> None:
    expected = {
        "program": info.rendered_program,
        "root_operator": info.canonical_tree["op"],
        "program_nodes": info.node_count,
        "program_depth": info.node_depth,
        "program_node_depth": info.node_depth,
        "program_transform_depth": info.transform_depth,
        "terminal_count": info.terminal_count,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise AnalysisError(
                "program_metadata_mismatch",
                f"{label}.{field}={record.get(field)!r}, expected {value!r}",
            )


def _program_sequence_sha256(lengths: Sequence[int], hashes: Sequence[str]) -> str:
    if len(lengths) != len(hashes):
        raise AnalysisError("program_evidence_mismatch", "program evidence arrays disagree")
    digest = hashlib.sha256()
    digest.update(PROGRAM_SEQUENCE_SPEC_ID.encode("ascii") + b"\x00")
    digest.update(struct.pack("<Q", len(lengths)))
    for index, (length, block_hash) in enumerate(zip(lengths, hashes, strict=True)):
        if not _is_int(length) or not 0 < length <= 0xFFFF_FFFF:
            raise AnalysisError("program_evidence_mismatch", f"block {index} bytecode length is invalid")
        _sha256(block_hash, f"block {index} bytecode SHA-256")
        digest.update(struct.pack("<I", length))
        digest.update(bytes.fromhex(block_hash))
    return digest.hexdigest()


def _dtype_family(dtype: str) -> str:
    if dtype in FLOAT_DTYPES:
        return "floating_point"
    if dtype.startswith("I"):
        return "signed_integer"
    return "unsigned_integer"


def _shape_class(shape: Sequence[int]) -> str:
    if not shape:
        return "scalar"
    if len(shape) == 1:
        return "vector"
    if len(shape) == 2:
        return "matrix"
    return "rank_3_plus"


def _validate_detail_reference(
    reference_value: Any,
    *,
    configuration_id: str,
    label: str,
) -> Mapping[str, Any]:
    reference = _mapping(reference_value, label)
    if (
        reference.get("scope") != "same_checkpoint_configuration"
        or reference.get("configuration_id") != configuration_id
        or reference.get("phase") not in ("warmup", "measured")
        or not _is_int(reference.get("run_index"))
        or reference.get("run_index") < 0
        or reference.get("report_field") != "bench_report"
        or reference.get("detail_fields") != list(BENCH_DETAIL_FIELDS)
    ):
        raise AnalysisError(
            "invalid_canonical_reference",
            f"{label} is not a checkpoint-local tensors/blocks reference",
        )
    _sha256(reference.get("semantic_sha256"), f"{label}.semantic_sha256")
    return reference


def _resolve_configuration_reports(
    configuration: Mapping[str, Any],
) -> tuple[ResolvedTechnicalReport, tuple[ResolvedTechnicalReport, ...]]:
    """Resolve and independently hash every schema-2 technical report."""

    configuration_id = configuration.get("id")
    if not isinstance(configuration_id, str) or not configuration_id:
        raise AnalysisError("invalid_configuration", "configuration id must be nonempty")
    consistency = _mapping(
        configuration.get("bench_report_detail_consistency"),
        f"configuration {configuration_id}.bench_report_detail_consistency",
    )
    if (
        consistency.get("status_code") != "consistent"
        or consistency.get("fingerprint_spec_id") != BENCH_DETAIL_FINGERPRINT_SPEC_ID
        or consistency.get("hash") != "sha256"
        or consistency.get("excluded_json_paths")
        != list(BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS)
        or consistency.get("detail_fields") != list(BENCH_DETAIL_FIELDS)
        or consistency.get("mismatching_reports") != 0
        or consistency.get("failed_or_incomplete_reports_retained") != 0
        or consistency.get("reports_unavailable") != 0
    ):
        raise AnalysisError(
            "invalid_detail_consistency",
            f"configuration {configuration_id!r} does not have complete, consistent canonical detail",
        )
    canonical_reference = _validate_detail_reference(
        consistency.get("canonical_reference"),
        configuration_id=configuration_id,
        label=f"configuration {configuration_id}.canonical_reference",
    )

    technical: list[tuple[Mapping[str, Any], str, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for collection_name, expected_phase in (("warmups", "warmup"), ("runs", "measured")):
        collection = _list(
            configuration.get(collection_name),
            f"configuration {configuration_id}.{collection_name}",
        )
        for position, run_value in enumerate(collection):
            run = _mapping(run_value, f"configuration {configuration_id}.{collection_name}[{position}]")
            if run.get("phase") != expected_phase:
                raise AnalysisError(
                    "invalid_run",
                    f"configuration {configuration_id}.{collection_name}[{position}] has wrong phase",
                )
            index = _nonnegative_int(run.get("index"), f"{expected_phase} run.index")
            key = (expected_phase, index)
            if key in seen_keys:
                raise AnalysisError(
                    "duplicate_repetition",
                    f"configuration {configuration_id!r} repeats technical run {key}",
                )
            seen_keys.add(key)
            technical.append((run, expected_phase, index))
    measured_count = sum(phase == "measured" for _, phase, _ in technical)
    if measured_count == 0:
        raise AnalysisError("no_measured_runs", f"configuration {configuration_id!r} has no measured runs")
    if (
        consistency.get("successful_reports") != len(technical)
        or consistency.get("matching_reports_including_canonical") != len(technical)
    ):
        raise AnalysisError(
            "invalid_detail_consistency",
            f"configuration {configuration_id!r} detail counters disagree with technical runs",
        )

    canonical_key = (
        str(canonical_reference["phase"]), int(canonical_reference["run_index"]),
    )
    canonical_matches = [
        item for item in technical if (item[1], item[2]) == canonical_key
    ]
    if len(canonical_matches) != 1:
        raise AnalysisError(
            "dangling_canonical_reference",
            f"configuration {configuration_id!r} canonical reference resolves to {len(canonical_matches)} runs",
        )
    canonical_run = canonical_matches[0][0]
    canonical_report = _mapping(canonical_run.get("bench_report"), "canonical bench report")
    if not all(isinstance(canonical_report.get(field), list) for field in BENCH_DETAIL_FIELDS):
        raise AnalysisError(
            "canonical_detail_not_inline", "canonical report does not inline tensors and blocks",
        )

    inline_sources = []
    for run, phase, index in technical:
        report = _mapping(run.get("bench_report"), f"{phase} run {index}.bench_report")
        present = [field in report for field in BENCH_DETAIL_FIELDS]
        if any(present) and not all(present):
            raise AnalysisError(
                "partial_inline_detail",
                f"{phase} run {index} contains only part of tensors/blocks detail",
            )
        if all(isinstance(report.get(field), list) for field in BENCH_DETAIL_FIELDS):
            inline_sources.append((phase, index))
    if inline_sources != [canonical_key]:
        raise AnalysisError(
            "duplicate_canonical_detail",
            f"configuration {configuration_id!r} inline detail sources are {inline_sources}, expected {[canonical_key]}",
        )

    resolved: list[ResolvedTechnicalReport] = []
    canonical_semantic_sha = _sha256(
        canonical_reference.get("semantic_sha256"), "canonical reference semantic SHA-256",
    )
    for run, phase, index in technical:
        is_canonical = run is canonical_run
        report = _mapping(run.get("bench_report"), f"{phase} run {index}.bench_report")
        detail = _mapping(run.get("bench_report_detail"), f"{phase} run {index}.bench_report_detail")
        if (
            detail.get("fingerprint_spec_id") != BENCH_DETAIL_FINGERPRINT_SPEC_ID
            or detail.get("hash") != "sha256"
            or detail.get("excluded_json_paths")
            != list(BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS)
            or detail.get("detail_fields") != list(BENCH_DETAIL_FIELDS)
            or detail.get("canonical_reference") != canonical_reference
            or detail.get("matches_canonical") is not True
        ):
            raise AnalysisError(
                "invalid_detail_reference",
                f"{phase} run {index} has malformed canonical-detail metadata",
            )
        if is_canonical:
            if (
                detail.get("status_code") != "inline_canonical"
                or detail.get("storage") != "inline_canonical"
                or detail.get("detail_fields_inline") is not True
                or detail.get("eligible_for_dsl_analysis") is not True
            ):
                raise AnalysisError(
                    "invalid_canonical_detail", "canonical run metadata does not identify inline canonical detail",
                )
            materialized = dict(report)
        else:
            if (
                detail.get("status_code") != "canonical_reference"
                or detail.get("storage") != "checkpoint_local_canonical_reference"
                or detail.get("detail_fields_inline") is not False
                or detail.get("eligible_for_dsl_analysis") is not False
                or any(field in report for field in BENCH_DETAIL_FIELDS)
            ):
                raise AnalysisError(
                    "invalid_detail_reference",
                    f"{phase} run {index} is not a strict compact canonical reference",
                )
            materialized = dict(report)
            for field in BENCH_DETAIL_FIELDS:
                materialized[field] = canonical_report[field]
        expected_sha = _sha256(
            detail.get("semantic_sha256"), f"{phase} run {index} semantic SHA-256",
        )
        actual_sha = _bench_detail_semantic_sha256(materialized)
        if expected_sha != actual_sha or expected_sha != canonical_semantic_sha:
            raise AnalysisError(
                "semantic_fingerprint_mismatch",
                f"{phase} run {index} materialized semantic SHA-256 is inconsistent",
            )
        resolved.append(ResolvedTechnicalReport(
            run=run,
            phase=phase,
            index=index,
            materialized_report=materialized,
            semantic_sha256=actual_sha,
            is_canonical=is_canonical,
        ))
    canonical_resolved = [item for item in resolved if item.is_canonical]
    if len(canonical_resolved) != 1:
        raise AnalysisError("duplicate_canonical_detail", "exactly one canonical report is required")
    return canonical_resolved[0], tuple(resolved)


def _validate_bench_report(
    report_value: Any,
    *,
    source: Mapping[str, Any],
    outer_configuration: Mapping[str, Any],
    archive_size: int,
) -> ValidatedBench:
    report = _mapping(report_value, "bench_report")
    if report.get("schema") != BENCH_SCHEMA_VERSION or report.get("kind") != BENCH_KIND:
        raise AnalysisError(
            "unsupported_bench_schema",
            "only brevis.bench-report schema 4 is accepted; future/older schemas are excluded",
        )
    source_size = _nonnegative_int(source.get("size_bytes"), "system source.size_bytes")
    source_sha = _sha256(source.get("sha256"), "system source.sha256")
    if report.get("input_size_bytes") != source_size or report.get("input_sha256") != source_sha:
        raise AnalysisError("source_mismatch", "bench report source size/SHA-256 differs from system source")

    config_id = outer_configuration.get("id")
    spec = _mapping(outer_configuration.get("spec"), f"configuration {config_id}.spec")
    if spec.get("id") != config_id:
        raise AnalysisError("configuration_mismatch", "outer configuration id and spec.id disagree")
    plan = spec.get("plan")
    prior_policy = spec.get("prior_policy")
    if plan not in ("fixed", "search") or prior_policy not in ("none", "canonical"):
        raise AnalysisError(
            "configuration_mismatch", "outer configuration has an invalid plan/prior policy",
        )
    effective_search = _mapping(
        outer_configuration.get("effective_search_configuration"),
        f"configuration {config_id}.effective_search_configuration",
    )
    if report.get("search") != effective_search:
        raise AnalysisError(
            "configuration_mismatch",
            "bench search configuration differs from the outer configuration probe",
        )
    effective_raw_only = effective_search.get("enabled_ops") == ["raw"]
    declared_raw_only = spec.get("require_raw_only")
    if declared_raw_only is not None and not isinstance(declared_raw_only, bool):
        raise AnalysisError(
            "configuration_mismatch", "outer configuration require_raw_only must be boolean",
        )
    if (
        isinstance(declared_raw_only, bool)
        and declared_raw_only is not effective_raw_only
    ):
        raise AnalysisError(
            "configuration_mismatch",
            "declared raw-only contract disagrees with the effective operator grammar",
        )
    expected_mode = "fixed" if plan == "fixed" else "phog" if prior_policy == "canonical" else "uniform"
    if report.get("mode") != expected_mode:
        raise AnalysisError(
            "configuration_mismatch",
            f"bench mode {report.get('mode')!r} does not match outer configuration {config_id!r}",
        )
    if effective_raw_only:
        search = _mapping(report.get("search"), "bench search")
        if search.get("enabled_ops") != ["raw"]:
            raise AnalysisError("configuration_mismatch", "raw-only contract does not have a raw-only grammar")
    requested_threads = _positive_int(report.get("requested_threads"), "bench requested_threads")
    if _positive_int(report.get("threads"), "bench threads") != requested_threads:
        raise AnalysisError("configuration_mismatch", "bench requested/effective threads disagree")
    for field in ("planning_workers_used", "encoding_workers_used"):
        workers = _nonnegative_int(report.get(field), f"bench {field}")
        if workers > requested_threads:
            raise AnalysisError("configuration_mismatch", f"bench {field} exceeds requested threads")
    _positive_int(report.get("target_block_bytes"), "bench target_block_bytes")
    if report.get("search_options_applied") is not (plan == "search"):
        raise AnalysisError(
            "configuration_mismatch", "bench search_options_applied disagrees with outer plan",
        )

    prior = _mapping(report.get("prior"), "bench prior")
    guidance_expected = prior_policy == "canonical"
    for field in ("supplied", "loaded", "applied", "guidance_active", "nonempty"):
        if prior.get(field) is not guidance_expected:
            raise AnalysisError("configuration_mismatch", f"bench prior.{field} disagrees with outer config")
    prior_sha = prior.get("sha256")
    if guidance_expected:
        _sha256(prior_sha, "bench prior.sha256")
    elif prior_sha is not None:
        raise AnalysisError("configuration_mismatch", "unguided report has a prior SHA-256")
    context_counts = _list(
        prior.get("context_counts_by_backoff_level"),
        "bench prior.context_counts_by_backoff_level",
    )
    if len(context_counts) != 3 or any(not _is_int(value) or value < 0 for value in context_counts):
        raise AnalysisError("configuration_mismatch", "bench prior context counts are invalid")

    tensors = _list(report.get("tensors"), "bench tensors")
    blocks = _list(report.get("blocks"), "bench blocks")
    top_fields = (
        "input_size_bytes", "raw_bytes", "tensor_data_bytes", "safetensors_prefix_bytes",
        "encoded_bytes_without_frame_headers",
        "block_frame_bytes_excluding_container_header_footer",
        "container_header_bytes", "container_footer_bytes", "projected_archive_bytes",
    )
    top = {field: _nonnegative_int(report.get(field), f"bench {field}") for field in top_fields}
    if top["raw_bytes"] != top["tensor_data_bytes"]:
        raise AnalysisError("accounting_mismatch", "bench raw_bytes and tensor_data_bytes disagree")
    if top["safetensors_prefix_bytes"] + top["tensor_data_bytes"] != top["input_size_bytes"]:
        raise AnalysisError("accounting_mismatch", "bench input prefix/data accounting is inconsistent")
    if (
        top["container_header_bytes"]
        + top["block_frame_bytes_excluding_container_header_footer"]
        + top["container_footer_bytes"]
        != top["projected_archive_bytes"]
    ):
        raise AnalysisError("accounting_mismatch", "bench projected archive accounting is inconsistent")
    if top["projected_archive_bytes"] != archive_size:
        raise AnalysisError("projection_mismatch", "bench projection differs from actual archive size")

    block_programs: list[ProgramInfo] = []
    bytecode_lengths: list[int] = []
    bytecode_hashes: list[str] = []
    block_raw = block_encoded = block_framed = 0
    normalized_blocks: list[dict[str, Any]] = []
    for position, raw_block in enumerate(blocks):
        block = _mapping(raw_block, f"bench blocks[{position}]")
        if _nonnegative_int(block.get("index"), f"blocks[{position}].index") != position:
            raise AnalysisError("block_index_mismatch", "block indices must be contiguous")
        tensor_index = _nonnegative_int(block.get("tensor_index"), f"blocks[{position}].tensor_index")
        if tensor_index >= len(tensors):
            raise AnalysisError("block_tensor_mismatch", "block references an unknown tensor")
        element_offset = _nonnegative_int(block.get("element_offset"), f"blocks[{position}].element_offset")
        element_count = _nonnegative_int(block.get("element_count"), f"blocks[{position}].element_count")
        raw_bytes = _nonnegative_int(block.get("raw_bytes"), f"blocks[{position}].raw_bytes")
        encoded = _nonnegative_int(
            block.get("encoded_bytes_without_frame_headers"),
            f"blocks[{position}].encoded_bytes_without_frame_headers",
        )
        bytecode = _positive_int(block.get("program_bytecode_bytes"), f"blocks[{position}].program_bytecode_bytes")
        bytecode_sha = _sha256(block.get("program_bytecode_sha256"), f"blocks[{position}].program_bytecode_sha256")
        payload = _nonnegative_int(
            block.get("packed_terminal_payload_bytes"),
            f"blocks[{position}].packed_terminal_payload_bytes",
        )
        frame_header = _nonnegative_int(block.get("frame_header_bytes"), f"blocks[{position}].frame_header_bytes")
        framed = _nonnegative_int(block.get("framed_bytes"), f"blocks[{position}].framed_bytes")
        if encoded != bytecode + payload or frame_header != bytecode + 12 or framed != encoded + 12:
            raise AnalysisError("accounting_mismatch", f"block {position} frame accounting is inconsistent")
        program = inspect_program_tree(block.get("program_tree"), f"blocks[{position}].program_tree")
        _validate_program_fields(block, program, f"blocks[{position}]")
        raw_classification = block.get("raw_classification")
        if raw_classification not in (None, "planned_raw", "fallback_raw"):
            raise AnalysisError("raw_classification_mismatch", f"block {position} has invalid raw classification")
        if (program.canonical_tree["op"] == "raw") is (raw_classification is None):
            raise AnalysisError("raw_classification_mismatch", f"block {position} raw classification disagrees with root")
        block_programs.append(program)
        bytecode_lengths.append(bytecode)
        bytecode_hashes.append(bytecode_sha)
        block_raw += raw_bytes
        block_encoded += encoded
        block_framed += framed
        normalized_blocks.append({
            "index": position,
            "tensor_index": tensor_index,
            "element_offset": element_offset,
            "element_count": element_count,
            "raw_bytes": raw_bytes,
            "encoded_bytes_without_frame_headers": encoded,
            "program_bytecode_bytes": bytecode,
            "program_bytecode_sha256": bytecode_sha,
            "packed_terminal_payload_bytes": payload,
            "frame_header_bytes": frame_header,
            "framed_bytes": framed,
            "raw_classification": raw_classification,
        })

    tensor_programs: list[ProgramInfo | None] = []
    covered_blocks: list[int] = []
    tensor_intervals: list[tuple[int, int, str]] = []
    tensor_raw = 0
    names: set[str] = set()
    for position, raw_tensor in enumerate(tensors):
        tensor = _mapping(raw_tensor, f"bench tensors[{position}]")
        if _nonnegative_int(tensor.get("index"), f"tensors[{position}].index") != position:
            raise AnalysisError("tensor_index_mismatch", "tensor indices must be contiguous")
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        if not isinstance(name, str) or not name or name in names:
            raise AnalysisError("invalid_tensor", "tensor names must be nonempty and unique")
        names.add(name)
        if dtype not in DTYPE_BYTES:
            raise AnalysisError("unsupported_dtype", f"tensor {name!r} has unsupported dtype {dtype!r}")
        shape = _list(tensor.get("shape"), f"tensor {name}.shape")
        if any(not _is_int(dimension) or dimension < 0 for dimension in shape):
            raise AnalysisError("invalid_tensor", f"tensor {name!r} shape is invalid")
        numel = _nonnegative_int(tensor.get("numel"), f"tensor {name}.numel")
        expected_numel = math.prod(shape)
        if numel != expected_numel:
            raise AnalysisError("invalid_tensor", f"tensor {name!r} numel disagrees with shape")
        raw_bytes = _nonnegative_int(tensor.get("raw_bytes"), f"tensor {name}.raw_bytes")
        if raw_bytes != numel * DTYPE_BYTES[dtype]:
            raise AnalysisError("invalid_tensor", f"tensor {name!r} raw bytes disagree with dtype/numel")
        start = _nonnegative_int(tensor.get("file_data_start_byte"), f"tensor {name}.file_data_start_byte")
        end = _nonnegative_int(
            tensor.get("file_data_end_byte_exclusive"), f"tensor {name}.file_data_end_byte_exclusive",
        )
        if end < start or end - start != raw_bytes or end > source_size:
            raise AnalysisError("invalid_tensor", f"tensor {name!r} file byte range is inconsistent")
        if start < top["safetensors_prefix_bytes"]:
            raise AnalysisError("invalid_tensor", f"tensor {name!r} overlaps the safetensors prefix")
        tensor_intervals.append((start, end, name))
        block_start = _nonnegative_int(tensor.get("block_start"), f"tensor {name}.block_start")
        block_count = _nonnegative_int(tensor.get("block_count"), f"tensor {name}.block_count")
        if block_start + block_count > len(blocks):
            raise AnalysisError("block_tensor_mismatch", f"tensor {name!r} block range is invalid")
        selected = normalized_blocks[block_start:block_start + block_count]
        if any(block["tensor_index"] != position for block in selected):
            raise AnalysisError("block_tensor_mismatch", f"tensor {name!r} block range has wrong owner")
        covered_blocks.extend(range(block_start, block_start + block_count))
        if sum(block["raw_bytes"] for block in selected) != raw_bytes:
            raise AnalysisError("accounting_mismatch", f"tensor {name!r} raw bytes disagree with blocks")
        expected_offset = 0
        for block in selected:
            if block["element_offset"] != expected_offset:
                raise AnalysisError("block_tensor_mismatch", f"tensor {name!r} blocks do not cover elements contiguously")
            if block["raw_bytes"] != block["element_count"] * DTYPE_BYTES[dtype]:
                raise AnalysisError("accounting_mismatch", f"tensor {name!r} block bytes disagree with dtype")
            expected_offset += block["element_count"]
        if expected_offset != numel:
            raise AnalysisError("block_tensor_mismatch", f"tensor {name!r} blocks do not cover numel")
        encoded = _nonnegative_int(
            tensor.get("encoded_bytes_without_frame_headers"),
            f"tensor {name}.encoded_bytes_without_frame_headers",
        )
        framed = _nonnegative_int(
            tensor.get("block_frame_bytes_excluding_container_header_footer"),
            f"tensor {name}.block_frame_bytes_excluding_container_header_footer",
        )
        if encoded != sum(block["encoded_bytes_without_frame_headers"] for block in selected):
            raise AnalysisError("accounting_mismatch", f"tensor {name!r} encoded bytes disagree with blocks")
        if framed != sum(block["framed_bytes"] for block in selected):
            raise AnalysisError("accounting_mismatch", f"tensor {name!r} framed bytes disagree with blocks")
        raw_roots = _nonnegative_int(tensor.get("raw_root_blocks"), f"tensor {name}.raw_root_blocks")
        planned_raw = _nonnegative_int(
            tensor.get("planned_raw_root_blocks"), f"tensor {name}.planned_raw_root_blocks",
        )
        fallback_raw = _nonnegative_int(
            tensor.get("fallback_raw_root_blocks"), f"tensor {name}.fallback_raw_root_blocks",
        )
        if raw_roots != planned_raw + fallback_raw or raw_roots > block_count:
            raise AnalysisError("raw_classification_mismatch", f"tensor {name!r} raw counts are inconsistent")
        actual_classifications = Counter(block["raw_classification"] for block in selected)
        if actual_classifications["planned_raw"] != planned_raw or actual_classifications["fallback_raw"] != fallback_raw:
            raise AnalysisError("raw_classification_mismatch", f"tensor {name!r} raw counts disagree with blocks")

        tree = tensor.get("program_tree")
        if tree is None:
            nullable_fields = (
                "program", "root_operator", "program_nodes", "program_depth",
                "program_node_depth", "program_transform_depth", "terminal_count",
            )
            if numel != 0 or block_count != 0 or any(tensor.get(field) is not None for field in nullable_fields):
                raise AnalysisError("invalid_program_tree", f"tensor {name!r} has an invalid null plan")
            program = None
        else:
            program = inspect_program_tree(tree, f"tensors[{position}].program_tree")
            _validate_program_fields(tensor, program, f"tensors[{position}]")
            if block_count == 0:
                raise AnalysisError("invalid_program_tree", f"tensor {name!r} has a plan but no blocks")
            planned_is_raw = program.canonical_tree["op"] == "raw"
            if planned_is_raw != (planned_raw == block_count):
                raise AnalysisError("raw_classification_mismatch", f"tensor {name!r} plan root disagrees with planned raw blocks")
        search_expansions = tensor.get("search_expansions")
        if program is not None:
            _nonnegative_int(search_expansions, f"tensor {name}.search_expansions")
        elif search_expansions is not None:
            raise AnalysisError("invalid_tensor", f"empty tensor {name!r} has search counters")
        search_only_fields = (
            "candidates_realized", "candidates_reranked", "probe_blocks_used",
            "selected_sample_rank_zero_based",
        )
        if plan == "search" and program is not None:
            for field in search_only_fields:
                _nonnegative_int(tensor.get(field), f"tensor {name}.{field}")
        elif any(tensor.get(field) is not None for field in search_only_fields):
            raise AnalysisError(
                "configuration_mismatch",
                f"tensor {name!r} has search-only counters when search is not active",
            )
        tensor_programs.append(program)
        tensor_raw += raw_bytes
    if covered_blocks != list(range(len(blocks))):
        raise AnalysisError("block_tensor_mismatch", "tensor ranges do not cover every block exactly once")
    cursor = top["safetensors_prefix_bytes"]
    for start, end, name in sorted(tensor_intervals):
        if start != cursor:
            raise AnalysisError(
                "invalid_tensor",
                f"tensor byte ranges have a gap/overlap before {name!r}: observed {start}, expected {cursor}",
            )
        cursor = end
    if cursor != source_size:
        raise AnalysisError("invalid_tensor", "tensor byte ranges do not cover the tensor-data region")
    if tensor_raw != top["tensor_data_bytes"] or block_raw != top["raw_bytes"]:
        raise AnalysisError("accounting_mismatch", "tensor/block raw totals disagree with report")
    if block_encoded != top["encoded_bytes_without_frame_headers"]:
        raise AnalysisError("accounting_mismatch", "block encoded total disagrees with report")
    if block_framed != top["block_frame_bytes_excluding_container_header_footer"]:
        raise AnalysisError("accounting_mismatch", "block framed total disagrees with report")

    evidence = _mapping(report.get("program_bytecode_evidence"), "program bytecode evidence")
    expected_sequence = _program_sequence_sha256(bytecode_lengths, bytecode_hashes)
    if (
        evidence.get("version") != 1
        or evidence.get("hash") != "sha256"
        or evidence.get("sequence_spec_id") != PROGRAM_SEQUENCE_SPEC_ID
        or evidence.get("block_count") != len(blocks)
        or evidence.get("sequence_sha256") != expected_sequence
    ):
        raise AnalysisError("program_evidence_mismatch", "bench aggregate program evidence is inconsistent")

    return ValidatedBench(
        report=report,
        tensor_programs=tuple(tensor_programs),
        block_programs=tuple(block_programs),
    )


def _validate_run(
    run_value: Any,
    *,
    source: Mapping[str, Any],
    configuration: Mapping[str, Any],
    materialized_report: Mapping[str, Any],
    validated_semantics: ValidatedBench | None = None,
) -> ValidatedBench:
    run = _mapping(run_value, "measured run")
    phase = run.get("phase")
    if phase not in ("warmup", "measured"):
        raise AnalysisError("invalid_run", "technical run phase must be warmup or measured")
    _nonnegative_int(run.get("index"), "measured run.index")
    if run.get("success") is not True or run.get("failure") is not None:
        raise AnalysisError("unsuccessful_run", "measured run is not successful")
    if run.get("archive_pipeline_success") is not True or run.get("diagnostic_success") is not True:
        raise AnalysisError("unsuccessful_run", "archive or diagnostic stage is not successful")
    verification = _mapping(run.get("verification"), "measured run.verification")
    if verification.get("attempted") is not True or verification.get("bit_exact") is not True:
        raise AnalysisError("not_bit_exact", "measured run lacks successful full-file bit-exact verification")
    if verification.get("error") is not None:
        raise AnalysisError("not_bit_exact", "bit-exact verification carries an error")
    source_sha = _sha256(source.get("sha256"), "system source.sha256")
    if verification.get("source_sha256") != source_sha or verification.get("restored_sha256") != source_sha:
        raise AnalysisError("not_bit_exact", "verification SHA-256 does not match the system source")
    if verification.get("restored_size_bytes") != source.get("size_bytes"):
        raise AnalysisError("not_bit_exact", "restored size does not match the system source")

    archive = _mapping(run.get("archive"), "measured run.archive")
    storage = _mapping(archive.get("storage"), "measured run.archive.storage")
    archive_size = _nonnegative_int(storage.get("logical_size_bytes"), "archive logical_size_bytes")
    archive_sha = _sha256(archive.get("sha256"), "archive SHA-256")
    if archive.get("sha256_after_decode") != archive_sha or archive.get("unchanged_during_decode") is not True:
        raise AnalysisError("archive_mutated", "archive was not stable through decode")

    projection = _mapping(
        run.get("bench_archive_projection_consistency"), "bench/archive projection consistency",
    )
    if (
        projection.get("status_code") != "consistent"
        or projection.get("matches_actual_archive") is not True
        or projection.get("projected_archive_bytes") != archive_size
        or projection.get("actual_archive_bytes") != archive_size
    ):
        raise AnalysisError("projection_mismatch", "bench projection was not verified against actual archive")
    program_consistency = _mapping(
        run.get("bench_archive_program_consistency"), "bench/archive program consistency",
    )
    if (
        program_consistency.get("status_code") != "consistent"
        or program_consistency.get("matches") is not True
        or program_consistency.get("mismatch_block_indices") != []
        or program_consistency.get("report_sequence_sha256")
        != program_consistency.get("archive_sequence_sha256")
    ):
        raise AnalysisError("program_evidence_mismatch", "bench programs do not match archive frames")
    archive_consistency = _mapping(run.get("archive_consistency"), "archive consistency")
    if phase == "measured":
        if archive_consistency.get("status_code") not in ("reference", "consistent"):
            raise AnalysisError("archive_replication_mismatch", "archive is not consistent across repetitions")
        if archive_consistency.get("matches_reference") is not True:
            raise AnalysisError("archive_replication_mismatch", "archive differs from repetition reference")
        if (
            archive_consistency.get("reference_size_bytes") != archive_size
            or archive_consistency.get("reference_sha256") != archive_sha
        ):
            raise AnalysisError("archive_replication_mismatch", "run archive reference fields disagree")
        config_consistency = _mapping(
            configuration.get("measured_archive_consistency"),
            "configuration measured archive consistency",
        )
        if (
            config_consistency.get("reference_size_bytes") != archive_size
            or config_consistency.get("reference_sha256") != archive_sha
        ):
            raise AnalysisError("archive_replication_mismatch", "configuration archive reference disagrees")
    elif (
        archive_consistency.get("status_code") != "not_applicable_warmup"
        or archive_consistency.get("matches_reference") is not None
    ):
        raise AnalysisError("archive_replication_mismatch", "warmup archive consistency metadata is invalid")

    validated = validated_semantics
    if validated is None:
        validated = _validate_bench_report(
            materialized_report,
            source=source,
            outer_configuration=configuration,
            archive_size=archive_size,
        )
    else:
        # The exact semantic fingerprint has already been recomputed against
        # the canonical detail.  Projection consistency binds this run's
        # archive size independently.
        if materialized_report.get("projected_archive_bytes") != archive_size:
            raise AnalysisError("projection_mismatch", "materialized report differs from this archive size")
    report_evidence = _mapping(
        materialized_report.get("program_bytecode_evidence"), "bench program evidence",
    )
    archive_evidence = _mapping(archive.get("program_bytecode_evidence"), "archive program evidence")
    report_blocks = _list(materialized_report.get("blocks"), "bench blocks")
    report_lengths = [block.get("program_bytecode_bytes") for block in report_blocks]
    report_hashes = [block.get("program_bytecode_sha256") for block in report_blocks]
    if (
        archive_evidence.get("version") != 1
        or archive_evidence.get("hash") != "sha256"
        or archive_evidence.get("sequence_spec_id") != PROGRAM_SEQUENCE_SPEC_ID
        or report_evidence.get("sequence_sha256") != archive_evidence.get("sequence_sha256")
        or report_evidence.get("block_count") != archive_evidence.get("block_count")
        or archive_evidence.get("archive_size_bytes") != archive_size
        or archive_evidence.get("bytecode_lengths") != report_lengths
        or archive_evidence.get("sha256_by_block") != report_hashes
    ):
        raise AnalysisError("program_evidence_mismatch", "bench/archive program evidence objects disagree")
    return validated


def _error_record(
    *,
    input_label: str,
    code: str,
    message: str,
    configuration_id: str | None = None,
    repetition_index: int | None = None,
) -> dict[str, Any]:
    return {
        **_analysis_header("error"),
        "input_label": input_label,
        "configuration_id": configuration_id,
        "repetition_index": repetition_index,
        "error_code": code,
        "message": message,
        "excluded_from_outputs": True,
        "paper_metrics_eligible": False,
    }


def _node_records(
    *,
    report_id: str,
    owner_id: str,
    tree_scope: str,
    program: ProgramInfo,
    program_raw_bytes: int,
    program_tensor_weight: float,
    variant_weight: float,
) -> list[dict[str, Any]]:
    seen_operators: set[str] = set()
    output: list[dict[str, Any]] = []
    for raw_node in program.nodes:
        node = dict(raw_node)
        operator = str(node["op"])
        first_operator_occurrence = operator not in seen_operators
        seen_operators.add(operator)
        terminal = bool(node["is_terminal"])
        output.append({
            **_analysis_header("node"),
            "report_id": report_id,
            "owner_id": owner_id,
            "tree_scope": tree_scope,
            "node_id": f"{owner_id}:node:{node['preorder_index']}",
            **node,
            "operator_kind": "terminal" if terminal else "transform",
            "program_structure_signature": program.structure_signature,
            "program_parameter_signature": program.parameter_signature,
            "program_structure_sha256": program.structure_sha256,
            "program_parameter_sha256": program.parameter_sha256,
            "first_occurrence_of_operator_in_program": first_operator_occurrence,
            "weights": {
                "semantic_variant_weight": variant_weight,
                "node_occurrence_weight": variant_weight,
                "operator_presence_program_weight": variant_weight if first_operator_occurrence else 0.0,
                "operator_presence_tensor_weight": (
                    program_tensor_weight * variant_weight if first_operator_occurrence else 0.0
                ),
                "operator_presence_raw_byte_weight": (
                    program_raw_bytes * variant_weight if first_operator_occurrence else 0.0
                ),
                "weight_semantics": (
                    "presence weights apply once per distinct operator per program; they are not "
                    "a partition of program bytes and must not be summed across operators"
                ),
            },
            "terminal_payload_attribution": {
                "terminal_present": terminal,
                "terminal_specific_payload_bytes": None,
                "terminal_specific_payload_share": None,
                "status": "unavailable_in_bench_schema_4_aggregate_payload_only",
                "terminal_presence_is_payload_share": False,
            },
        })
    return output


def _raw_terminal_identity(configuration: Mapping[str, Any]) -> str | None:
    """Describe semantic raw-only evidence without trusting a configuration id.

    Early system-schema-2 checkpoints predate the explicit ``require_raw_only``
    spec field.  Their probed effective grammar remains sufficient evidence.
    Newer schema-2 records carry both sources, whose agreement is checked by
    ``_validate_bench_report`` before this function is called.
    """

    spec = _mapping(configuration.get("spec"), "configuration spec")
    effective = _mapping(
        configuration.get("effective_search_configuration"),
        "effective search configuration",
    )
    if effective.get("enabled_ops") != ["raw"]:
        return None
    if spec.get("require_raw_only") is True:
        return "explicit_raw_only_contract_confirmed_by_effective_enabled_ops"
    return "legacy_schema2_effective_enabled_ops_is_raw_only_evidence"


def _emit_group(
    bundle: AnalysisBundle,
    *,
    input_label: str,
    source_document_sha256: str,
    document: Mapping[str, Any],
    configuration: Mapping[str, Any],
    fingerprint: str,
    canonical: ResolvedTechnicalReport,
    technical_reports: Sequence[ResolvedTechnicalReport],
    validated: ValidatedBench,
    rules: RoleRules,
) -> None:
    report = validated.report
    source = _mapping(document.get("source"), "system source")
    run_classification = _mapping(document.get("run_classification"), "run classification")
    config_id = str(configuration["id"])
    repetition_indices = sorted(
        technical.index for technical in technical_reports if technical.phase == "measured"
    )
    technical_repetitions = [
        {
            "phase": technical.phase,
            "index": technical.index,
            "semantic_sha256": technical.semantic_sha256,
            "canonical_detail_source": technical.is_canonical,
        }
        for technical in technical_reports
    ]
    variant_weight = 1.0
    paper_eligible = (
        run_classification.get("run_class") == "formal"
        and run_classification.get("formal_eligible") is True
    )
    report_id = "report-" + _canonical_sha256({
        "source_sha256": source["sha256"],
        "outer_configuration_id": config_id,
        "semantic_fingerprint_sha256": fingerprint,
    })
    tensors = _list(report.get("tensors"), "bench tensors")
    blocks = _list(report.get("blocks"), "bench blocks")
    role_rules_record = {
        "schema": {"id": ROLE_RULES_SCHEMA_ID, "version": ROLE_RULES_SCHEMA_VERSION},
        "file_sha256": rules.file_sha256,
    }
    bundle.reports.append({
        **_analysis_header("report"),
        "report_id": report_id,
        "input_label": input_label,
        "source_document_sha256": source_document_sha256,
        "source": {
            "size_bytes": source["size_bytes"],
            "sha256": source["sha256"],
        },
        "outer_configuration_id": config_id,
        "outer_configuration_spec": configuration.get("spec"),
        "bench_mode": report.get("mode"),
        "raw_terminal_identity": _raw_terminal_identity(configuration),
        "semantic_fingerprint_sha256": fingerprint,
        "semantic_variant_index": 0,
        "semantic_variant_count": 1,
        "canonical_detail_source": {
            "phase": canonical.phase,
            "index": canonical.index,
        },
        "technical_repetitions": technical_repetitions,
        "technical_repetition_indices": repetition_indices,
        "technical_repetition_count": len(technical_reports),
        "warmup_repetition_count": sum(
            technical.phase == "warmup" for technical in technical_reports
        ),
        "measured_repetition_count": len(repetition_indices),
        "technical_repetitions_are_independent_tensor_samples": False,
        "semantic_variant_weight": variant_weight,
        "run_class": run_classification.get("run_class"),
        "formal_eligible": run_classification.get("formal_eligible"),
        "paper_metrics_eligible": paper_eligible,
        "role_rules": role_rules_record,
        "counts": {
            "tensors": len(tensors),
            "blocks": len(blocks),
            "tensor_plan_nodes": sum(
                program.node_count for program in validated.tensor_programs if program is not None
            ),
            "realized_block_nodes": sum(program.node_count for program in validated.block_programs),
        },
        "accounting": {
            field: report.get(field) for field in (
                "input_size_bytes", "raw_bytes", "tensor_data_bytes", "safetensors_prefix_bytes",
                "encoded_bytes_without_frame_headers",
                "block_frame_bytes_excluding_container_header_footer",
                "container_header_bytes", "container_footer_bytes", "projected_archive_bytes",
            )
        },
        "search": report.get("search"),
        "program_bytecode_evidence": report.get("program_bytecode_evidence"),
        "terminal_payload_semantics": (
            "packed_terminal_payload_bytes is available only per complete block program; "
            "terminal-node presence is not a terminal-specific payload share"
        ),
    })

    for tensor_index, (tensor_value, tensor_program) in enumerate(
        zip(tensors, validated.tensor_programs, strict=True)
    ):
        tensor = _mapping(tensor_value, f"tensor {tensor_index}")
        tensor_id = f"{report_id}:tensor:{tensor_index}"
        block_start = int(tensor["block_start"])
        block_count = int(tensor["block_count"])
        tensor_blocks = blocks[block_start:block_start + block_count]
        block_programs = validated.block_programs[block_start:block_start + block_count]
        raw_bytes = int(tensor["raw_bytes"])
        encoded = int(tensor["encoded_bytes_without_frame_headers"])
        framed = int(tensor["block_frame_bytes_excluding_container_header_footer"])
        role = classify_tensor_role(str(tensor["name"]), rules)
        structure_counts = Counter(program.structure_signature for program in block_programs)
        parameter_counts = Counter(program.parameter_signature for program in block_programs)
        terminal_presence = _unique_preorder(
            terminal for program in block_programs for terminal in program.terminal_codecs_preorder
        )
        payload_bytes = sum(int(_mapping(block, "block")["packed_terminal_payload_bytes"])
                            for block in tensor_blocks)
        bundle.tensors.append({
            **_analysis_header("tensor"),
            "report_id": report_id,
            "tensor_id": tensor_id,
            "tensor_index": tensor_index,
            "name": tensor["name"],
            "role": role,
            "dtype": tensor["dtype"],
            "dtype_family": _dtype_family(str(tensor["dtype"])),
            "shape": tensor["shape"],
            "shape_class": _shape_class(tensor["shape"]),
            "rank": len(tensor["shape"]),
            "numel": tensor["numel"],
            "file_data_start_byte": tensor["file_data_start_byte"],
            "file_data_end_byte_exclusive": tensor["file_data_end_byte_exclusive"],
            "block_start": block_start,
            "block_count": block_count,
            "weights": {
                "semantic_variant_weight": variant_weight,
                "tensor_count_weight": variant_weight,
                "raw_byte_weight": raw_bytes * variant_weight,
            },
            "accounting": {
                "raw_bytes": raw_bytes,
                "encoded_bytes_without_frame_headers": encoded,
                "block_frame_bytes_excluding_container_header_footer": framed,
                "compression_ratio_without_frame_headers": _ratio(raw_bytes, encoded),
                "compression_ratio_with_block_frames": _ratio(raw_bytes, framed),
                "saving_fraction_with_block_frames": _saving_fraction(raw_bytes, framed),
                "packed_terminal_payload_bytes_all_terminals": payload_bytes,
                "packed_payload_fraction_of_encoded_program": _ratio(payload_bytes, encoded),
            },
            "search": {
                field: tensor.get(field) for field in (
                    "search_expansions", "candidates_realized", "candidates_reranked",
                    "probe_blocks_used", "selected_sample_rank_zero_based",
                )
            },
            "raw_root_blocks": tensor["raw_root_blocks"],
            "planned_raw_root_blocks": tensor["planned_raw_root_blocks"],
            "fallback_raw_root_blocks": tensor["fallback_raw_root_blocks"],
            "tensor_plan": None if tensor_program is None else {
                "structure_signature": tensor_program.structure_signature,
                "parameter_signature": tensor_program.parameter_signature,
                "structure_sha256": tensor_program.structure_sha256,
                "parameter_sha256": tensor_program.parameter_sha256,
                "node_count": tensor_program.node_count,
                "node_depth": tensor_program.node_depth,
                "transform_depth": tensor_program.transform_depth,
                "terminal_count": tensor_program.terminal_count,
                "operators_preorder": list(tensor_program.operators_preorder),
                "terminal_codecs_preorder": list(tensor_program.terminal_codecs_preorder),
            },
            "realized_block_programs": {
                "structure_signature_counts": dict(sorted(structure_counts.items())),
                "parameter_signature_counts": dict(sorted(parameter_counts.items())),
                "terminal_codecs_present": terminal_presence,
            },
            "terminal_presence_is_payload_share": False,
            "paper_metrics_eligible": paper_eligible,
        })
        if tensor_program is not None:
            bundle.nodes.extend(_node_records(
                report_id=report_id,
                owner_id=tensor_id,
                tree_scope="tensor_plan_template",
                program=tensor_program,
                program_raw_bytes=raw_bytes,
                program_tensor_weight=1.0,
                variant_weight=variant_weight,
            ))

    for block_index, (block_value, program) in enumerate(
        zip(blocks, validated.block_programs, strict=True)
    ):
        block = _mapping(block_value, f"block {block_index}")
        tensor_index = int(block["tensor_index"])
        tensor = _mapping(tensors[tensor_index], f"tensor {tensor_index}")
        block_id = f"{report_id}:block:{block_index}"
        tensor_block_count = int(tensor["block_count"])
        tensor_equal_weight = 1.0 / tensor_block_count
        raw_bytes = int(block["raw_bytes"])
        encoded = int(block["encoded_bytes_without_frame_headers"])
        framed = int(block["framed_bytes"])
        payload = int(block["packed_terminal_payload_bytes"])
        bundle.blocks.append({
            **_analysis_header("block"),
            "report_id": report_id,
            "block_id": block_id,
            "block_index": block_index,
            "tensor_id": f"{report_id}:tensor:{tensor_index}",
            "tensor_index": tensor_index,
            "element_offset": block["element_offset"],
            "element_count": block["element_count"],
            "weights": {
                "semantic_variant_weight": variant_weight,
                "block_count_weight": variant_weight,
                "tensor_equal_weight": tensor_equal_weight * variant_weight,
                "raw_byte_weight": raw_bytes * variant_weight,
            },
            "accounting": {
                "raw_bytes": raw_bytes,
                "encoded_bytes_without_frame_headers": encoded,
                "program_bytecode_bytes": block["program_bytecode_bytes"],
                "packed_terminal_payload_bytes_all_terminals": payload,
                "frame_header_bytes": block["frame_header_bytes"],
                "framed_bytes": framed,
                "compression_ratio_without_frame_headers": _ratio(raw_bytes, encoded),
                "compression_ratio_with_block_frame": _ratio(raw_bytes, framed),
                "saving_fraction_with_block_frame": _saving_fraction(raw_bytes, framed),
                "packed_payload_fraction_of_encoded_program": _ratio(payload, encoded),
            },
            "raw_classification": block["raw_classification"],
            "program": {
                "structure_signature": program.structure_signature,
                "parameter_signature": program.parameter_signature,
                "structure_sha256": program.structure_sha256,
                "parameter_sha256": program.parameter_sha256,
                "node_count": program.node_count,
                "node_depth": program.node_depth,
                "transform_depth": program.transform_depth,
                "terminal_count": program.terminal_count,
                "operators_preorder": list(program.operators_preorder),
                "operators_present": _unique_preorder(program.operators_preorder),
                "terminal_codecs_preorder": list(program.terminal_codecs_preorder),
                "terminal_codecs_present": _unique_preorder(program.terminal_codecs_preorder),
            },
            "program_bytecode_sha256": block["program_bytecode_sha256"],
            "terminal_presence_is_payload_share": False,
            "paper_metrics_eligible": paper_eligible,
        })
        bundle.nodes.extend(_node_records(
            report_id=report_id,
            owner_id=block_id,
            tree_scope="realized_block",
            program=program,
            program_raw_bytes=raw_bytes,
            program_tensor_weight=tensor_equal_weight,
            variant_weight=variant_weight,
        ))


def analyze_documents(
    documents: Iterable[tuple[str, Any]],
    *,
    allow_pilot: bool = False,
    role_rules_path: os.PathLike[str] | str = DEFAULT_ROLE_RULES,
) -> AnalysisBundle:
    """Analyze parsed system documents without treating repetitions as samples."""

    rules = load_role_rules(role_rules_path)
    bundle = AnalysisBundle(role_rules={
        "schema": {"id": ROLE_RULES_SCHEMA_ID, "version": ROLE_RULES_SCHEMA_VERSION},
        "file_sha256": rules.file_sha256,
        "scope": rules.document.get("scope"),
    })
    for input_label, raw_document in documents:
        try:
            document = _mapping(raw_document, "system document")
            if document.get("schema") != {
                "id": SYSTEM_SCHEMA_ID, "version": SYSTEM_SCHEMA_VERSION,
            }:
                raise AnalysisError(
                    "unsupported_system_schema",
                    "only brevis.system-benchmark schema 2 is accepted; future/older schemas are excluded",
                )
            if (
                document.get("status") != "complete"
                or document.get("success") is not True
                or document.get("failure") is not None
            ):
                raise AnalysisError("unsuccessful_system_benchmark", "system benchmark is not complete and successful")
            source = _mapping(document.get("source"), "system source")
            _nonnegative_int(source.get("size_bytes"), "system source.size_bytes")
            _sha256(source.get("sha256"), "system source.sha256")
            classification = _mapping(document.get("run_classification"), "run classification")
            formal = (
                classification.get("run_class") == "formal"
                and classification.get("formal_eligible") is True
            )
            if not formal and not allow_pilot:
                raise AnalysisError(
                    "pilot_excluded",
                    "pilot benchmark excluded by default; pass allow_pilot only for exploratory outputs",
                )
            document_sha = _canonical_sha256(document)
            configurations = _list(document.get("configurations"), "system configurations")
            if not configurations:
                raise AnalysisError("no_configurations", "system benchmark has no configurations")
        except AnalysisError as exc:
            bundle.errors.append(_error_record(
                input_label=input_label, code=exc.code, message=str(exc),
            ))
            continue

        for config_position, raw_configuration in enumerate(configurations):
            config_id: str | None = None
            try:
                configuration = _mapping(raw_configuration, f"configuration {config_position}")
                config_id_value = configuration.get("id")
                if not isinstance(config_id_value, str) or not config_id_value:
                    raise AnalysisError("invalid_configuration", "configuration id must be nonempty")
                config_id = config_id_value
                if configuration.get("status_code") != "ok" or configuration.get("failure") is not None:
                    raise AnalysisError("unsuccessful_configuration", f"configuration {config_id!r} is not successful")
                archive_consistency = _mapping(
                    configuration.get("measured_archive_consistency"),
                    f"configuration {config_id}.measured_archive_consistency",
                )
                if (
                    archive_consistency.get("status_code") != "consistent"
                    or archive_consistency.get("inconsistent_runs") != 0
                ):
                    raise AnalysisError(
                        "archive_replication_mismatch",
                        f"configuration {config_id!r} lacks consistent measured archives",
                    )
                canonical, technical_reports = _resolve_configuration_reports(configuration)
                validated = _validate_run(
                    canonical.run,
                    source=source,
                    configuration=configuration,
                    materialized_report=canonical.materialized_report,
                )
                for technical in technical_reports:
                    if technical.is_canonical:
                        continue
                    _validate_run(
                        technical.run,
                        source=source,
                        configuration=configuration,
                        materialized_report=technical.materialized_report,
                        validated_semantics=validated,
                    )
                _emit_group(
                    bundle,
                    input_label=input_label,
                    source_document_sha256=document_sha,
                    document=document,
                    configuration=configuration,
                    fingerprint=canonical.semantic_sha256,
                    canonical=canonical,
                    technical_reports=technical_reports,
                    validated=validated,
                    rules=rules,
                )
            except AnalysisError as exc:
                bundle.errors.append(_error_record(
                    input_label=input_label,
                    configuration_id=config_id,
                    code=exc.code,
                    message=str(exc),
                ))
                bundle.errors.append(_error_record(
                    input_label=input_label,
                    configuration_id=config_id,
                    code="configuration_excluded_invalid_evidence",
                    message=(
                        f"configuration {config_id!r} excluded in full because canonical or "
                        "technical-repetition evidence failed validation"
                    ),
                ))
    return bundle


def analyze_files(
    paths: Sequence[os.PathLike[str] | str],
    *,
    allow_pilot: bool = False,
    role_rules_path: os.PathLike[str] | str = DEFAULT_ROLE_RULES,
) -> AnalysisBundle:
    parsed: list[tuple[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for raw_path in paths:
        path = pathlib.Path(raw_path)
        try:
            document = json.loads(path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_errors.append(_error_record(
                input_label=str(path),
                code="invalid_json_input",
                message=f"cannot read/parse input: {type(exc).__name__}: {exc}",
            ))
        else:
            parsed.append((str(path), document))
    bundle = analyze_documents(
        parsed, allow_pilot=allow_pilot, role_rules_path=role_rules_path,
    )
    bundle.errors[0:0] = parse_errors
    return bundle


def render_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    lines = [_canonical_json(record) for record in records]
    return "" if not lines else "\n".join(lines) + "\n"


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_outputs(
    bundle: AnalysisBundle,
    output_dir: os.PathLike[str] | str,
    *,
    force: bool = False,
) -> None:
    directory = pathlib.Path(output_dir)
    outputs = {
        "reports.jsonl": render_jsonl(bundle.reports).encode("utf-8"),
        "tensors.jsonl": render_jsonl(bundle.tensors).encode("utf-8"),
        "blocks.jsonl": render_jsonl(bundle.blocks).encode("utf-8"),
        "nodes.jsonl": render_jsonl(bundle.nodes).encode("utf-8"),
        "errors.jsonl": render_jsonl(bundle.errors).encode("utf-8"),
    }
    manifest = {
        "analysis_schema": {"id": ANALYSIS_SCHEMA_ID, "version": ANALYSIS_SCHEMA_VERSION},
        "summary": bundle.summary(),
        "role_rules": bundle.role_rules,
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for name, payload in outputs.items()
        },
        "paper_metrics_policy": (
            "only records with paper_metrics_eligible=true may enter formal paper metrics; "
            "pilot-derived records remain exploratory even when --allow-pilot is used"
        ),
    }
    outputs["manifest.json"] = (json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n").encode("utf-8")
    existing = [directory / name for name in outputs if (directory / name).exists()]
    if existing and not force:
        raise FileExistsError("refusing to overwrite outputs: " + ", ".join(map(str, existing)))
    for name, payload in outputs.items():
        _atomic_write(directory / name, payload)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="schema-2 Brevis system benchmark JSON files")
    parser.add_argument("--output-dir", required=True, help="directory for canonical JSONL outputs")
    parser.add_argument("--role-rules", default=str(DEFAULT_ROLE_RULES))
    parser.add_argument(
        "--allow-pilot", action="store_true",
        help="emit successful pilot-derived exploratory records (never paper-metrics eligible)",
    )
    parser.add_argument("--allow-exclusions", action="store_true", help="return success even if errors.jsonl is nonempty")
    parser.add_argument("--force", action="store_true", help="replace analyzer-owned output files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        bundle = analyze_files(
            args.inputs,
            allow_pilot=args.allow_pilot,
            role_rules_path=args.role_rules,
        )
        write_outputs(bundle, args.output_dir, force=args.force)
    except (AnalysisError, OSError, ValueError) as exc:
        print(f"generated-DSL analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(_canonical_json({"summary": bundle.summary(), "output_dir": args.output_dir}))
    return 0 if args.allow_exclusions or not bundle.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
