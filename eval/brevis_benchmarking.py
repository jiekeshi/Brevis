#!/usr/bin/env python3
"""Auditable repeated benchmarks for the Brevis system pipeline.

This harness keeps four distinct measurements:

* input-local PHOG calibration;
* a ``brevis bench --format json`` diagnostic replay, which reports planning
  and block-encoding time but does not write an archive;
* complete archive compression; and
* complete archive decompression followed by an untimed byte-for-byte scan.

The diagnostic replay must not be presented as a decomposition of the separate
compression invocation.  Complete ``.brv`` files, including all framing and
metadata, are the sole source for storage-effectiveness results.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import benchmarking as common


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BINARY = ROOT / "zig-out" / "bin" / "brevis"
SCHEMA_ID = "brevis.system-benchmark"
SCHEMA_VERSION = 2
DEFAULT_WARMUPS = 1
DEFAULT_REPETITIONS = 6
DEFAULT_SCHEDULE_SEED = 2701
DEFAULT_TIMEOUT_SECONDS = 3600.0
DEFAULT_CALIBRATION_TENSORS = 200
DEFAULT_CALIBRATION_SEED = 0x5EED_B10C
DEFAULT_DISK_RESERVE_FRACTION = 0.30
DEFAULT_TEMP_MULTIPLIER = 3.0
DEFAULT_TEMP_FIXED_BYTES = 256 << 20
DEFAULT_CANONICAL_REPORT_FRACTION_PER_CONFIGURATION = 0.25
DEFAULT_COMPACT_REPORT_BYTES_PER_ITERATION = 128 << 10
DEFAULT_CHECKPOINT_FIXED_BYTES = 64 << 20
CORE_CONFIGURATION_IDS = ("raw-terminal", "fixed", "uniform", "phog")
FROZEN_MAX_REALIZATIONS = 32
FROZEN_MAX_REALIZATIONS_SCOPE = "single_stream_search_only"
FROZEN_TENSOR_SEARCH_USES_MAX_REALIZATIONS = False
FROZEN_TARGET_BLOCK_BYTES = 256 << 10
SEARCH_ARGUMENT_OPTIONS = {
    "--max-expansions": "max_expansions",
    "--max-nodes": "max_nodes",
    "--max-depth": "max_depth",
    "--sample-elems": "sample_elems",
    "--rerank-candidates": "rerank_candidates",
    "--rerank-blocks": "rerank_blocks",
}
EXPECTED_EFFECTIVE_CONFIG_KEYS = (
    "max_expansions",
    "max_nodes",
    "max_depth",
    "sample_elems",
    "rerank_candidates",
    "rerank_blocks",
    "enabled_ops",
    "rerank_enabled",
    "max_realizations",
    "max_realizations_scope",
    "tensor_search_uses_max_realizations",
    "target_block_bytes",
)
PROGRAM_SEQUENCE_SPEC_ID = "brevis.program-bytecode-sequence.v1"
BENCH_DETAIL_FINGERPRINT_SPEC_ID = "brevis.bench-detail-semantics.v1"
BENCH_DETAIL_FIELDS = ("tensors", "blocks")
BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS = (
    "/input",
    "/planning_wall_ms",
    "/encoding_wall_ms",
    "/prior/path",
)


class BenchmarkError(RuntimeError):
    """Invalid benchmark setup or an invariant violation."""


def _parse_search_arguments(search_args: Sequence[str]) -> tuple[dict[str, int], tuple[str, ...]]:
    """Parse the deliberately small search CLI surface without accepting ambiguity."""

    if len(search_args) % 2:
        raise ValueError("search_args must contain option/value pairs")
    settings: dict[str, int] = {}
    disabled: list[str] = []
    for index in range(0, len(search_args), 2):
        option, value = search_args[index:index + 2]
        if option == "--disable-op":
            if value == "raw":
                raise ValueError("raw is mandatory and cannot be disabled")
            if re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
                raise ValueError(f"invalid --disable-op value {value!r}")
            if value in disabled:
                raise ValueError(f"duplicate --disable-op value {value!r}")
            disabled.append(value)
            continue
        key = SEARCH_ARGUMENT_OPTIONS.get(option)
        if key is None:
            raise ValueError(f"unsupported or harness-owned search option {option!r}")
        if key in settings:
            raise ValueError(f"duplicate search option {option!r}")
        if not value.isdecimal() or str(int(value)) != value:
            raise ValueError(f"search option {option!r} requires a canonical decimal integer")
        parsed = int(value)
        if key in {"max_expansions", "max_nodes", "sample_elems"} and parsed < 1:
            raise ValueError(f"search option {option!r} must be positive")
        settings[key] = parsed
    return settings, tuple(disabled)


def _expected_effective_dict(spec: "BrevisSpec") -> dict[str, Any]:
    return dict(spec.expected_effective_config)


@dataclasses.dataclass(frozen=True)
class BrevisSpec:
    """One fully specified Brevis configuration."""

    identifier: str
    plan: str
    prior_policy: str
    search_args: tuple[str, ...] = ()
    notes: str = ""
    require_raw_only: bool = False
    expected_effective_config: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.identifier, str)
            or not re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?", self.identifier)
        ):
            raise ValueError(
                "Brevis configuration identifiers must be 1--128 lowercase "
                "filename-safe ASCII letters, digits, underscores, or hyphens"
            )
        if self.plan not in ("fixed", "search"):
            raise ValueError(f"invalid plan for {self.identifier}: {self.plan}")
        if self.prior_policy not in ("none", "canonical"):
            raise ValueError(
                f"invalid prior policy for {self.identifier}: {self.prior_policy}"
            )
        if self.plan == "fixed" and self.search_args:
            raise ValueError(
                "fixed mode ignores search/operator options; refusing an inert configuration"
            )
        if self.plan == "fixed" and self.prior_policy != "none":
            raise ValueError("fixed mode cannot consume a prior")
        if not isinstance(self.search_args, tuple) or not all(
            isinstance(argument, str) and argument for argument in self.search_args
        ):
            raise ValueError("search_args must be a tuple of nonempty strings")
        parsed_settings, disabled_ops = _parse_search_arguments(self.search_args)
        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")
        if not isinstance(self.require_raw_only, bool):
            raise ValueError("require_raw_only must be a boolean")
        if self.require_raw_only and (
            self.plan != "search" or self.prior_policy != "none"
        ):
            raise ValueError(
                "a required raw-only grammar must use unguided search"
            )
        if not isinstance(self.expected_effective_config, tuple) or not all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in self.expected_effective_config
        ):
            raise ValueError("expected_effective_config must be a tuple of key/value pairs")
        expected = _expected_effective_dict(self)
        if len(expected) != len(self.expected_effective_config):
            raise ValueError("expected_effective_config keys must be unique")
        if expected:
            if tuple(expected) != EXPECTED_EFFECTIVE_CONFIG_KEYS:
                raise ValueError(
                    "expected_effective_config must contain every frozen key in canonical order"
                )
            for key in (
                "max_expansions", "max_nodes", "max_depth", "sample_elems",
                "rerank_candidates", "rerank_blocks", "max_realizations",
                "target_block_bytes",
            ):
                value = expected[key]
                minimum = (
                    1 if key in {
                        "max_expansions", "max_nodes", "sample_elems",
                        "max_realizations", "target_block_bytes",
                    } else 0
                )
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(
                        f"expected effective {key} must be an integer >= {minimum}"
                    )
            enabled_ops = expected["enabled_ops"]
            if (
                not isinstance(enabled_ops, tuple)
                or not enabled_ops
                or not all(isinstance(operator, str) and operator for operator in enabled_ops)
                or len(enabled_ops) != len(set(enabled_ops))
                or "raw" not in enabled_ops
            ):
                raise ValueError("expected effective enabled_ops must be a unique tuple including raw")
            if not isinstance(expected["rerank_enabled"], bool):
                raise ValueError("expected effective rerank_enabled must be boolean")
            if expected["max_realizations"] != FROZEN_MAX_REALIZATIONS:
                raise ValueError("expected effective max_realizations disagrees with the frozen value")
            if expected["max_realizations_scope"] != FROZEN_MAX_REALIZATIONS_SCOPE:
                raise ValueError("expected effective max_realizations_scope is unsupported")
            if (
                expected["tensor_search_uses_max_realizations"]
                is not FROZEN_TENSOR_SEARCH_USES_MAX_REALIZATIONS
            ):
                raise ValueError("expected tensor-search max_realizations semantics are unsupported")
            if expected["target_block_bytes"] != FROZEN_TARGET_BLOCK_BYTES:
                raise ValueError("expected target block size disagrees with the frozen value")
            for key, value in parsed_settings.items():
                if expected[key] != value:
                    raise ValueError(
                        f"expected effective {key} disagrees with its requested CLI value"
                    )
            if expected["rerank_enabled"] is not (
                expected["rerank_candidates"] > 0 and expected["rerank_blocks"] > 0
            ):
                raise ValueError("expected rerank_enabled disagrees with rerank settings")
            if any(operator in expected["enabled_ops"] for operator in disabled_ops):
                raise ValueError("a disabled operator remains in expected enabled_ops")
            if self.require_raw_only and expected["enabled_ops"] != ("raw",):
                raise ValueError("raw-only contract requires expected enabled_ops=('raw',)")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "plan": self.plan,
            "prior_policy": self.prior_policy,
            "search_args": list(self.search_args),
            "notes": self.notes,
            "require_raw_only": self.require_raw_only,
            "expected_effective_config": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in self.expected_effective_config
            },
        }


def _disable_args(operators: Iterable[str]) -> tuple[str, ...]:
    return tuple(part for operator in operators for part in ("--disable-op", operator))


def core_specs(enabled_operators: Sequence[str]) -> tuple[BrevisSpec, ...]:
    """Materialize the four core configurations against the running binary.

    The current Zig CLI has no ``--plan raw``.  The internal raw baseline is
    therefore search with every non-raw production disabled.  It still writes
    and decodes a normal Brevis archive, so its size includes framing.
    """

    operators = tuple(enabled_operators)
    if "raw" not in operators:
        raise BenchmarkError("brevis config did not expose the mandatory raw operator")
    if len(operators) != len(set(operators)):
        raise BenchmarkError("brevis config exposed duplicate operator names")
    non_raw = tuple(operator for operator in operators if operator != "raw")
    return (
        BrevisSpec(
            "raw-terminal",
            "search",
            "none",
            _disable_args(non_raw),
            "Brevis archive constrained to raw terminal frames; includes archive overhead.",
            True,
        ),
        BrevisSpec(
            "fixed",
            "fixed",
            "none",
            notes="Fixed dtype-specific DSL template with its normal raw safety fallback.",
        ),
        BrevisSpec(
            "uniform",
            "search",
            "none",
            notes="Bounded search with the uniform conditional grammar.",
        ),
        BrevisSpec(
            "phog",
            "search",
            "canonical",
            notes="The uniform configuration with only the canonical input-local prior added.",
        ),
    )


def select_specs(
    specs: Sequence[BrevisSpec], identifiers: Iterable[str] | None,
) -> tuple[BrevisSpec, ...]:
    if identifiers is None:
        return tuple(specs)
    by_id = {spec.identifier: spec for spec in specs}
    selected: list[BrevisSpec] = []
    for identifier in identifiers:
        try:
            selected.append(by_id[identifier])
        except KeyError as exc:
            raise ValueError(
                f"unknown Brevis configuration {identifier!r}; choose from: "
                + ", ".join(by_id)
            ) from exc
    if not selected:
        raise ValueError("at least one Brevis configuration must be selected")
    if len({spec.identifier for spec in selected}) != len(selected):
        raise ValueError("selected Brevis configurations must be unique")
    return tuple(selected)


def _validate_explicit_specs(specs: Sequence[BrevisSpec]) -> tuple[BrevisSpec, ...]:
    """Validate an API-supplied configuration sequence before running processes."""

    selected = tuple(specs)
    if not selected:
        raise ValueError("explicit Brevis specs must be nonempty")
    if not all(isinstance(spec, BrevisSpec) for spec in selected):
        raise TypeError("explicit specs must contain only BrevisSpec values")
    identifiers = [spec.identifier for spec in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("explicit Brevis configuration identifiers must be unique")
    for spec in selected:
        if not spec.expected_effective_config:
            raise ValueError(
                "explicit Brevis specs must bind a complete expected_effective_config"
            )
        if spec.identifier == "raw-terminal" and not spec.require_raw_only:
            raise ValueError(
                "the reserved raw-terminal identifier requires an explicit raw-only contract"
            )
    canonical_search_args = {
        spec.search_args for spec in selected if spec.prior_policy == "canonical"
    }
    if len(canonical_search_args) > 1:
        raise BenchmarkError(
            "canonical-prior configurations in one benchmark must have exactly "
            "identical search_args; refusing to share a mismatched prior"
        )
    return selected


def _process_ok(process: Mapping[str, Any]) -> bool:
    return (
        process.get("error") is None
        and not process.get("timed_out")
        and process.get("exit_code") == 0
    )


def _captured_stdout_bytes(process: Mapping[str, Any]) -> bytes:
    captured = process.get("stdout")
    if not isinstance(captured, Mapping) or not isinstance(captured.get("base64"), str):
        raise BenchmarkError("process record has no byte-exact stdout")
    try:
        return base64.b64decode(captured["base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise BenchmarkError(f"process stdout is not valid Base64: {exc}") from exc


def _embed_json_stdout(
    process: dict[str, Any], *, minimum_schema: int, expected_kind: str | None = None,
) -> dict[str, Any]:
    """Parse a JSON report and replace its duplicate captured stdout with a digest.

    The complete parsed object is returned and embedded without field filtering.
    This accepts future report schemas while preserving all of their fields.
    """

    raw = _captured_stdout_bytes(process)
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"invalid JSON report: {exc}") from exc
    if not isinstance(report, dict):
        raise BenchmarkError("JSON report must be an object")
    schema = report.get("schema")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema < minimum_schema:
        raise BenchmarkError(
            f"JSON report schema must be an integer >= {minimum_schema}; observed {schema!r}"
        )
    if expected_kind is not None and report.get("kind") != expected_kind:
        raise BenchmarkError(
            f"JSON report kind must be {expected_kind!r}; observed {report.get('kind')!r}"
        )
    process["stdout"] = {
        "embedded_as_parsed_json": True,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "text": None,
        "base64": None,
    }
    return report


def _bench_detail_semantic_projection(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return the schema-4 report semantics used to identify DSL repetitions.

    Only the two internal wall-clock observations and machine-local input/prior
    paths are excluded.  In particular, the tensor and block arrays, unknown
    future fields, search settings, accounting, program evidence, and every
    other top-level value remain part of the fingerprint.
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
    """Fingerprint all stable schema-4 report semantics in canonical JSON."""

    try:
        encoded = json.dumps(
            _bench_detail_semantic_projection(report),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(
            f"bench report cannot be semantically fingerprinted: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _new_bench_detail_consistency() -> dict[str, Any]:
    return {
        "status_code": "awaiting_successful_report",
        "fingerprint_spec_id": BENCH_DETAIL_FINGERPRINT_SPEC_ID,
        "hash": "sha256",
        "canonical_json_encoding": "utf-8 sorted-key compact JSON; finite values only",
        "excluded_json_paths": list(BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS),
        "detail_fields": list(BENCH_DETAIL_FIELDS),
        "canonical_reference": None,
        "successful_reports": 0,
        "matching_reports_including_canonical": 0,
        "mismatching_reports": 0,
        "failed_or_incomplete_reports_retained": 0,
        "reports_unavailable": 0,
        "analysis_policy": (
            "resolve tensors/blocks from the single checkpoint-local canonical report "
            "once per configuration; warmup/measured diagnostic repetitions are technical "
            "replicates for timing and consistency, not independent tensors"
        ),
    }


def _refresh_bench_detail_consistency_status(consistency: dict[str, Any]) -> None:
    if consistency["mismatching_reports"]:
        consistency["status_code"] = "semantic_mismatch"
    elif consistency["canonical_reference"] is None:
        consistency["status_code"] = "awaiting_successful_report"
    elif (
        consistency["failed_or_incomplete_reports_retained"]
        or consistency["reports_unavailable"]
    ):
        consistency["status_code"] = "incomplete_reports_retained"
    else:
        consistency["status_code"] = "consistent"


def _bench_detail_reference(
    configuration: Mapping[str, Any], run: Mapping[str, Any], semantic_sha256: str,
) -> dict[str, Any]:
    phase = run.get("phase")
    index = run.get("index")
    if phase not in ("warmup", "measured"):
        raise BenchmarkError("bench detail source has an invalid phase")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise BenchmarkError("bench detail source has an invalid repetition index")
    return {
        "scope": "same_checkpoint_configuration",
        "configuration_id": configuration.get("id"),
        "phase": phase,
        "run_index": index,
        "report_field": "bench_report",
        "detail_fields": list(BENCH_DETAIL_FIELDS),
        "semantic_sha256": semantic_sha256,
        "resolution": (
            "find the unique run with this phase/run_index in the same configuration, "
            "then read bench_report.tensors and bench_report.blocks"
        ),
    }


def _find_bench_detail_source(
    configuration: Mapping[str, Any], reference: Mapping[str, Any],
) -> Mapping[str, Any]:
    if reference.get("configuration_id") != configuration.get("id"):
        raise BenchmarkError("bench detail reference targets a different configuration")
    phase = reference.get("phase")
    collection_name = "warmups" if phase == "warmup" else "runs" if phase == "measured" else None
    if collection_name is None:
        raise BenchmarkError("bench detail reference has an invalid phase")
    collection = configuration.get(collection_name)
    if not isinstance(collection, list):
        raise BenchmarkError("bench detail reference collection is unavailable")
    matches = [
        run for run in collection
        if isinstance(run, Mapping)
        and run.get("index") == reference.get("run_index")
    ]
    if len(matches) != 1:
        raise BenchmarkError("bench detail reference does not resolve to exactly one run")
    report = matches[0].get("bench_report")
    if not isinstance(report, Mapping):
        raise BenchmarkError("bench detail reference source has no report")
    if not all(isinstance(report.get(field), list) for field in BENCH_DETAIL_FIELDS):
        raise BenchmarkError("bench detail reference source has no inline tensor/block arrays")
    return report


def _materialize_bench_report_detail(
    configuration: Mapping[str, Any], run: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a compact report using only records in the same checkpoint."""

    report = run.get("bench_report")
    detail = run.get("bench_report_detail")
    if not isinstance(report, Mapping) or not isinstance(detail, Mapping):
        raise BenchmarkError("run has no materializable bench report detail")
    materialized = dict(report)
    inline = all(isinstance(materialized.get(field), list) for field in BENCH_DETAIL_FIELDS)
    if not inline:
        reference = detail.get("canonical_reference")
        if not isinstance(reference, Mapping):
            raise BenchmarkError("compact bench report has no canonical reference")
        canonical = _find_bench_detail_source(configuration, reference)
        for field in BENCH_DETAIL_FIELDS:
            materialized[field] = canonical[field]
    expected = detail.get("semantic_sha256")
    actual = _bench_detail_semantic_sha256(materialized)
    if expected != actual:
        raise BenchmarkError("materialized bench report semantic SHA-256 is inconsistent")
    return materialized


def _apply_bench_detail_compaction(
    configuration: dict[str, Any], run: dict[str, Any],
) -> str | None:
    """Deduplicate one valid diagnostic report, retaining failures verbatim.

    Returns a failure message when an otherwise successful diagnostic differs
    semantically from the configuration's canonical report.
    """

    consistency = configuration["bench_report_detail_consistency"]
    canonical = consistency.get("canonical_reference")
    report = run.get("bench_report")
    metadata: dict[str, Any] = {
        "status_code": "report_unavailable",
        "fingerprint_spec_id": BENCH_DETAIL_FINGERPRINT_SPEC_ID,
        "hash": "sha256",
        "semantic_sha256": None,
        "excluded_json_paths": list(BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS),
        "detail_fields": list(BENCH_DETAIL_FIELDS),
        "detail_fields_inline": False,
        "canonical_reference": canonical,
        "matches_canonical": None,
        "eligible_for_dsl_analysis": False,
    }
    run["bench_report_detail"] = metadata
    if not isinstance(report, Mapping):
        consistency["reports_unavailable"] += 1
        _refresh_bench_detail_consistency_status(consistency)
        return None

    try:
        semantic_sha256 = _bench_detail_semantic_sha256(report)
    except BenchmarkError as exc:
        consistency["failed_or_incomplete_reports_retained"] += 1
        _refresh_bench_detail_consistency_status(consistency)
        metadata.update({
            "status_code": "unfingerprintable_report_retained",
            "detail_fields_inline": all(
                isinstance(report.get(field), list) for field in BENCH_DETAIL_FIELDS
            ),
        })
        return str(exc)

    detail_inline = all(isinstance(report.get(field), list) for field in BENCH_DETAIL_FIELDS)
    metadata.update({
        "semantic_sha256": semantic_sha256,
        "detail_fields_inline": detail_inline,
    })
    complete_success = (
        run.get("archive_pipeline_success") is True
        and run.get("diagnostic_success") is True
        and run.get("failure") is None
        and detail_inline
    )
    if not complete_success:
        consistency["failed_or_incomplete_reports_retained"] += 1
        _refresh_bench_detail_consistency_status(consistency)
        metadata.update({
            "status_code": "failed_or_incomplete_report_retained",
            "storage": "inline_uncompacted" if detail_inline else "detail_fields_unavailable",
        })
        return None

    if canonical is None:
        reference = _bench_detail_reference(configuration, run, semantic_sha256)
        consistency.update({
            "status_code": "consistent",
            "canonical_reference": reference,
            "successful_reports": consistency["successful_reports"] + 1,
            "matching_reports_including_canonical": 1,
        })
        metadata.update({
            "status_code": "inline_canonical",
            "storage": "inline_canonical",
            "canonical_reference": reference,
            "matches_canonical": True,
            "eligible_for_dsl_analysis": True,
        })
        # Earlier failed/incomplete reports may have been checkpointed before a
        # canonical source existed. Their audit metadata can now point at the
        # self-contained canonical detail without claiming a semantic match.
        for collection_name in ("warmups", "runs"):
            collection = configuration.get(collection_name)
            if not isinstance(collection, list):
                continue
            for previous in collection:
                previous_detail = (
                    previous.get("bench_report_detail")
                    if isinstance(previous, Mapping) else None
                )
                if (
                    isinstance(previous_detail, dict)
                    and previous_detail.get("canonical_reference") is None
                ):
                    previous_detail["canonical_reference"] = reference
        _refresh_bench_detail_consistency_status(consistency)
        return None

    if not isinstance(canonical, Mapping):
        raise BenchmarkError("bench detail canonical reference is malformed")
    canonical_sha256 = canonical.get("semantic_sha256")
    if semantic_sha256 == canonical_sha256:
        # Resolve before dropping the duplicate arrays, ensuring that the next
        # atomic checkpoint contains no dangling or external reference.
        _find_bench_detail_source(configuration, canonical)
        for field in BENCH_DETAIL_FIELDS:
            report.pop(field, None)
        consistency["successful_reports"] += 1
        consistency["matching_reports_including_canonical"] += 1
        _refresh_bench_detail_consistency_status(consistency)
        metadata.update({
            "status_code": "canonical_reference",
            "storage": "checkpoint_local_canonical_reference",
            "detail_fields_inline": False,
            "canonical_reference": canonical,
            "matches_canonical": True,
            "eligible_for_dsl_analysis": False,
        })
        # Exercise the resolver and semantic check before checkpointing.
        _materialize_bench_report_detail(configuration, run)
        return None

    consistency["mismatching_reports"] += 1
    _refresh_bench_detail_consistency_status(consistency)
    metadata.update({
        "status_code": "semantic_mismatch_retained",
        "storage": "inline_mismatch",
        "canonical_reference": canonical,
        "matches_canonical": False,
        "eligible_for_dsl_analysis": False,
    })
    return (
        "diagnostic bench report semantics differ from the configuration canonical "
        f"detail ({semantic_sha256} != {canonical_sha256}); full mismatching "
        "tensors/blocks were retained"
    )


def _process_failure(label: str, process: Mapping[str, Any]) -> str | None:
    if process.get("timed_out"):
        return f"{label} timed out after {process.get('timeout_seconds')} seconds"
    if process.get("error") is not None:
        return f"{label} launch/wait failed: {process['error']}"
    if process.get("exit_code") != 0:
        return f"{label} exited with status {process.get('exit_code')}"
    return None


def _safe_label(identifier: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in identifier)


def _run_config(
    binary: pathlib.Path,
    search_args: Sequence[str],
    log_dir: pathlib.Path,
    label: str,
    timeout_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    process = common._run_process(
        [str(binary), "config", *search_args], log_dir, label, timeout_seconds,
    )
    failure = _process_failure("configuration probe", process)
    if failure is not None:
        return process, None, failure
    try:
        raw = _captured_stdout_bytes(process)
        payload = json.loads(raw)
    except (BenchmarkError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return process, None, f"invalid configuration JSON: {exc}"
    if not isinstance(payload, dict):
        return process, None, "configuration JSON must be an object"
    return process, payload, None


def _validate_effective_configuration(
    spec: BrevisSpec, effective: Mapping[str, Any],
) -> None:
    """Compare a config probe with independently frozen, hash-bound expectations."""

    expected = _expected_effective_dict(spec)
    if not expected:
        return
    for key in EXPECTED_EFFECTIVE_CONFIG_KEYS:
        observed = effective.get(key)
        expected_value = expected[key]
        if key == "enabled_ops" and isinstance(observed, list):
            observed = tuple(observed)
        if type(observed) is not type(expected_value) or observed != expected_value:
            raise BenchmarkError(
                f"effective configuration {key} drifted from the frozen expectation: "
                f"expected {expected_value!r}, observed {observed!r}"
            )


def _binary_record(path: pathlib.Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "requested_path": str(path),
        "resolved_path": None,
        "sha256_before_runs": None,
        "sha256_after_runs": None,
        "unchanged_during_benchmark": None,
        "executable": False,
    }
    try:
        resolved = path.resolve(strict=True)
        record["resolved_path"] = str(resolved)
        record["executable"] = resolved.is_file() and os.access(resolved, os.X_OK)
        if resolved.is_file():
            record["sha256_before_runs"] = common._sha256_file(resolved)
    except OSError:
        pass
    return record


def _tool_version(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command), check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (completed.stdout or completed.stderr).splitlines()
    return lines[0].strip() if lines else None


def _gpu_info() -> list[dict[str, Any]] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    query = "name,uuid,driver_version,memory.total"
    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    records = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            records.append({
                "name": fields[0],
                "uuid": fields[1],
                "driver_version": fields[2],
                "memory_total_mib": int(fields[3]) if fields[3].isdigit() else fields[3],
            })
    return records


def _git_tracked_snapshot() -> dict[str, Any]:
    """Fingerprint committed identity plus staged/unstaged tracked changes.

    Untracked outputs are intentionally excluded because a checkpoint written
    inside the repository may itself be untracked.  The formal clean-tree gate
    still checks untracked files before the benchmark starts.
    """

    git = shutil.which("git")
    result: dict[str, Any] = {
        "commit": None,
        "tracked_diff_sha256": None,
        "tracked_dirty": None,
        "status_porcelain_untracked_excluded": None,
    }
    if git is None:
        return result
    try:
        commit = subprocess.run(
            [git, "-C", str(ROOT), "rev-parse", "HEAD"],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        )
        diff = subprocess.run(
            [git, "-C", str(ROOT), "diff", "--binary", "HEAD", "--", "."],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        )
        status = subprocess.run(
            [
                git, "-C", str(ROOT), "status", "--porcelain",
                "--untracked-files=no",
            ],
            check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if commit.returncode == 0:
        result["commit"] = commit.stdout.decode("utf-8", errors="replace").strip()
    if diff.returncode == 0:
        result["tracked_diff_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
    if status.returncode == 0:
        status_text = status.stdout.decode("utf-8", errors="replace").rstrip("\n")
        result["tracked_dirty"] = bool(status_text)
        result["status_porcelain_untracked_excluded"] = status_text
    return result


def _finalize_provenance_checks(
    provenance: dict[str, Any],
    *,
    source_path: pathlib.Path,
    source_sha256: str,
    integrity: Mapping[str, Any],
) -> list[str]:
    """Rehash immutable inputs and benchmark code after all measured work."""

    failures: list[str] = []
    checks: dict[str, Any] = {}

    def check_file(label: str, path: pathlib.Path, expected: str) -> None:
        try:
            actual = common._sha256_file(path)
        except OSError as exc:
            checks[label] = {
                "path": str(path), "expected_sha256": expected,
                "actual_sha256": None, "unchanged": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(f"cannot rehash {label}: {type(exc).__name__}: {exc}")
            return
        unchanged = actual == expected
        checks[label] = {
            "path": str(path), "expected_sha256": expected,
            "actual_sha256": actual, "unchanged": unchanged, "error": None,
        }
        if not unchanged:
            failures.append(f"{label} SHA-256 changed during the benchmark")

    check_file("source", source_path, source_sha256)
    for key in ("harness", "generic_helper"):
        record = provenance[key]
        check_file(key, pathlib.Path(record["path"]), record["sha256"])

    manifest = integrity.get("manifest")
    if isinstance(manifest, Mapping) and manifest.get("path") and manifest.get("actual_sha256"):
        check_file(
            "manifest",
            pathlib.Path(str(manifest["path"])),
            str(manifest["actual_sha256"]),
        )

    start_git = provenance["git_tracked_start"]
    end_git = _git_tracked_snapshot()
    provenance["git_end"] = common._git_provenance()
    same_commit = start_git.get("commit") == end_git.get("commit")
    same_diff = (
        start_git.get("tracked_diff_sha256")
        == end_git.get("tracked_diff_sha256")
    )
    checks["git_tracked_state"] = {
        "start": start_git,
        "end": end_git,
        "same_commit": same_commit,
        "same_tracked_diff": same_diff,
        "unchanged": same_commit and same_diff,
        "untracked_files_excluded_from_end_comparison": True,
    }
    if not same_commit:
        failures.append("Git commit changed during the benchmark")
    if not same_diff:
        failures.append("tracked Git diff changed during the benchmark")

    provenance["end_checks"] = {
        "performed": True,
        "checks": checks,
        "success": not failures,
        "failures": list(failures),
    }
    return failures


def _disk_gate(
    work_path: pathlib.Path,
    *,
    checkpoint_path: pathlib.Path | None,
    source_size: int,
    iterations: int,
    configurations: int,
    keep_artifacts: bool,
    reserve_fraction: float,
    temp_multiplier: float,
    fixed_bytes: int,
) -> dict[str, Any]:
    try:
        scaled_source = source_size * temp_multiplier
        if not math.isfinite(scaled_source):
            raise OverflowError
        one_iteration = math.ceil(scaled_source) + fixed_bytes
    except (OverflowError, ValueError) as exc:
        raise ValueError("temp_multiplier produces an unrepresentable disk estimate") from exc
    work_estimate = (
        one_iteration * max(1, iterations) if keep_artifacts else one_iteration
    )

    work_usage = shutil.disk_usage(work_path)
    work_device = os.stat(work_path).st_dev
    checkpoint_parent = checkpoint_path.parent if checkpoint_path is not None else None
    checkpoint_usage = (
        shutil.disk_usage(checkpoint_parent) if checkpoint_parent is not None else None
    )
    checkpoint_device = (
        os.stat(checkpoint_parent).st_dev if checkpoint_parent is not None else None
    )
    # One full tensor/block report is retained per successful configuration. Each
    # repetition retains its top-level report, process evidence, and a local
    # canonical reference. Atomic replacement temporarily needs old plus new JSON.
    estimated_final_checkpoint = (
        math.ceil(source_size * DEFAULT_CANONICAL_REPORT_FRACTION_PER_CONFIGURATION)
        * max(1, configurations)
        + DEFAULT_COMPACT_REPORT_BYTES_PER_ITERATION * max(1, iterations)
        + DEFAULT_CHECKPOINT_FIXED_BYTES
        if checkpoint_parent is not None else 0
    )
    checkpoint_atomic_peak = estimated_final_checkpoint * 2
    same_filesystem = checkpoint_device is not None and checkpoint_device == work_device

    work_reserve = math.ceil(work_usage.total * reserve_fraction)
    combined_work_estimate = work_estimate + (checkpoint_atomic_peak if same_filesystem else 0)
    work_projected_free = work_usage.free - combined_work_estimate
    work_passes = work_projected_free >= work_reserve
    checkpoint_record: dict[str, Any] | None = None
    checkpoint_passes = True
    if checkpoint_parent is not None and checkpoint_usage is not None:
        checkpoint_reserve = math.ceil(checkpoint_usage.total * reserve_fraction)
        checkpoint_charge = 0 if same_filesystem else checkpoint_atomic_peak
        checkpoint_projected_free = checkpoint_usage.free - checkpoint_charge
        checkpoint_passes = checkpoint_projected_free >= checkpoint_reserve
        checkpoint_record = {
            "path": str(checkpoint_parent),
            "device": checkpoint_device,
            "filesystem_total_bytes": checkpoint_usage.total,
            "filesystem_free_bytes_before": checkpoint_usage.free,
            "estimated_final_json_bytes": estimated_final_checkpoint,
            "estimated_atomic_replace_peak_bytes": checkpoint_atomic_peak,
            "charged_here_bytes": checkpoint_charge,
            "required_free_reserve_bytes": checkpoint_reserve,
            "projected_free_bytes_after_estimate": checkpoint_projected_free,
            "passes": checkpoint_passes,
        }
    passes = work_passes and checkpoint_passes
    return {
        "status_code": "ok" if passes else "skipped_resource",
        "passes": passes,
        "source_size_bytes": source_size,
        "keep_artifacts": keep_artifacts,
        "scheduled_pipeline_iterations": iterations,
        "scheduled_configurations": configurations,
        "reserve_fraction": reserve_fraction,
        "same_work_and_checkpoint_filesystem": same_filesystem,
        "work": {
            "path": str(work_path),
            "device": work_device,
            "filesystem_total_bytes": work_usage.total,
            "filesystem_free_bytes_before": work_usage.free,
            "estimated_peak_or_retained_temporary_bytes": work_estimate,
            "checkpoint_atomic_peak_charged_here_bytes": (
                checkpoint_atomic_peak if same_filesystem else 0
            ),
            "combined_estimated_bytes": combined_work_estimate,
            "estimate_formula": (
                "ceil(source_size_bytes * temp_multiplier) + fixed_bytes; multiplied by "
                "scheduled iterations only when artifacts are retained"
            ),
            "temp_multiplier": temp_multiplier,
            "fixed_bytes": fixed_bytes,
            "required_free_reserve_bytes": work_reserve,
            "projected_free_bytes_after_estimate": work_projected_free,
            "passes": work_passes,
        },
        "checkpoint": checkpoint_record,
        "checkpoint_estimate_policy": {
            "canonical_report_fraction_of_source_per_configuration": (
                DEFAULT_CANONICAL_REPORT_FRACTION_PER_CONFIGURATION
            ),
            "compact_top_level_bytes_per_pipeline_iteration": (
                DEFAULT_COMPACT_REPORT_BYTES_PER_ITERATION
            ),
            "fixed_bytes": DEFAULT_CHECKPOINT_FIXED_BYTES,
            "atomic_replace_copies_at_peak": 2,
            "note": (
                "one successful canonical tensors/blocks report per configuration; "
                "matching technical repetitions retain top-level reports and local references"
            ),
        },
        "failure": None if passes else (
            "estimated work/checkpoint footprint would violate the configured post-run "
            "free-space reserve"
        ),
    }


def _cleanup_directory(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        return {
            "label": label, "path": str(path), "attempted": True,
            "success": True, "already_absent": True, "error": None,
        }
    except OSError as exc:
        return {
            "label": label, "path": str(path), "attempted": True,
            "success": False, "already_absent": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "label": label, "path": str(path), "attempted": True,
        "success": not path.exists(), "already_absent": False,
        "error": None if not path.exists() else "directory still exists after rmtree",
    }


def _program_sequence_sha256(
    bytecode_lengths: Sequence[int], sha256_by_block: Sequence[str],
) -> str:
    if len(bytecode_lengths) != len(sha256_by_block):
        raise BenchmarkError("program evidence length/hash arrays disagree")
    digest = hashlib.sha256()
    digest.update(PROGRAM_SEQUENCE_SPEC_ID.encode("ascii"))
    digest.update(b"\x00")
    digest.update(struct.pack("<Q", len(bytecode_lengths)))
    for index, (length, block_sha256) in enumerate(zip(bytecode_lengths, sha256_by_block)):
        if isinstance(length, bool) or not isinstance(length, int) or not 0 < length <= 0xFFFFFFFF:
            raise BenchmarkError(f"program evidence block {index} has an invalid bytecode length")
        if (
            not isinstance(block_sha256, str)
            or len(block_sha256) != 64
            or block_sha256 != block_sha256.lower()
            or any(character not in "0123456789abcdef" for character in block_sha256)
        ):
            raise BenchmarkError(f"program evidence block {index} has a noncanonical SHA-256")
        try:
            raw_digest = bytes.fromhex(block_sha256)
        except (TypeError, ValueError) as exc:
            raise BenchmarkError(
                f"program evidence block {index} has an invalid SHA-256"
            ) from exc
        if len(raw_digest) != 32:
            raise BenchmarkError(f"program evidence block {index} has an invalid SHA-256")
        digest.update(struct.pack("<I", length))
        digest.update(raw_digest)
    return digest.hexdigest()


def _read_exact(stream: Any, size: int, label: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise BenchmarkError(f"truncated archive while reading {label}")
    return payload


def _archive_program_bytecode_evidence(path: pathlib.Path) -> dict[str, Any]:
    """Scan current-writer frames without decoding or charging timed work."""

    file_size = path.stat().st_size
    if file_size < 20:
        raise BenchmarkError("archive is too short to contain its header and footer")
    lengths: list[int] = []
    hashes: list[str] = []
    with path.open("rb") as archive:
        header = _read_exact(archive, 8, "container header")
        if header != b"BRV\x03\x06\x00\x00\x00":
            raise BenchmarkError(
                "program evidence supports current schema-6 archives without legacy references"
            )
        archive.seek(file_size - 12)
        footer_tail = _read_exact(archive, 12, "footer tail")
        if footer_tail[:4] != b"BRVF":
            raise BenchmarkError("archive footer magic is invalid")
        index_offset = struct.unpack("<Q", footer_tail[4:])[0]
        if index_offset < 8 or index_offset > file_size - 12:
            raise BenchmarkError("archive footer index offset is outside the file")
        archive.seek(8)
        position = 8
        while position < index_offset:
            program_length = struct.unpack(
                "<I", _read_exact(archive, 4, "program length")
            )[0]
            position += 4
            if program_length == 0:
                raise BenchmarkError(
                    "legacy program back-references are unsupported for formal program evidence"
                )
            if program_length > index_offset - position:
                raise BenchmarkError("archive program bytecode extends beyond the frame region")
            program_digest = hashlib.sha256()
            remaining = program_length
            while remaining:
                chunk = _read_exact(
                    archive, min(remaining, 1 << 20), "program bytecode",
                )
                program_digest.update(chunk)
                remaining -= len(chunk)
            position += program_length
            payload_length = struct.unpack(
                "<Q", _read_exact(archive, 8, "payload length")
            )[0]
            position += 8
            if payload_length > index_offset - position:
                raise BenchmarkError("archive payload extends beyond the frame region")
            archive.seek(payload_length, os.SEEK_CUR)
            position += payload_length
            lengths.append(program_length)
            hashes.append(program_digest.hexdigest())
        if position != index_offset:
            raise BenchmarkError("archive frame scan did not end at the footer index")
    return {
        "version": 1,
        "hash": "sha256",
        "sequence_spec_id": PROGRAM_SEQUENCE_SPEC_ID,
        "archive_header_hex": header.hex(),
        "archive_size_bytes": file_size,
        "index_offset": index_offset,
        "block_count": len(lengths),
        "bytecode_lengths": lengths,
        "sha256_by_block": hashes,
        "sequence_sha256": _program_sequence_sha256(lengths, hashes),
    }


def _validate_calibration_report(
    report: Mapping[str, Any],
    *,
    source_path: pathlib.Path,
    source_size: int,
    source_sha256: str,
    prior_path: pathlib.Path,
    prior_sha256: str,
    jobs: int,
    calibration_tensors: int,
    expected_search_configuration: Mapping[str, Any],
) -> None:
    input_record = report.get("input")
    output_record = report.get("output_prior")
    if not isinstance(input_record, Mapping):
        raise BenchmarkError("calibration report has no input object")
    if input_record.get("sha256") != source_sha256:
        raise BenchmarkError("calibration report input SHA-256 does not match the source")
    if input_record.get("size_bytes") != source_size:
        raise BenchmarkError("calibration report input size does not match the source")
    try:
        reported_input = pathlib.Path(str(input_record.get("path"))).resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"calibration report has an invalid input path: {exc}") from exc
    if reported_input != source_path.resolve(strict=False):
        raise BenchmarkError("calibration report input path does not match the source")
    if not isinstance(output_record, Mapping):
        raise BenchmarkError("calibration report has no output_prior object")
    if output_record.get("sha256") != prior_sha256:
        raise BenchmarkError("calibration report prior SHA-256 does not match the file")
    try:
        reported_path = pathlib.Path(str(output_record.get("path"))).resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"calibration report has an invalid prior path: {exc}") from exc
    if reported_path != prior_path.resolve(strict=False):
        raise BenchmarkError("calibration report prior path does not match the requested output")
    if output_record.get("nonempty") is not True:
        raise BenchmarkError("calibration produced an empty prior")

    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise BenchmarkError("calibration report has no configuration object")
    if configuration.get("max_tensors") != calibration_tensors:
        raise BenchmarkError("calibration report max_tensors does not match the command")
    if configuration.get("requested_threads") != jobs:
        raise BenchmarkError("calibration report requested thread count does not match the command")
    if configuration.get("seed") != DEFAULT_CALIBRATION_SEED:
        raise BenchmarkError("calibration report seed does not match the pinned default")
    if configuration.get("search") != dict(expected_search_configuration):
        raise BenchmarkError("calibration report search configuration does not match the probe")

    observed = report.get("observed")
    if not isinstance(observed, Mapping):
        raise BenchmarkError("calibration report has no observed object")
    available = observed.get("available_tensors")
    sampled = observed.get("sampled_tensors")
    threads_used = configuration.get("threads_used")
    for label, value in (
        ("available_tensors", available),
        ("sampled_tensors", sampled),
        ("threads_used", threads_used),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise BenchmarkError(f"calibration report {label} must be a positive integer")
    assert isinstance(available, int) and isinstance(sampled, int)
    assert isinstance(threads_used, int)
    if sampled > available or sampled > calibration_tensors:
        raise BenchmarkError("calibration report sampled_tensors exceeds its declared bound")
    if threads_used != min(jobs, sampled):
        raise BenchmarkError("calibration report threads_used is inconsistent with the sample")


def _run_calibration(
    *,
    binary: pathlib.Path,
    source: pathlib.Path,
    source_sha256: str,
    prior_path: pathlib.Path,
    run_dir: pathlib.Path,
    phase: str,
    index: int,
    jobs: int,
    calibration_tensors: int,
    search_args: Sequence[str],
    expected_search_configuration: Mapping[str, Any],
    timeout_seconds: float | None,
    keep_artifacts: bool,
) -> dict[str, Any]:
    command = [
        str(binary), "calibrate", str(source), str(prior_path),
        "--format", "json", "--tensors", str(calibration_tensors),
        "--jobs", str(jobs), *search_args,
    ]
    process = common._run_process(command, run_dir, "calibration", timeout_seconds)
    failure = _process_failure("calibration", process)
    report: dict[str, Any] | None = None
    prior: dict[str, Any] | None = None
    if failure is None:
        try:
            report = _embed_json_stdout(
                process, minimum_schema=1, expected_kind="brevis.calibration-report",
            )
            if not prior_path.is_file():
                raise BenchmarkError("calibration exited successfully but produced no prior")
            prior_sha256 = common._sha256_file(prior_path)
            _validate_calibration_report(
                report,
                source_path=source,
                source_size=source.stat().st_size,
                source_sha256=source_sha256,
                prior_path=prior_path,
                prior_sha256=prior_sha256,
                jobs=jobs,
                calibration_tensors=calibration_tensors,
                expected_search_configuration=expected_search_configuration,
            )
            prior = {
                "path": str(prior_path) if keep_artifacts else None,
                "retained": keep_artifacts,
                "sha256": prior_sha256,
                "storage": common._file_allocation(prior_path),
            }
        except (OSError, BenchmarkError) as exc:
            failure = f"invalid calibration output: {exc}"
    return {
        "status_code": "ok" if failure is None else "failed_calibration",
        "phase": phase,
        "index": index,
        "command": command,
        "process": process,
        "report": report,
        "prior": prior,
        "success": failure is None,
        "failure": failure,
    }


def _validate_bench_report(
    report: Mapping[str, Any],
    *,
    spec: BrevisSpec,
    source_path: pathlib.Path,
    source_size: int,
    source_sha256: str,
    canonical_prior: pathlib.Path | None,
    prior_sha256: str | None,
    jobs: int,
    expected_search_configuration: Mapping[str, Any],
) -> None:
    schema = report.get("schema")
    if isinstance(schema, int) and not isinstance(schema, bool) and schema >= 4:
        if report.get("kind") != "brevis.bench-report":
            raise BenchmarkError(
                "bench schema >=4 must identify kind='brevis.bench-report'"
            )
    if report.get("input_sha256") != source_sha256:
        raise BenchmarkError("bench report input SHA-256 does not match the source")
    if report.get("input_size_bytes") != source_size:
        raise BenchmarkError("bench report input size does not match the source")
    try:
        reported_input = pathlib.Path(str(report.get("input"))).resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"bench report has an invalid input path: {exc}") from exc
    if reported_input != source_path.resolve(strict=False):
        raise BenchmarkError("bench report input path does not match the source")
    expected_mode = (
        "fixed" if spec.plan == "fixed"
        else "phog" if spec.prior_policy == "canonical"
        else "uniform"
    )
    if report.get("mode") != expected_mode:
        raise BenchmarkError(
            f"bench report mode is {report.get('mode')!r}, expected {expected_mode!r}"
        )
    if report.get("requested_threads") != jobs:
        raise BenchmarkError("bench report requested thread count does not match the command")
    if report.get("threads") != jobs:
        raise BenchmarkError("bench report effective thread count does not match the command")
    for field in ("planning_workers_used", "encoding_workers_used"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= jobs:
            raise BenchmarkError(f"bench report {field} is outside the requested worker bound")
    if report.get("search_options_applied") is not (spec.plan == "search"):
        raise BenchmarkError("bench report disagrees about whether search options were applied")
    if report.get("search") != dict(expected_search_configuration):
        raise BenchmarkError("bench report search configuration does not match the probe")
    expected_effective = _expected_effective_dict(spec)
    if (
        expected_effective
        and "target_block_bytes" in report
        and (
            type(report.get("target_block_bytes"))
            is not type(expected_effective["target_block_bytes"])
            or report.get("target_block_bytes")
            != expected_effective["target_block_bytes"]
        )
    ):
        raise BenchmarkError("bench report target_block_bytes drifted from the frozen expectation")
    prior = report.get("prior")
    if not isinstance(prior, Mapping):
        raise BenchmarkError("bench report has no prior object")
    guidance_expected = spec.prior_policy == "canonical"
    for field in ("supplied", "loaded", "applied", "guidance_active", "nonempty"):
        if prior.get(field) is not guidance_expected:
            raise BenchmarkError(
                f"bench report prior.{field} disagrees with the requested guidance mode"
            )
    if guidance_expected:
        if prior.get("sha256") != prior_sha256:
            raise BenchmarkError("bench report prior SHA-256 does not match the canonical prior")
        try:
            reported_prior = pathlib.Path(str(prior.get("path"))).resolve(strict=False)
        except (OSError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"bench report has an invalid prior path: {exc}") from exc
        if canonical_prior is None or reported_prior != canonical_prior.resolve(strict=False):
            raise BenchmarkError("bench report prior path does not match the canonical prior")
    elif prior.get("path") is not None or prior.get("sha256") is not None:
        raise BenchmarkError("unguided bench report unexpectedly records a prior")
    if spec.require_raw_only:
        search = report.get("search")
        if not isinstance(search, Mapping) or search.get("enabled_ops") != ["raw"]:
            raise BenchmarkError(
                "configuration requiring a raw-only grammar did not apply exactly raw"
            )


def _nonnegative_report_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"bench report {label} must be a non-negative integer")
    return value


def _validate_schema_four_accounting(report: Mapping[str, Any]) -> None:
    """Fail closed on the structured fields consumed by DSL analysis."""

    tensors = report.get("tensors")
    blocks = report.get("blocks")
    if not isinstance(tensors, list) or not isinstance(blocks, list):
        raise BenchmarkError("bench schema 4 requires tensor and block arrays")

    top = {
        key: _nonnegative_report_int(report.get(key), key)
        for key in (
            "input_size_bytes",
            "raw_bytes",
            "tensor_data_bytes",
            "safetensors_prefix_bytes",
            "encoded_bytes_without_frame_headers",
            "block_frame_bytes_excluding_container_header_footer",
            "container_header_bytes",
            "container_footer_bytes",
            "projected_archive_bytes",
        )
    }
    if top["raw_bytes"] != top["tensor_data_bytes"]:
        raise BenchmarkError("bench report raw_bytes and tensor_data_bytes disagree")
    if top["safetensors_prefix_bytes"] + top["tensor_data_bytes"] != top["input_size_bytes"]:
        raise BenchmarkError("bench report input prefix/data accounting is inconsistent")
    if (
        top["container_header_bytes"]
        + top["block_frame_bytes_excluding_container_header_footer"]
        + top["container_footer_bytes"]
        != top["projected_archive_bytes"]
    ):
        raise BenchmarkError("bench report projected archive accounting is inconsistent")

    block_raw = 0
    block_encoded = 0
    block_framed = 0
    program_lengths: list[int] = []
    program_hashes: list[str] = []
    for position, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise BenchmarkError(f"bench report block {position} is not an object")
        if _nonnegative_report_int(block.get("index"), f"blocks[{position}].index") != position:
            raise BenchmarkError("bench report block indices are not contiguous")
        tensor_index = _nonnegative_report_int(
            block.get("tensor_index"), f"blocks[{position}].tensor_index",
        )
        if tensor_index >= len(tensors):
            raise BenchmarkError("bench report block references an unknown tensor")
        raw = _nonnegative_report_int(block.get("raw_bytes"), f"blocks[{position}].raw_bytes")
        encoded = _nonnegative_report_int(
            block.get("encoded_bytes_without_frame_headers"),
            f"blocks[{position}].encoded_bytes_without_frame_headers",
        )
        bytecode = _nonnegative_report_int(
            block.get("program_bytecode_bytes"),
            f"blocks[{position}].program_bytecode_bytes",
        )
        if bytecode == 0:
            raise BenchmarkError("bench report block program bytecode cannot be empty")
        bytecode_sha256 = block.get("program_bytecode_sha256")
        if not isinstance(bytecode_sha256, str):
            raise BenchmarkError("bench report block has no program bytecode SHA-256")
        payload = _nonnegative_report_int(
            block.get("packed_terminal_payload_bytes"),
            f"blocks[{position}].packed_terminal_payload_bytes",
        )
        frame_header = _nonnegative_report_int(
            block.get("frame_header_bytes"), f"blocks[{position}].frame_header_bytes",
        )
        framed = _nonnegative_report_int(
            block.get("framed_bytes"), f"blocks[{position}].framed_bytes",
        )
        if encoded != bytecode + payload:
            raise BenchmarkError("bench report block bytecode/payload accounting is inconsistent")
        if frame_header != bytecode + 12 or framed != encoded + 12:
            raise BenchmarkError("bench report block frame accounting is inconsistent")
        block_raw += raw
        block_encoded += encoded
        block_framed += framed
        program_lengths.append(bytecode)
        program_hashes.append(bytecode_sha256)

    tensor_raw = 0
    covered_blocks: list[int] = []
    for position, tensor in enumerate(tensors):
        if not isinstance(tensor, Mapping):
            raise BenchmarkError(f"bench report tensor {position} is not an object")
        if _nonnegative_report_int(tensor.get("index"), f"tensors[{position}].index") != position:
            raise BenchmarkError("bench report tensor indices are not contiguous")
        if not isinstance(tensor.get("name"), str) or not isinstance(tensor.get("dtype"), str):
            raise BenchmarkError("bench report tensor name and dtype must be strings")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
            for dimension in shape
        ):
            raise BenchmarkError("bench report tensor shape must contain non-negative integers")
        _nonnegative_report_int(tensor.get("numel"), f"tensors[{position}].numel")
        raw = _nonnegative_report_int(
            tensor.get("raw_bytes"), f"tensors[{position}].raw_bytes",
        )
        start = _nonnegative_report_int(
            tensor.get("file_data_start_byte"),
            f"tensors[{position}].file_data_start_byte",
        )
        end = _nonnegative_report_int(
            tensor.get("file_data_end_byte_exclusive"),
            f"tensors[{position}].file_data_end_byte_exclusive",
        )
        if end < start or end - start != raw or end > top["input_size_bytes"]:
            raise BenchmarkError("bench report tensor file offsets are inconsistent")
        first = _nonnegative_report_int(
            tensor.get("block_start"), f"tensors[{position}].block_start",
        )
        count = _nonnegative_report_int(
            tensor.get("block_count"), f"tensors[{position}].block_count",
        )
        if first + count > len(blocks):
            raise BenchmarkError("bench report tensor block range exceeds the block array")
        selected = blocks[first:first + count]
        if any(block.get("tensor_index") != position for block in selected):
            raise BenchmarkError("bench report tensor block range has the wrong tensor index")
        covered_blocks.extend(range(first, first + count))
        selected_raw = sum(int(block["raw_bytes"]) for block in selected)
        selected_encoded = sum(
            int(block["encoded_bytes_without_frame_headers"]) for block in selected
        )
        selected_framed = sum(int(block["framed_bytes"]) for block in selected)
        if selected_raw != raw:
            raise BenchmarkError("bench report tensor raw bytes disagree with its blocks")
        tensor_encoded = _nonnegative_report_int(
            tensor.get("encoded_bytes_without_frame_headers"),
            f"tensors[{position}].encoded_bytes_without_frame_headers",
        )
        tensor_framed = _nonnegative_report_int(
            tensor.get("block_frame_bytes_excluding_container_header_footer"),
            f"tensors[{position}].block_frame_bytes_excluding_container_header_footer",
        )
        if tensor_encoded != selected_encoded:
            raise BenchmarkError("bench report tensor encoded bytes disagree with its blocks")
        if tensor_framed != selected_framed:
            raise BenchmarkError("bench report tensor framed bytes disagree with its blocks")
        planned_raw = _nonnegative_report_int(
            tensor.get("planned_raw_root_blocks"),
            f"tensors[{position}].planned_raw_root_blocks",
        )
        fallback_raw = _nonnegative_report_int(
            tensor.get("fallback_raw_root_blocks"),
            f"tensors[{position}].fallback_raw_root_blocks",
        )
        raw_roots = _nonnegative_report_int(
            tensor.get("raw_root_blocks"), f"tensors[{position}].raw_root_blocks",
        )
        if raw_roots != planned_raw + fallback_raw or raw_roots > count:
            raise BenchmarkError("bench report tensor raw-root classification is inconsistent")
        tensor_raw += raw

    if covered_blocks != list(range(len(blocks))):
        raise BenchmarkError("bench report tensor ranges do not cover each block exactly once")
    if tensor_raw != top["tensor_data_bytes"] or block_raw != top["raw_bytes"]:
        raise BenchmarkError("bench report tensor/block raw totals are inconsistent")
    if block_encoded != top["encoded_bytes_without_frame_headers"]:
        raise BenchmarkError("bench report encoded block total is inconsistent")
    if block_framed != top["block_frame_bytes_excluding_container_header_footer"]:
        raise BenchmarkError("bench report framed block total is inconsistent")
    evidence = report.get("program_bytecode_evidence")
    if not isinstance(evidence, Mapping):
        raise BenchmarkError("bench report has no program_bytecode_evidence object")
    if (
        evidence.get("version") != 1
        or evidence.get("hash") != "sha256"
        or evidence.get("sequence_spec_id") != PROGRAM_SEQUENCE_SPEC_ID
        or evidence.get("block_count") != len(blocks)
    ):
        raise BenchmarkError("bench report program evidence metadata is unsupported")
    expected_sequence = _program_sequence_sha256(program_lengths, program_hashes)
    if evidence.get("sequence_sha256") != expected_sequence:
        raise BenchmarkError("bench report aggregate program bytecode digest is inconsistent")


def _compare_program_bytecode_evidence(
    report: Mapping[str, Any], archive_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report_blocks = report.get("blocks")
    report_top = report.get("program_bytecode_evidence")
    if not isinstance(report_blocks, list) or not isinstance(report_top, Mapping):
        raise BenchmarkError("bench report program evidence is unavailable")
    if archive_evidence is None:
        return {
            "status_code": "not_checked_archive_unavailable",
            "matches": None,
            "mismatch_block_indices": [],
            "report_sequence_sha256": report_top.get("sequence_sha256"),
            "archive_sequence_sha256": None,
        }
    archive_lengths = archive_evidence.get("bytecode_lengths")
    archive_hashes = archive_evidence.get("sha256_by_block")
    if not isinstance(archive_lengths, list) or not isinstance(archive_hashes, list):
        raise BenchmarkError("archive program evidence arrays are unavailable")
    report_lengths = [block.get("program_bytecode_bytes") for block in report_blocks]
    report_hashes = [block.get("program_bytecode_sha256") for block in report_blocks]
    mismatch_indices = [
        index
        for index in range(max(len(report_lengths), len(archive_lengths)))
        if index >= len(report_lengths)
        or index >= len(archive_lengths)
        or report_lengths[index] != archive_lengths[index]
        or report_hashes[index] != archive_hashes[index]
    ]
    matches = (
        not mismatch_indices
        and report_top.get("block_count") == archive_evidence.get("block_count")
        and report_top.get("sequence_sha256") == archive_evidence.get("sequence_sha256")
    )
    return {
        "status_code": "consistent" if matches else "program_mismatch",
        "matches": matches,
        "mismatch_block_indices": mismatch_indices,
        "report_block_count": report_top.get("block_count"),
        "archive_block_count": archive_evidence.get("block_count"),
        "report_sequence_sha256": report_top.get("sequence_sha256"),
        "archive_sequence_sha256": archive_evidence.get("sequence_sha256"),
        "sequence_spec_id": PROGRAM_SEQUENCE_SPEC_ID,
    }


def _search_command_args(
    spec: BrevisSpec,
    canonical_prior: pathlib.Path | None,
    canonical_prior_sha256: str | None,
) -> tuple[list[str], str | None]:
    common_args = ["--plan", spec.plan, *spec.search_args]
    if spec.prior_policy != "canonical":
        return common_args, None
    if canonical_prior is None or canonical_prior_sha256 is None:
        return common_args, "PHOG configuration has no successful measured canonical prior"
    common_args.extend(("--prior", str(canonical_prior)))
    return common_args, None


def _run_archive_pipeline(
    *,
    binary: pathlib.Path,
    source: pathlib.Path,
    source_size: int,
    source_sha256: str,
    spec: BrevisSpec,
    canonical_prior: pathlib.Path | None,
    canonical_prior_sha256: str | None,
    run_dir: pathlib.Path,
    phase: str,
    index: int,
    execution_order: int,
    order_within_repetition: int,
    jobs: int,
    timeout_seconds: float | None,
    keep_artifacts: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    search_args, prior_failure = _search_command_args(
        spec, canonical_prior, canonical_prior_sha256,
    )
    common_args = [*search_args[:2], "--jobs", str(jobs), *search_args[2:]]
    if prior_failure is not None:
        return {
            "status_code": "canonical_prior_unavailable",
            "phase": phase,
            "index": index,
            "execution_order": execution_order,
            "archive_execution_order": execution_order,
            "diagnostic_execution_order": None,
            "order_within_repetition": order_within_repetition,
            "commands": {"bench": None, "compress": None, "decompress": None},
            "bench_process": None,
            "bench_report": None,
            "compression": None,
            "archive": None,
            "bench_archive_projection_consistency": {
                "status_code": "diagnostic_not_run",
                "projected_archive_bytes": None,
                "actual_archive_bytes": None,
                "matches_actual_archive": None,
            },
            "bench_archive_program_consistency": {
                "status_code": "diagnostic_not_run",
                "matches": None,
                "mismatch_block_indices": [],
            },
            "decompression": None,
            "verification": None,
            "archive_pipeline_success": False,
            "diagnostic_success": None,
            "success": False,
            "failure": prior_failure,
            "artifact_directory": str(run_dir) if keep_artifacts else None,
        }

    archive_path = run_dir / "archive.brv"
    compress_command = [
        str(binary), "compress", str(source), str(archive_path), *common_args,
    ]
    compression = common._run_process(
        compress_command, run_dir, "compression", timeout_seconds,
    )
    compression_failure = _process_failure("compression", compression)
    archive: dict[str, Any] | None = None
    if compression_failure is None:
        if not archive_path.is_file():
            compression_failure = "compression exited successfully but produced no archive"
        else:
            try:
                archive = {
                    "path": str(archive_path) if keep_artifacts else None,
                    "storage": common._file_allocation(archive_path),
                    "sha256": common._sha256_file(archive_path),
                    "sha256_after_decode": None,
                    "unchanged_during_decode": None,
                    "program_bytecode_evidence": _archive_program_bytecode_evidence(
                        archive_path
                    ),
                }
                compression["output_size_bytes"] = archive["storage"]["logical_size_bytes"]
                compression["output_storage"] = archive["storage"]
            except (OSError, BenchmarkError) as exc:
                compression_failure = f"archive inspection failed: {type(exc).__name__}: {exc}"
    if compression_failure is not None:
        failures.append(compression_failure)

    decompression: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    if compression_failure is None and archive is not None:
        restored_path = run_dir / "restored.safetensors"
        decompress_command = [
            str(binary), "decompress", str(archive_path), str(restored_path),
            "--jobs", str(jobs),
        ]
        decompression = common._run_process(
            decompress_command, run_dir, "decompression", timeout_seconds,
        )
        decompression_failure = _process_failure("decompression", decompression)
        if decompression_failure is None:
            verification = common._verify_file(source, restored_path, source_sha256)
            if restored_path.is_file():
                decompression["output_size_bytes"] = restored_path.stat().st_size
                decompression["output_storage"] = common._file_allocation(restored_path)
            if not verification.get("bit_exact"):
                decompression_failure = (
                    "full post-timing byte verification failed: "
                    + str(verification.get("error"))
                )
        if decompression_failure is not None:
            failures.append(decompression_failure)
        try:
            archive_after_decode = common._sha256_file(archive_path)
        except OSError as exc:
            failures.append(f"archive post-decode rehash failed: {type(exc).__name__}: {exc}")
        else:
            archive["sha256_after_decode"] = archive_after_decode
            archive["unchanged_during_decode"] = archive_after_decode == archive["sha256"]
            if not archive["unchanged_during_decode"]:
                failures.append("archive SHA-256 changed while it was being decoded")

    return {
        "status_code": "awaiting_diagnostic" if not failures else "archive_pipeline_failed",
        "phase": phase,
        "index": index,
        "execution_order": execution_order,
        "archive_execution_order": execution_order,
        "diagnostic_execution_order": None,
        "order_within_repetition": order_within_repetition,
        "commands": {
            "bench": None,
            "compress": compress_command,
            "decompress": (
                decompression.get("command") if decompression is not None else None
            ),
        },
        "bench_process": None,
        "bench_report": None,
        "compression": compression,
        "archive": archive,
        "bench_archive_projection_consistency": {
            "status_code": "diagnostic_pending",
            "projected_archive_bytes": None,
            "actual_archive_bytes": (
                archive["storage"]["logical_size_bytes"] if archive is not None else None
            ),
            "matches_actual_archive": None,
        },
        "bench_archive_program_consistency": {
            "status_code": "diagnostic_pending",
            "matches": None,
            "mismatch_block_indices": [],
        },
        "decompression": decompression,
        "verification": verification,
        "archive_pipeline_success": not failures,
        "diagnostic_success": None,
        "success": None,
        "failure": "; ".join(failures) if failures else None,
        "artifact_directory": str(run_dir) if keep_artifacts else None,
        "timing_relationship": (
            "bench is a separate diagnostic replay executed only after every timed archive "
            "pipeline; its internal times do not decompose this compression invocation"
        ),
    }


def _run_bench_diagnostic(
    *,
    binary: pathlib.Path,
    source: pathlib.Path,
    source_size: int,
    source_sha256: str,
    spec: BrevisSpec,
    canonical_prior: pathlib.Path | None,
    canonical_prior_sha256: str | None,
    expected_search_configuration: Mapping[str, Any],
    actual_archive_bytes: int | None,
    archive_program_evidence: Mapping[str, Any] | None,
    run_dir: pathlib.Path,
    jobs: int,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    search_args, prior_failure = _search_command_args(
        spec, canonical_prior, canonical_prior_sha256,
    )
    common_args = [*search_args[:2], "--jobs", str(jobs), *search_args[2:]]
    if prior_failure is not None:
        return {
            "command": None,
            "process": None,
            "report": None,
            "projection_consistency": {
                "status_code": "diagnostic_not_run",
                "projected_archive_bytes": None,
                "actual_archive_bytes": actual_archive_bytes,
                "matches_actual_archive": None,
            },
            "program_consistency": {
                "status_code": "diagnostic_not_run",
                "matches": None,
                "mismatch_block_indices": [],
            },
            "success": False,
            "failure": prior_failure,
        }

    command = [str(binary), "bench", str(source), "--format", "json", *common_args]
    process = common._run_process(command, run_dir, "bench", timeout_seconds)
    report: dict[str, Any] | None = None
    failure = _process_failure("diagnostic bench replay", process)
    projection = {
        "status_code": "report_unavailable",
        "projected_archive_bytes": None,
        "actual_archive_bytes": actual_archive_bytes,
        "matches_actual_archive": None,
    }
    program_consistency: dict[str, Any] = {
        "status_code": "report_unavailable",
        "matches": None,
        "mismatch_block_indices": [],
    }
    if failure is None:
        try:
            report = _embed_json_stdout(
                process, minimum_schema=4, expected_kind="brevis.bench-report",
            )
            if report.get("schema") != 4:
                raise BenchmarkError(
                    f"formal DSL evidence supports bench schema 4 exactly; observed {report.get('schema')!r}"
                )
            _validate_bench_report(
                report,
                spec=spec,
                source_path=source,
                source_size=source_size,
                source_sha256=source_sha256,
                canonical_prior=canonical_prior,
                prior_sha256=canonical_prior_sha256,
                jobs=jobs,
                expected_search_configuration=expected_search_configuration,
            )
            _validate_schema_four_accounting(report)
            program_consistency = _compare_program_bytecode_evidence(
                report, archive_program_evidence,
            )
            if archive_program_evidence is not None and not program_consistency["matches"]:
                raise BenchmarkError(
                    "diagnostic program bytecode does not match the actual archive frames"
                )
        except BenchmarkError as exc:
            failure = f"invalid diagnostic bench report: {exc}"

    if report is not None:
        projected = report.get("projected_archive_bytes")
        projection.update({
            "status_code": "not_checked_archive_unavailable",
            "projected_archive_bytes": projected,
        })
        if isinstance(projected, bool) or not isinstance(projected, int) or projected < 0:
            projection["status_code"] = "invalid_projection"
            failure = failure or "bench schema 4 has no valid projected_archive_bytes"
        elif actual_archive_bytes is not None:
            matches = projected == actual_archive_bytes
            projection.update({
                "status_code": "consistent" if matches else "projection_mismatch",
                "matches_actual_archive": matches,
            })
            if not matches:
                message = "bench projected_archive_bytes does not match the actual .brv size"
                failure = f"{failure}; {message}" if failure else message

    return {
        "command": command,
        "process": process,
        "report": report,
        "projection_consistency": projection,
        "program_consistency": program_consistency,
        "success": failure is None,
        "failure": failure,
    }


def _new_archive_consistency() -> dict[str, Any]:
    return {
        "status_code": "not_checked",
        "reference_size_bytes": None,
        "reference_sha256": None,
        "consistent_runs": 0,
        "inconsistent_runs": 0,
    }


def _apply_archive_consistency(
    configuration: dict[str, Any], run: dict[str, Any], phase: str,
) -> None:
    consistency = configuration["measured_archive_consistency"]
    if phase != "measured":
        run["archive_consistency"] = {
            "status_code": "not_applicable_warmup",
            "matches_reference": None,
        }
        return
    archive = run.get("archive")
    verification = run.get("verification")
    if archive is None or not isinstance(verification, Mapping) or not verification.get("bit_exact"):
        run["archive_consistency"] = {
            "status_code": "not_checked_iteration_failed",
            "matches_reference": None,
        }
        return
    size = archive["storage"]["logical_size_bytes"]
    digest = archive["sha256"]
    if consistency["reference_sha256"] is None:
        consistency.update({
            "status_code": "consistent",
            "reference_size_bytes": size,
            "reference_sha256": digest,
            "consistent_runs": 1,
        })
        run["archive_consistency"] = {
            "status_code": "reference",
            "matches_reference": True,
            "reference_size_bytes": size,
            "reference_sha256": digest,
        }
        return
    matches = (
        size == consistency["reference_size_bytes"]
        and digest == consistency["reference_sha256"]
    )
    run["archive_consistency"] = {
        "status_code": "consistent" if matches else "archive_inconsistent",
        "matches_reference": matches,
        "reference_size_bytes": consistency["reference_size_bytes"],
        "reference_sha256": consistency["reference_sha256"],
    }
    if matches:
        consistency["consistent_runs"] += 1
    else:
        consistency["status_code"] = "archive_inconsistent"
        consistency["inconsistent_runs"] += 1
        run["status_code"] = "archive_inconsistent"
        run["archive_pipeline_success"] = False
        run["success"] = False
        message = "measured archive size/SHA-256 differs from the first valid repetition"
        run["failure"] = f"{run['failure']}; {message}" if run.get("failure") else message


def _calibration_consistency(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digests = [
        run["prior"]["sha256"]
        for run in runs
        if run.get("success") and isinstance(run.get("prior"), Mapping)
    ]
    context_counts = []
    for run in runs:
        report = run.get("report")
        output_prior = report.get("output_prior") if isinstance(report, Mapping) else None
        counts = (
            output_prior.get("context_counts_by_backoff_level")
            if isinstance(output_prior, Mapping) else None
        )
        if isinstance(counts, list):
            context_counts.append(tuple(counts))
    distinct_context_counts = list(dict.fromkeys(context_counts))
    if not digests:
        return {
            "status_code": "no_successful_measured_prior",
            "reference_sha256": None,
            "distinct_sha256": [],
            "byte_consistent": None,
            "distinct_context_counts_by_backoff_level": [
                list(counts) for counts in distinct_context_counts
            ],
            "context_counts_consistent": None,
        }
    distinct = list(dict.fromkeys(digests))
    return {
        "status_code": "consistent" if len(distinct) == 1 else "byte_inconsistent",
        "reference_sha256": digests[0],
        "distinct_sha256": distinct,
        "byte_consistent": len(distinct) == 1,
        "distinct_context_counts_by_backoff_level": [
            list(counts) for counts in distinct_context_counts
        ],
        "context_counts_consistent": len(distinct_context_counts) == 1,
        "note": (
            "Equal context counts are only a coarse check, not proof of semantic equality. "
            "Formal runs therefore fail closed when prior bytes or context counts differ."
        ),
    }


def _rotated_warmup_orders(
    specs: Sequence[BrevisSpec], warmups: int,
) -> list[tuple[BrevisSpec, ...]]:
    if not specs:
        return []
    return [
        tuple((*specs[index % len(specs):], *specs[:index % len(specs)]))
        for index in range(warmups)
    ]


def benchmark_file(
    source: os.PathLike[str] | str,
    *,
    binary: os.PathLike[str] | str = DEFAULT_BINARY,
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
    configuration_ids: Iterable[str] | None = None,
    specs: Sequence[BrevisSpec] | None = None,
    jobs: int = 1,
    calibration_tensors: int = DEFAULT_CALIBRATION_TENSORS,
    work_dir: os.PathLike[str] | str | None = None,
    keep_artifacts: bool = False,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    input_metadata: Mapping[str, Any] | None = None,
    expected_source_size_bytes: int | None = None,
    expected_source_sha256: str | None = None,
    manifest_path: os.PathLike[str] | str | None = None,
    expected_manifest_sha256: str | None = None,
    checkpoint_path: os.PathLike[str] | str | None = None,
    force_checkpoint: bool = False,
    require_declared_integrity: bool = True,
    require_clean_git: bool = True,
    build_binary: bool = True,
    additional_formal_ineligibility_reasons: Sequence[str] = (),
    disk_reserve_fraction: float = DEFAULT_DISK_RESERVE_FRACTION,
    temp_multiplier: float = DEFAULT_TEMP_MULTIPLIER,
    temp_fixed_bytes: int = DEFAULT_TEMP_FIXED_BYTES,
) -> dict[str, Any]:
    """Run the complete repeated Brevis benchmark for one immutable shard."""

    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    if (
        isinstance(calibration_tensors, bool)
        or not isinstance(calibration_tensors, int)
        or calibration_tensors < 1
    ):
        raise ValueError("calibration_tensors must be a positive integer")
    common._validate_timeout("timeout_seconds", timeout_seconds)
    if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
        raise ValueError("schedule_seed must be an integer")
    if not math.isfinite(disk_reserve_fraction) or not 0 <= disk_reserve_fraction < 1:
        raise ValueError("disk_reserve_fraction must be finite and in [0, 1)")
    if not math.isfinite(temp_multiplier) or temp_multiplier <= 0:
        raise ValueError("temp_multiplier must be finite and positive")
    if isinstance(temp_fixed_bytes, bool) or not isinstance(temp_fixed_bytes, int) or temp_fixed_bytes < 0:
        raise ValueError("temp_fixed_bytes must be a non-negative integer")
    extra_ineligibility = tuple(additional_formal_ineligibility_reasons)
    if (
        len(extra_ineligibility) != len(set(extra_ineligibility))
        or not all(
            isinstance(reason, str)
            and re.fullmatch(r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?", reason)
            for reason in extra_ineligibility
        )
    ):
        raise ValueError(
            "additional formal ineligibility reasons must be unique filename-safe identifiers"
        )
    if specs is not None and configuration_ids is not None:
        raise ValueError("configuration_ids and explicit specs are mutually exclusive")
    explicit_specs = _validate_explicit_specs(specs) if specs is not None else None

    source_path = pathlib.Path(source).resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"source is not a regular file: {source_path}")
    requested_binary = pathlib.Path(binary).resolve(strict=False)
    if build_binary and requested_binary != DEFAULT_BINARY.resolve(strict=False):
        raise ValueError("a custom binary requires build_binary=False/--skip-build")
    checkpoint = pathlib.Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        if common._paths_conflict(source_path, checkpoint):
            raise ValueError("source and checkpoint/output paths must differ")
        if checkpoint.exists() and not force_checkpoint:
            raise FileExistsError(f"checkpoint/output already exists: {checkpoint}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(input_metadata or {})
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input_metadata must be JSON serializable: {exc}") from exc

    started_at_utc = common._utc_now()
    source_size = source_path.stat().st_size
    source_sha256 = common._sha256_file(source_path)
    integrity = common._verify_declared_integrity(
        source_path=source_path,
        source_size=source_size,
        source_sha256=source_sha256,
        metadata=metadata,
        expected_source_size_bytes=expected_source_size_bytes,
        expected_source_sha256=expected_source_sha256,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if require_declared_integrity and not (
        integrity["source"].get("verified")
        and isinstance(integrity.get("manifest"), Mapping)
        and integrity["manifest"].get("verified")
        and isinstance(integrity.get("manifest_binding"), Mapping)
        and integrity["manifest_binding"].get("verified")
    ):
        raise BenchmarkError(
            "formal Brevis benchmarks require expected source size/SHA-256 and a "
            "SHA-verified unique model-manifest binding"
        )

    git = common._git_provenance()
    if require_clean_git and git.get("dirty") is not False:
        state = "unknown" if git.get("dirty") is None else "dirty"
        raise BenchmarkError(
            f"formal Brevis benchmarks require a clean Git tree; observed {state}; "
            "use allow_dirty only for explicitly labelled pilot runs"
        )

    parent = pathlib.Path(work_dir).resolve() if work_dir is not None else None
    if parent is None:
        parent = checkpoint.parent.resolve() if checkpoint is not None else pathlib.Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)

    requested_ids = (
        tuple(spec.identifier for spec in explicit_specs)
        if explicit_specs is not None
        else tuple(configuration_ids) if configuration_ids is not None
        else CORE_CONFIGURATION_IDS
    )
    if not requested_ids or len(set(requested_ids)) != len(requested_ids):
        raise ValueError("configuration_ids must be nonempty and unique")
    unknown = set(requested_ids) - set(CORE_CONFIGURATION_IDS)
    if explicit_specs is None and unknown:
        raise ValueError(f"unknown Brevis configuration(s): {', '.join(sorted(unknown))}")
    total_pipeline_iterations = (warmups + repetitions) * len(requested_ids)
    disk_gate = _disk_gate(
        parent,
        checkpoint_path=checkpoint,
        source_size=source_size,
        iterations=total_pipeline_iterations,
        configurations=len(requested_ids),
        keep_artifacts=keep_artifacts,
        reserve_fraction=disk_reserve_fraction,
        temp_multiplier=temp_multiplier,
        fixed_bytes=temp_fixed_bytes,
    )

    formal_ineligibility_reasons = []
    if not require_declared_integrity:
        formal_ineligibility_reasons.append("declared_integrity_gate_disabled")
    if not require_clean_git:
        formal_ineligibility_reasons.append("clean_git_gate_disabled")
    if not build_binary:
        formal_ineligibility_reasons.append("releasefast_build_not_performed_by_harness")
    formal_ineligibility_reasons.extend(extra_ineligibility)
    run_classification = {
        "run_class": "formal" if not formal_ineligibility_reasons else "pilot",
        "formal_eligible": not formal_ineligibility_reasons,
        "formal_ineligibility_reasons": formal_ineligibility_reasons,
        "policy": (
            "formal requires a manifest-bound immutable input, a clean Git tree, and a "
            "ReleaseFast build performed and hashed by this harness"
        ),
    }

    environment = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "zig": _tool_version(("zig", "version")),
        "gpu": _gpu_info(),
        "clock": "time.perf_counter_ns",
        "rss_source": (
            "os.wait4 direct-child ru_maxrss; Brevis threads share that process, but future "
            "descendant-process RSS would not be aggregated"
        ),
        "cpu": common._cpu_info(),
        "memory": common._memory_info(),
        "filesystems": {
            "source": common._filesystem_info(source_path),
            "work": common._filesystem_info(parent),
            "checkpoint": common._filesystem_info(checkpoint.parent) if checkpoint else None,
        },
    }
    harness_path = pathlib.Path(__file__).resolve()
    provenance: dict[str, Any] = {
        "git": git,
        "git_tracked_start": _git_tracked_snapshot(),
        "harness": {"path": str(harness_path), "sha256": common._sha256_file(harness_path)},
        "generic_helper": {
            "path": str(pathlib.Path(common.__file__).resolve()),
            "sha256": common._sha256_file(pathlib.Path(common.__file__).resolve()),
        },
        "build": {
            "requested": build_binary,
            "mode": "ReleaseFast" if build_binary else "not_verified_by_harness",
            "command": None,
            "process": None,
            "success": None,
        },
        "binary": _binary_record(requested_binary),
    }
    canonical_specs = tuple(
        spec for spec in explicit_specs or () if spec.prior_policy == "canonical"
    )
    calibration_required = (
        bool(canonical_specs) if explicit_specs is not None else "phog" in requested_ids
    )
    calibration_search_args = (
        canonical_specs[0].search_args if canonical_specs else ()
    )
    calibration: dict[str, Any] = {
        "required": calibration_required,
        "status_code": "pending" if calibration_required else "not_required",
        "success": None if calibration_required else True,
        "failure_count": 0,
        "failure": None,
        "configuration": {
            "max_tensors": calibration_tensors,
            "requested_jobs": jobs,
            "search_args": list(calibration_search_args),
        },
        "warmups": [],
        "runs": [],
        "execution_schedule": [],
        "measured_prior_consistency": _calibration_consistency([]),
        "canonical_prior": None,
    }
    if not calibration["required"]:
        calibration["measured_prior_consistency"] = {
            "status_code": "not_required",
            "reference_sha256": None,
            "distinct_sha256": [],
            "byte_consistent": None,
            "distinct_context_counts_by_backoff_level": [],
            "context_counts_consistent": None,
        }
    configurations: list[dict[str, Any]] = []
    execution_schedule: list[dict[str, Any]] = []
    artifact_root: pathlib.Path | None = None
    checkpoint_written = False
    global_failure: str | None = None
    cleanup: dict[str, Any] = {
        "requested": not keep_artifacts,
        "per_run": [],
        "artifact_root": {
            "attempted": False,
            "success": None,
            "path": None,
            "error": None,
        },
    }

    def build_document(*, status: str, success: bool | None) -> dict[str, Any]:
        return {
            "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
            "status": status,
            "success": success,
            "failure": global_failure,
            "run_classification": run_classification,
            "started_at_utc": started_at_utc,
            "ended_at_utc": common._utc_now(),
            "source": {
                "path": str(source_path),
                "size_bytes": source_size,
                "sha256": source_sha256,
            },
            "integrity": integrity,
            "input_metadata": metadata,
            "disk_gate": disk_gate,
            "configuration": {
                "requested_configuration_ids": list(requested_ids),
                "configuration_source": (
                    "explicit_brevis_specs" if explicit_specs is not None else "core_registry"
                ),
                "requested_specs": (
                    [spec.to_dict() for spec in explicit_specs]
                    if explicit_specs is not None else None
                ),
                "warmups": warmups,
                "repetitions": repetitions,
                "jobs": jobs,
                "calibration_tensors": calibration_tensors,
                "timeout_seconds_per_process": timeout_seconds,
                "schedule_seed": schedule_seed,
                "schedule_balance": (
                    "paired" if repetitions % 2 == 0 else "unpaired_final_repetition"
                ),
                "keep_artifacts": keep_artifacts,
                "artifact_root": str(artifact_root) if keep_artifacts and artifact_root else None,
                "checkpoint_path": str(checkpoint) if checkpoint else None,
                "require_declared_integrity": require_declared_integrity,
                "require_clean_git": require_clean_git,
                "cache_policy": (
                    "best-effort buffered I/O; integrity hashing and prior runs can warm pages; "
                    "the harness neither drops host caches nor guarantees page residency"
                ),
                "io_policy": (
                    "Brevis reads the immutable source directly and writes regular on-disk archive/"
                    "restored files; hashing and full verification occur after each timed process"
                ),
                "timing_scope": (
                    "external wall/RSS: immediately before process spawn through direct-child reap; "
                    "internal bench/calibration fields retain their CLI-defined narrower scopes"
                ),
                "scheduling_policy": (
                    "serial execution; all archive pipelines precede all diagnostic replays; "
                    "within each stage, warmups rotate and deterministic randomized measured "
                    "orders are paired with their reverse"
                ),
                "execution_schedule": list(execution_schedule),
            },
            "provenance": provenance,
            "environment": environment,
            "calibration": calibration,
            "configurations": configurations,
            "cleanup": cleanup,
            "limitations": [
                "bench is a separate deterministic diagnostic replay, not a timing decomposition "
                "of the archive-producing compression process; every diagnostic follows all "
                "timed archive tasks",
                "raw-terminal is a framed Brevis archive constrained to raw leaves; it is not the "
                "unframed raw/copy generic baseline",
                "cache residency is best effort and machine-specific",
                "PHOG calibration is input-local and its cost is reported separately",
                "schema-4 tensor/block arrays are stored once per successful configuration; "
                "matching diagnostic repetitions retain all top-level fields and resolve the "
                "checkpoint-local canonical detail by semantic SHA-256",
            ],
        }

    def save_checkpoint(*, status: str = "in_progress", success: bool | None = None) -> None:
        nonlocal checkpoint_written
        if checkpoint is None:
            return
        common.write_json(
            checkpoint,
            build_document(status=status, success=success),
            force=force_checkpoint or checkpoint_written,
        )
        checkpoint_written = True

    def add_global_failure(message: str | None) -> None:
        nonlocal global_failure
        if not message:
            return
        global_failure = f"{global_failure}; {message}" if global_failure else message

    def clean_run_directory(
        run_dir: pathlib.Path, record: dict[str, Any], label: str, field: str = "cleanup",
    ) -> None:
        if keep_artifacts:
            record[field] = {
                "label": label,
                "path": str(run_dir),
                "attempted": False,
                "success": True,
                "retained_by_request": True,
                "error": None,
            }
            return
        cleanup_record = _cleanup_directory(run_dir, label)
        cleanup["per_run"].append(cleanup_record)
        record[field] = cleanup_record
        if not cleanup_record["success"]:
            message = f"temporary-directory cleanup failed for {label}: {cleanup_record['error']}"
            add_global_failure(message)
            record["status_code"] = "cleanup_failed"
            record["success"] = False
            record["failure"] = (
                f"{record['failure']}; {message}" if record.get("failure") else message
            )
            record["artifact_directory"] = str(run_dir)

    def clean_artifact_root() -> None:
        if artifact_root is None:
            return
        cleanup["artifact_root"]["path"] = str(artifact_root)
        if keep_artifacts:
            cleanup["artifact_root"].update({
                "attempted": False,
                "success": True,
                "retained_by_request": True,
                "error": None,
            })
            return
        cleanup_record = _cleanup_directory(artifact_root, "artifact-root")
        cleanup["artifact_root"] = cleanup_record
        if not cleanup_record["success"]:
            add_global_failure(
                "artifact-root cleanup failed: " + str(cleanup_record.get("error"))
            )

    if not disk_gate["passes"]:
        global_failure = disk_gate["failure"]
        document = build_document(status="skipped_resource", success=False)
        save_checkpoint(status="skipped_resource", success=False)
        return document

    artifact_root = pathlib.Path(tempfile.mkdtemp(prefix="brevis-system-", dir=parent))
    canonical_prior_path: pathlib.Path | None = None
    canonical_prior_sha256: str | None = None
    try:
        if build_binary:
            zig = shutil.which("zig")
            if zig is None:
                global_failure = "zig executable not found; cannot build ReleaseFast binary"
            else:
                build_command = [
                    zig, "build", "--build-file", str(ROOT / "build.zig"),
                    "--prefix", str(ROOT / "zig-out"), "-Doptimize=ReleaseFast",
                ]
                provenance["build"]["command"] = build_command
                build_process = common._run_process(
                    build_command, artifact_root, "releasefast-build", timeout_seconds,
                )
                provenance["build"]["process"] = build_process
                provenance["build"]["success"] = _process_ok(build_process)
                if not _process_ok(build_process):
                    global_failure = _process_failure("ReleaseFast build", build_process)
        else:
            provenance["build"]["success"] = None

        provenance["binary"] = _binary_record(requested_binary)
        if global_failure is None and (
            not provenance["binary"]["executable"]
            or provenance["binary"]["sha256_before_runs"] is None
        ):
            global_failure = f"Brevis binary is missing, non-executable, or unhashable: {requested_binary}"

        default_config: dict[str, Any] | None = None
        if global_failure is None:
            probe_process, default_config, probe_failure = _run_config(
                requested_binary, (), artifact_root, "config-default", timeout_seconds,
            )
            provenance["binary"]["default_config_probe"] = probe_process
            provenance["binary"]["default_config"] = default_config
            if probe_failure is not None:
                global_failure = probe_failure

        selected_specs: tuple[BrevisSpec, ...] = ()
        if global_failure is None:
            assert default_config is not None
            enabled_ops = default_config.get("enabled_ops")
            if not isinstance(enabled_ops, list) or not all(
                isinstance(operator, str) for operator in enabled_ops
            ):
                global_failure = "brevis config has no string-valued enabled_ops array"
            elif explicit_specs is not None:
                selected_specs = explicit_specs
            else:
                selected_specs = select_specs(core_specs(enabled_ops), requested_ids)

        configuration_by_id: dict[str, dict[str, Any]] = {}
        for spec in selected_specs:
            probe, effective, failure = _run_config(
                requested_binary,
                spec.search_args,
                artifact_root,
                f"config-{_safe_label(spec.identifier)}",
                timeout_seconds,
            )
            if failure is None:
                try:
                    if not isinstance(effective, Mapping):
                        raise BenchmarkError("configuration probe returned no effective object")
                    _validate_effective_configuration(spec, effective)
                except BenchmarkError as exc:
                    failure = str(exc)
            if failure is None and spec.require_raw_only:
                if effective is None or effective.get("enabled_ops") != ["raw"]:
                    failure = (
                        "configuration requires a raw-only grammar, but its probe did not "
                        "leave exactly raw enabled"
                    )
            record = {
                "id": spec.identifier,
                "spec": spec.to_dict(),
                "configuration_probe": probe,
                "effective_search_configuration": effective,
                "warmups": [],
                "runs": [],
                "measured_archive_consistency": _new_archive_consistency(),
                "bench_report_detail_consistency": _new_bench_detail_consistency(),
                "status_code": "ready" if failure is None else "configuration_failed",
                "failure": failure,
            }
            configurations.append(record)
            configuration_by_id[spec.identifier] = record

        configuration_probe_failures = [
            record for record in configurations if record["failure"] is not None
        ]
        if configuration_probe_failures:
            global_failure = (
                "configuration probe failures before calibration/measurement: "
                + ", ".join(record["id"] for record in configuration_probe_failures)
            )

        selected_canonical_specs = tuple(
            spec for spec in selected_specs if spec.prior_policy == "canonical"
        )
        canonical_arg_sets = {
            spec.search_args for spec in selected_canonical_specs
        }
        if len(canonical_arg_sets) > 1:
            global_failure = (
                "canonical-prior configurations resolved to different search_args; "
                "refusing to share one calibration prior"
            )
        calibration_effective_config: Mapping[str, Any] = default_config or {}
        if selected_canonical_specs:
            canonical_search_args = selected_canonical_specs[0].search_args
            calibration["configuration"]["search_args"] = list(canonical_search_args)
            canonical_effective = [
                configuration_by_id[spec.identifier].get(
                    "effective_search_configuration"
                )
                for spec in selected_canonical_specs
            ]
            available_effective = [
                value for value in canonical_effective if isinstance(value, Mapping)
            ]
            if available_effective:
                calibration_effective_config = available_effective[0]
                if any(
                    dict(value) != dict(calibration_effective_config)
                    for value in available_effective[1:]
                ):
                    global_failure = (
                        "identical canonical search_args produced inconsistent effective "
                        "configuration probes; refusing to share one calibration prior"
                    )

        if global_failure is not None:
            clean_artifact_root()
            document = build_document(status="complete", success=False)
            save_checkpoint(status="complete", success=False)
            return document

        assert default_config is not None
        warmup_orders = _rotated_warmup_orders(selected_specs, warmups)
        measured_orders = common._balanced_measured_orders(
            selected_specs, repetitions, schedule_seed,
        )
        scheduled_tasks: dict[str, list[tuple[dict[str, Any], BrevisSpec]]] = {
            "archive_pipeline": [],
            "bench_diagnostic": [],
        }
        next_execution_order = 0
        for stage in ("archive_pipeline", "bench_diagnostic"):
            for phase, orders in (("warmup", warmup_orders), ("measured", measured_orders)):
                for index, order in enumerate(orders):
                    for order_within_repetition, spec in enumerate(order):
                        schedule_entry = {
                            "execution_order": next_execution_order,
                            "stage": stage,
                            "phase": phase,
                            "repetition": index,
                            "pair": index // 2 if phase == "measured" else None,
                            "direction": (
                                "forward" if phase == "measured" and index % 2 == 0
                                else "reverse" if phase == "measured"
                                else "rotation"
                            ),
                            "order_within_repetition": order_within_repetition,
                            "configuration": spec.identifier,
                            "status": "pending",
                            "started_at_utc": None,
                            "completed_at_utc": None,
                            "success": None,
                        }
                        execution_schedule.append(schedule_entry)
                        scheduled_tasks[stage].append((schedule_entry, spec))
                        next_execution_order += 1
        if calibration["required"]:
            for phase, count in (("warmup", warmups), ("measured", repetitions)):
                for index in range(count):
                    calibration["execution_schedule"].append({
                        "execution_order": len(calibration["execution_schedule"]),
                        "phase": phase,
                        "repetition": index,
                        "status": "pending",
                        "started_at_utc": None,
                        "completed_at_utc": None,
                        "success": None,
                    })

        # Publish every calibration, archive, and diagnostic task before the
        # first measured process starts. Running/completed states are checkpointed.
        save_checkpoint()

        if calibration["required"]:
            for schedule_entry in calibration["execution_schedule"]:
                phase = str(schedule_entry["phase"])
                index = int(schedule_entry["repetition"])
                destination = calibration["warmups"] if phase == "warmup" else calibration["runs"]
                schedule_entry["status"] = "running"
                schedule_entry["started_at_utc"] = common._utc_now()
                save_checkpoint()
                run_dir = pathlib.Path(tempfile.mkdtemp(
                    prefix=f"calibration-{phase}-{index}-", dir=artifact_root,
                ))
                prior_path = run_dir / "prior.bin"
                record: dict[str, Any] | None = None
                try:
                    record = _run_calibration(
                        binary=requested_binary,
                        source=source_path,
                        source_sha256=source_sha256,
                        prior_path=prior_path,
                        run_dir=run_dir,
                        phase=phase,
                        index=index,
                        jobs=jobs,
                        calibration_tensors=calibration_tensors,
                        search_args=tuple(calibration["configuration"]["search_args"]),
                        expected_search_configuration=calibration_effective_config,
                        timeout_seconds=timeout_seconds,
                        keep_artifacts=keep_artifacts,
                    )
                    if (
                        phase == "measured"
                        and canonical_prior_path is None
                        and record["success"]
                    ):
                        canonical_prior_path = artifact_root / "canonical-prior.bin"
                        common._stage_file(prior_path, canonical_prior_path)
                        canonical_prior_sha256 = common._sha256_file(canonical_prior_path)
                        calibration["canonical_prior"] = {
                            "source_measured_run": index,
                            "path": str(canonical_prior_path) if keep_artifacts else None,
                            "retained": keep_artifacts,
                            "sha256": canonical_prior_sha256,
                            "sha256_after_uses": None,
                            "unchanged_during_benchmark": None,
                            "storage": common._file_allocation(canonical_prior_path),
                        }
                except Exception as exc:
                    message = f"calibration harness error: {type(exc).__name__}: {exc}"
                    record = record or {
                        "status_code": "harness_error",
                        "phase": phase,
                        "index": index,
                        "process": None,
                        "report": None,
                        "prior": None,
                        "success": False,
                        "failure": message,
                    }
                    record["status_code"] = "harness_error"
                    record["success"] = False
                    record["failure"] = message
                assert record is not None
                clean_run_directory(run_dir, record, f"calibration-{phase}-{index}")
                destination.append(record)
                schedule_entry["status"] = "completed"
                schedule_entry["completed_at_utc"] = common._utc_now()
                schedule_entry["success"] = bool(record.get("success"))
                save_checkpoint()

            consistency = _calibration_consistency(calibration["runs"])
            calibration["measured_prior_consistency"] = consistency
            failed_calibrations = [
                run for run in (*calibration["warmups"], *calibration["runs"])
                if not run.get("success")
            ]
            consistency_failed = (
                consistency.get("status_code") != "consistent"
                or consistency.get("byte_consistent") is not True
                or consistency.get("context_counts_consistent") is not True
            )
            calibration["failure_count"] = len(failed_calibrations)
            calibration["success"] = (
                canonical_prior_path is not None
                and not failed_calibrations
                and not consistency_failed
            )
            if canonical_prior_path is None:
                calibration["status_code"] = "canonical_prior_unavailable"
                calibration["failure"] = "no successful measured calibration produced a canonical prior"
            elif failed_calibrations:
                calibration["status_code"] = "iteration_failed"
                calibration["failure"] = (
                    f"{len(failed_calibrations)} calibration warmup/measured run(s) failed"
                )
            elif consistency_failed:
                calibration["status_code"] = "prior_replication_inconsistent"
                calibration["failure"] = (
                    "measured calibration priors are not byte- and context-count consistent; "
                    "semantic equivalence is not assumed"
                )
            else:
                calibration["status_code"] = "ok"

        for schedule_entry, spec in scheduled_tasks["archive_pipeline"]:
            phase = str(schedule_entry["phase"])
            index = int(schedule_entry["repetition"])
            order_within_repetition = int(schedule_entry["order_within_repetition"])
            execution_order = int(schedule_entry["execution_order"])
            configuration = configuration_by_id[spec.identifier]
            destination = configuration["warmups"] if phase == "warmup" else configuration["runs"]
            schedule_entry["status"] = "running"
            schedule_entry["started_at_utc"] = common._utc_now()
            save_checkpoint()
            run_dir = pathlib.Path(tempfile.mkdtemp(
                prefix=f"{_safe_label(spec.identifier)}-{phase}-{index}-archive-",
                dir=artifact_root,
            ))
            try:
                if configuration["failure"] is not None:
                    record = {
                        "status_code": "configuration_unavailable",
                        "phase": phase,
                        "index": index,
                        "execution_order": execution_order,
                        "archive_execution_order": execution_order,
                        "diagnostic_execution_order": None,
                        "order_within_repetition": order_within_repetition,
                        "archive_pipeline_success": False,
                        "diagnostic_success": False,
                        "success": False,
                        "failure": configuration["failure"],
                        "artifact_directory": str(run_dir) if keep_artifacts else None,
                    }
                else:
                    record = _run_archive_pipeline(
                        binary=requested_binary,
                        source=source_path,
                        source_size=source_size,
                        source_sha256=source_sha256,
                        spec=spec,
                        canonical_prior=canonical_prior_path,
                        canonical_prior_sha256=canonical_prior_sha256,
                        run_dir=run_dir,
                        phase=phase,
                        index=index,
                        execution_order=execution_order,
                        order_within_repetition=order_within_repetition,
                        jobs=jobs,
                        timeout_seconds=timeout_seconds,
                        keep_artifacts=keep_artifacts,
                    )
                    _apply_archive_consistency(configuration, record, phase)
            except Exception as exc:
                record = {
                    "status_code": "harness_error",
                    "phase": phase,
                    "index": index,
                    "execution_order": execution_order,
                    "archive_execution_order": execution_order,
                    "diagnostic_execution_order": None,
                    "order_within_repetition": order_within_repetition,
                    "archive_pipeline_success": False,
                    "diagnostic_success": None,
                    "success": False,
                    "failure": f"archive pipeline harness error: {type(exc).__name__}: {exc}",
                    "artifact_directory": str(run_dir) if keep_artifacts else None,
                }
            clean_run_directory(run_dir, record, f"{spec.identifier}-{phase}-{index}-archive")
            destination.append(record)
            schedule_entry["status"] = "completed"
            schedule_entry["completed_at_utc"] = common._utc_now()
            schedule_entry["success"] = bool(record.get("archive_pipeline_success"))
            save_checkpoint()

        # Diagnostic search/encoding replays are deliberately isolated after
        # every timed archive pipeline so they cannot precondition those timings.
        for schedule_entry, spec in scheduled_tasks["bench_diagnostic"]:
            phase = str(schedule_entry["phase"])
            index = int(schedule_entry["repetition"])
            configuration = configuration_by_id[spec.identifier]
            destination = configuration["warmups"] if phase == "warmup" else configuration["runs"]
            record = next((run for run in destination if run.get("index") == index), None)
            if record is None or configuration["failure"] is not None:
                schedule_entry["status"] = "skipped_configuration"
                schedule_entry["completed_at_utc"] = common._utc_now()
                schedule_entry["success"] = False
                save_checkpoint()
                continue
            schedule_entry["status"] = "running"
            schedule_entry["started_at_utc"] = common._utc_now()
            save_checkpoint()
            run_dir = pathlib.Path(tempfile.mkdtemp(
                prefix=f"{_safe_label(spec.identifier)}-{phase}-{index}-diagnostic-",
                dir=artifact_root,
            ))
            archive = record.get("archive")
            actual_archive_bytes = (
                archive.get("storage", {}).get("logical_size_bytes")
                if isinstance(archive, Mapping) else None
            )
            archive_program_evidence = (
                archive.get("program_bytecode_evidence")
                if isinstance(archive, Mapping) else None
            )
            try:
                expected_search_configuration = configuration["effective_search_configuration"]
                if not isinstance(expected_search_configuration, Mapping):
                    raise BenchmarkError("configuration has no effective search probe")
                diagnostic = _run_bench_diagnostic(
                    binary=requested_binary,
                    source=source_path,
                    source_size=source_size,
                    source_sha256=source_sha256,
                    spec=spec,
                    canonical_prior=canonical_prior_path,
                    canonical_prior_sha256=canonical_prior_sha256,
                    expected_search_configuration=expected_search_configuration,
                    actual_archive_bytes=actual_archive_bytes,
                    archive_program_evidence=(
                        archive_program_evidence
                        if isinstance(archive_program_evidence, Mapping) else None
                    ),
                    run_dir=run_dir,
                    jobs=jobs,
                    timeout_seconds=timeout_seconds,
                )
                record.setdefault("commands", {})["bench"] = diagnostic["command"]
                record["bench_process"] = diagnostic["process"]
                record["bench_report"] = diagnostic["report"]
                record["bench_archive_projection_consistency"] = diagnostic[
                    "projection_consistency"
                ]
                record["bench_archive_program_consistency"] = diagnostic[
                    "program_consistency"
                ]
                record["diagnostic_execution_order"] = schedule_entry["execution_order"]
                record["diagnostic_success"] = diagnostic["success"]
                if diagnostic["failure"]:
                    record["failure"] = (
                        f"{record['failure']}; {diagnostic['failure']}"
                        if record.get("failure") else diagnostic["failure"]
                    )
                record["success"] = bool(record.get("archive_pipeline_success")) and bool(
                    diagnostic["success"]
                )
                record["status_code"] = "ok" if record["success"] else "iteration_failed"
            except Exception as exc:
                message = f"diagnostic harness error: {type(exc).__name__}: {exc}"
                record["diagnostic_success"] = False
                record["success"] = False
                record["status_code"] = "harness_error"
                record["failure"] = f"{record['failure']}; {message}" if record.get("failure") else message
            record["diagnostic_artifact_directory"] = str(run_dir) if keep_artifacts else None
            clean_run_directory(
                run_dir, record, f"{spec.identifier}-{phase}-{index}-diagnostic",
                field="diagnostic_cleanup",
            )
            try:
                detail_failure = _apply_bench_detail_compaction(configuration, record)
            except Exception as exc:
                detail_failure = (
                    "bench detail compaction harness error: "
                    f"{type(exc).__name__}: {exc}"
                )
                record.setdefault("bench_report_detail", {
                    "status_code": "compaction_harness_error",
                    "semantic_sha256": None,
                    "canonical_reference": configuration[
                        "bench_report_detail_consistency"
                    ].get("canonical_reference"),
                    "eligible_for_dsl_analysis": False,
                })
            if detail_failure is not None:
                record["diagnostic_success"] = False
                record["success"] = False
                record["status_code"] = "bench_detail_semantic_failure"
                record["failure"] = (
                    f"{record['failure']}; {detail_failure}"
                    if record.get("failure") else detail_failure
                )
            schedule_entry["status"] = "completed"
            schedule_entry["completed_at_utc"] = common._utc_now()
            schedule_entry["success"] = bool(record.get("diagnostic_success"))
            save_checkpoint()

        for configuration in configurations:
            detail_consistency = configuration["bench_report_detail_consistency"]
            _refresh_bench_detail_consistency_status(detail_consistency)
            if detail_consistency["canonical_reference"] is None:
                detail_consistency["status_code"] = "no_successful_canonical_report"
            failures = [
                run for run in (*configuration["warmups"], *configuration["runs"])
                if not run.get("success")
            ]
            if configuration["failure"] is None and failures:
                configuration["status_code"] = "iteration_failed"
                configuration["failure"] = (
                    f"{len(failures)} warmup/measured iteration(s) failed; see raw records"
                )
            elif configuration["failure"] is None:
                configuration["status_code"] = "ok"

        if canonical_prior_path is not None and calibration.get("canonical_prior") is not None:
            try:
                prior_after = common._sha256_file(canonical_prior_path)
            except OSError as exc:
                add_global_failure(
                    f"cannot rehash canonical prior after use: {type(exc).__name__}: {exc}"
                )
            else:
                calibration["canonical_prior"]["sha256_after_uses"] = prior_after
                unchanged = prior_after == canonical_prior_sha256
                calibration["canonical_prior"]["unchanged_during_benchmark"] = unchanged
                if not unchanged:
                    add_global_failure("canonical prior SHA-256 changed during benchmark use")

        resolved_binary = provenance["binary"].get("resolved_path")
        if resolved_binary is not None:
            try:
                after_hash = common._sha256_file(pathlib.Path(resolved_binary))
            except OSError as exc:
                add_global_failure(
                    f"cannot hash Brevis binary after runs: {type(exc).__name__}: {exc}"
                )
            else:
                provenance["binary"]["sha256_after_runs"] = after_hash
                unchanged = after_hash == provenance["binary"]["sha256_before_runs"]
                provenance["binary"]["unchanged_during_benchmark"] = unchanged
                if not unchanged:
                    add_global_failure("Brevis binary SHA-256 changed during the benchmark")

        provenance_failures = _finalize_provenance_checks(
            provenance,
            source_path=source_path,
            source_sha256=source_sha256,
            integrity=integrity,
        )
        if provenance_failures:
            message = "; ".join(provenance_failures)
            add_global_failure(message)

        calibration_iteration_failed = any(
            not run.get("success")
            for run in (*calibration["warmups"], *calibration["runs"])
        )
        calibration_failed = (
            calibration["required"]
            and (
                canonical_prior_path is None
                or calibration_iteration_failed
                or calibration.get("success") is not True
            )
        )
        configurations_failed = any(record["failure"] is not None for record in configurations)
        if calibration_failed:
            add_global_failure(calibration["failure"] or "required calibration failed")
        if configurations_failed:
            failed_ids = [record["id"] for record in configurations if record["failure"] is not None]
            add_global_failure(
                "configuration failures: " + ", ".join(failed_ids)
            )
        clean_artifact_root()
        success = global_failure is None and not calibration_failed and not configurations_failed
        document = build_document(status="complete", success=success)
        save_checkpoint(status="complete", success=success)
        return document
    except BaseException as exc:
        add_global_failure(f"benchmark interrupted: {type(exc).__name__}: {exc}")
        clean_artifact_root()
        try:
            save_checkpoint(status="interrupted", success=False)
        except Exception:
            pass
        raise
    finally:
        if (
            artifact_root is not None
            and not keep_artifacts
            and artifact_root.exists()
        ):
            fallback_cleanup = _cleanup_directory(artifact_root, "artifact-root-finally")
            if not fallback_cleanup["success"]:
                cleanup["artifact_root"] = fallback_cleanup


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--jobs", type=int, required=False, default=1)
    parser.add_argument("--tensors", type=int, default=DEFAULT_CALIBRATION_TENSORS)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--schedule-seed", type=int, default=DEFAULT_SCHEDULE_SEED)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--allow-unverified-input",
        action="store_true",
        help="pilot only: permit a source without a verified manifest binding",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="use an existing binary; its build mode is not verified by this harness",
    )
    parser.add_argument(
        "--configuration", action="append", choices=CORE_CONFIGURATION_IDS,
        help="select a core configuration; repeat to select multiple",
    )
    parser.add_argument("--disk-reserve-fraction", type=float, default=DEFAULT_DISK_RESERVE_FRACTION)
    parser.add_argument("--temp-multiplier", type=float, default=DEFAULT_TEMP_MULTIPLIER)
    parser.add_argument("--model-tag")
    parser.add_argument("--model-repo")
    parser.add_argument("--model-revision")
    parser.add_argument("--manifest")
    parser.add_argument("--shard")
    parser.add_argument("--expected-source-size", type=int)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps({
            "configuration_ids": list(CORE_CONFIGURATION_IDS),
            "raw_terminal_note": (
                "the running binary's non-raw operators are dynamically disabled and recorded"
            ),
        }, indent=2))
        return 0
    if args.source is None or args.output is None:
        parser.error("source and --output are required unless --list is used")
    metadata = {
        key: value
        for key, value in {
            "model_tag": args.model_tag,
            "model_repo": args.model_repo,
            "model_revision": args.model_revision,
            "manifest": args.manifest,
            "shard": args.shard,
        }.items()
        if value is not None
    }
    try:
        document = benchmark_file(
            args.source,
            binary=args.binary,
            warmups=args.warmups,
            repetitions=args.repetitions,
            configuration_ids=args.configuration,
            jobs=args.jobs,
            calibration_tensors=args.tensors,
            work_dir=args.work_dir,
            keep_artifacts=args.keep_artifacts,
            timeout_seconds=args.timeout_seconds,
            schedule_seed=args.schedule_seed,
            input_metadata=metadata,
            expected_source_size_bytes=args.expected_source_size,
            expected_source_sha256=args.expected_source_sha256,
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            checkpoint_path=args.output,
            force_checkpoint=args.force,
            require_declared_integrity=not args.allow_unverified_input,
            require_clean_git=not args.allow_dirty,
            build_binary=not args.skip_build,
            disk_reserve_fraction=args.disk_reserve_fraction,
            temp_multiplier=args.temp_multiplier,
        )
    except (OSError, ValueError, BenchmarkError) as exc:
        print(f"brevis benchmark: {exc}", file=sys.stderr)
        return 2
    return 0 if document.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
