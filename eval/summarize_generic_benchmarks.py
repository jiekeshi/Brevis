#!/usr/bin/env python3
"""Validate and aggregate formal generic-codec campaign results.

Every observation is supplied as a pair: the raw JSON checkpoint written by
``benchmarking.py`` and the adjacent validation receipt written by
``campaign_runner.py``.  This tool deliberately does not accept an unreceipted
benchmark result.  It revalidates the receipt binding, the complete frozen
19-row registry, every warmup and measured iteration, archive determinism,
source identity, and the timing/resource fields used below.

The two bzip2 level-9 spellings remain visible as rows, but are assigned one
independent-observation identifier.  No cross-method average or significance
claim treats those aliases as two independent compressor configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import statistics
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import benchmarking


INPUT_SCHEMA = {"id": benchmarking.SCHEMA_ID, "version": benchmarking.SCHEMA_VERSION}
RECEIPT_SCHEMA = {"id": "brevis.generic-campaign-validation", "version": 1}
OUTPUT_SCHEMA = {"id": "brevis.generic-baseline-aggregate", "version": 1}
TASK_SPEC_ID = "brevis.campaign-task-semantics.v1"
REGISTRY_ID = "all_19_pinned_configurations"
WARMUPS = 1
REPETITIONS = 6
SCHEDULE_SEED = 2701
DISK_RESERVE_FRACTION = 0.30
SHA256_LENGTH = 64

RECEIPT_KEYS = frozenset((
    "schema", "status", "task_semantic_sha256", "task_semantics", "raw_result",
    "receipt_path", "attempt_contract_valid", "result_success", "formal_eligible",
    "postvalidation", "method_outcomes", "disk_gate", "implementation_binding",
    "generic_registry", "exclusive_lock_path", "stage_override_reasons",
    "benchmark_exception",
))
RECEIPT_RAW_KEYS = frozenset((
    "path", "present", "size_bytes", "sha256", "schema", "status",
    "preserved_without_selective_omission",
))
RAW_TOP_KEYS = frozenset((
    "schema", "status", "started_at_utc", "ended_at_utc", "source", "integrity",
    "input_metadata", "configuration", "provenance", "environment", "methods",
))
METHOD_KEYS = frozenset((
    "id", "spec", "available", "resolved_executable", "executable_realpath",
    "executable_sha256", "executable_sha256_after_runs",
    "executable_unchanged_during_benchmark", "version", "version_probe", "warmups",
    "runs", "failure", "status_code", "measured_archive_consistency",
))
ITERATION_KEYS = frozenset((
    "status_code", "started_at_utc", "ended_at_utc", "phase", "index",
    "execution_order", "order_within_repetition", "compression",
    "compressed_size_bytes", "archive_storage", "archive_sha256",
    "archive_hash_error", "archive_consistency", "decompression", "verification",
    "bit_exact", "success", "failure", "staging", "artifact_directory",
))
METHOD_OUTCOME_KEYS = frozenset((
    "id", "spec_matches", "available", "status_ok", "tool_provenance",
    "executable_unchanged",
    "warmup_count", "measured_count", "all_iterations_bit_exact",
    "measured_archive_consistent", "valid",
))
DISK_GATE_KEYS = frozenset((
    "spec_id", "path", "filesystem_total_bytes", "filesystem_free_bytes",
    "reserve_fraction", "reserved_bytes", "source_bytes", "source_multiplier",
    "fixed_bytes", "checkpoint_bytes_per_iteration", "method_count",
    "iterations_per_method", "estimated_work_and_checkpoint_bytes",
    "projected_free_bytes", "passes", "limitation",
))
POST_RUN_DISK_GATE_KEYS = DISK_GATE_KEYS | frozenset((
    "post_run_filesystem_total_bytes", "post_run_free_bytes",
    "post_run_reserved_bytes", "post_run_passes", "post_run_error",
))

METRIC_KEYS = (
    "archive_bytes",
    "compression_ratio_raw_over_archive",
    "saving_fraction",
    "compression_wall_time_seconds",
    "decompression_wall_time_seconds",
    "compression_direct_child_peak_rss_bytes",
    "decompression_direct_child_peak_rss_bytes",
)
PERFORMANCE_METRIC_KEYS = (
    "compression_wall_time_seconds",
    "decompression_wall_time_seconds",
    "compression_direct_child_peak_rss_bytes",
    "decompression_direct_child_peak_rss_bytes",
)
METRIC_UNITS = {
    "archive_bytes": "bytes",
    "compression_ratio_raw_over_archive": "ratio",
    "saving_fraction": "fraction",
    "compression_wall_time_seconds": "seconds",
    "decompression_wall_time_seconds": "seconds",
    "compression_direct_child_peak_rss_bytes": "bytes",
    "decompression_direct_child_peak_rss_bytes": "bytes",
}

ALIAS_GROUPS = {
    "bzip2/default": {
        "independent_observation_id": "bzip2/level-9",
        "alias_group": "bzip2-level-9",
        "alias_role": "representative",
        "alias_of": None,
    },
    "bzip2/ratio": {
        "independent_observation_id": "bzip2/level-9",
        "alias_group": "bzip2-level-9",
        "alias_role": "equivalent_spelling",
        "alias_of": "bzip2/default",
    },
}


class SummaryError(ValueError):
    """Fail-closed input or aggregation error with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _reject_constant(value: str) -> None:
    raise SummaryError("nonfinite_json", f"JSON constant {value!r} is not permitted")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SummaryError("duplicate_json_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SummaryError("noncanonical_json", str(exc)) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_bytes(path: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise SummaryError("unreadable_input", f"cannot read {path}: {exc}") from exc
    return payload, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_json(path: pathlib.Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    payload, identity = _file_bytes(path)
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except SummaryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError("invalid_json", f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SummaryError("invalid_document", f"top-level JSON in {path} must be an object")
    return value, identity


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError("invalid_field", f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SummaryError("invalid_field", f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SummaryError("invalid_field", f"{label} must be nonempty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SummaryError("invalid_field", f"{label} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str) or len(value) != SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SummaryError("invalid_sha256", f"{label} must be a lowercase SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise SummaryError(
            "schema_drift",
            f"{label} keys drifted; missing={sorted(set(expected) - actual)}, "
            f"unsupported={sorted(actual - set(expected))}",
        )


def _same_path(left: Any, right: pathlib.Path, label: str) -> None:
    text = _string(left, label)
    if pathlib.Path(text).resolve(strict=False) != right.resolve(strict=False):
        raise SummaryError("path_binding_mismatch", f"{label} does not name {right}")


def _registry() -> tuple[list[dict[str, Any]], list[str], str]:
    specs = [spec.to_dict() for spec in benchmarking.BASELINE_SPECS]
    identifiers = [spec.identifier for spec in benchmarking.BASELINE_SPECS]
    if len(specs) != 19 or len(set(identifiers)) != 19:
        raise SummaryError("registry_drift", "local generic baseline registry is not 19 unique rows")
    return specs, identifiers, _canonical_sha256(specs)


def _alias_metadata(identifier: str) -> dict[str, Any]:
    if identifier in ALIAS_GROUPS:
        return dict(ALIAS_GROUPS[identifier])
    return {
        "independent_observation_id": identifier,
        "alias_group": None,
        "alias_role": "unique",
        "alias_of": None,
    }


def _stats(values: Sequence[float | int], unit: str) -> dict[str, Any]:
    if not values:
        raise SummaryError("empty_metric", "cannot summarize an empty metric")
    floats = [float(value) for value in values]
    if not all(math.isfinite(value) for value in floats):
        raise SummaryError("nonfinite_metric", "metric contains a non-finite value")
    return {
        "unit": unit,
        "n": len(floats),
        "mean": statistics.fmean(floats),
        "sample_stddev": statistics.stdev(floats) if len(floats) > 1 else None,
        "median": statistics.median(floats),
        "min": min(floats),
        "max": max(floats),
    }


def _validate_argv(
    value: Any, *, executable: str, template: Any, label: str,
) -> None:
    command = _array(value, label)
    arguments = _array(template, f"{label}.template")
    if len(command) != len(arguments) + 1 or command[0] != executable:
        raise SummaryError("command_drift", f"{label} does not use the bound executable/argv")
    for index, (actual, expected) in enumerate(zip(command[1:], arguments, strict=True)):
        if not isinstance(actual, str) or not actual or not isinstance(expected, str):
            raise SummaryError("command_drift", f"{label}[{index + 1}] is invalid")
        if expected in {"{input}", "{output}"}:
            if not pathlib.Path(actual).is_absolute():
                raise SummaryError("command_drift", f"{label}[{index + 1}] is not an absolute artifact path")
        elif "{" in expected or "}" in expected or actual != expected:
            raise SummaryError("command_drift", f"{label}[{index + 1}] drifted from the frozen spec")


def _validate_process(
    value: Any, label: str, *, expected_output_bytes: int, executable: str,
    argv_template: Any, timeout_seconds: Any,
) -> tuple[int, int]:
    process = _mapping(value, label)
    if process.get("status_code") != "ok" or process.get("exit_code") != 0:
        raise SummaryError("iteration_failed", f"{label} process did not exit successfully")
    if process.get("timed_out") is not False or process.get("error") is not None:
        raise SummaryError("iteration_failed", f"{label} timed out or recorded an error")
    if process.get("output_size_bytes") != expected_output_bytes:
        raise SummaryError("output_size_mismatch", f"{label} output size disagrees with record")
    if process.get("timeout_seconds") != timeout_seconds:
        raise SummaryError("configuration_drift", f"{label} timeout differs from the task")
    if not all((
        process.get("process_group_isolated") is True,
        process.get("started_new_session") is True,
    )):
        raise SummaryError("process_isolation_drift", f"{label} lacks process-group isolation")
    _validate_argv(
        process.get("command"), executable=executable, template=argv_template,
        label=f"{label}.command",
    )
    wall_ns = _integer(process.get("wall_time_ns"), f"{label}.wall_time_ns")
    rss_bytes = _integer(
        process.get("direct_child_max_rss_bytes"),
        f"{label}.direct_child_max_rss_bytes",
    )
    storage = _mapping(process.get("output_storage"), f"{label}.output_storage")
    if storage.get("logical_size_bytes") != expected_output_bytes:
        raise SummaryError("output_size_mismatch", f"{label} storage size disagrees")
    return wall_ns, rss_bytes


def _validate_iteration(
    value: Any, *, label: str, phase: str, index: int, source_bytes: int,
    source_sha256: str, expected_spec: Mapping[str, Any], executable: str,
    timeout_seconds: Any,
) -> dict[str, Any]:
    iteration = _mapping(value, label)
    _exact_keys(iteration, ITERATION_KEYS, label)
    if (
        iteration.get("status_code") != "ok" or iteration.get("success") is not True
        or iteration.get("bit_exact") is not True or iteration.get("failure") is not None
        or iteration.get("archive_hash_error") is not None
        or iteration.get("phase") != phase or iteration.get("index") != index
    ):
        raise SummaryError("iteration_failed", f"{label} is not a successful {phase} iteration")
    execution_order = _integer(iteration.get("execution_order"), f"{label}.execution_order")
    order_within = _integer(
        iteration.get("order_within_repetition"), f"{label}.order_within_repetition",
    )
    archive_bytes = _integer(
        iteration.get("compressed_size_bytes"), f"{label}.compressed_size_bytes",
    )
    archive_sha256 = _sha256(iteration.get("archive_sha256"), f"{label}.archive_sha256")
    archive_storage = _mapping(iteration.get("archive_storage"), f"{label}.archive_storage")
    if archive_storage.get("logical_size_bytes") != archive_bytes:
        raise SummaryError("archive_size_mismatch", f"{label} archive storage size disagrees")

    compression_wall, compression_rss = _validate_process(
        iteration.get("compression"), f"{label}.compression",
        expected_output_bytes=archive_bytes, executable=executable,
        argv_template=expected_spec.get("compress_args"), timeout_seconds=timeout_seconds,
    )
    decompression_wall, decompression_rss = _validate_process(
        iteration.get("decompression"), f"{label}.decompression",
        expected_output_bytes=source_bytes, executable=executable,
        argv_template=expected_spec.get("decompress_args"), timeout_seconds=timeout_seconds,
    )
    verification = _mapping(iteration.get("verification"), f"{label}.verification")
    if not all((
        verification.get("attempted") is True,
        verification.get("bit_exact") is True,
        verification.get("source_sha256") == source_sha256,
        verification.get("restored_sha256") == source_sha256,
        verification.get("restored_size_bytes") == source_bytes,
        verification.get("error") is None,
    )):
        raise SummaryError("verification_mismatch", f"{label} is not bit-exact to the bound source")
    consistency = _mapping(
        iteration.get("archive_consistency"), f"{label}.archive_consistency",
    )
    if phase == "warmup":
        if consistency != {"status_code": "not_applicable_warmup", "matches_reference": None}:
            raise SummaryError("archive_consistency_mismatch", f"{label} warmup consistency drifted")
    elif consistency.get("matches_reference") is not True:
        raise SummaryError("archive_consistency_mismatch", f"{label} did not match archive reference")
    staging = _mapping(iteration.get("staging"), f"{label}.staging")
    if staging != {
        "source": "independent_copy", "archive": "independent_copy",
        "excluded_from_timing": True,
    }:
        raise SummaryError("staging_drift", f"{label} did not use excluded independent copies")
    return {
        "phase": phase,
        "repetition": index,
        "execution_order": execution_order,
        "order_within_repetition": order_within,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "compression_wall_time_ns": compression_wall,
        "decompression_wall_time_ns": decompression_wall,
        "compression_direct_child_peak_rss_bytes": compression_rss,
        "decompression_direct_child_peak_rss_bytes": decompression_rss,
    }


def _validate_method(
    value: Any, *, expected_spec: Mapping[str, Any], source_bytes: int,
    source_sha256: str, timeout_seconds: Any, version_probe_timeout_seconds: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identifier = str(expected_spec["id"])
    label = f"method {identifier}"
    method = _mapping(value, label)
    _exact_keys(method, METHOD_KEYS, label)
    if method.get("id") != identifier or method.get("spec") != expected_spec:
        raise SummaryError("registry_drift", f"{label} does not match the frozen registry")
    if not all((
        method.get("available") is True,
        method.get("status_code") == "ok",
        method.get("failure") is None,
        method.get("executable_unchanged_during_benchmark") is True,
    )):
        raise SummaryError("method_failed", f"{label} is unavailable, failed, or changed executable")
    resolved = _string(method.get("resolved_executable"), f"{label}.resolved_executable")
    realpath = _string(method.get("executable_realpath"), f"{label}.executable_realpath")
    if not pathlib.Path(realpath).is_absolute():
        raise SummaryError("tool_provenance_mismatch", f"{label} executable realpath is not absolute")
    before_hash = _sha256(method.get("executable_sha256"), f"{label}.executable_sha256")
    after_hash = _sha256(
        method.get("executable_sha256_after_runs"), f"{label}.executable_sha256_after_runs",
    )
    if before_hash != after_hash:
        raise SummaryError("executable_drift", f"{label} executable hash changed")
    _string(method.get("version"), f"{label}.version")
    version_probe = _mapping(method.get("version_probe"), f"{label}.version_probe")
    if not all((
        version_probe.get("status_code") == "ok",
        version_probe.get("exit_code") == 0,
        version_probe.get("timed_out") is False,
        version_probe.get("error") is None,
        version_probe.get("timeout_seconds") == version_probe_timeout_seconds,
    )):
        raise SummaryError("tool_provenance_mismatch", f"{label} version probe did not pass")
    _validate_argv(
        version_probe.get("command"), executable=resolved,
        template=expected_spec.get("version_args"), label=f"{label}.version_probe.command",
    )

    warmups = _array(method.get("warmups"), f"{label}.warmups")
    runs = _array(method.get("runs"), f"{label}.runs")
    if len(warmups) != WARMUPS or len(runs) != REPETITIONS:
        raise SummaryError("iteration_count", f"{label} must contain exactly 1+6 iterations")
    warmup = _validate_iteration(
        warmups[0], label=f"{label}.warmups[0]", phase="warmup", index=0,
        source_bytes=source_bytes, source_sha256=source_sha256,
        expected_spec=expected_spec, executable=resolved, timeout_seconds=timeout_seconds,
    )
    samples = [
        _validate_iteration(
            run, label=f"{label}.runs[{index}]", phase="measured", index=index,
            source_bytes=source_bytes, source_sha256=source_sha256,
            expected_spec=expected_spec, executable=resolved, timeout_seconds=timeout_seconds,
        )
        for index, run in enumerate(runs)
    ]
    sizes = {sample["archive_bytes"] for sample in samples}
    hashes = {sample["archive_sha256"] for sample in samples}
    if len(sizes) != 1 or len(hashes) != 1:
        raise SummaryError("unstable_archive", f"{label} measured archives are not stable")
    consistency = _mapping(
        method.get("measured_archive_consistency"),
        f"{label}.measured_archive_consistency",
    )
    archive_bytes = next(iter(sizes))
    archive_sha256 = next(iter(hashes))
    if not all((
        consistency.get("status_code") == "consistent",
        consistency.get("reference_size_bytes") == archive_bytes,
        consistency.get("reference_sha256") == archive_sha256,
        consistency.get("consistent_runs") == REPETITIONS,
        consistency.get("inconsistent_runs") == 0,
    )):
        raise SummaryError("unstable_archive", f"{label} archive consistency summary disagrees")
    if identifier == "raw/copy" and archive_bytes != source_bytes:
        raise SummaryError("raw_copy_size_mismatch", "raw/copy must preserve logical source size")

    measured_samples: list[dict[str, Any]] = []
    metric_vectors: dict[str, list[float | int]] = defaultdict(list)
    for sample in samples:
        if sample["archive_bytes"] <= 0:
            raise SummaryError("invalid_archive_size", f"{label} archive must be nonempty")
        ratio = source_bytes / sample["archive_bytes"]
        saving = (source_bytes - sample["archive_bytes"]) / source_bytes
        row = {
            **sample,
            "compression_ratio_raw_over_archive": ratio,
            "saving_fraction": saving,
        }
        measured_samples.append(row)
        for key in METRIC_KEYS:
            if key == "compression_wall_time_seconds":
                value_for_metric: float | int = sample["compression_wall_time_ns"] / 1e9
            elif key == "decompression_wall_time_seconds":
                value_for_metric = sample["decompression_wall_time_ns"] / 1e9
            else:
                value_for_metric = row[key]
            metric_vectors[key].append(value_for_metric)
    metrics = {
        key: _stats(metric_vectors[key], METRIC_UNITS[key]) for key in METRIC_KEYS
    }
    return ({
        "id": identifier,
        "method": expected_spec["method"],
        "profile": expected_spec["profile"],
        **_alias_metadata(identifier),
        "spec": dict(expected_spec),
        "executable": {
            "resolved_path": resolved,
            "realpath": realpath,
            "sha256": before_hash,
            "version": method["version"],
        },
        "storage": {
            "source_bytes": source_bytes,
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha256,
            "compression_ratio_raw_over_archive": source_bytes / archive_bytes,
            "saving_fraction": (source_bytes - archive_bytes) / source_bytes,
        },
        "warmup": warmup,
        "measured_samples": measured_samples,
        "metrics": metrics,
    }, samples)


def _validate_disk_gate(
    value: Any, source_bytes: int, *, require_post_run: bool,
) -> Mapping[str, Any]:
    gate = _mapping(value, "disk_gate")
    _exact_keys(
        gate, POST_RUN_DISK_GATE_KEYS if require_post_run else DISK_GATE_KEYS,
        "disk_gate",
    )
    if not all((
        gate.get("spec_id") == "brevis.generic-disk-gate.v1",
        gate.get("reserve_fraction") == DISK_RESERVE_FRACTION,
        gate.get("source_bytes") == source_bytes,
        gate.get("method_count") == 19,
        gate.get("iterations_per_method") == WARMUPS + REPETITIONS,
        gate.get("passes") is True,
    )):
        raise SummaryError("disk_gate_mismatch", "formal generic disk gate is invalid")
    total = _integer(gate.get("filesystem_total_bytes"), "disk_gate.filesystem_total_bytes", minimum=1)
    free = _integer(gate.get("filesystem_free_bytes"), "disk_gate.filesystem_free_bytes")
    reserved = _integer(gate.get("reserved_bytes"), "disk_gate.reserved_bytes")
    estimate = _integer(
        gate.get("estimated_work_and_checkpoint_bytes"),
        "disk_gate.estimated_work_and_checkpoint_bytes",
    )
    projected = gate.get("projected_free_bytes")
    if isinstance(projected, bool) or not isinstance(projected, int):
        raise SummaryError("disk_gate_mismatch", "disk_gate.projected_free_bytes must be integer")
    expected_estimate = (
        source_bytes * _integer(gate.get("source_multiplier"), "disk_gate.source_multiplier")
        + _integer(gate.get("fixed_bytes"), "disk_gate.fixed_bytes")
        + 19 * 7 * _integer(
            gate.get("checkpoint_bytes_per_iteration"),
            "disk_gate.checkpoint_bytes_per_iteration",
        )
    )
    if not all((
        reserved == math.ceil(total * DISK_RESERVE_FRACTION),
        estimate == expected_estimate,
        projected == free - estimate,
        projected >= reserved,
    )):
        raise SummaryError("disk_gate_mismatch", "disk gate arithmetic is inconsistent")
    _string(gate.get("path"), "disk_gate.path")
    _string(gate.get("limitation"), "disk_gate.limitation")
    if require_post_run:
        post_total = _integer(
            gate.get("post_run_filesystem_total_bytes"),
            "disk_gate.post_run_filesystem_total_bytes", minimum=1,
        )
        post_reserved = _integer(
            gate.get("post_run_reserved_bytes"), "disk_gate.post_run_reserved_bytes",
        )
        if not all((
            post_reserved == math.ceil(post_total * DISK_RESERVE_FRACTION),
            gate.get("post_run_passes") is True,
            gate.get("post_run_error") is None,
        )):
            raise SummaryError("disk_gate_mismatch", "post-run disk gate did not pass")
        post_free = _integer(gate.get("post_run_free_bytes"), "disk_gate.post_run_free_bytes")
        if post_free < post_reserved:
            raise SummaryError("disk_gate_mismatch", "post-run disk reserve is below its bound")
    return gate


def _validate_configuration(
    value: Any, *, raw_path: pathlib.Path, task: Mapping[str, Any], identifiers: list[str],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    config = _mapping(value, "raw.configuration")
    task_timeout = task.get("timeout_seconds_per_process")
    task_probe_timeout = task.get("version_probe_timeout_seconds")
    if not all((
        config.get("warmups") == WARMUPS,
        config.get("repetitions") == REPETITIONS,
        config.get("schedule_seed") == SCHEDULE_SEED,
        config.get("schedule_balance") == "paired",
        config.get("keep_artifacts") is False,
        config.get("artifact_root") is None,
        config.get("timeout_seconds") == task_timeout,
        config.get("version_probe_timeout_seconds") == task_probe_timeout,
        config.get("termination_grace_seconds") == benchmarking.PROCESS_TERMINATION_GRACE_SECONDS,
    )):
        raise SummaryError("configuration_drift", "raw benchmark configuration drifted")
    _same_path(config.get("checkpoint_path"), raw_path, "configuration.checkpoint_path")
    expected_text = {
        "timing_scope": "after stdout/stderr log open, immediately before process spawn, through process reap",
        "verification_scope": "complete post-timing byte scan on every successful decompression",
    }
    for key, expected in expected_text.items():
        if config.get(key) != expected:
            raise SummaryError("configuration_drift", f"configuration.{key} drifted")
    for key, needles in {
        "cache_policy": ("warmups precede measurements", "--reflink=never", "--sparse=never"),
        "io_policy": ("--no-sparse", "--no-asyncio"),
        "scheduling_policy": ("serial execution", "paired forward/reverse"),
    }.items():
        text = _string(config.get(key), f"configuration.{key}")
        if not all(needle in text for needle in needles):
            raise SummaryError("configuration_drift", f"configuration.{key} lost required semantics")

    orders = _array(config.get("measured_orders"), "configuration.measured_orders")
    if len(orders) != REPETITIONS:
        raise SummaryError("schedule_drift", "measured schedule must have six repetitions")
    normalized_orders: list[list[str]] = []
    for index, raw_order in enumerate(orders):
        order = _mapping(raw_order, f"measured_orders[{index}]")
        methods = _array(order.get("methods"), f"measured_orders[{index}].methods")
        if not all((
            order.get("repetition") == index,
            order.get("pair") == index // 2,
            order.get("direction") == ("forward" if index % 2 == 0 else "reverse"),
            len(methods) == 19,
            set(methods) == set(identifiers),
        )):
            raise SummaryError("schedule_drift", f"measured order {index} is invalid")
        normalized_orders.append(list(methods))
        if index % 2 == 1 and normalized_orders[index] != normalized_orders[index - 1][::-1]:
            raise SummaryError("schedule_drift", f"measured pair {index // 2} is not reversed")
    expected_orders = [
        [spec.identifier for spec in order]
        for order in benchmarking._balanced_measured_orders(
            benchmarking.BASELINE_SPECS, REPETITIONS, SCHEDULE_SEED,
        )
    ]
    if normalized_orders != expected_orders:
        raise SummaryError(
            "schedule_seed_drift",
            "measured orders are balanced but do not match the exact seed-2701 schedule",
        )
    schedule = _array(config.get("execution_schedule"), "configuration.execution_schedule")
    expected_schedule: list[dict[str, Any]] = []
    execution_order = 0
    for order_within_repetition, identifier in enumerate(identifiers):
        expected_schedule.append({
            "execution_order": execution_order,
            "phase": "warmup",
            "repetition": 0,
            "order_within_repetition": order_within_repetition,
            "method": identifier,
        })
        execution_order += 1
    for repetition, methods in enumerate(expected_orders):
        for order_within_repetition, identifier in enumerate(methods):
            expected_schedule.append({
                "execution_order": execution_order,
                "phase": "measured",
                "repetition": repetition,
                "order_within_repetition": order_within_repetition,
                "method": identifier,
            })
            execution_order += 1
    if schedule != expected_schedule:
        raise SummaryError(
            "schedule_drift", "execution schedule is not the exact seed-bound 133-entry schedule",
        )
    return config, schedule


def _validate_method_outcomes(
    receipt: Mapping[str, Any], identifiers: list[str], disk_gate: Mapping[str, Any],
) -> None:
    outcomes = _array(receipt.get("method_outcomes"), "receipt.method_outcomes")
    if len(outcomes) != 19:
        raise SummaryError("receipt_method_drift", "receipt must contain 19 method outcomes")
    for index, (value, identifier) in enumerate(zip(outcomes, identifiers, strict=True)):
        outcome = _mapping(value, f"method_outcomes[{index}]")
        _exact_keys(outcome, METHOD_OUTCOME_KEYS, f"method_outcomes[{index}]")
        if not all((
            outcome.get("id") == identifier,
            outcome.get("spec_matches") is True,
            outcome.get("available") is True,
            outcome.get("status_ok") is True,
            outcome.get("tool_provenance") is True,
            outcome.get("executable_unchanged") is True,
            outcome.get("warmup_count") == WARMUPS,
            outcome.get("measured_count") == REPETITIONS,
            outcome.get("all_iterations_bit_exact") is True,
            outcome.get("measured_archive_consistent") is True,
            outcome.get("valid") is True,
        )):
            raise SummaryError("receipt_method_drift", f"receipt outcome {identifier} is not valid")
    post = _mapping(receipt.get("postvalidation"), "receipt.postvalidation")
    if (
        post.get("status_code") != "passed"
        or post.get("failures") != []
        or post.get("checks") is None
    ):
        raise SummaryError("receipt_failed", "receipt postvalidation did not pass cleanly")
    checks = _mapping(post.get("checks"), "receipt.postvalidation.checks")
    preflight_gate = {key: disk_gate[key] for key in DISK_GATE_KEYS}
    if (
        checks.get("methods") != outcomes
        or checks.get("disk_gate") != preflight_gate
        or checks.get("post_run_disk_gate") != disk_gate
    ):
        raise SummaryError("receipt_check_drift", "receipt checks disagree with outcomes or disk gates")
    post_environment = _mapping(
        checks.get("post_run_codec_environment"),
        "receipt.postvalidation.checks.post_run_codec_environment",
    )
    if set(post_environment) != set(benchmarking.CODEC_ENVIRONMENT_VARIABLES):
        raise SummaryError("receipt_check_drift", "post-run codec environment keys drifted")
    for name, value in post_environment.items():
        record = _mapping(value, f"post_run_codec_environment.{name}")
        if record != {"present": False, "value": None}:
            raise SummaryError("codec_environment_present", f"post-run environment contains {name}")
    for key, value in checks.items():
        if key in {
            "methods", "disk_gate", "post_run_disk_gate", "post_run_codec_environment",
        }:
            continue
        if value is not True:
            raise SummaryError("receipt_check_failed", f"receipt check {key!r} is not true")


def _validate_task(
    receipt: Mapping[str, Any], raw: Mapping[str, Any], registry: list[dict[str, Any]],
    registry_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    task = _mapping(receipt.get("task_semantics"), "receipt.task_semantics")
    task_sha = _sha256(
        receipt.get("task_semantic_sha256"), "receipt.task_semantic_sha256",
    )
    if _canonical_sha256(task) != task_sha:
        raise SummaryError("task_hash_mismatch", "receipt task semantics do not hash to task ID")
    expected_ids = [str(spec["id"]) for spec in registry]
    if not all((
        task.get("spec_id") == TASK_SPEC_ID,
        task.get("campaign_id") == "small-generic-codecs-v1",
        task.get("campaign_kind") == "generic_codecs",
        task.get("campaign_stage") == 1,
        task.get("campaign_resource_gate") is None,
        task.get("comparison_group_id") == "generic-codec-comparison",
        task.get("dimension") == {
            "registry": REGISTRY_ID,
            "method_ids": expected_ids,
            "comparison_semantics": (
                "all 19 pinned registry entries share one balanced harness invocation"
            ),
        },
        task.get("comparison_schedule") == {
            "within_group_balanced": True,
            "outer_axis_balanced": True,
            "status_code": "within_group_balanced_outer_axis_balanced",
            "limitation": None,
        },
        task.get("brevis_specs") == [],
        task.get("baseline_specs") == registry,
        task.get("jobs") == 1,
        task.get("calibration_tensors") == 200,
        task.get("warmups") == WARMUPS,
        task.get("repetitions") == REPETITIONS,
        task.get("timeout_seconds_per_process") == 3600,
        task.get("version_probe_timeout_seconds") == 5.0,
        task.get("schedule_seed") == SCHEDULE_SEED,
        task.get("disk_reserve_fraction") == DISK_RESERVE_FRACTION,
    )):
        raise SummaryError("task_contract_drift", "task is not the frozen generic formal contract")
    executor = _mapping(task.get("executor"), "task.executor")
    if not all((
        executor.get("kind") == "generic_baseline_harness",
        executor.get("supported") is True,
        executor.get("status_code") == "ready",
        executor.get("implementation") == "benchmarking.benchmark_file",
    )):
        raise SummaryError("task_contract_drift", "task executor is not the generic harness")
    expected_formal_gates = {
        "require_declared_integrity": True,
        "require_clean_git": True,
        "require_codec_environment_unset": True,
        "require_exact_registry": True,
        "require_bit_exact_all_iterations": True,
        "require_stable_measured_archives": True,
        "require_unchanged_executables": True,
        "force_checkpoint": False,
    }
    if executor.get("formal_gates") != expected_formal_gates:
        raise SummaryError("formal_gate_drift", "task generic formal gates drifted")
    lock_policy = _mapping(
        executor.get("exclusive_execution_policy"),
        "task.executor.exclusive_execution_policy",
    )
    expected_lock_path = "/tmp/brevis-machine-benchmark.lock"
    if not all((
        lock_policy.get("policy_id") == "brevis.machine-advisory-execution-lock.v1",
        lock_policy.get("required") is True,
        lock_policy.get("backend") == "fcntl.flock",
        lock_policy.get("acquisition") == "exclusive_nonblocking",
        lock_policy.get("lock_path") == expected_lock_path,
    )):
        raise SummaryError("lock_policy_drift", "task exclusive execution policy drifted")
    if receipt.get("exclusive_lock_path") != lock_policy.get("lock_path"):
        raise SummaryError("lock_path_mismatch", "receipt lock path disagrees with task policy")
    model = _mapping(task.get("model"), "task.model")
    task_input = _mapping(task.get("input"), "task.input")
    _string(model.get("tag"), "task.model.tag")
    _string(model.get("repo"), "task.model.repo")
    _string(model.get("revision"), "task.model.revision")
    _string(task_input.get("file"), "task.input.file")
    _integer(task_input.get("bytes"), "task.input.bytes", minimum=1)
    _sha256(task_input.get("sha256"), "task.input.sha256")
    _sha256(task.get("experiment_matrix_sha256"), "task.experiment_matrix_sha256")
    _sha256(task.get("model_manifest_sha256"), "task.model_manifest_sha256")
    implementation = _mapping(task.get("implementation_binding"), "task.implementation_binding")
    if receipt.get("implementation_binding") != implementation:
        raise SummaryError("implementation_binding_mismatch", "receipt implementation binding drifted")
    for key in (
        "campaign_runner_sha256", "brevis_benchmarking_sha256",
        "generic_benchmarking_sha256",
    ):
        _sha256(implementation.get(key), f"implementation.{key}")
    commit = _string(implementation.get("git_commit"), "implementation.git_commit")
    if len(commit) not in range(40, 65) or any(c not in "0123456789abcdef" for c in commit):
        raise SummaryError("invalid_commit", "implementation.git_commit is not canonical hexadecimal")

    metadata = _mapping(raw.get("input_metadata"), "raw.input_metadata")
    expected_metadata = {
        "model_tag": model.get("tag"),
        "model_repo": model.get("repo"),
        "model_revision": model.get("revision"),
        "shard": task_input.get("file"),
        "campaign_id": task.get("campaign_id"),
        "campaign_comparison_group_id": task.get("comparison_group_id"),
        "campaign_task_semantic_sha256": task_sha,
        "campaign_experiment_matrix_sha256": task.get("experiment_matrix_sha256"),
        "campaign_runner_sha256": implementation.get("campaign_runner_sha256"),
        "generic_benchmarking_sha256": implementation.get("generic_benchmarking_sha256"),
        "generic_registry_sha256": registry_sha256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise SummaryError("metadata_binding_mismatch", f"raw metadata {key!r} drifted")
    if set(metadata) != set(expected_metadata) | {"manifest", "campaign_disk_gate"}:
        raise SummaryError("schema_drift", "raw input_metadata fields drifted")
    _string(metadata.get("manifest"), "raw.input_metadata.manifest")
    return task, model, task_input


def _validate_source_and_provenance(
    raw: Mapping[str, Any], *, task: Mapping[str, Any], model: Mapping[str, Any],
    task_input: Mapping[str, Any], receipt: Mapping[str, Any], raw_path: pathlib.Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = _mapping(raw.get("source"), "raw.source")
    source_path = pathlib.Path(_string(source.get("path"), "raw.source.path"))
    shard_parts = pathlib.PurePosixPath(str(task_input.get("file"))).parts
    source_has_shard_suffix = (
        bool(shard_parts) and len(source_path.parts) >= len(shard_parts)
        and source_path.parts[-len(shard_parts):] == shard_parts
    )
    if not all((
        source.get("size_bytes") == task_input.get("bytes"),
        source.get("sha256") == task_input.get("sha256"),
        source_has_shard_suffix,
    )):
        raise SummaryError("source_identity_mismatch", "raw source disagrees with task identity")
    integrity = _mapping(raw.get("integrity"), "raw.integrity")
    source_integrity = _mapping(integrity.get("source"), "integrity.source")
    manifest = _mapping(integrity.get("manifest"), "integrity.manifest")
    binding = _mapping(integrity.get("manifest_binding"), "integrity.manifest_binding")
    if not all((
        integrity.get("model_metadata_declared") is True,
        source_integrity.get("verified") is True,
        source_integrity.get("expected_size_bytes") == task_input.get("bytes"),
        source_integrity.get("actual_size_bytes") == task_input.get("bytes"),
        source_integrity.get("expected_sha256") == task_input.get("sha256"),
        source_integrity.get("actual_sha256") == task_input.get("sha256"),
        manifest.get("verified") is True,
        manifest.get("expected_sha256") == task.get("model_manifest_sha256"),
        manifest.get("actual_sha256") == task.get("model_manifest_sha256"),
        binding.get("verified") is True,
        binding.get("tag") == model.get("tag"),
        binding.get("repo") == model.get("repo"),
        binding.get("revision") == model.get("revision"),
        binding.get("shard") == task_input.get("file"),
        binding.get("bytes") == task_input.get("bytes"),
        binding.get("sha256") == task_input.get("sha256"),
    )):
        raise SummaryError("source_identity_mismatch", "source/manifest integrity binding drifted")
    metadata = _mapping(raw.get("input_metadata"), "raw.input_metadata")
    if metadata.get("manifest") != manifest.get("path"):
        raise SummaryError("manifest_path_mismatch", "manifest path differs across raw records")

    provenance = _mapping(raw.get("provenance"), "raw.provenance")
    git = _mapping(provenance.get("git"), "raw.provenance.git")
    script = _mapping(provenance.get("benchmark_script"), "raw.provenance.benchmark_script")
    implementation = _mapping(receipt.get("implementation_binding"), "receipt.implementation")
    if not all((
        git.get("commit") == implementation.get("git_commit"),
        git.get("dirty") is False,
        script.get("sha256") == implementation.get("generic_benchmarking_sha256"),
    )):
        raise SummaryError("provenance_mismatch", "raw implementation provenance drifted")
    environment = _mapping(raw.get("environment"), "raw.environment")
    codec_environment = _mapping(
        environment.get("codec_environment_variables"),
        "raw.environment.codec_environment_variables",
    )
    if (
        set(codec_environment) != set(benchmarking.CODEC_ENVIRONMENT_VARIABLES)
        or any(codec_environment[name] is not None for name in benchmarking.CODEC_ENVIRONMENT_VARIABLES)
    ):
        raise SummaryError("codec_environment_present", "codec-affecting environment was not unset")
    checkpoint = _mapping(environment.get("filesystems"), "environment.filesystems").get("checkpoint")
    if checkpoint is None:
        raise SummaryError("missing_checkpoint_environment", "checkpoint filesystem provenance is absent")
    return source, environment


def _environment_fingerprint(environment: Mapping[str, Any]) -> dict[str, Any]:
    cpu = _mapping(environment.get("cpu"), "environment.cpu")
    _exact_keys(cpu, frozenset((
        "model", "logical_cores_host", "physical_cores_host",
        "affinity_logical_cores", "quota",
    )), "environment.cpu")
    quota = _mapping(cpu.get("quota"), "environment.cpu.quota")
    _exact_keys(quota, frozenset((
        "source", "quota_microseconds", "period_microseconds", "quota_cores",
    )), "environment.cpu.quota")
    memory = _mapping(environment.get("memory"), "environment.memory")
    _exact_keys(memory, frozenset((
        "total_bytes", "available_bytes_at_start", "cgroup_limit_bytes",
        "cgroup_limit_source",
    )), "environment.memory")
    return {
        "system": _string(environment.get("system"), "environment.system"),
        "release": _string(environment.get("release"), "environment.release"),
        "machine": _string(environment.get("machine"), "environment.machine"),
        "hostname": _string(environment.get("hostname"), "environment.hostname"),
        "cpu": {
            "model": cpu.get("model"),
            "logical_cores_host": cpu.get("logical_cores_host"),
            "physical_cores_host": cpu.get("physical_cores_host"),
            "affinity_logical_cores": cpu.get("affinity_logical_cores"),
            "quota": {
                "source": quota.get("source"),
                "quota_microseconds": quota.get("quota_microseconds"),
                "period_microseconds": quota.get("period_microseconds"),
                "quota_cores": quota.get("quota_cores"),
            },
        },
        "memory": {
            "cgroup_limit_bytes": memory.get("cgroup_limit_bytes"),
            "cgroup_limit_source": memory.get("cgroup_limit_source"),
        },
    }


def _schedule_tuple(value: Any, label: str) -> tuple[int, str, int, int, str]:
    row = _mapping(value, label)
    return (
        _integer(row.get("execution_order"), f"{label}.execution_order"),
        _string(row.get("phase"), f"{label}.phase"),
        _integer(row.get("repetition"), f"{label}.repetition"),
        _integer(row.get("order_within_repetition"), f"{label}.order_within_repetition"),
        _string(row.get("method"), f"{label}.method"),
    )


def _consume_pair(raw_argument: os.PathLike[str] | str, receipt_argument: os.PathLike[str] | str) -> dict[str, Any]:
    try:
        raw_path = pathlib.Path(raw_argument).resolve(strict=True)
        receipt_path = pathlib.Path(receipt_argument).resolve(strict=True)
    except OSError as exc:
        raise SummaryError("missing_input", f"cannot resolve raw/receipt pair: {exc}") from exc
    if not raw_path.is_file() or not receipt_path.is_file():
        raise SummaryError("missing_input", "raw result and receipt must be regular files")
    raw, raw_identity = _load_json(raw_path)
    receipt, receipt_identity = _load_json(receipt_path)
    _exact_keys(raw, RAW_TOP_KEYS, "raw result")
    _exact_keys(receipt, RECEIPT_KEYS, "validation receipt")
    if raw.get("schema") != INPUT_SCHEMA or raw.get("status") != "complete":
        raise SummaryError("raw_schema", "raw result is not complete generic benchmark schema 2")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != "complete":
        raise SummaryError("receipt_schema", "validation receipt is not complete schema 1")
    if not all((
        receipt.get("attempt_contract_valid") is True,
        receipt.get("result_success") is True,
        receipt.get("formal_eligible") is True,
        receipt.get("stage_override_reasons") == [],
        receipt.get("benchmark_exception") is None,
    )):
        raise SummaryError("receipt_ineligible", "validation receipt is not unconditionally formal")
    _same_path(receipt.get("receipt_path"), receipt_path, "receipt.receipt_path")
    raw_binding = _mapping(receipt.get("raw_result"), "receipt.raw_result")
    _exact_keys(raw_binding, RECEIPT_RAW_KEYS, "receipt.raw_result")
    _same_path(raw_binding.get("path"), raw_path, "receipt.raw_result.path")
    if not all((
        raw_binding.get("present") is True,
        raw_binding.get("size_bytes") == raw_identity["bytes"],
        raw_binding.get("sha256") == raw_identity["sha256"],
        raw_binding.get("schema") == INPUT_SCHEMA,
        raw_binding.get("status") == "complete",
        raw_binding.get("preserved_without_selective_omission") is True,
    )):
        raise SummaryError("raw_receipt_mismatch", "receipt does not bind exact raw bytes/schema/status")

    registry, identifiers, registry_sha256 = _registry()
    registry_receipt = _mapping(receipt.get("generic_registry"), "receipt.generic_registry")
    if registry_receipt != {"id": REGISTRY_ID, "count": 19, "sha256": registry_sha256}:
        raise SummaryError("registry_drift", "receipt registry identity drifted")
    task, model, task_input = _validate_task(receipt, raw, registry, registry_sha256)
    source, environment = _validate_source_and_provenance(
        raw, task=task, model=model, task_input=task_input, receipt=receipt,
        raw_path=raw_path,
    )
    source_bytes = int(source["size_bytes"])
    source_sha = str(source["sha256"])
    disk_gate = _validate_disk_gate(
        receipt.get("disk_gate"), source_bytes, require_post_run=True,
    )
    preflight_disk_gate = _validate_disk_gate(
        raw["input_metadata"].get("campaign_disk_gate"), source_bytes,
        require_post_run=False,
    )
    if preflight_disk_gate != {key: disk_gate[key] for key in DISK_GATE_KEYS}:
        raise SummaryError("disk_gate_mismatch", "raw metadata and receipt disk gates differ")
    _same_path(disk_gate.get("path"), raw_path.parent, "disk_gate.path")
    _validate_method_outcomes(receipt, identifiers, disk_gate)
    config, execution_schedule = _validate_configuration(
        raw.get("configuration"), raw_path=raw_path, task=task, identifiers=identifiers,
    )

    methods_raw = _array(raw.get("methods"), "raw.methods")
    if len(methods_raw) != 19:
        raise SummaryError("method_count", "raw result must contain exactly 19 methods")
    methods: list[dict[str, Any]] = []
    observed_schedule: list[tuple[int, str, int, int, str]] = []
    for index, expected_spec in enumerate(registry):
        method, _ = _validate_method(
            methods_raw[index], expected_spec=expected_spec,
            source_bytes=source_bytes, source_sha256=source_sha,
            timeout_seconds=config.get("timeout_seconds"),
            version_probe_timeout_seconds=config.get("version_probe_timeout_seconds"),
        )
        methods.append(method)
        for sample in (method["warmup"], *method["measured_samples"]):
            observed_schedule.append((
                sample["execution_order"], sample["phase"], sample["repetition"],
                sample["order_within_repetition"], method["id"],
            ))
    declared_schedule = [
        _schedule_tuple(row, f"execution_schedule[{index}]")
        for index, row in enumerate(execution_schedule)
    ]
    observed_schedule.sort()
    declared_schedule.sort()
    if declared_schedule != observed_schedule or [row[0] for row in observed_schedule] != list(range(133)):
        raise SummaryError("schedule_drift", "execution schedule and iteration records disagree")

    by_id = {method["id"]: method for method in methods}
    default = by_id["bzip2/default"]
    ratio = by_id["bzip2/ratio"]
    for key in ("archive_bytes", "archive_sha256"):
        left = [sample[key] for sample in default["measured_samples"]]
        right = [sample[key] for sample in ratio["measured_samples"]]
        if left != right:
            raise SummaryError(
                "alias_drift",
                "bzip2/default (-9) and bzip2/ratio (--best) did not produce identical archives",
            )
    if default["executable"]["sha256"] != ratio["executable"]["sha256"]:
        raise SummaryError("alias_drift", "bzip2 aliases used different executable bytes")

    return {
        "task_semantic_sha256": receipt["task_semantic_sha256"],
        "raw_result": raw_identity,
        "validation_receipt": receipt_identity,
        "model": {
            "tag": model["tag"], "repo": model["repo"],
            "revision": model["revision"], "scope": model.get("scope"),
        },
        "source": {
            "path": source["path"], "file": task_input["file"],
            "bytes": source_bytes, "sha256": source_sha,
        },
        "manifest": {
            "path": raw["input_metadata"]["manifest"],
            "sha256": task["model_manifest_sha256"],
        },
        "experiment_matrix_sha256": task["experiment_matrix_sha256"],
        "campaign": {
            "id": task["campaign_id"], "kind": task["campaign_kind"],
            "stage": task.get("campaign_stage"),
            "resource_gate": task.get("campaign_resource_gate"),
            "comparison_group_id": task["comparison_group_id"],
        },
        "critical_configuration": {
            "warmups": config["warmups"],
            "repetitions": config["repetitions"],
            "schedule_seed": config["schedule_seed"],
            "timeout_seconds": config["timeout_seconds"],
            "version_probe_timeout_seconds": config["version_probe_timeout_seconds"],
            "registry_sha256": registry_sha256,
        },
        "configuration": dict(config),
        "implementation_binding": dict(receipt["implementation_binding"]),
        "environment": dict(environment),
        "environment_fingerprint": _environment_fingerprint(environment),
        "toolchain_fingerprint": [{
            "id": method["id"],
            "realpath": method["executable"]["realpath"],
            "sha256": method["executable"]["sha256"],
            "version": method["executable"]["version"],
        } for method in methods],
        "methods": methods,
    }


def _weighted(values: Sequence[tuple[float, float]], label: str) -> float:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        raise SummaryError("invalid_weight", f"{label} has no positive weight")
    return sum(value * weight for value, weight in values) / total_weight


def _view(
    rows: Sequence[Mapping[str, Any]], weights: Sequence[float], label: str,
    metric_keys: Sequence[str],
) -> dict[str, Any]:
    if len(rows) != len(weights) or not rows:
        raise SummaryError("invalid_weight", f"{label} rows and weights are inconsistent")
    return {
        "observation_count": len(rows),
        "weight_sum": float(sum(weights)),
        "weighted_point_estimates": {
            key: _weighted(
                [(float(row["metrics"][key]["mean"]), float(weight))
                 for row, weight in zip(rows, weights, strict=True)],
                f"{label}.{key}",
            )
            for key in metric_keys
        },
    }


def _pooled_storage(inputs: Sequence[Mapping[str, Any]], identifier: str) -> dict[str, Any]:
    raw_bytes = sum(int(row["source"]["bytes"]) for row in inputs)
    archive_bytes = 0
    for row in inputs:
        method = next(method for method in row["methods"] if method["id"] == identifier)
        archive_bytes += int(method["storage"]["archive_bytes"])
    return {
        "raw_bytes": raw_bytes,
        "archive_bytes": archive_bytes,
        "compression_ratio_raw_over_archive": raw_bytes / archive_bytes,
        "saving_fraction": (raw_bytes - archive_bytes) / raw_bytes,
    }


def _serial_complete_model_performance(
    inputs: Sequence[Mapping[str, Any]], methods: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(inputs) != len(methods) or not inputs:
        raise SummaryError("invalid_model_performance", "model inputs and method rows disagree")
    raw_bytes = sum(int(row["source"]["bytes"]) for row in inputs)
    compression_seconds = sum(
        float(method["metrics"]["compression_wall_time_seconds"]["mean"])
        for method in methods
    )
    decompression_seconds = sum(
        float(method["metrics"]["decompression_wall_time_seconds"]["mean"])
        for method in methods
    )
    if compression_seconds <= 0 or decompression_seconds <= 0:
        raise SummaryError(
            "invalid_model_performance", "serial expected wall time must be positive",
        )
    compression_peak = max(
        int(sample["compression_direct_child_peak_rss_bytes"])
        for method in methods for sample in method["measured_samples"]
    )
    decompression_peak = max(
        int(sample["decompression_direct_child_peak_rss_bytes"])
        for method in methods for sample in method["measured_samples"]
    )
    return {
        "execution_model": "all model shards processed sequentially",
        "shard_count": len(inputs),
        "raw_bytes": raw_bytes,
        "compression": {
            "expected_wall_seconds": compression_seconds,
            "throughput_raw_bytes_per_second": raw_bytes / compression_seconds,
            "peak_rss_bytes": compression_peak,
        },
        "decompression": {
            "expected_wall_seconds": decompression_seconds,
            "throughput_raw_bytes_per_second": raw_bytes / decompression_seconds,
            "peak_rss_bytes": decompression_peak,
        },
        "dispersion": {
            "combined_sample_stddev": None,
            "status": "not_computed_unpaired_shard_repetitions",
            "reason": (
                "measured repetitions are paired within a shard, not across independently "
                "executed shards; summing shard variances would assert an unsupported pairing"
            ),
        },
    }


def _aggregate_views(inputs: Sequence[Mapping[str, Any]], identifiers: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in inputs:
        model = row["model"]
        grouped[(model["tag"], model["repo"], model["revision"])].append(row)
    per_model: list[dict[str, Any]] = []
    for key, model_inputs in sorted(grouped.items()):
        method_rows = []
        for identifier in identifiers:
            observations = [
                next(method for method in row["methods"] if method["id"] == identifier)
                for row in model_inputs
            ]
            pooled_storage = _pooled_storage(model_inputs, identifier)
            serial_performance = _serial_complete_model_performance(
                model_inputs, observations,
            )
            method_rows.append({
                "id": identifier,
                **_alias_metadata(identifier),
                "serial_complete_model_performance": serial_performance,
                "descriptive_shard_mean_views": {
                    "warning": (
                        "weighted averages of per-shard means; these are not complete-model "
                        "latency or complete-model peak RSS"
                    ),
                    "input_equal": _view(
                        observations, [1.0] * len(observations), "input_equal",
                        PERFORMANCE_METRIC_KEYS,
                    ),
                    "raw_byte_weighted": _view(
                        observations,
                        [float(row["source"]["bytes"]) for row in model_inputs],
                        "raw_byte_weighted",
                        PERFORMANCE_METRIC_KEYS,
                    ),
                },
                "pooled_storage": pooled_storage,
            })
        per_model.append({
            "model": {"tag": key[0], "repo": key[1], "revision": key[2]},
            "input_count": len(model_inputs),
            "total_raw_bytes": sum(int(row["source"]["bytes"]) for row in model_inputs),
            "methods": method_rows,
        })

    cross_methods = []
    for identifier in identifiers:
        all_observations = [
            next(method for method in row["methods"] if method["id"] == identifier)
            for row in inputs
        ]
        storage_model_points: list[dict[str, Any]] = []
        performance_model_points: list[dict[str, Any]] = []
        for model_row in per_model:
            method_row = next(row for row in model_row["methods"] if row["id"] == identifier)
            serial = method_row["serial_complete_model_performance"]
            performance_model_points.append({
                "model": dict(model_row["model"]),
                "raw_bytes": serial["raw_bytes"],
                "compression_expected_wall_seconds": serial["compression"][
                    "expected_wall_seconds"
                ],
                "decompression_expected_wall_seconds": serial["decompression"][
                    "expected_wall_seconds"
                ],
                "compression_throughput_raw_bytes_per_second": serial["compression"][
                    "throughput_raw_bytes_per_second"
                ],
                "decompression_throughput_raw_bytes_per_second": serial["decompression"][
                    "throughput_raw_bytes_per_second"
                ],
                "compression_peak_rss_bytes": serial["compression"]["peak_rss_bytes"],
                "decompression_peak_rss_bytes": serial["decompression"]["peak_rss_bytes"],
            })
            storage_model_points.append({
                "model": dict(model_row["model"]),
                **dict(method_row["pooled_storage"]),
            })
        model_equal_storage = {
            key: statistics.fmean(float(point[key]) for point in storage_model_points)
            for key in (
                "raw_bytes", "archive_bytes", "compression_ratio_raw_over_archive",
                "saving_fraction",
            )
        }
        pooled_all_inputs = _pooled_storage(inputs, identifier)
        model_equal_performance = {
            key: statistics.fmean(float(point[key]) for point in performance_model_points)
            for key in (
                "compression_throughput_raw_bytes_per_second",
                "decompression_throughput_raw_bytes_per_second",
                "compression_peak_rss_bytes", "decompression_peak_rss_bytes",
            )
        }
        total_raw_bytes = sum(int(point["raw_bytes"]) for point in performance_model_points)
        total_compression_seconds = sum(
            float(point["compression_expected_wall_seconds"])
            for point in performance_model_points
        )
        total_decompression_seconds = sum(
            float(point["decompression_expected_wall_seconds"])
            for point in performance_model_points
        )
        serial_global_performance = {
            "execution_model": "all model shards and models processed sequentially",
            "raw_bytes": total_raw_bytes,
            "compression": {
                "expected_wall_seconds": total_compression_seconds,
                "throughput_raw_bytes_per_second": total_raw_bytes / total_compression_seconds,
                "peak_rss_bytes": max(
                    int(point["compression_peak_rss_bytes"])
                    for point in performance_model_points
                ),
            },
            "decompression": {
                "expected_wall_seconds": total_decompression_seconds,
                "throughput_raw_bytes_per_second": total_raw_bytes / total_decompression_seconds,
                "peak_rss_bytes": max(
                    int(point["decompression_peak_rss_bytes"])
                    for point in performance_model_points
                ),
            },
            "combined_sample_stddev": None,
            "dispersion_status": "not_computed_unpaired_shard_and_model_repetitions",
        }
        cross_methods.append({
            "id": identifier,
            **_alias_metadata(identifier),
            "views": {
                "model_equal": {
                    "weighting_unit": "one model",
                    "storage": {
                        "point_definition": (
                            "each model point pools shard bytes first; ratio and saving are then "
                            "derived from sum(raw_bytes)/sum(archive_bytes)"
                        ),
                        "model_points": storage_model_points,
                        "point_estimates": model_equal_storage,
                    },
                    "performance": {
                        "point_definition": (
                            "one serial-complete-model throughput and observed sequential peak-RSS "
                            "point per model; models receive equal weight"
                        ),
                        "model_points": performance_model_points,
                        "point_estimates": model_equal_performance,
                    },
                },
                "raw_byte_weighted": {
                    "weighting_unit": "source logical byte across inputs",
                    "storage": {
                        "point_definition": "all input bytes are pooled before deriving ratio and saving",
                        "pooled_storage": pooled_all_inputs,
                    },
                    "performance": serial_global_performance,
                },
            },
            "pooled_storage": pooled_all_inputs,
            "descriptive_shard_mean_view": {
                "warning": (
                    "raw-byte-weighted shard means are descriptive only; they are neither "
                    "complete-model total latency nor sequential peak RSS"
                ),
                "raw_byte_weighted": _view(
                    all_observations,
                    [float(row["source"]["bytes"]) for row in inputs],
                    "cross_descriptive_shard_mean", PERFORMANCE_METRIC_KEYS,
                ),
            },
        })
    return per_model, {
        "model_count": len(per_model),
        "input_count": len(inputs),
        "total_raw_bytes": sum(int(row["source"]["bytes"]) for row in inputs),
        "methods": cross_methods,
    }


def aggregate_pairs(
    pairs: Sequence[tuple[os.PathLike[str] | str, os.PathLike[str] | str]],
) -> dict[str, Any]:
    if not pairs:
        raise SummaryError("missing_input", "at least one raw/receipt pair is required")
    resolved_pairs = [
        (pathlib.Path(raw).resolve(strict=False), pathlib.Path(receipt).resolve(strict=False))
        for raw, receipt in pairs
    ]
    if len(resolved_pairs) != len(set(resolved_pairs)):
        raise SummaryError("duplicate_input", "the same raw/receipt pair was supplied more than once")
    inputs = [_consume_pair(raw, receipt) for raw, receipt in resolved_pairs]
    inputs.sort(key=lambda row: str(row["task_semantic_sha256"]))
    logical_sources: dict[tuple[str, str, str], str] = {}
    for row in inputs:
        logical_key = (
            str(row["model"]["repo"]), str(row["model"]["revision"]),
            str(row["source"]["file"]),
        )
        source_sha256 = str(row["source"]["sha256"])
        if logical_key in logical_sources:
            if logical_sources[logical_key] == source_sha256:
                raise SummaryError(
                    "duplicate_observation",
                    "the same (repo, revision, file) source was supplied more than once",
                )
            raise SummaryError(
                "source_identity_conflict",
                "the same (repo, revision, file) was supplied with conflicting SHA-256 identities",
            )
        logical_sources[logical_key] = source_sha256
    tasks = [row["task_semantic_sha256"] for row in inputs]
    if len(tasks) != len(set(tasks)):
        raise SummaryError("duplicate_task", "a formal campaign task was supplied more than once")

    def homogeneous_binding(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "implementation_binding": row["implementation_binding"],
            "experiment_matrix_sha256": row["experiment_matrix_sha256"],
            "model_manifest_sha256": row["manifest"]["sha256"],
            "campaign": row["campaign"],
            "critical_configuration": row["critical_configuration"],
            "toolchain_fingerprint": row["toolchain_fingerprint"],
            "environment_fingerprint": row["environment_fingerprint"],
        }

    common_binding = homogeneous_binding(inputs[0])
    for index, row in enumerate(inputs[1:], start=1):
        candidate = homogeneous_binding(row)
        for key, expected in common_binding.items():
            if candidate.get(key) != expected:
                raise SummaryError(
                    "heterogeneous_input",
                    f"input {index} differs in performance-fairness binding {key!r}",
                )

    registry, identifiers, registry_sha = _registry()
    per_model, cross_model = _aggregate_views(inputs, identifiers)
    fingerprint_projection = {
        "schema": OUTPUT_SCHEMA,
        "registry_sha256": registry_sha,
        "pairs": [{
            "task_semantic_sha256": row["task_semantic_sha256"],
            "raw_sha256": row["raw_result"]["sha256"],
            "receipt_sha256": row["validation_receipt"]["sha256"],
        } for row in inputs],
    }
    tool_payload, tool_identity = _file_bytes(pathlib.Path(__file__).resolve())
    del tool_payload
    independent_ids = {
        _alias_metadata(identifier)["independent_observation_id"] for identifier in identifiers
    }
    return {
        "schema": OUTPUT_SCHEMA,
        "aggregate_input_fingerprint_sha256": _canonical_sha256(fingerprint_projection),
        "tool": {
            "path": str(pathlib.Path(__file__).resolve()),
            "bytes": tool_identity["bytes"],
            "sha256": tool_identity["sha256"],
            "hash_scope": "exact script bytes",
        },
        "scope": {
            "formal_eligible_only": True,
            "input_count": len(inputs),
            "model_count": len(per_model),
            "registry_row_count": len(identifiers),
            "independent_configuration_count": len(independent_ids),
            "warmups_per_method_per_input": WARMUPS,
            "measured_repetitions_per_method_per_input": REPETITIONS,
        },
        "homogeneous_execution_binding": {
            **common_binding,
            "sha256": _canonical_sha256(common_binding),
            "policy": (
                "all aggregated pairs share one implementation, matrix, manifest, campaign, "
                "critical configuration, codec toolchain, and stable host/resource fingerprint"
            ),
        },
        "registry": {
            "id": REGISTRY_ID,
            "sha256": registry_sha,
            "rows": [{**spec, **_alias_metadata(str(spec["id"]))} for spec in registry],
            "alias_policy": {
                "bzip2-level-9": {
                    "members": ["bzip2/default", "bzip2/ratio"],
                    "representative": "bzip2/default",
                    "reason": "bzip2 -9 and --best are equivalent level-9 spellings",
                    "count_as_independent_configurations": 1,
                },
            },
        },
        "metric_definitions": {
            "archive_bytes": "logical archive bytes; all six measured archives must agree",
            "compression_ratio_raw_over_archive": "source logical bytes divided by archive logical bytes",
            "saving_fraction": "(source logical bytes - archive logical bytes) / source logical bytes",
            "wall_time": "process spawn through direct-child reap; staging, hashing, and verification excluded",
            "serial_complete_model_wall_time": (
                "sum of per-shard measured mean wall times under sequential shard execution"
            ),
            "throughput": "sum(source bytes) divided by serial expected wall seconds",
            "peak_rss": (
                "maximum direct-child ru_maxrss across measured shard trials under sequential "
                "execution; descendant RSS is not aggregated"
            ),
            "dispersion": (
                "per-input sample_stddev uses n-1; no complete-model SD is formed because "
                "technical repetitions are not paired across shards"
            ),
        },
        "weighting_policy": {
            "per_input_metrics": "six measured technical repetitions; no warmup contributes",
            "per_model_serial_performance": (
                "expected wall time sums shard means, throughput uses total raw bytes over that "
                "sum, and sequential peak RSS is the maximum measured shard-trial RSS"
            ),
            "descriptive_shard_mean_views": (
                "input-equal and raw-byte-weighted shard means are retained only as descriptive "
                "views; neither represents complete-model latency or peak RSS"
            ),
            "per_model_storage": (
                "source and archive bytes are pooled across shards before deriving ratio and saving"
            ),
            "cross_model_model_equal": (
                "storage gives each pooled model ratio/saving one vote; performance gives each "
                "serial-complete-model throughput and sequential peak-RSS point one vote"
            ),
            "cross_model_raw_byte_weighted": (
                "storage pools all bytes; global serial throughput is total raw bytes divided by "
                "the sum of model expected times, and RSS is the global maximum model peak"
            ),
            "pooled_storage": "sums source and stable archive bytes before deriving ratio and saving",
            "alias_warning": (
                "bzip2/default and bzip2/ratio rows are retained but share one independent-observation ID"
            ),
        },
        "traceability": {
            "fingerprint_projection": fingerprint_projection,
            "inputs": [{
                "task_semantic_sha256": row["task_semantic_sha256"],
                "raw_result": row["raw_result"],
                "validation_receipt": row["validation_receipt"],
                "model": row["model"],
                "source": row["source"],
                "manifest": row["manifest"],
                "experiment_matrix_sha256": row["experiment_matrix_sha256"],
                "implementation_binding": row["implementation_binding"],
            } for row in inputs],
        },
        "inputs": inputs,
        "per_model": per_model,
        "cross_model": cross_model,
    }


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SummaryError("output_exists", f"refusing to overwrite {path}") from exc
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
            pass
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", nargs=2, action="append", required=True,
        metavar=("RAW_JSON", "VALIDATION_JSON"),
        help="formal raw result and its campaign validation receipt; repeat for more inputs",
    )
    parser.add_argument(
        "--output", help="atomically create this JSON file; existing files are never overwritten",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = aggregate_pairs([(raw, receipt) for raw, receipt in args.pair])
        payload = (json.dumps(
            result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True,
        ) + "\n").encode("utf-8")
        if args.output:
            _atomic_write(pathlib.Path(args.output), payload)
        else:
            sys.stdout.buffer.write(payload)
    except SummaryError as exc:
        print(f"error[{exc.code}]: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
