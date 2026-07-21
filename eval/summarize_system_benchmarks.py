#!/usr/bin/env python3
"""Build paper-facing, auditable summaries of Brevis system benchmarks.

This tool is intentionally stricter than a generic JSON aggregator.  By
default it accepts only complete, successful, formal-eligible
``brevis.system-benchmark`` schema-2 documents containing schema-4 bench
reports.  It delegates semantic/archive/program validation to
``analyze_generated_dsl`` and independently validates the timing processes,
declared repetition counts, source/manifest integrity, and code/binary
provenance.

Each input result remains a separate shard record.  The tool never combines
files into a model-level observation.  Compression and decompression metrics
come only from the archive-producing pipeline.  Calibration and ``bench``
timings are emitted in explicitly separate diagnostic sections and are never
presented as a decomposition of compression time.
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
from collections.abc import Mapping, Sequence
from typing import Any

import analyze_generated_dsl as dsl


SUMMARY_SCHEMA_ID = "brevis.system-benchmark-summary"
SUMMARY_SCHEMA_VERSION = 1
SHA256_RE = dsl.SHA256_RE
DIAGNOSTIC_TIMING_RELATIONSHIP = (
    "bench is a separate diagnostic replay executed only after every timed archive "
    "pipeline; its internal times do not decompose this compression invocation"
)


class SummaryError(ValueError):
    """A fail-closed summary validation error with a stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryError("missing_or_invalid_field", f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SummaryError("missing_or_invalid_field", f"{label} must be an array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SummaryError(
            "missing_or_invalid_field", f"{label} must be an integer >= {minimum}",
        )
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError("missing_or_invalid_field", f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise SummaryError(
            "missing_or_invalid_field", f"{label} must be finite and >= {minimum}",
        )
    return numeric


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SummaryError("invalid_identity", f"{label} must be a lowercase SHA-256")
    return value


def _git_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SummaryError("invalid_identity", f"{label} must be a Git object id")
    return value


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _trace(result_sha256: str, pointer: str) -> dict[str, str]:
    key_material = f"{result_sha256}\0{pointer}".encode("utf-8")
    return {
        "result_file_sha256": result_sha256,
        "json_pointer": pointer,
        "stable_trace_key": "trace-" + hashlib.sha256(key_material).hexdigest(),
    }


def _statistics(values: Sequence[int | float]) -> dict[str, int | float | None]:
    if not values:
        raise SummaryError("no_measured_trials", "cannot summarize an empty trial sequence")
    numeric = [float(value) for value in values]
    mean = statistics.fmean(numeric)
    sample_sd = statistics.stdev(numeric) if len(numeric) >= 2 else None
    return {
        "n": len(numeric),
        "mean": mean,
        "sample_sd": sample_sd,
        "median": statistics.median(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "coefficient_of_variation": (
            sample_sd / mean if sample_sd is not None and mean != 0.0 else None
        ),
    }


def _validate_process(process_value: Any, label: str) -> Mapping[str, Any]:
    process = _mapping(process_value, label)
    if (
        process.get("status_code") != "ok"
        or process.get("timed_out") is not False
        or process.get("error") is not None
        or process.get("exit_code") != 0
    ):
        raise SummaryError("unsuccessful_process", f"{label} is not a successful process")
    _integer(process.get("wall_time_ns"), f"{label}.wall_time_ns", minimum=1)
    rss = process.get("direct_child_max_rss_bytes")
    if rss is not None:
        _integer(rss, f"{label}.direct_child_max_rss_bytes")
    return process


def _validate_integrity_and_provenance(
    document: Mapping[str, Any], source: Mapping[str, Any],
) -> dict[str, Any]:
    source_size = _integer(source.get("size_bytes"), "source.size_bytes", minimum=1)
    source_sha = _sha256(source.get("sha256"), "source.sha256")
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise SummaryError("unverified_source_identity", "source.path is missing")
    integrity = _mapping(document.get("integrity"), "integrity")
    source_integrity = _mapping(integrity.get("source"), "integrity.source")
    if (
        source_integrity.get("verified") is not True
        or source_integrity.get("expected_size_bytes") != source_size
        or source_integrity.get("actual_size_bytes") != source_size
        or source_integrity.get("expected_sha256") != source_sha
        or source_integrity.get("actual_sha256") != source_sha
    ):
        raise SummaryError(
            "unverified_source_identity", "formal source integrity is absent or inconsistent",
        )

    manifest = _mapping(integrity.get("manifest"), "integrity.manifest")
    manifest_sha = _sha256(manifest.get("actual_sha256"), "manifest.actual_sha256")
    if manifest.get("verified") is not True or manifest.get("expected_sha256") != manifest_sha:
        raise SummaryError("unverified_manifest", "manifest hash was not verified")
    _integer(manifest.get("size_bytes"), "manifest.size_bytes", minimum=1)
    if not isinstance(manifest.get("path"), str) or not manifest["path"]:
        raise SummaryError("unverified_manifest", "manifest path is missing")

    binding = _mapping(integrity.get("manifest_binding"), "integrity.manifest_binding")
    if (
        binding.get("verified") is not True
        or binding.get("bytes") != source_size
        or binding.get("sha256") != source_sha
    ):
        raise SummaryError("unverified_manifest_binding", "manifest shard binding is inconsistent")
    for field in ("tag", "repo", "revision", "shard"):
        if not isinstance(binding.get(field), str) or not binding[field]:
            raise SummaryError(
                "unverified_manifest_binding", f"manifest binding {field} is missing",
            )
    _sha256(binding.get("entry_sha256"), "manifest_binding.entry_sha256")

    metadata = _mapping(document.get("input_metadata"), "input_metadata")
    for metadata_key, binding_key in (
        ("model_tag", "tag"), ("model_repo", "repo"),
        ("model_revision", "revision"), ("shard", "shard"),
    ):
        if metadata.get(metadata_key) != binding.get(binding_key):
            raise SummaryError(
                "unverified_manifest_binding",
                f"input_metadata.{metadata_key} disagrees with manifest binding",
            )

    provenance = _mapping(document.get("provenance"), "provenance")
    git = _mapping(provenance.get("git"), "provenance.git")
    commit = _git_commit(git.get("commit"), "provenance.git.commit")
    if git.get("dirty") is not False:
        raise SummaryError("unverified_code_identity", "formal benchmark Git tree was not clean")

    build = _mapping(provenance.get("build"), "provenance.build")
    if (
        build.get("requested") is not True
        or build.get("mode") != "ReleaseFast"
        or build.get("success") is not True
    ):
        raise SummaryError("unverified_build", "ReleaseFast build provenance is incomplete")

    binary = _mapping(provenance.get("binary"), "provenance.binary")
    binary_before = _sha256(binary.get("sha256_before_runs"), "binary.sha256_before_runs")
    binary_after = _sha256(binary.get("sha256_after_runs"), "binary.sha256_after_runs")
    if (
        binary.get("executable") is not True
        or binary.get("unchanged_during_benchmark") is not True
        or binary_before != binary_after
    ):
        raise SummaryError("unverified_binary", "Brevis binary identity changed or is incomplete")

    harness = _mapping(provenance.get("harness"), "provenance.harness")
    helper = _mapping(provenance.get("generic_helper"), "provenance.generic_helper")
    harness_sha = _sha256(harness.get("sha256"), "provenance.harness.sha256")
    helper_sha = _sha256(helper.get("sha256"), "provenance.generic_helper.sha256")
    tracked_start = _mapping(provenance.get("git_tracked_start"), "provenance.git_tracked_start")
    if _git_commit(tracked_start.get("commit"), "git_tracked_start.commit") != commit:
        raise SummaryError("unverified_code_identity", "Git identities disagree at start")

    end_checks = _mapping(provenance.get("end_checks"), "provenance.end_checks")
    if (
        end_checks.get("performed") is not True
        or end_checks.get("success") is not True
        or end_checks.get("failures") != []
    ):
        raise SummaryError("failed_end_checks", "post-run provenance checks did not pass")
    checks = _mapping(end_checks.get("checks"), "provenance.end_checks.checks")
    expected_checks = {
        "source": source_sha,
        "manifest": manifest_sha,
        "harness": harness_sha,
        "generic_helper": helper_sha,
    }
    for name, expected_sha in expected_checks.items():
        check = _mapping(checks.get(name), f"provenance.end_checks.checks.{name}")
        if (
            check.get("unchanged") is not True
            or check.get("error") is not None
            or check.get("expected_sha256") != expected_sha
            or check.get("actual_sha256") != expected_sha
        ):
            raise SummaryError("failed_end_checks", f"post-run {name} identity check failed")
    git_check = _mapping(checks.get("git_tracked_state"), "end_checks.git_tracked_state")
    if (
        git_check.get("same_commit") is not True
        or git_check.get("same_tracked_diff") is not True
        or git_check.get("unchanged") is not True
    ):
        raise SummaryError("failed_end_checks", "tracked Git state changed during benchmark")

    return {
        "source": {
            "path": source.get("path"),
            "size_bytes": source_size,
            "sha256": source_sha,
            "json_pointer": "/source",
        },
        "manifest": {
            "path_recorded": manifest["path"],
            "size_bytes": manifest["size_bytes"],
            "sha256": manifest_sha,
            "binding": dict(binding),
            "manifest_json_pointer": "/integrity/manifest",
            "binding_json_pointer": "/integrity/manifest_binding",
        },
        "implementation": {
            "git_commit": commit,
            "git_json_pointer": "/provenance/git",
            "binary_sha256": binary_before,
            "binary_json_pointer": "/provenance/binary",
            "build_mode": "ReleaseFast",
            "build_json_pointer": "/provenance/build",
            "harness_sha256": harness_sha,
            "generic_helper_sha256": helper_sha,
            "end_checks_json_pointer": "/provenance/end_checks",
        },
    }


def _validate_declared_counts(document: Mapping[str, Any]) -> tuple[int, int, list[str]]:
    settings = _mapping(document.get("configuration"), "configuration")
    warmups = _integer(settings.get("warmups"), "configuration.warmups")
    repetitions = _integer(
        settings.get("repetitions"), "configuration.repetitions", minimum=1,
    )
    requested = _list(
        settings.get("requested_configuration_ids"),
        "configuration.requested_configuration_ids",
    )
    if not requested or not all(isinstance(value, str) and value for value in requested):
        raise SummaryError("invalid_repetition_contract", "requested configuration ids are invalid")
    if len(requested) != len(set(requested)):
        raise SummaryError("invalid_repetition_contract", "requested configuration ids repeat")
    return warmups, repetitions, requested


def _validate_calibration(
    document: Mapping[str, Any], *, expected_warmups: int, expected_repetitions: int,
    result_sha256: str,
) -> dict[str, Any]:
    calibration = _mapping(document.get("calibration"), "calibration")
    required = calibration.get("required")
    if not isinstance(required, bool):
        raise SummaryError("invalid_calibration", "calibration.required must be boolean")
    warmups = _list(calibration.get("warmups"), "calibration.warmups")
    runs = _list(calibration.get("runs"), "calibration.runs")
    if not required:
        if (
            calibration.get("status_code") != "not_required"
            or calibration.get("success") is not True
            or calibration.get("failure") is not None
            or warmups
            or runs
        ):
            raise SummaryError("invalid_calibration", "not-required calibration is inconsistent")
        return {
            "required": False,
            "status": "not_required",
            "cost_scope": "none",
            "measured_trials": [],
            "warmup_trials": [],
            "external_wall_time_ns": None,
            "compression_time_component": False,
            "json_pointer": "/calibration",
        }

    if (
        calibration.get("status_code") != "ok"
        or calibration.get("success") is not True
        or calibration.get("failure") is not None
        or len(warmups) != expected_warmups
        or len(runs) != expected_repetitions
    ):
        raise SummaryError("invalid_calibration", "required calibration did not complete exactly")
    consistency = _mapping(
        calibration.get("measured_prior_consistency"),
        "calibration.measured_prior_consistency",
    )
    if (
        consistency.get("status_code") != "consistent"
        or consistency.get("byte_consistent") is not True
        or consistency.get("context_counts_consistent") is not True
    ):
        raise SummaryError("invalid_calibration", "calibration priors are inconsistent")
    reference_sha = _sha256(
        consistency.get("reference_sha256"), "calibration prior reference SHA-256",
    )
    canonical = _mapping(calibration.get("canonical_prior"), "calibration.canonical_prior")
    if (
        canonical.get("sha256") != reference_sha
        or canonical.get("sha256_after_uses") != reference_sha
        or canonical.get("unchanged_during_benchmark") is not True
    ):
        raise SummaryError("invalid_calibration", "canonical prior identity is not stable")

    output: dict[str, Any] = {
        "required": True,
        "status": "complete_consistent",
        "cost_scope": (
            "input-local prior construction; reported once per shard and never included in "
            "compression or decompression timing"
        ),
        "canonical_prior_sha256": reference_sha,
        "compression_time_component": False,
        "json_pointer": "/calibration",
    }
    for collection_name, collection in (("warmup_trials", warmups), ("measured_trials", runs)):
        phase = "warmup" if collection_name == "warmup_trials" else "measured"
        trials = []
        for position, value in enumerate(collection):
            record = _mapping(value, f"calibration.{phase}[{position}]")
            if (
                record.get("phase") != phase
                or record.get("index") != position
                or record.get("status_code") != "ok"
                or record.get("success") is not True
                or record.get("failure") is not None
            ):
                raise SummaryError("invalid_calibration", f"calibration {phase} {position} failed")
            process = _validate_process(record.get("process"), f"calibration {phase} {position}")
            prior = _mapping(record.get("prior"), f"calibration {phase} {position}.prior")
            prior_sha = _sha256(prior.get("sha256"), f"calibration {phase} {position} prior")
            if phase == "measured" and prior_sha != reference_sha:
                raise SummaryError(
                    "invalid_calibration",
                    f"calibration measured prior {position} differs from the reference",
                )
            pointer = f"/calibration/{'warmups' if phase == 'warmup' else 'runs'}/{position}"
            trials.append({
                "phase": phase,
                "repetition_index": position,
                "external_wall_time_ns": process["wall_time_ns"],
                "prior_sha256": prior_sha,
                "trace": _trace(result_sha256, pointer),
            })
        output[collection_name] = trials
    output["external_wall_time_ns"] = _statistics([
        trial["external_wall_time_ns"] for trial in output["measured_trials"]
    ])
    return output


def _trial(
    *, run: Mapping[str, Any], phase: str, position: int, config_position: int,
    source_size: int, result_sha256: str,
) -> dict[str, Any]:
    prefix = (
        f"/configurations/{config_position}/"
        f"{'warmups' if phase == 'warmup' else 'runs'}/{position}"
    )
    if (
        run.get("phase") != phase
        or run.get("index") != position
        or run.get("status_code") != "ok"
        or run.get("success") is not True
        or run.get("failure") is not None
        or run.get("archive_pipeline_success") is not True
        or run.get("diagnostic_success") is not True
    ):
        raise SummaryError("unsuccessful_repetition", f"{prefix} is not fully successful")
    if run.get("timing_relationship") != DIAGNOSTIC_TIMING_RELATIONSHIP:
        raise SummaryError(
            "ambiguous_timing_scope", f"{prefix} lacks the separate-replay timing contract",
        )
    compression = _validate_process(run.get("compression"), f"{prefix}/compression")
    decompression = _validate_process(run.get("decompression"), f"{prefix}/decompression")
    diagnostic = _validate_process(run.get("bench_process"), f"{prefix}/bench_process")
    archive = _mapping(run.get("archive"), f"{prefix}/archive")
    archive_storage = _mapping(archive.get("storage"), f"{prefix}/archive/storage")
    archive_size = _integer(
        archive_storage.get("logical_size_bytes"),
        f"{prefix}/archive/storage/logical_size_bytes", minimum=1,
    )
    archive_sha = _sha256(archive.get("sha256"), f"{prefix}/archive/sha256")
    if (
        archive.get("sha256_after_decode") != archive_sha
        or archive.get("unchanged_during_decode") is not True
    ):
        raise SummaryError("archive_mutated", f"{prefix} archive changed during decode")
    if compression.get("output_size_bytes") != archive_size:
        raise SummaryError("archive_size_drift", f"{prefix} compression output size disagrees")
    if decompression.get("output_size_bytes") != source_size:
        raise SummaryError("decode_size_mismatch", f"{prefix} decompressed size disagrees")

    bench_report = _mapping(run.get("bench_report"), f"{prefix}/bench_report")
    planning_ms = _number(
        bench_report.get("planning_wall_ms"), f"{prefix}/bench_report/planning_wall_ms",
    )
    encoding_ms = _number(
        bench_report.get("encoding_wall_ms"), f"{prefix}/bench_report/encoding_wall_ms",
    )
    compression_ns = _integer(compression["wall_time_ns"], "compression.wall_time_ns", minimum=1)
    decompression_ns = _integer(
        decompression["wall_time_ns"], "decompression.wall_time_ns", minimum=1,
    )
    return {
        "phase": phase,
        "repetition_index": position,
        "archive_execution_order": run.get("archive_execution_order"),
        "diagnostic_execution_order": run.get("diagnostic_execution_order"),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha,
        "compression": {
            "wall_time_ns": compression_ns,
            "throughput_bytes_per_second": source_size * 1_000_000_000 / compression_ns,
            "direct_child_max_rss_bytes": compression.get("direct_child_max_rss_bytes"),
            "trace": _trace(result_sha256, prefix + "/compression"),
        },
        "decompression": {
            "wall_time_ns": decompression_ns,
            "throughput_bytes_per_second": source_size * 1_000_000_000 / decompression_ns,
            "direct_child_max_rss_bytes": decompression.get("direct_child_max_rss_bytes"),
            "trace": _trace(result_sha256, prefix + "/decompression"),
        },
        "search_diagnostic_replay": {
            "external_wall_time_ns": diagnostic["wall_time_ns"],
            "planning_wall_ms": planning_ms,
            "encoding_wall_ms": encoding_ms,
            "compression_time_component": False,
            "trace": _trace(result_sha256, prefix + "/bench_process"),
            "bench_report_trace": _trace(result_sha256, prefix + "/bench_report"),
        },
        "trace": _trace(result_sha256, prefix),
    }


def _metric_summary(trials: Sequence[Mapping[str, Any]], stage: str) -> dict[str, Any]:
    wall = [trial[stage]["wall_time_ns"] for trial in trials]
    throughput = [trial[stage]["throughput_bytes_per_second"] for trial in trials]
    rss = [
        trial[stage]["direct_child_max_rss_bytes"] for trial in trials
        if trial[stage]["direct_child_max_rss_bytes"] is not None
    ]
    return {
        "wall_time_ns": _statistics(wall),
        "throughput_bytes_per_second": _statistics(throughput),
        "direct_child_max_rss_bytes": _statistics(rss) if rss else None,
        "rss_observation_count": len(rss),
        "rss_scope": "direct child process; threads share the process",
    }


def _diagnostic_summary(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    diagnostic = [trial["search_diagnostic_replay"] for trial in trials]
    return {
        "scope": (
            "separate deterministic bench replay run after archive pipelines; these values are "
            "search/encoding diagnostics and are not compression-time components"
        ),
        "compression_time_component": False,
        "external_wall_time_ns": _statistics([
            trial["external_wall_time_ns"] for trial in diagnostic
        ]),
        "planning_wall_ms": _statistics([trial["planning_wall_ms"] for trial in diagnostic]),
        "encoding_wall_ms": _statistics([trial["encoding_wall_ms"] for trial in diagnostic]),
    }


def summarize_document(
    input_label: str,
    document_value: Any,
    *,
    result_file_sha256: str,
    allow_pilot: bool = False,
) -> dict[str, Any]:
    """Validate and summarize one result without merging it with another file."""

    result_sha = _sha256(result_file_sha256, "result file SHA-256")
    document = _mapping(document_value, "system benchmark")
    if (
        document.get("status") != "complete"
        or document.get("success") is not True
        or document.get("failure") is not None
    ):
        raise SummaryError("unsuccessful_system_benchmark", "result is not complete and successful")
    classification = _mapping(document.get("run_classification"), "run_classification")
    formal = (
        classification.get("run_class") == "formal"
        and classification.get("formal_eligible") is True
    )
    if not formal and not allow_pilot:
        raise SummaryError("pilot_excluded", "non-formal result is excluded by default")

    configurations = _list(document.get("configurations"), "configurations")
    if not configurations:
        raise SummaryError("no_configurations", "result has no configurations")
    expected_warmups, expected_repetitions, requested = _validate_declared_counts(document)
    observed_ids = [
        config.get("id") if isinstance(config, Mapping) else None for config in configurations
    ]
    if observed_ids != requested:
        raise SummaryError(
            "configuration_contract_mismatch",
            "configuration records do not exactly match requested_configuration_ids",
        )

    dsl_bundle = dsl.analyze_documents(
        [(input_label, document)], allow_pilot=allow_pilot,
    )
    if dsl_bundle.errors:
        details = "; ".join(
            f"{error['error_code']}: {error['message']}" for error in dsl_bundle.errors
        )
        raise SummaryError("dsl_validation_failed", details)
    dsl_by_config = {
        report["outer_configuration_id"]: report for report in dsl_bundle.reports
    }
    if len(dsl_by_config) != len(configurations) or set(dsl_by_config) != set(requested):
        raise SummaryError(
            "dsl_validation_failed",
            "DSL validator did not accept exactly one report per configuration",
        )

    source = _mapping(document.get("source"), "source")
    environment = _mapping(document.get("environment"), "environment")
    if not environment:
        raise SummaryError("missing_or_invalid_field", "environment must not be empty")
    identity = _validate_integrity_and_provenance(document, source)
    source_size = identity["source"]["size_bytes"]
    calibration = _validate_calibration(
        document,
        expected_warmups=expected_warmups,
        expected_repetitions=expected_repetitions,
        result_sha256=result_sha,
    )
    canonical_prior_configs = [
        config.get("id") for config in configurations
        if isinstance(config, Mapping)
        and isinstance(config.get("spec"), Mapping)
        and config["spec"].get("prior_policy") == "canonical"
    ]
    if bool(canonical_prior_configs) is not calibration["required"]:
        raise SummaryError(
            "invalid_calibration",
            "calibration requirement disagrees with configuration prior policies",
        )

    summaries = []
    for config_position, config_value in enumerate(configurations):
        configuration = _mapping(config_value, f"configurations[{config_position}]")
        config_id = str(configuration["id"])
        warmups = _list(configuration.get("warmups"), f"configuration {config_id}.warmups")
        runs = _list(configuration.get("runs"), f"configuration {config_id}.runs")
        if len(warmups) != expected_warmups or len(runs) != expected_repetitions:
            raise SummaryError(
                "incomplete_repetitions",
                f"configuration {config_id!r} has {len(warmups)} warmups/{len(runs)} runs; "
                f"expected {expected_warmups}/{expected_repetitions}",
            )
        warmup_trials = [
            _trial(
                run=_mapping(run, f"configuration {config_id} warmup {position}"),
                phase="warmup", position=position, config_position=config_position,
                source_size=source_size, result_sha256=result_sha,
            )
            for position, run in enumerate(warmups)
        ]
        measured_trials = [
            _trial(
                run=_mapping(run, f"configuration {config_id} run {position}"),
                phase="measured", position=position, config_position=config_position,
                source_size=source_size, result_sha256=result_sha,
            )
            for position, run in enumerate(runs)
        ]
        archive_sizes = {trial["archive_size_bytes"] for trial in measured_trials}
        archive_hashes = {trial["archive_sha256"] for trial in measured_trials}
        if len(archive_sizes) != 1 or len(archive_hashes) != 1:
            raise SummaryError(
                "archive_size_drift", f"configuration {config_id!r} measured archives drift",
            )
        archive_size = next(iter(archive_sizes))
        consistency = _mapping(
            configuration.get("measured_archive_consistency"),
            f"configuration {config_id}.measured_archive_consistency",
        )
        if (
            consistency.get("status_code") != "consistent"
            or consistency.get("reference_size_bytes") != archive_size
            or consistency.get("reference_sha256") != next(iter(archive_hashes))
            or consistency.get("consistent_runs") != expected_repetitions
            or consistency.get("inconsistent_runs") != 0
        ):
            raise SummaryError(
                "archive_size_drift", f"configuration {config_id!r} consistency counters disagree",
            )

        dsl_report = dsl_by_config[config_id]
        summaries.append({
            "configuration_id": config_id,
            "configuration_json_pointer": f"/configurations/{config_position}",
            "spec": configuration.get("spec"),
            "effective_search_configuration": configuration.get(
                "effective_search_configuration"
            ),
            "formal_eligible": formal,
            "paper_metrics_eligible": formal,
            "declared_counts": {
                "warmups": expected_warmups,
                "measured_repetitions": expected_repetitions,
            },
            "warmup_trials": warmup_trials,
            "measured_trials": measured_trials,
            "archive": {
                "size_bytes": archive_size,
                "sha256": next(iter(archive_hashes)),
                "source_size_bytes": source_size,
                "compression_ratio_input_over_archive": source_size / archive_size,
                "storage_saved_bytes": source_size - archive_size,
                "storage_saving_fraction": 1.0 - archive_size / source_size,
                "measured_repetitions_byte_identical": True,
                "consistency_json_pointer": (
                    f"/configurations/{config_position}/measured_archive_consistency"
                ),
            },
            "compression": _metric_summary(measured_trials, "compression"),
            "decompression": _metric_summary(measured_trials, "decompression"),
            "search_diagnostic_replay": _diagnostic_summary(measured_trials),
            "calibration_reference": {
                "required_for_configuration": config_id in canonical_prior_configs,
                "shared_shard_level_json_pointer": "/calibration",
                "cost_included_in_compression_time": False,
            },
            "dsl_evidence": {
                "analysis_schema": dsl_report["analysis_schema"],
                "report_id": dsl_report["report_id"],
                "semantic_fingerprint_sha256": dsl_report[
                    "semantic_fingerprint_sha256"
                ],
                "canonical_detail_source": dsl_report["canonical_detail_source"],
                "measured_repetition_count": dsl_report["measured_repetition_count"],
                "technical_repetitions_are_independent_tensor_samples": False,
            },
            "trace": _trace(result_sha, f"/configurations/{config_position}"),
        })

    return {
        "record_type": "shard",
        "input_label": input_label,
        "result_file": {
            "sha256": result_sha,
            "hash_scope": "exact input file bytes",
            "root_json_pointer": "",
        },
        **identity,
        "input_metadata": document.get("input_metadata"),
        "input_metadata_json_pointer": "/input_metadata",
        "benchmark_configuration": document.get("configuration"),
        "benchmark_configuration_json_pointer": "/configuration",
        "environment": dict(environment),
        "environment_json_pointer": "/environment",
        "run_class": classification.get("run_class"),
        "formal_eligible": classification.get("formal_eligible"),
        "paper_metrics_eligible": formal,
        "calibration": calibration,
        "configurations": summaries,
        "aggregation": {
            "model_level_aggregation_performed": False,
            "cross_file_aggregation_performed": False,
            "unit": "one system-result file / one source shard",
        },
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _read_result(path: pathlib.Path) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
        document = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SummaryError("invalid_result_file", f"cannot read {path}: {exc}") from exc
    return document, hashlib.sha256(raw).hexdigest()


def summarize_files(
    paths: Sequence[os.PathLike[str] | str], *, allow_pilot: bool = False,
) -> dict[str, Any]:
    if not paths:
        raise SummaryError("no_inputs", "at least one result file is required")
    shards = []
    for path_value in paths:
        path = pathlib.Path(path_value)
        document, result_sha = _read_result(path)
        shards.append(summarize_document(
            str(path), document, result_file_sha256=result_sha, allow_pilot=allow_pilot,
        ))
    tool_path = pathlib.Path(__file__).resolve()
    validator_path = pathlib.Path(dsl.__file__).resolve()
    return {
        "schema": {"id": SUMMARY_SCHEMA_ID, "version": SUMMARY_SCHEMA_VERSION},
        "generator": {
            "path": str(tool_path),
            "sha256": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
            "dsl_validator_schema": {
                "id": dsl.ANALYSIS_SCHEMA_ID,
                "version": dsl.ANALYSIS_SCHEMA_VERSION,
            },
            "dsl_validator_path": str(validator_path),
            "dsl_validator_sha256": hashlib.sha256(
                validator_path.read_bytes()
            ).hexdigest(),
            "accepted_system_schema": {
                "id": dsl.SYSTEM_SCHEMA_ID,
                "version": dsl.SYSTEM_SCHEMA_VERSION,
            },
            "accepted_bench_schema": dsl.BENCH_SCHEMA_VERSION,
        },
        "shards": shards,
        "input_file_count": len(shards),
        "cross_file_aggregation_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-pilot", action="store_true",
        help="emit exploratory summaries marked ineligible for paper metrics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = summarize_files(args.inputs, allow_pilot=args.allow_pilot)
        encoded = json.dumps(
            summary, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True,
        ) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if args.force else "x"
            with args.output.open(mode, encoding="utf-8") as output:
                output.write(encoded)
    except (SummaryError, FileExistsError, OSError, ValueError) as exc:
        code = exc.code if isinstance(exc, SummaryError) else type(exc).__name__
        print(f"{code}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
