#!/usr/bin/env python3
"""Repeated engineering benchmarks for canonical BRTA version 1 archives.

The harness times complete ``brevis compress`` and ``brevis decompress``
processes. Hashing and the full byte-for-byte comparison happen after each
operation's clock has stopped. Results are always labelled engineering evidence;
this script does not make a run eligible for paper metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import benchmarking as generic


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BINARY = ROOT / "zig-out" / "bin" / "brevis"
SCHEMA_ID = "brevis.brta-v1-engineering-benchmark"
SCHEMA_VERSION = 1
DEFAULT_WARMUPS = 1
DEFAULT_REPETITIONS = 3
DEFAULT_TIMEOUT_SECONDS = 3600.0

SEARCH_OPTIONS = (
    ("max_expansions", "--max-expansions", 0),
    ("max_nodes", "--max-nodes", 1),
    ("max_depth", "--max-depth", 0),
    ("seed_float_fields", "--seed-float-fields", 0),
    ("max_repeat_period", "--max-repeat-period", 0),
    ("max_concat_splits", "--max-concat-splits", 0),
    ("max_map_constants", "--max-map-constants", 0),
    ("max_rotations", "--max-rotations", 0),
    ("max_field_splits", "--max-field-splits", 0),
)
RESOURCE_OPTIONS = (
    ("max_total_bytes", "--max-total-bytes", 1),
    ("max_tensor_bytes", "--max-tensor-bytes", 1),
    ("max_prefix_bytes", "--max-prefix-bytes", 1),
)
ALL_OPTIONS = (*SEARCH_OPTIONS, *RESOURCE_OPTIONS)


class BenchmarkError(RuntimeError):
    """The benchmark setup or a required BRTA invariant is invalid."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _normalize_positive(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _normalize_options(options: Mapping[str, int] | None) -> dict[str, int]:
    requested = dict(options or {})
    known = {name for name, _, _ in ALL_OPTIONS}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError("unknown Brevis option(s): " + ", ".join(unknown))
    normalized: dict[str, int] = {}
    for name, _, minimum in ALL_OPTIONS:
        if name in requested and requested[name] is not None:
            normalized[name] = _normalize_positive(name, requested[name], minimum)
            if name == "seed_float_fields" and normalized[name] not in (0, 1):
                raise ValueError("seed_float_fields must be 0 or 1")
    return normalized


def _option_args(
    options: Mapping[str, int],
    definitions: Sequence[tuple[str, str, int]],
) -> list[str]:
    return [
        part
        for name, flag, _ in definitions
        if name in options
        for part in (flag, str(options[name]))
    ]


def _process_failure(operation: str, record: Mapping[str, Any]) -> str | None:
    if record.get("timed_out"):
        return f"{operation} timed out after {record.get('timeout_seconds')} seconds"
    if record.get("error") is not None:
        return f"{operation} launch failed: {record['error']}"
    if record.get("exit_code") != 0:
        return f"{operation} exited with status {record.get('exit_code')}"
    return None


def _run_checked_json(
    command: Sequence[str],
    log_dir: pathlib.Path,
    label: str,
    timeout_seconds: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = generic._run_process(command, log_dir, label, timeout_seconds)
    failure = _process_failure(label, record)
    if failure is not None:
        detail = record["stderr"]["text"].strip()
        raise BenchmarkError(f"{failure}: {detail}" if detail else failure)
    try:
        payload = json.loads(record["stdout"]["text"])
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{label} did not emit valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"{label} JSON must be an object")
    return payload, record


def _validate_effective_config(
    config: Mapping[str, Any],
    options: Mapping[str, int],
    workers: int,
) -> None:
    expected_fixed = {
        "synthesis_unit": "complete_tensor",
        "objective": "canonical_program_bytes",
        "literal_fallback": True,
        "phog_role": "queue_order_only",
        "archive": "BRTA-v1",
        "workers": workers,
    }
    for key, expected in expected_fixed.items():
        if config.get(key) != expected:
            raise BenchmarkError(
                f"effective config {key!r} is {config.get(key)!r}, expected {expected!r}"
            )
    for name, _, _ in ALL_OPTIONS:
        expected = (
            bool(options[name])
            if name == "seed_float_fields" and name in options
            else options.get(name)
        )
        if name in options and (
            type(config.get(name)) is not type(expected)
            or config.get(name) != expected
        ):
            raise BenchmarkError(
                f"effective config {name!r} is {config.get(name)!r}, "
                f"requested {expected!r}"
            )
    if (
        "max_tensor_bytes" in options
        and config.get("max_decomposition_bytes") != options["max_tensor_bytes"]
    ):
        raise BenchmarkError(
            "effective max_decomposition_bytes does not match max_tensor_bytes"
        )


def _git_provenance(
    excluded_generated_paths: Sequence[pathlib.Path] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "repository": str(ROOT),
        "commit": None,
        "branch": None,
        "dirty": None,
        "status_porcelain": None,
        "tracked_diff_sha256": None,
    }
    git = shutil.which("git")
    if git is None:
        return result
    status_arguments = ["status", "--porcelain", "--untracked-files=normal"]
    exclusions: list[str] = []
    for path in excluded_generated_paths:
        try:
            relative = path.resolve(strict=False).relative_to(ROOT)
        except ValueError:
            continue
        exclusions.append(relative.as_posix())
    if exclusions:
        status_arguments.extend((
            "--",
            ".",
            *(f":(exclude){path}" for path in exclusions),
        ))
    commands = {
        "commit": ("rev-parse", "HEAD"),
        "branch": ("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": tuple(status_arguments),
    }
    try:
        for key, arguments in commands.items():
            process = subprocess.run(
                [git, "-C", str(ROOT), *arguments],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            if process.returncode == 0:
                result[key] = process.stdout.decode("utf-8", errors="replace").rstrip("\n")
        diff = subprocess.run(
            [git, "-C", str(ROOT), "diff", "--binary", "HEAD", "--"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        if diff.returncode == 0:
            result["tracked_diff_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if result["status_porcelain"] is not None:
        result["dirty"] = bool(result["status_porcelain"])
    result["excluded_generated_paths"] = exclusions
    return result


def _query_config(
    binary: pathlib.Path,
    options: Mapping[str, int],
    workers: int,
    log_dir: pathlib.Path,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    command = [
        str(binary),
        "config",
        *_option_args(options, ALL_OPTIONS),
        "--workers",
        str(workers),
    ]
    effective, process = _run_checked_json(
        command, log_dir, f"config-workers-{workers}", timeout_seconds,
    )
    _validate_effective_config(effective, options, workers)
    return {"command": command, "effective": effective, "process": process}


def _run_calibration(
    binary: pathlib.Path,
    source: pathlib.Path,
    prior: pathlib.Path,
    tensors: int,
    options: Mapping[str, int],
    log_dir: pathlib.Path,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    command = [
        str(binary),
        "calibrate",
        str(source),
        str(prior),
        "--tensors",
        str(tensors),
        *_option_args(options, ALL_OPTIONS),
    ]
    process = generic._run_process(command, log_dir, "calibration", timeout_seconds)
    failure = _process_failure("calibration", process)
    if failure is not None:
        detail = process["stderr"]["text"].strip()
        raise BenchmarkError(f"{failure}: {detail}" if detail else failure)
    if not prior.is_file():
        raise BenchmarkError("calibration succeeded but did not create a prior")
    return {
        "command": command,
        "max_tensors": tensors,
        "process": process,
        "prior": {
            "path": str(prior),
            "size_bytes": prior.stat().st_size,
            "sha256": generic._sha256_file(prior),
        },
    }


def _run_iteration(
    *,
    binary: pathlib.Path,
    source: pathlib.Path,
    source_sha256: str,
    prior: pathlib.Path | None,
    options: Mapping[str, int],
    workers: int,
    phase: str,
    index: int,
    execution_order: int,
    run_dir: pathlib.Path,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True)
    archive = run_dir / "model.brta"
    restored = run_dir / "restored.safetensors"
    compress_command = [
        str(binary),
        "compress",
        str(source),
        str(archive),
    ]
    if prior is not None:
        compress_command.extend(("--prior", str(prior)))
    compress_command.extend(_option_args(options, ALL_OPTIONS))
    compress_command.extend(("--workers", str(workers)))
    compression = generic._run_process(
        compress_command, run_dir, "compression", timeout_seconds,
    )

    failure = _process_failure("compression", compression)
    archive_record: dict[str, Any] | None = None
    decompression: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    decompress_command: list[str] | None = None
    if failure is None:
        if not archive.is_file():
            failure = "compression succeeded but did not create an archive"
        else:
            archive_record = {
                "path": str(archive),
                "size_bytes": archive.stat().st_size,
                "storage": generic._file_allocation(archive),
                "sha256": generic._sha256_file(archive),
                "sha256_after_decode": None,
                "unchanged_during_decode": None,
            }

    if failure is None:
        decompress_command = [
            str(binary),
            "decompress",
            str(archive),
            str(restored),
            *_option_args(options, RESOURCE_OPTIONS),
            "--workers",
            str(workers),
        ]
        decompression = generic._run_process(
            decompress_command, run_dir, "decompression", timeout_seconds,
        )
        failure = _process_failure("decompression", decompression)
        if failure is None:
            verification = generic._verify_file(source, restored, source_sha256)
            if not verification["bit_exact"]:
                failure = verification["error"] or "restored bytes differ from source"
        archive_after = generic._sha256_file(archive)
        archive_record["sha256_after_decode"] = archive_after
        archive_record["unchanged_during_decode"] = (
            archive_after == archive_record["sha256"]
        )
        if not archive_record["unchanged_during_decode"] and failure is None:
            failure = "archive changed during decompression"

    return {
        "phase": phase,
        "index": index,
        "workers": workers,
        "execution_order": execution_order,
        "commands": {
            "compress": compress_command,
            "decompress": decompress_command,
        },
        "compression": compression,
        "archive": archive_record,
        "decompression": decompression,
        "verification": verification,
        "bit_exact": bool(verification and verification["bit_exact"]),
        "success": failure is None,
        "failure": failure,
        "artifact_directory": str(run_dir),
    }


def _apply_archive_consistency(
    worker_results: Sequence[dict[str, Any]],
    repetitions: int,
) -> dict[str, Any]:
    measured = [
        run
        for result in worker_results
        for run in result["runs"]
    ]
    successful = [
        run for run in measured
        if run["success"] and run["archive"] is not None
    ]
    identities = sorted({
        (run["archive"]["size_bytes"], run["archive"]["sha256"])
        for run in successful
    })
    reference = identities[0] if identities else None
    for run in measured:
        archive = run["archive"]
        matches = bool(
            run["success"]
            and archive is not None
            and reference is not None
            and (archive["size_bytes"], archive["sha256"]) == reference
        )
        run["archive_consistency"] = {
            "matches_reference": matches if run["success"] else None,
            "reference_size_bytes": reference[0] if reference else None,
            "reference_sha256": reference[1] if reference else None,
        }
        if run["success"] and not matches:
            run["success"] = False
            run["failure"] = (
                "canonical archive differs across measured repetitions or worker counts"
            )

    expected = len(worker_results) * repetitions
    complete = len(successful) == expected
    consistent = complete and len(identities) == 1
    for result in worker_results:
        result["status"] = (
            "complete"
            if all(run["success"] for run in result["runs"])
            else "complete_with_failures"
        )
        result["timing_observations_ns"] = {
            "compression_wall": [
                run["compression"]["wall_time_ns"]
                for run in result["runs"]
                if run["compression"]["wall_time_ns"] is not None
            ],
            "decompression_wall": [
                run["decompression"]["wall_time_ns"]
                for run in result["runs"]
                if run["decompression"] is not None
                and run["decompression"]["wall_time_ns"] is not None
            ],
        }
    return {
        "status": (
            "consistent"
            if consistent
            else "incomplete"
            if not complete
            else "inconsistent"
        ),
        "expected_measured_runs": expected,
        "successful_exact_runs": len(successful),
        "distinct_archives": [
            {"size_bytes": size, "sha256": digest}
            for size, digest in identities
        ],
        "reference_size_bytes": reference[0] if reference else None,
        "reference_sha256": reference[1] if reference else None,
    }


def benchmark_file(
    source: os.PathLike[str] | str,
    *,
    binary: os.PathLike[str] | str = DEFAULT_BINARY,
    workers: Sequence[int] = (1,),
    warmups: int = DEFAULT_WARMUPS,
    repetitions: int = DEFAULT_REPETITIONS,
    options: Mapping[str, int] | None = None,
    prior_path: os.PathLike[str] | str | None = None,
    calibrate_tensors: int | None = None,
    work_dir: os.PathLike[str] | str | None = None,
    keep_artifacts: bool = False,
    timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    expected_input_size_bytes: int | None = None,
    expected_input_sha256: str | None = None,
    input_label: str | None = None,
    generic_methods: Sequence[str] = (),
) -> dict[str, Any]:
    """Return a JSON-serializable BRTA v1 engineering result."""

    _normalize_positive("warmups", warmups, 0)
    _normalize_positive("repetitions", repetitions, 1)
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive, or None")
    selected_workers = tuple(workers)
    if not selected_workers:
        raise ValueError("at least one worker count is required")
    if len(set(selected_workers)) != len(selected_workers):
        raise ValueError("worker counts must be unique")
    for worker_count in selected_workers:
        _normalize_positive("workers", worker_count, 1)
    if (expected_input_size_bytes is None) != (expected_input_sha256 is None):
        raise ValueError(
            "expected_input_size_bytes and expected_input_sha256 must be provided together"
        )
    if prior_path is not None and calibrate_tensors is not None:
        raise ValueError("prior_path and calibrate_tensors are mutually exclusive")
    if calibrate_tensors is not None:
        _normalize_positive("calibrate_tensors", calibrate_tensors, 1)

    requested_options = _normalize_options(options)
    source_path = pathlib.Path(source).resolve(strict=True)
    binary_path = pathlib.Path(binary).resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"source is not a regular file: {source_path}")
    if not binary_path.is_file() or not os.access(binary_path, os.X_OK):
        raise ValueError(f"binary is not an executable file: {binary_path}")
    external_prior = (
        pathlib.Path(prior_path).resolve(strict=True)
        if prior_path is not None else None
    )
    if external_prior is not None and not external_prior.is_file():
        raise ValueError(f"prior is not a regular file: {external_prior}")

    source_size = source_path.stat().st_size
    source_sha256 = generic._sha256_file(source_path)
    if expected_input_size_bytes is not None:
        _normalize_positive("expected_input_size_bytes", expected_input_size_bytes, 0)
        expected_hash = generic._normalize_sha256(
            expected_input_sha256, "expected_input_sha256",
        )
        if source_size != expected_input_size_bytes:
            raise ValueError(
                f"input size mismatch: expected {expected_input_size_bytes}, "
                f"observed {source_size}"
            )
        if source_sha256 != expected_hash:
            raise ValueError(
                f"input SHA-256 mismatch: expected {expected_hash}, "
                f"observed {source_sha256}"
            )

    git = _git_provenance()
    parent = pathlib.Path(work_dir).resolve() if work_dir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    artifact_root = pathlib.Path(tempfile.mkdtemp(prefix="brta-v1-", dir=parent))
    started_at = _utc_now()
    binary_sha256 = generic._sha256_file(binary_path)
    harness_path = pathlib.Path(__file__).resolve()
    harness_sha256 = generic._sha256_file(harness_path)
    process_helper_path = pathlib.Path(generic.__file__).resolve()
    process_helper_sha256 = generic._sha256_file(process_helper_path)
    worker_results = [
        {
            "workers": worker_count,
            "config_probe": None,
            "warmups": [],
            "runs": [],
            "status": "pending",
        }
        for worker_count in selected_workers
    ]
    by_workers = {result["workers"]: result for result in worker_results}
    schedule: list[dict[str, Any]] = []
    calibration: dict[str, Any] | None = None
    prior = external_prior
    prior_initial_sha256: str | None = None
    generic_result: dict[str, Any] | None = None

    try:
        for result in worker_results:
            worker_count = result["workers"]
            result["config_probe"] = _query_config(
                binary_path,
                requested_options,
                worker_count,
                artifact_root,
                timeout_seconds,
            )

        if calibrate_tensors is not None:
            prior = artifact_root / "calibrated.brgp"
            calibration = _run_calibration(
                binary_path,
                source_path,
                prior,
                calibrate_tensors,
                requested_options,
                artifact_root,
                timeout_seconds,
            )
        if prior is not None:
            prior_initial_sha256 = generic._sha256_file(prior)

        execution_order = 0
        for phase, count in (("warmup", warmups), ("measured", repetitions)):
            for index in range(count):
                order = (
                    selected_workers
                    if index % 2 == 0
                    else tuple(reversed(selected_workers))
                )
                schedule.append({
                    "phase": phase,
                    "index": index,
                    "workers": list(order),
                    "direction": "forward" if index % 2 == 0 else "reverse",
                })
                for worker_count in order:
                    execution_order += 1
                    run_dir = (
                        artifact_root
                        / phase
                        / f"{index:02d}-workers-{worker_count}"
                    )
                    run = _run_iteration(
                        binary=binary_path,
                        source=source_path,
                        source_sha256=source_sha256,
                        prior=prior,
                        options=requested_options,
                        workers=worker_count,
                        phase=phase,
                        index=index,
                        execution_order=execution_order,
                        run_dir=run_dir,
                        timeout_seconds=timeout_seconds,
                    )
                    destination = (
                        by_workers[worker_count]["warmups"]
                        if phase == "warmup"
                        else by_workers[worker_count]["runs"]
                    )
                    destination.append(run)

        archive_consistency = _apply_archive_consistency(
            worker_results, repetitions,
        )
        source_sha256_after = generic._sha256_file(source_path)
        binary_sha256_after = generic._sha256_file(binary_path)
        harness_sha256_after = generic._sha256_file(harness_path)
        process_helper_sha256_after = generic._sha256_file(process_helper_path)
        prior_sha256_after = generic._sha256_file(prior) if prior is not None else None

        if generic_methods:
            generic_result = generic.benchmark_file(
                source_path,
                warmups=warmups,
                repetitions=repetitions,
                specs=generic.select_specs(generic_methods),
                work_dir=parent,
                keep_artifacts=keep_artifacts,
                timeout_seconds=timeout_seconds,
                evidence_policy={
                    "run_class": "engineering",
                    "paper_eligible": False,
                    "reason": "embedded comparison outside a frozen formal campaign",
                },
                expected_source_size_bytes=expected_input_size_bytes,
                expected_source_sha256=expected_input_sha256,
            )

        generated_paths = [artifact_root]
        if generic_result is not None:
            generic_configuration = generic_result.get("configuration")
            generic_artifact = (
                generic_configuration.get("artifact_root")
                if isinstance(generic_configuration, Mapping)
                else None
            )
            if generic_artifact:
                generated_paths.append(pathlib.Path(generic_artifact))
        git_after = _git_provenance(generated_paths)
        git_unchanged = all(
            git.get(key) == git_after.get(key)
            for key in ("commit", "status_porcelain", "tracked_diff_sha256")
        )
        brta_runs = [
            run
            for result in worker_results
            for run in (*result["warmups"], *result["runs"])
        ]
        invariants_ok = (
            all(run["success"] for run in brta_runs)
            and archive_consistency["status"] == "consistent"
            and source_sha256_after == source_sha256
            and binary_sha256_after == binary_sha256
            and harness_sha256_after == harness_sha256
            and process_helper_sha256_after == process_helper_sha256
            and prior_sha256_after == prior_initial_sha256
            and git_unchanged
        )
        generic_ok = (
            generic_result is None
            or all(
                method.get("available") and method.get("failure") is None
                for method in generic_result["methods"]
            )
        )
        document = {
            "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
            "status": (
                "complete" if invariants_ok and generic_ok
                else "complete_with_failures"
            ),
            "started_at_utc": started_at,
            "ended_at_utc": _utc_now(),
            "evidence_policy": {
                "run_class": "engineering",
                "paper_metrics_eligible": False,
                "reason": (
                    "This harness records reproducible engineering measurements "
                    "but does not certify a preregistered clean formal campaign."
                ),
            },
            "source": {
                "label": input_label,
                "path": str(source_path),
                "size_bytes": source_size,
                "sha256": source_sha256,
                "sha256_after_runs": source_sha256_after,
                "unchanged_during_benchmark": source_sha256_after == source_sha256,
                "expected_size_bytes": expected_input_size_bytes,
                "expected_sha256": (
                    generic._normalize_sha256(
                        expected_input_sha256, "expected_input_sha256",
                    )
                    if expected_input_sha256 is not None else None
                ),
                "declared_identity_verified": expected_input_size_bytes is not None,
            },
            "configuration": {
                "workers": list(selected_workers),
                "warmups_per_worker": warmups,
                "measured_repetitions_per_worker": repetitions,
                "timeout_seconds_per_operation": timeout_seconds,
                "requested_cli_options": requested_options,
                "prior_policy": (
                    "calibrated_once"
                    if calibrate_tensors is not None
                    else "external"
                    if external_prior is not None
                    else "none"
                ),
                "execution_schedule": schedule,
                "timing_scope": (
                    "wall clock starts immediately before process spawn and stops "
                    "after direct-child reap; input/output hashing and exact byte "
                    "comparison are outside the timed region"
                ),
                "io_policy": (
                    "Brevis file commands sync each successful temporary output "
                    "before atomic replacement inside the measured child process; "
                    "the harness does not issue a second fsync or directory fsync"
                ),
                "cache_policy": (
                    "best-effort buffered I/O; the harness does not flush host page "
                    "cache or claim cold-cache or guaranteed warm-cache timing"
                ),
                "artifact_root": str(artifact_root) if keep_artifacts else None,
                "artifacts_retained": keep_artifacts,
            },
            "prior": (
                {
                    "path": str(prior),
                    "size_bytes": prior.stat().st_size,
                    "sha256": prior_initial_sha256,
                    "sha256_after_runs": prior_sha256_after,
                    "unchanged_during_uses": prior_sha256_after == prior_initial_sha256,
                }
                if prior is not None else None
            ),
            "calibration": calibration,
            "provenance": {
                "git": {
                    **git,
                    "after": git_after,
                    "unchanged_during_benchmark": git_unchanged,
                },
                "harness": {
                    "path": str(harness_path),
                    "sha256": harness_sha256,
                    "sha256_after_runs": harness_sha256_after,
                    "unchanged_during_benchmark": (
                        harness_sha256_after == harness_sha256
                    ),
                },
                "process_helper": {
                    "path": str(process_helper_path),
                    "sha256": process_helper_sha256,
                    "sha256_after_runs": process_helper_sha256_after,
                    "unchanged_during_benchmark": (
                        process_helper_sha256_after == process_helper_sha256
                    ),
                },
                "binary": {
                    "path": str(binary_path),
                    "sha256": binary_sha256,
                    "sha256_after_runs": binary_sha256_after,
                    "unchanged_during_benchmark": binary_sha256_after == binary_sha256,
                },
            },
            "environment": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python": platform.python_version(),
                "cpu": generic._cpu_info(),
                "memory": generic._memory_info(),
                "filesystems": {
                    "source": generic._filesystem_info(source_path),
                    "work": generic._filesystem_info(artifact_root),
                },
                "thread_environment_variables": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
            },
            "worker_results": worker_results,
            "measured_archive_consistency": archive_consistency,
            "generic_baselines": (
                {
                    "evidence_policy": "embedded engineering comparison only",
                    "result": generic_result,
                }
                if generic_result is not None else None
            ),
            "limitations": [
                "The result remains engineering evidence even for a clean tree.",
                "The binary hash binds the executed artifact, but this harness does "
                "not rebuild it or infer its compiler and optimization flags.",
                "Direct-child resource counters do not aggregate descendant processes.",
                "Worker configurations are alternated but BRTA and optional generic "
                "methods are scheduled in separate phases.",
                "The harness does not drop or pin the host page cache.",
            ],
        }
        return document
    finally:
        if not keep_artifacts:
            shutil.rmtree(artifact_root, ignore_errors=True)


def _write_json(path: pathlib.Path, document: Mapping[str, Any], force: bool) -> None:
    output = path.resolve(strict=False)
    if output.exists() and not force:
        raise FileExistsError(f"output already exists (pass --force): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--workers",
        type=int,
        action="append",
        help="worker count; repeat to benchmark multiple counts (default: 1)",
    )
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prior", type=pathlib.Path)
    parser.add_argument("--calibrate-tensors", type=int)
    parser.add_argument("--expected-input-size", type=int)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--input-label")
    parser.add_argument(
        "--generic",
        action="append",
        metavar="METHOD/PROFILE",
        help="also run one registry entry from benchmarking.py; repeat as needed",
    )
    for name, flag, _ in ALL_OPTIONS:
        parser.add_argument(flag, dest=name, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.resolve(strict=False) == args.source.resolve(strict=False):
        parser.error("source and output must differ")
    options = {
        name: getattr(args, name)
        for name, _, _ in ALL_OPTIONS
        if getattr(args, name) is not None
    }
    try:
        document = benchmark_file(
            args.source,
            binary=args.binary,
            workers=args.workers or (1,),
            warmups=args.warmups,
            repetitions=args.repetitions,
            options=options,
            prior_path=args.prior,
            calibrate_tensors=args.calibrate_tensors,
            work_dir=args.work_dir,
            keep_artifacts=args.keep_artifacts,
            timeout_seconds=args.timeout_seconds,
            expected_input_size_bytes=args.expected_input_size,
            expected_input_sha256=args.expected_input_sha256,
            input_label=args.input_label,
            generic_methods=args.generic or (),
        )
        _write_json(args.output, document, args.force)
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"brta_benchmarking: {exc}", file=sys.stderr)
        return 2
    return 0 if document["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
