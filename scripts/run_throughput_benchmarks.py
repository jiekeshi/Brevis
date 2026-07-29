#!/usr/bin/env python3
"""Repeatable phase-level compression and decompression throughput benchmarks.

This harness is intentionally separate from ``run_benchmarks.py``.  The paper
corpus harness records useful per-shard command timings, while this harness
measures the wall-clock makespan from launching a whole checkpoint phase until
every output has been fsynced.  Cache conditioning, exactness verification, and
artifact cleanup happen outside the measured interval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Sequence

import run_benchmarks as base


SCHEMA_VERSION = 2
DEFAULT_WARMUPS = 1
DEFAULT_REPETITIONS = 5
DEFAULT_ZIPNN_WORKERS = 16
PROFILE_NAMES = ("practical", "resource-matched", "single-core")
BREVIS_EXECUTIONS = ("threads", "processes")
ZIPNN_EXECUTIONS = ("threads", "processes")
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ResourceSpec:
    """Declared CPU resource envelope for one method and profile.

    ``shard_jobs`` is the maximum number of simultaneously launched codec
    commands. ``processes_per_command`` includes adapter children (notably the
    Python + libdeflate-gzip pair), while ``cpu_slots_per_command`` counts
    simultaneously CPU-active slots. ``codec_workers`` is passed to the
    codec's worker/thread option. The distinction lets the raw data represent
    both one ZipNN process with 16 internal threads and a pool of up to 16
    one-thread ZipNN shard processes.
    """

    profile: str
    method: str
    execution_model: str
    declared_cpu_slots: int
    shard_jobs: int
    processes_per_command: int
    cpu_slots_per_command: int
    codec_workers: int
    expected_threads_per_process: int
    cpu_affinity: str | None
    environment: dict[str, str]
    runtime_controls: dict[str, Any]
    note: str

    def provenance(self, shard_count: int) -> dict[str, Any]:
        effective_jobs = min(self.shard_jobs, shard_count)
        return {
            **asdict(self),
            "configured_process_concurrency": (
                self.shard_jobs * self.processes_per_command
            ),
            "effective_shard_jobs": effective_jobs,
            "effective_process_concurrency": (
                effective_jobs * self.processes_per_command
            ),
            "effective_cpu_slot_upper_bound": min(
                self.declared_cpu_slots,
                effective_jobs * self.cpu_slots_per_command,
            ),
            "affinity_enforced": self.cpu_affinity is not None,
        }


@dataclass(frozen=True)
class CommandMeasurement:
    shard_index: int
    shard: str
    command: list[str]
    environment: dict[str, str]
    started_offset_seconds: float
    finished_offset_seconds: float
    wall_seconds: float
    peak_process_tree_rss_bytes: int | None
    input_path: str
    output_path: str
    input_bytes: int
    output_bytes: int
    stdout_path: str
    stderr_path: str


@dataclass(frozen=True)
class ArtifactWorkspace:
    root: Path
    marker: Path
    archives: tuple[Path, ...]
    restored: tuple[Path, ...]
    pair_id: str
    attempt_id: str


CommandBuilder = Callable[
    [
        str,
        str,
        Path,
        Path,
        ResourceSpec,
        Path,
        base.BrevisConfig,
    ],
    list[str],
]


def stable_hash(value: Any, length: int | None = None) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    return digest[:length] if length is not None else digest


def quantile_type7(values: Sequence[float], probability: float) -> float:
    """Return the R-7/NumPy-linear sample quantile."""

    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution requires at least one value")
    q1 = quantile_type7(values, 0.25)
    q3 = quantile_type7(values, 0.75)
    return {
        "n": len(values),
        "median": median(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(values),
        "max": max(values),
    }


def parse_cpu_list(value: str) -> tuple[str, int]:
    """Validate Linux taskset syntax and return its number of unique CPUs."""

    cpus: set[int] = set()
    try:
        for component in value.split(","):
            component = component.strip()
            if not component:
                raise ValueError
            if "-" in component:
                first_text, last_text = component.split("-", 1)
                first, last = int(first_text), int(last_text)
                if first < 0 or last < first:
                    raise ValueError
                cpus.update(range(first, last + 1))
            else:
                cpu = int(component)
                if cpu < 0:
                    raise ValueError
                cpus.add(cpu)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "CPU list must look like 0-7,16-23"
        ) from exc
    if not cpus:
        raise argparse.ArgumentTypeError("CPU list must not be empty")
    return value, len(cpus)


def expand_cpu_list(value: str) -> tuple[int, ...]:
    parse_cpu_list(value)
    cpus: set[int] = set()
    for component in value.split(","):
        component = component.strip()
        if "-" in component:
            first_text, last_text = component.split("-", 1)
            cpus.update(range(int(first_text), int(last_text) + 1))
        else:
            cpus.add(int(component))
    return tuple(sorted(cpus))


def scheduler_cpu_set() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        try:
            return tuple(sorted(os.sched_getaffinity(0)))
        except OSError:
            pass
    return tuple(range(os.cpu_count() or 1))


def compact_cpu_list(cpus: Sequence[int]) -> str:
    if not cpus:
        raise base.BenchmarkError("CPU affinity cannot be empty")
    ordered = sorted(set(cpus))
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def bind_resource_affinity(
    resources: dict[tuple[str, str], ResourceSpec],
    cpu_pool: Sequence[int],
    *,
    taskset_available: bool,
) -> dict[tuple[str, str], ResourceSpec]:
    """Pin every declared slot to CPUs allowed by the current scheduler."""

    if not taskset_available:
        matched = [
            key for key in resources if key[0] == "resource-matched"
        ]
        if matched:
            raise base.BenchmarkError(
                "resource-matched requires taskset-enforced CPU affinity"
            )
        return resources
    ordered_pool = tuple(sorted(set(cpu_pool)))
    output = {}
    for key, resource in resources.items():
        if resource.declared_cpu_slots > len(ordered_pool):
            raise base.BenchmarkError(
                f"{key[0]}/{key[1]} requests {resource.declared_cpu_slots} "
                f"CPU slots, but scheduler affinity exposes {len(ordered_pool)}"
            )
        selected = ordered_pool[: resource.declared_cpu_slots]
        output[key] = replace(
            resource,
            cpu_affinity=compact_cpu_list(selected),
        )
    return output


def thread_environment(workers: int) -> dict[str, str]:
    value = str(workers)
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "RAYON_NUM_THREADS": value,
    }


def resolve_resource(
    method: str,
    profile: str,
    *,
    matched_workers: int,
    practical_workers: int,
    brevis_workers: int,
    zipnn_workers: int,
    zipnn_execution: str,
    cpu_affinity: str | None,
    brevis_execution: str = "threads",
) -> ResourceSpec:
    """Resolve a method/profile pair to an explicit resource envelope."""

    if method not in base.GENERIC_METHODS:
        raise base.BenchmarkError(
            f"throughput harness supports generic reversible codecs only: {method}"
        )
    if profile not in PROFILE_NAMES:
        raise base.BenchmarkError(f"unknown resource profile: {profile}")
    for name, value in (
        ("matched_workers", matched_workers),
        ("practical_workers", practical_workers),
        ("brevis_workers", brevis_workers),
        ("zipnn_workers", zipnn_workers),
    ):
        if value < 1:
            raise base.BenchmarkError(f"{name} must be positive")
    if zipnn_execution not in ZIPNN_EXECUTIONS:
        raise base.BenchmarkError(
            f"unknown ZipNN execution model: {zipnn_execution}"
        )
    if brevis_execution not in BREVIS_EXECUTIONS:
        raise base.BenchmarkError(
            f"unknown Brevis execution model: {brevis_execution}"
        )

    if profile == "single-core":
        slots = 1
    elif profile == "resource-matched":
        slots = matched_workers
    elif method == "brevis":
        slots = brevis_workers
    elif method == "zipnn":
        slots = zipnn_workers
    else:
        slots = practical_workers

    if method == "brevis" and brevis_execution == "threads":
        return ResourceSpec(
            profile=profile,
            method=method,
            execution_model="single_process_internal_workers",
            declared_cpu_slots=slots,
            shard_jobs=1,
            processes_per_command=1,
            cpu_slots_per_command=slots,
            codec_workers=slots,
            expected_threads_per_process=slots,
            cpu_affinity=cpu_affinity,
            environment=thread_environment(slots),
            runtime_controls={},
            note="One Brevis process; --workers consumes the declared CPU slots.",
        )

    if method == "brevis":
        return ResourceSpec(
            profile=profile,
            method=method,
            execution_model="one_worker_brevis_process_pool_across_shards",
            declared_cpu_slots=slots,
            shard_jobs=slots,
            processes_per_command=1,
            cpu_slots_per_command=1,
            codec_workers=1,
            expected_threads_per_process=1,
            cpu_affinity=cpu_affinity,
            environment=thread_environment(1),
            runtime_controls={"brevis_workers_per_process": 1},
            note=(
                "Up to the declared number of independent one-worker Brevis "
                "processes, one per shard. Effective process count is bounded "
                "by checkpoint shard count; no nested worker parallelism."
            ),
        )

    if method == "zipnn" and zipnn_execution == "threads":
        return ResourceSpec(
            profile=profile,
            method=method,
            execution_model="single_process_internal_threads",
            declared_cpu_slots=slots,
            shard_jobs=1,
            processes_per_command=1,
            cpu_slots_per_command=slots,
            codec_workers=slots,
            expected_threads_per_process=slots,
            cpu_affinity=cpu_affinity,
            environment=thread_environment(1),
            runtime_controls={
                "torch_intra_op_threads": 1,
                "torch_inter_op_threads": 1,
                "zipnn_native_threads": slots,
            },
            note=(
                "One ZipNN adapter process at a time; ZipNN threads equals "
                "the declared CPU slots (16 in its practical profile). "
                "PyTorch intra-op and inter-op are forced to one separately."
            ),
        )

    if method == "zipnn":
        return ResourceSpec(
            profile=profile,
            method=method,
            execution_model="one_thread_process_pool_across_shards",
            declared_cpu_slots=slots,
            shard_jobs=slots,
            processes_per_command=1,
            cpu_slots_per_command=1,
            codec_workers=1,
            expected_threads_per_process=1,
            cpu_affinity=cpu_affinity,
            environment=thread_environment(1),
            runtime_controls={
                "torch_intra_op_threads": 1,
                "torch_inter_op_threads": 1,
                "zipnn_native_threads": 1,
            },
            note=(
                "Up to the declared number of independent one-thread ZipNN "
                "adapter processes, one per shard. Effective process count is "
                "bounded by checkpoint shard count."
            ),
        )

    libdeflate = method == "libdeflate-1"
    return ResourceSpec(
        profile=profile,
        method=method,
        execution_model=(
            "python_adapter_with_codec_child_pool_across_shards"
            if libdeflate
            else "one_thread_process_pool_across_shards"
        ),
        declared_cpu_slots=slots,
        shard_jobs=slots,
        processes_per_command=2 if libdeflate else 1,
        cpu_slots_per_command=1,
        codec_workers=1,
        expected_threads_per_process=1,
        cpu_affinity=cpu_affinity,
        environment=thread_environment(1),
        runtime_controls={},
        note=(
            (
                "Each shard launches one mostly-idle Python adapter plus one "
                "single-thread libdeflate-gzip child; CPU-active slots remain "
                "one per shard."
            )
            if libdeflate
            else (
                "Single-thread codec commands are parallelized across shards "
                "up to the declared CPU-slot limit."
            )
        ),
    )


def build_command(
    method: str,
    operation: str,
    source: Path,
    output: Path,
    resource: ResourceSpec,
    brevis_bin: Path,
    brevis_config: base.BrevisConfig,
) -> list[str]:
    if method == "zipnn":
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_zipnn-adapter",
            operation,
            str(source),
            str(output),
            "--threads",
            str(resource.codec_workers),
        ]
    else:
        command = base.command_for(
            method,
            operation,
            source,
            output,
            resource.codec_workers,
            brevis_bin,
            brevis_config,
        )
    if resource.cpu_affinity:
        command = ["taskset", "--cpu-list", resource.cpu_affinity, *command]
    return command


def zipnn_adapter_main(argv: Sequence[str]) -> int:
    """Run ZipNN with PyTorch's unrelated thread pools pinned to one."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("operation", choices=("compress", "decompress"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import benchmark_codecs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    codec = benchmark_codecs.CODECS["zipnn"]
    operation = (
        codec.compress if args.operation == "compress" else codec.decompress
    )
    operation(args.source, args.output, args.threads)
    return 0


def _sample_peak_rss(process: subprocess.Popen[bytes], peak: list[int | None]) -> None:
    value = base.process_tree_rss(process.pid)
    if value is not None:
        peak[0] = max(peak[0] or 0, value)


def measure_command(
    *,
    command: list[str],
    environment: dict[str, str],
    log_dir: Path,
    command_id: str,
    shard_index: int,
    shard: str,
    input_path: Path,
    output_path: Path,
    phase_started: float,
    rss_poll_seconds: float = 0.01,
) -> CommandMeasurement:
    """Measure one command without adding a fixed polling delay to completion."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{command_id}.stdout"
    stderr_path = log_dir / f"{command_id}.stderr"
    merged_environment = {**os.environ, **environment}
    command_started = time.perf_counter()
    peak: list[int | None] = [None]
    stop = threading.Event()

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env=merged_environment,
        )
        _sample_peak_rss(process, peak)

        def monitor() -> None:
            while not stop.wait(rss_poll_seconds):
                _sample_peak_rss(process, peak)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        return_code = process.wait()
        _sample_peak_rss(process, peak)
        stop.set()
        monitor_thread.join()

    if return_code:
        raise base.BenchmarkError(
            f"command failed ({return_code}): {shlex.join(command)}\n"
            f"{base.tail(stderr_path)}"
        )
    if output_path.is_symlink() or not output_path.is_file():
        raise base.BenchmarkError(
            f"command produced no safe regular output file: {output_path}"
        )
    base.fsync_output(output_path)
    command_finished = time.perf_counter()
    return CommandMeasurement(
        shard_index=shard_index,
        shard=shard,
        command=command,
        environment=environment,
        started_offset_seconds=command_started - phase_started,
        finished_offset_seconds=command_finished - phase_started,
        wall_seconds=command_finished - command_started,
        peak_process_tree_rss_bytes=peak[0],
        input_path=str(input_path),
        output_path=str(output_path),
        input_bytes=input_path.stat().st_size,
        output_bytes=output_path.stat().st_size,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def condition_cache(
    inputs: Iterable[Path],
    cache_mode: str,
    drop_caches_command: str | None,
) -> float:
    started = time.perf_counter()
    if cache_mode == "hot":
        for path in inputs:
            base.warm_page_cache(path)
    elif cache_mode == "cold":
        if not base.drop_caches(drop_caches_command):
            raise base.BenchmarkError("cold-cache control is unavailable")
    elif cache_mode != "unconditioned":
        raise base.BenchmarkError(f"unknown cache mode: {cache_mode}")
    return time.perf_counter() - started


def run_phase(
    *,
    method: str,
    operation: str,
    tasks: Sequence[tuple[int, Path, Path]],
    resource: ResourceSpec,
    brevis_bin: Path,
    brevis_config: base.BrevisConfig,
    cache_mode: str,
    drop_caches_command: str | None,
    log_dir: Path,
    phase_id: str,
    artifact_workspace: ArtifactWorkspace,
    command_builder: CommandBuilder = build_command,
) -> dict[str, Any]:
    """Run all checkpoint shards and return phase-makespan measurements."""

    if not tasks:
        raise base.BenchmarkError("cannot measure an empty phase")
    for _, _, output in tasks:
        prepare_artifact_output(artifact_workspace, output)

    cache_seconds = condition_cache(
        (input_path for _, input_path, _ in tasks),
        cache_mode,
        drop_caches_command,
    )
    phase_started_at = base.utc_now()
    phase_started = time.perf_counter()
    measurements: list[CommandMeasurement] = []
    max_workers = min(resource.shard_jobs, len(tasks))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for shard_index, input_path, output_path in tasks:
            command = command_builder(
                method,
                operation,
                input_path,
                output_path,
                resource,
                brevis_bin,
                brevis_config,
            )
            command_id = f"{phase_id}-shard-{shard_index:05d}"
            future = executor.submit(
                measure_command,
                command=command,
                environment=resource.environment,
                log_dir=log_dir,
                command_id=command_id,
                shard_index=shard_index,
                shard=input_path.name,
                input_path=input_path,
                output_path=output_path,
                phase_started=phase_started,
            )
            futures[future] = shard_index
        for future in as_completed(futures):
            measurements.append(future.result())

    phase_finished = time.perf_counter()
    measurements.sort(key=lambda item: item.shard_index)
    makespan = phase_finished - phase_started
    logical_bytes = sum(
        (
            source.stat().st_size
            if operation == "compress"
            else output.stat().st_size
        )
        for _, source, output in tasks
    )
    return {
        "phase_started_at": phase_started_at,
        "phase_finished_at": base.utc_now(),
        "phase_makespan_seconds": makespan,
        "cache_conditioning_seconds_excluded": cache_seconds,
        "logical_uncompressed_bytes": logical_bytes,
        "physical_input_bytes": sum(item.input_bytes for item in measurements),
        "output_bytes": sum(item.output_bytes for item in measurements),
        "throughput_bytes_per_second": logical_bytes / makespan,
        "max_command_peak_process_tree_rss_bytes": max(
            (
                item.peak_process_tree_rss_bytes
                for item in measurements
                if item.peak_process_tree_rss_bytes is not None
            ),
            default=None,
        ),
        "sum_command_wall_seconds_diagnostic_only": sum(
            item.wall_seconds for item in measurements
        ),
        "commands": [asdict(item) for item in measurements],
    }


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def checkpoint_relative_path(
    checkpoint: base.Checkpoint,
    path: Path,
) -> str:
    try:
        relative = path.absolute().relative_to(checkpoint.directory.absolute())
    except ValueError as exc:
        raise base.BenchmarkError(
            f"checkpoint shard is outside its directory: {path}"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise base.BenchmarkError(f"unsafe checkpoint shard path: {path}")
    return relative.as_posix()


def checkpoint_descriptor(checkpoint: base.Checkpoint) -> dict[str, Any]:
    """Build a content-authenticated checkpoint identity once per invocation."""

    manifest_path = checkpoint.directory / "download-manifest.json"
    manifest: dict[str, Any] | None = None
    manifest_sha256: str | None = None
    manifest_reported_verified = False
    declared: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise base.BenchmarkError(
                f"invalid checkpoint manifest: {manifest_path}"
            ) from exc
        manifest_sha256 = base.file_sha256(manifest_path)
        weights = manifest.get("weights")
        if not isinstance(weights, list):
            raise base.BenchmarkError(
                f"checkpoint manifest has no valid weights: {manifest_path}"
            )
        for item in weights:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise base.BenchmarkError(
                    f"checkpoint manifest has an invalid weight row: {manifest_path}"
                )
            if item["path"] in declared:
                raise base.BenchmarkError(
                    f"checkpoint manifest repeats {item['path']}"
                )
            declared[item["path"]] = item
        manifest_reported_verified = manifest.get("sha256_verified") is True

    relative_paths = [
        checkpoint_relative_path(checkpoint, path)
        for path in checkpoint.files
    ]
    if manifest is not None and set(relative_paths) != set(declared):
        raise base.BenchmarkError(
            f"manifest/checkpoint shard set differs: {manifest_path}"
        )

    files = []
    for path, relative in zip(checkpoint.files, relative_paths):
        actual_size = path.stat().st_size
        item = declared.get(relative)
        if item is not None and item.get("size") != actual_size:
            raise base.BenchmarkError(
                f"manifest size mismatch for {path}: "
                f"{item.get('size')} != {actual_size}"
            )
        digest = base.file_sha256(path)
        if item is not None:
            declared_digest = item.get("sha256")
            if not valid_sha256(declared_digest):
                raise base.BenchmarkError(
                    f"manifest has invalid SHA-256 for {path}"
                )
            if digest.lower() != declared_digest.lower():
                raise base.BenchmarkError(
                    f"manifest SHA-256 mismatch for {path}: "
                    f"{declared_digest} != {digest}"
                )
            digest_source = "computed_and_matched_manifest"
        else:
            digest_source = "computed_content"
        files.append(
            {
                "path": str(path),
                "relative_path": relative,
                "name": path.name,
                "bytes": actual_size,
                "sha256": digest,
                "sha256_source": digest_source,
            }
        )

    fingerprint_basis = {
        "files": [
            {
                "relative_path": item["relative_path"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for item in files
        ],
    }
    return {
        "name": checkpoint.name,
        "directory": str(checkpoint.directory),
        "repo_id": checkpoint.repo_id,
        "revision": checkpoint.revision,
        "manifest_path": str(manifest_path) if manifest is not None else None,
        "manifest_sha256": manifest_sha256,
        "manifest_reported_sha256_verified": manifest_reported_verified,
        "content_sha256_recomputed": True,
        "identity_source": (
            "computed_content_sha256_matched_manifest"
            if manifest is not None
            else "computed_content_sha256"
        ),
        "files": files,
        "content_fingerprint_algorithm": (
            "sha256(canonical(relative_path,bytes,actual_sha256)[])"
        ),
        "fingerprint": stable_hash(fingerprint_basis),
    }


def pair_identity(
    *,
    experiment_id: str,
    checkpoint: base.Checkpoint,
    method: str,
    profile: str,
    sample_kind: str,
    sample_index: int,
    cache_mode: str,
    resource: ResourceSpec,
    descriptor: dict[str, Any],
    expected_measured_repetitions: int,
    round_trip_check_performed: bool,
) -> dict[str, Any]:
    verification = (
        exactness_protocol(method)
        if round_trip_check_performed
        else skipped_exactness_protocol()
    )
    return {
        "experiment_id": experiment_id,
        "checkpoint": checkpoint.name,
        "checkpoint_fingerprint": descriptor["fingerprint"],
        "method": method,
        "profile": profile,
        "execution_model": resource.execution_model,
        "sample_kind": sample_kind,
        "sample_index": sample_index,
        "expected_measured_repetitions": expected_measured_repetitions,
        "round_trip_check_performed": round_trip_check_performed,
        "exactness_scope": verification["scope"],
        "exactness_verification": verification,
        "cache_mode": cache_mode,
        "resource": resource.provenance(len(checkpoint.files)),
    }


def safe_component(value: str) -> str:
    rendered = base.slug(value)
    if not rendered or rendered in (".", "..") or rendered.startswith("."):
        return f"item-{stable_hash(value, 12)}"
    return rendered


def ensure_artifact_contained(root: Path, path: Path) -> None:
    root_resolved = root.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise base.BenchmarkError(
            f"artifact path escapes owned workspace: {path}"
        )


def ensure_no_artifact_symlinks(root: Path, path: Path) -> None:
    ensure_artifact_contained(root, path)
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise base.BenchmarkError(
            f"artifact path is not lexically contained: {path}"
        ) from exc
    current = root
    if current.is_symlink():
        raise base.BenchmarkError(f"artifact root is a symlink: {current}")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise base.BenchmarkError(
                f"artifact path contains a symlink: {current}"
            )


def archive_suffix(method: str) -> str:
    return {
        "brevis": ".brv",
        "zipnn": ".safetensors.znn",
        "zstd-9": ".zst",
        "lz4-hc-9": ".lz4",
        "libdeflate-1": ".gz",
        "snappy": ".snappy",
    }[method]


def validate_artifact_results_root(results: Path) -> Path:
    results_absolute = results.absolute()
    if results_absolute.is_symlink():
        raise base.BenchmarkError(
            f"results directory must not be a symlink: {results}"
        )
    results_absolute.mkdir(parents=True, exist_ok=True)
    work_root = results_absolute / "work"
    if work_root.is_symlink():
        raise base.BenchmarkError(
            f"artifact work root must not be a symlink: {work_root}"
        )
    return results_absolute


def validate_results_output_path(results: Path, path: Path) -> None:
    root = validate_artifact_results_root(results)
    ensure_artifact_contained(root, path)
    ensure_no_artifact_symlinks(root, path.parent)
    if path.is_symlink():
        raise base.BenchmarkError(
            f"results output must not be a symlink: {path}"
        )
    if path.exists() and not path.is_file():
        raise base.BenchmarkError(
            f"results output is not a regular file: {path}"
        )


def create_artifact_workspace(
    *,
    results: Path,
    checkpoint: base.Checkpoint,
    method: str,
    profile: str,
    sample_kind: str,
    sample_index: int,
    pair_id: str,
    attempt_id: str,
) -> ArtifactWorkspace:
    results_absolute = validate_artifact_results_root(results)
    work_root = results_absolute / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    workspace = (
        work_root
        / safe_component(checkpoint.name)
        / safe_component(profile)
        / safe_component(method)
        / f"{sample_kind}-{sample_index:03d}-{pair_id}"
        / f"attempt-{attempt_id}"
    )
    ensure_no_artifact_symlinks(work_root, workspace.parent)
    if workspace.exists() or workspace.is_symlink():
        raise base.BenchmarkError(
            f"refusing to reuse artifact workspace: {workspace}"
        )
    workspace.mkdir(parents=True, exist_ok=False)
    ensure_no_artifact_symlinks(work_root, workspace)
    marker = workspace / ".brevis-throughput-owner.json"
    marker_payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "brevis-throughput-artifact-workspace",
        "pair_id": pair_id,
        "attempt_id": attempt_id,
        "workspace": str(workspace.resolve()),
    }
    with marker.open("x", encoding="utf-8") as output:
        output.write(json.dumps(marker_payload, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    suffix = archive_suffix(method)
    archives = tuple(
        workspace
        / "archives"
        / f"{index:05d}-{safe_component(source.name)}{suffix}"
        for index, source in enumerate(checkpoint.files)
    )
    restored = tuple(
        workspace / "restored" / f"{index:05d}-{safe_component(source.name)}"
        for index, source in enumerate(checkpoint.files)
    )
    return ArtifactWorkspace(
        root=workspace,
        marker=marker,
        archives=archives,
        restored=restored,
        pair_id=pair_id,
        attempt_id=attempt_id,
    )


def validate_artifact_workspace(workspace: ArtifactWorkspace) -> None:
    ensure_no_artifact_symlinks(workspace.root, workspace.marker)
    if workspace.marker.is_symlink() or not workspace.marker.is_file():
        raise base.BenchmarkError(
            f"artifact ownership marker is missing: {workspace.marker}"
        )
    try:
        marker = json.loads(workspace.marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise base.BenchmarkError(
            f"artifact ownership marker is invalid: {workspace.marker}"
        ) from exc
    expected = {
        "kind": "brevis-throughput-artifact-workspace",
        "pair_id": workspace.pair_id,
        "attempt_id": workspace.attempt_id,
        "workspace": str(workspace.root.resolve()),
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        raise base.BenchmarkError(
            f"artifact ownership marker does not match: {workspace.marker}"
        )


def prepare_artifact_output(
    workspace: ArtifactWorkspace,
    output: Path,
) -> None:
    validate_artifact_workspace(workspace)
    ensure_no_artifact_symlinks(workspace.root, output.parent)
    if output.exists() or output.is_symlink():
        raise base.BenchmarkError(
            f"refusing to replace an existing artifact: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_artifact_symlinks(workspace.root, output.parent)


def safe_unlink_artifact(
    workspace: ArtifactWorkspace,
    path: Path,
) -> None:
    validate_artifact_workspace(workspace)
    ensure_artifact_contained(workspace.root, path)
    ensure_no_artifact_symlinks(workspace.root, path)
    if path.is_symlink():
        raise base.BenchmarkError(f"refusing to unlink symlink artifact: {path}")
    if path.exists():
        if not path.is_file():
            raise base.BenchmarkError(
                f"refusing to unlink non-file artifact: {path}"
            )
        path.unlink()


def cleanup_artifact_workspace(workspace: ArtifactWorkspace) -> None:
    """Remove every regular file from an owned attempt without following links."""

    validate_artifact_workspace(workspace)
    files: list[Path] = []
    directories: list[Path] = []
    for current, directory_names, file_names in os.walk(
        workspace.root,
        topdown=False,
        followlinks=False,
    ):
        current_path = Path(current)
        for name in file_names:
            path = current_path / name
            ensure_artifact_contained(workspace.root, path)
            if path.is_symlink() or not path.is_file():
                raise base.BenchmarkError(
                    f"refusing to clean unsafe artifact file: {path}"
                )
            files.append(path)
        for name in directory_names:
            path = current_path / name
            ensure_artifact_contained(workspace.root, path)
            if path.is_symlink() or not path.is_dir():
                raise base.BenchmarkError(
                    f"refusing to clean unsafe artifact directory: {path}"
                )
            directories.append(path)

    for path in files:
        if path != workspace.marker:
            path.unlink()
    validate_artifact_workspace(workspace)
    workspace.marker.unlink()
    for path in directories:
        path.rmdir()
    workspace.root.rmdir()


def exactness_protocol(method: str) -> dict[str, Any]:
    if base.CODEC_POLICIES[method].tensor_exact:
        return {
            "scope": "safetensors_tensor_bit_exact",
            "algorithm": "source_vs_restored_tensor_payload_byte_comparison",
            "source_identity": "live_source_safetensors_tensor_payloads",
            "source_reread_during_final_verification": True,
            "restored_reread_during_final_verification": True,
            "metadata_checked": True,
            "round_trip_check_performed": True,
        }
    return {
        "scope": "whole_file_byte_exact",
        "algorithm": "restored_sha256_equals_precomputed_source_sha256",
        "hash": "SHA-256",
        "source_identity": (
            "checkpoint_descriptor.files[].sha256 computed by a complete "
            "pre-run source read"
        ),
        "source_reread_during_final_verification": False,
        "restored_reread_during_final_verification": True,
        "size_checked_before_hash": True,
        "round_trip_check_performed": True,
    }


def skipped_exactness_protocol() -> dict[str, Any]:
    return {
        "scope": "not_checked",
        "algorithm": "not_performed_by_explicit_throughput_fast_mode",
        "source_identity": "not_used_for_final_round_trip_check",
        "source_reread_during_final_verification": False,
        "restored_reread_during_final_verification": False,
        "round_trip_check_performed": False,
        "verified_shards": 0,
    }


def verify_round_trip(
    *,
    method: str,
    checkpoint: base.Checkpoint,
    descriptor: dict[str, Any],
    restored: Sequence[Path],
) -> tuple[bool, dict[str, Any]]:
    protocol = exactness_protocol(method)
    if len(restored) != len(checkpoint.files):
        raise base.BenchmarkError("restored shard count differs from source")
    if base.CODEC_POLICIES[method].tensor_exact:
        verified_shards = 0
        for source, restored_path in zip(checkpoint.files, restored):
            if not base.tensor_exact(source, restored_path):
                return False, {
                    **protocol,
                    "round_trip_check_performed": True,
                    "verified_shards": verified_shards,
                }
            verified_shards += 1
        return True, {
            **protocol,
            "round_trip_check_performed": True,
            "verified_shards": verified_shards,
        }

    descriptor_files = descriptor.get("files")
    if (
        not isinstance(descriptor_files, list)
        or len(descriptor_files) != len(checkpoint.files)
    ):
        raise base.BenchmarkError(
            "checkpoint descriptor does not cover every source shard"
        )
    verified_shards = 0
    for source, restored_path, expected in zip(
        checkpoint.files,
        restored,
        descriptor_files,
    ):
        relative = checkpoint_relative_path(checkpoint, source)
        if expected.get("relative_path") != relative:
            raise base.BenchmarkError(
                f"checkpoint descriptor order/path mismatch for {source}"
            )
        expected_digest = expected.get("sha256")
        expected_size = expected.get("bytes")
        if not valid_sha256(expected_digest) or not isinstance(expected_size, int):
            raise base.BenchmarkError(
                f"checkpoint descriptor lacks content identity for {source}"
            )
        if restored_path.stat().st_size != expected_size:
            return False, {
                **protocol,
                "round_trip_check_performed": True,
                "verified_shards": verified_shards,
            }
        if base.file_sha256(restored_path).lower() != expected_digest.lower():
            return False, {
                **protocol,
                "round_trip_check_performed": True,
                "verified_shards": verified_shards,
            }
        verified_shards += 1
    return True, {
        **protocol,
        "round_trip_check_performed": True,
        "verified_shards": verified_shards,
    }


def phase_record(
    *,
    identity: dict[str, Any],
    pair_id: str,
    attempt_id: str,
    operation: str,
    phase: dict[str, Any],
    schedule_position: int,
    exact: bool | None,
    verification: dict[str, Any],
) -> dict[str, Any]:
    operation_id = stable_hash(
        {
            "pair_id": pair_id,
            "attempt_id": attempt_id,
            "operation": operation,
        },
        20,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        **identity,
        "run_id": operation_id,
        "pair_id": pair_id,
        "attempt_id": attempt_id,
        "status": "ok" if exact is not False else "failed",
        "operation": operation,
        "schedule_position": schedule_position,
        "round_trip_check_performed": verification[
            "round_trip_check_performed"
        ],
        "round_trip_exact": exact,
        "exactness_scope": verification["scope"],
        "exactness_verification": verification,
        **phase,
    }


def latest_attempt_for_pair(
    records: Sequence[dict[str, Any]],
    pair_id: str,
) -> list[dict[str, Any]]:
    attempts: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        if record.get("pair_id") != pair_id or not record.get("attempt_id"):
            continue
        attempts.setdefault(record["attempt_id"], []).append((index, record))
    if not attempts:
        return []
    latest = max(
        attempts.values(),
        key=lambda rows: max(index for index, _ in rows),
    )
    return [record for _, record in sorted(latest)]


def complete_attempt_phases(
    records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]] | None:
    if not records:
        return None
    attempt_ids = {record.get("attempt_id") for record in records}
    if len(attempt_ids) != 1 or None in attempt_ids:
        return None
    operations: dict[str, dict[str, Any]] = {}
    for record in records:
        operation = record.get("operation")
        if operation in ("compress", "decompress"):
            operations[operation] = record
    if set(operations) != {"compress", "decompress"}:
        return None
    check_modes = {
        record.get("round_trip_check_performed")
        for record in operations.values()
    }
    if len(check_modes) != 1 or None in check_modes:
        return None
    check_performed = next(iter(check_modes))
    if not isinstance(check_performed, bool):
        return None
    if not all(record.get("status") == "ok" for record in operations.values()):
        return None
    if check_performed and not all(
        record.get("round_trip_exact") is True
        for record in operations.values()
    ):
        return None
    if not check_performed and not all(
        record.get("round_trip_exact") is None
        for record in operations.values()
    ):
        return None
    return operations


def run_round(
    *,
    results: Path,
    log: base.ResultLog,
    experiment_id: str,
    checkpoint: base.Checkpoint,
    descriptor: dict[str, Any],
    method: str,
    profile: str,
    resource: ResourceSpec,
    sample_kind: str,
    sample_index: int,
    expected_measured_repetitions: int,
    schedule_position: int,
    cache_mode: str,
    drop_caches_command: str | None,
    brevis_bin: Path,
    brevis_config: base.BrevisConfig,
    keep_artifacts: bool,
    keep_failed_artifacts: bool,
    skip_round_trip_check: bool,
    rerun: bool,
    command_builder: CommandBuilder = build_command,
) -> list[dict[str, Any]]:
    identity = pair_identity(
        experiment_id=experiment_id,
        checkpoint=checkpoint,
        method=method,
        profile=profile,
        sample_kind=sample_kind,
        sample_index=sample_index,
        cache_mode=cache_mode,
        resource=resource,
        descriptor=descriptor,
        expected_measured_repetitions=expected_measured_repetitions,
        round_trip_check_performed=not skip_round_trip_check,
    )
    pair_id = stable_hash(identity, 20)
    latest_attempt = latest_attempt_for_pair(log.records, pair_id)
    complete_previous = complete_attempt_phases(latest_attempt)
    if not rerun and complete_previous is not None:
        print(
            f"  resume {checkpoint.name}/{profile}/{method}/"
            f"{sample_kind}-{sample_index}"
        )
        return [
            complete_previous[operation]
            for operation in ("compress", "decompress")
        ]

    attempt_id = uuid.uuid4().hex
    operation_ids = {
        operation: stable_hash(
            {
                "pair_id": pair_id,
                "attempt_id": attempt_id,
                "operation": operation,
            },
            20,
        )
        for operation in ("compress", "decompress")
    }
    log.append(
        {
            "schema_version": SCHEMA_VERSION,
            **identity,
            "run_id": stable_hash(
                {
                    "pair_id": pair_id,
                    "attempt_id": attempt_id,
                    "operation": "pair",
                },
                20,
            ),
            "pair_id": pair_id,
            "attempt_id": attempt_id,
            "status": "started",
            "operation": "pair",
            "schedule_position": schedule_position,
            "started_at": base.utc_now(),
        }
    )
    print(
        f"  run {checkpoint.name}/{profile}/{method}/"
        f"{sample_kind}-{sample_index}"
    )

    current_operation = "compress"
    phase_records_appended = False
    workspace: ArtifactWorkspace | None = None
    try:
        workspace = create_artifact_workspace(
            results=results,
            checkpoint=checkpoint,
            method=method,
            profile=profile,
            sample_kind=sample_kind,
            sample_index=sample_index,
            pair_id=pair_id,
            attempt_id=attempt_id,
        )
        compress_tasks = [
            (index, source, workspace.archives[index])
            for index, source in enumerate(checkpoint.files)
        ]
        decompress_tasks = [
            (index, workspace.archives[index], workspace.restored[index])
            for index in range(len(checkpoint.files))
        ]
        compress = run_phase(
            method=method,
            operation="compress",
            tasks=compress_tasks,
            resource=resource,
            brevis_bin=brevis_bin,
            brevis_config=brevis_config,
            cache_mode=cache_mode,
            drop_caches_command=drop_caches_command,
            log_dir=results / "logs",
            phase_id=operation_ids["compress"],
            artifact_workspace=workspace,
            command_builder=command_builder,
        )
        current_operation = "decompress"
        decompress = run_phase(
            method=method,
            operation="decompress",
            tasks=decompress_tasks,
            resource=resource,
            brevis_bin=brevis_bin,
            brevis_config=brevis_config,
            cache_mode=cache_mode,
            drop_caches_command=drop_caches_command,
            log_dir=results / "logs",
            phase_id=operation_ids["decompress"],
            artifact_workspace=workspace,
            command_builder=command_builder,
        )
        if skip_round_trip_check:
            exact: bool | None = None
            verification = skipped_exactness_protocol()
        else:
            exact, verification = verify_round_trip(
                method=method,
                checkpoint=checkpoint,
                descriptor=descriptor,
                restored=workspace.restored,
            )
        compress_record = phase_record(
            identity=identity,
            pair_id=pair_id,
            attempt_id=attempt_id,
            operation="compress",
            phase=compress,
            schedule_position=schedule_position,
            exact=exact,
            verification=verification,
        )
        decompress_record = phase_record(
            identity=identity,
            pair_id=pair_id,
            attempt_id=attempt_id,
            operation="decompress",
            phase=decompress,
            schedule_position=schedule_position,
            exact=exact,
            verification=verification,
        )
        log.append(compress_record)
        log.append(decompress_record)
        phase_records_appended = True
        if exact is False:
            raise base.BenchmarkError(
                f"{checkpoint.name}/{method}: round-trip exactness failed"
            )
    except Exception as exc:
        failure_id = operation_ids[current_operation]
        if not phase_records_appended:
            log.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    **identity,
                    "run_id": failure_id,
                    "pair_id": pair_id,
                    "attempt_id": attempt_id,
                    "status": "failed",
                    "operation": current_operation,
                    "schedule_position": schedule_position,
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": base.utc_now(),
                }
            )
        if workspace is not None and not keep_failed_artifacts:
            try:
                cleanup_artifact_workspace(workspace)
            except Exception as cleanup_exc:
                raise base.BenchmarkError(
                    f"{type(exc).__name__}: {exc}; safe failed-artifact "
                    f"cleanup also failed: {type(cleanup_exc).__name__}: "
                    f"{cleanup_exc}"
                ) from exc
        raise

    if not keep_artifacts and workspace is not None:
        cleanup_artifact_workspace(workspace)
    return [compress_record, decompress_record]


def scheduled_methods(
    methods: Sequence[str],
    *,
    checkpoint: str,
    profile: str,
    sample_kind: str,
    sample_index: int,
    seed: int,
) -> list[str]:
    ordered = list(methods)
    salt = stable_hash(
        {
            "checkpoint": checkpoint,
            "profile": profile,
            "sample_kind": sample_kind,
            "sample_index": sample_index,
            "seed": seed,
        }
    )
    random.Random(int(salt[:16], 16)).shuffle(ordered)
    return ordered


def execute_schedule(
    *,
    results: Path,
    checkpoints: Sequence[base.Checkpoint],
    profiles: Sequence[str],
    methods: Sequence[str],
    resources: dict[tuple[str, str], ResourceSpec],
    experiment_id: str,
    warmups: int,
    repetitions: int,
    cache_mode: str,
    drop_caches_command: str | None,
    brevis_bin: Path,
    brevis_config: base.BrevisConfig,
    keep_artifacts: bool,
    rerun: bool,
    order_seed: int,
    keep_failed_artifacts: bool = False,
    skip_round_trip_check: bool = False,
    checkpoint_descriptors: dict[str, dict[str, Any]] | None = None,
    command_builder: CommandBuilder = build_command,
) -> list[dict[str, Any]]:
    if warmups < 0 or repetitions < 1:
        raise base.BenchmarkError(
            "warmups must be non-negative and repetitions must be positive"
        )
    validate_artifact_results_root(results)
    raw_path = results / "raw" / "throughput-runs.jsonl"
    validate_results_output_path(results, raw_path)
    log = base.ResultLog(raw_path)
    descriptors = checkpoint_descriptors or {
        checkpoint.name: checkpoint_descriptor(checkpoint)
        for checkpoint in checkpoints
    }
    records: list[dict[str, Any]] = []
    rounds = [
        *(("warmup", index) for index in range(warmups)),
        *(("measured", index) for index in range(repetitions)),
    ]
    for checkpoint in checkpoints:
        for profile in profiles:
            for sample_kind, sample_index in rounds:
                order = scheduled_methods(
                    methods,
                    checkpoint=checkpoint.name,
                    profile=profile,
                    sample_kind=sample_kind,
                    sample_index=sample_index,
                    seed=order_seed,
                )
                for position, method in enumerate(order):
                    records.extend(
                        run_round(
                            results=results,
                            log=log,
                            experiment_id=experiment_id,
                            checkpoint=checkpoint,
                            descriptor=descriptors[checkpoint.name],
                            method=method,
                            profile=profile,
                            resource=resources[(profile, method)],
                            sample_kind=sample_kind,
                            sample_index=sample_index,
                            expected_measured_repetitions=repetitions,
                            schedule_position=position,
                            cache_mode=cache_mode,
                            drop_caches_command=drop_caches_command,
                            brevis_bin=brevis_bin,
                            brevis_config=brevis_config,
                            keep_artifacts=keep_artifacts,
                            keep_failed_artifacts=keep_failed_artifacts,
                            skip_round_trip_check=skip_round_trip_check,
                            rerun=rerun,
                            command_builder=command_builder,
                        )
                    )
    return records


def latest_records(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in base.read_jsonl(path):
        latest[record["run_id"]] = record
    return list(latest.values())


def prefixed_distribution(
    prefix: str,
    values: Sequence[float],
) -> dict[str, float | int]:
    return {
        f"{key}_{prefix}": value
        for key, value in distribution(values).items()
        if key != "n"
    }


SUMMARY_PAIR_KEYS = (
    "experiment_id",
    "checkpoint",
    "checkpoint_fingerprint",
    "method",
    "profile",
    "cache_mode",
    "round_trip_check_performed",
)


def latest_attempts_by_pair(
    records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    attempts: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(records):
        pair_id = record.get("pair_id")
        attempt_id = record.get("attempt_id")
        if pair_id and attempt_id:
            attempts.setdefault((pair_id, attempt_id), []).append((index, record))
    latest_keys: dict[str, tuple[str, int]] = {}
    for (pair_id, attempt_id), rows in attempts.items():
        last_index = max(index for index, _ in rows)
        current = latest_keys.get(pair_id)
        if current is None or last_index > current[1]:
            latest_keys[pair_id] = (attempt_id, last_index)
    return {
        pair_id: [
            record
            for _, record in sorted(attempts[(pair_id, attempt_id)])
        ]
        for pair_id, (attempt_id, _) in latest_keys.items()
    }


def environment_expected_groups(
    environments: Sequence[dict[str, Any]],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    expected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for environment in environments:
        experiment_id = environment.get("experiment_id")
        protocol = environment.get("protocol", {})
        repetitions = protocol.get("repetitions")
        checkpoints = environment.get("checkpoints")
        profiles = protocol.get("profiles")
        methods = protocol.get("methods")
        cache_mode = protocol.get("cache_mode")
        resource_matrix = environment.get("resource_matrix")
        round_trip_check = protocol.get("round_trip_check")
        exactness_protocols = (
            round_trip_check.get("protocols")
            if isinstance(round_trip_check, dict)
            else None
        )
        if (
            not experiment_id
            or not isinstance(repetitions, int)
            or repetitions < 1
            or not isinstance(checkpoints, list)
            or not checkpoints
            or not isinstance(profiles, list)
            or not profiles
            or not isinstance(methods, list)
            or not methods
            or not isinstance(cache_mode, str)
            or not isinstance(resource_matrix, dict)
            or not isinstance(round_trip_check, dict)
            or not isinstance(round_trip_check.get("performed"), bool)
            or not isinstance(exactness_protocols, dict)
        ):
            raise base.BenchmarkError(
                f"incomplete throughput environment snapshot: {experiment_id}"
            )
        for checkpoint in checkpoints:
            if (
                not isinstance(checkpoint, dict)
                or not checkpoint.get("name")
                or not checkpoint.get("fingerprint")
                or not isinstance(checkpoint.get("files"), list)
                or not checkpoint["files"]
            ):
                raise base.BenchmarkError(
                    f"invalid checkpoint in environment: {experiment_id}"
                )
            for profile in profiles:
                for method in methods:
                    verification = exactness_protocols.get(method)
                    if not isinstance(verification, dict):
                        raise base.BenchmarkError(
                            f"environment lacks exactness protocol for {method}"
                        )
                    key = (
                        experiment_id,
                        checkpoint["name"],
                        checkpoint["fingerprint"],
                        method,
                        profile,
                        cache_mode,
                        round_trip_check["performed"],
                    )
                    resource_data = resource_matrix.get(
                        f"{profile}/{method}"
                    )
                    if not resource_data:
                        raise base.BenchmarkError(
                            f"environment lacks resource profile "
                            f"{profile}/{method}"
                        )
                    try:
                        resource = ResourceSpec(**resource_data).provenance(
                            len(checkpoint["files"])
                        )
                    except TypeError as exc:
                        raise base.BenchmarkError(
                            f"invalid environment resource profile "
                            f"{profile}/{method}"
                        ) from exc
                    expected[key] = {
                        "expected_repetitions": repetitions,
                        "resource": resource,
                        "source_bytes": sum(
                            item.get("bytes", 0)
                            for item in checkpoint.get("files", ())
                        ),
                        "round_trip_check_performed": round_trip_check[
                            "performed"
                        ],
                        "exactness_verification": verification,
                    }
    return expected


def summary_rows(
    records: Iterable[dict[str, Any]],
    experiment_id: str | None = None,
    expected_groups: dict[tuple[Any, ...], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = list(records)
    plans = dict(expected_groups or {})
    valid: dict[
        tuple[Any, ...],
        dict[str, dict[int, dict[str, Any]]],
    ] = {}
    seen: dict[tuple[Any, ...], set[int]] = {}
    invalid: dict[tuple[Any, ...], set[int]] = {}

    for attempt in latest_attempts_by_pair(records).values():
        representative = attempt[-1]
        if (
            representative.get("sample_kind") != "measured"
            or (
                experiment_id is not None
                and representative.get("experiment_id") != experiment_id
            )
        ):
            continue
        key = tuple(representative.get(name) for name in SUMMARY_PAIR_KEYS)
        sample_index = representative.get("sample_index")
        expected = representative.get("expected_measured_repetitions")
        if not isinstance(sample_index, int) or not isinstance(expected, int):
            continue
        plans.setdefault(
            key,
            {
                "expected_repetitions": expected,
                "resource": representative.get("resource"),
                "source_bytes": representative.get(
                    "logical_uncompressed_bytes"
                ),
                "round_trip_check_performed": representative.get(
                    "round_trip_check_performed"
                ),
                "exactness_verification": representative.get(
                    "exactness_verification"
                ),
            },
        )
        if plans[key]["expected_repetitions"] != expected:
            raise base.BenchmarkError(
                f"inconsistent repetition count for {key}"
            )
        seen.setdefault(key, set()).add(sample_index)
        phases = complete_attempt_phases(attempt)
        if phases is None:
            invalid.setdefault(key, set()).add(sample_index)
            continue
        for operation, record in phases.items():
            valid.setdefault(
                key,
                {"compress": {}, "decompress": {}},
            )[operation][sample_index] = record

    output = []
    for key, plan in sorted(plans.items()):
        if experiment_id is not None and key[0] != experiment_id:
            continue
        expected_count = plan["expected_repetitions"]
        expected_indices = set(range(expected_count))
        seen_indices = seen.get(key, set())
        invalid_indices = invalid.get(key, set())
        valid_operations = valid.get(
            key,
            {"compress": {}, "decompress": {}},
        )
        pair_valid_indices = (
            set(valid_operations["compress"])
            & set(valid_operations["decompress"])
        )
        missing_indices = expected_indices - seen_indices
        unexpected_indices = seen_indices - expected_indices
        complete = (
            pair_valid_indices == expected_indices
            and not invalid_indices
            and not unexpected_indices
        )
        check_performed = plan.get("round_trip_check_performed")
        if not isinstance(check_performed, bool):
            raise base.BenchmarkError(
                f"missing round-trip check mode for {key}"
            )
        for operation in ("compress", "decompress"):
            rows = [
                valid_operations[operation][index]
                for index in sorted(pair_valid_indices & expected_indices)
            ]
            result: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                **dict(zip(SUMMARY_PAIR_KEYS, key)),
                "operation": operation,
                "summary_status": "complete" if complete else "incomplete",
                "expected_repetitions": expected_count,
                "n": len(rows),
                "valid_repetition_indices": sorted(pair_valid_indices),
                "missing_repetition_indices": sorted(missing_indices),
                "invalid_repetition_indices": sorted(invalid_indices),
                "unexpected_repetition_indices": sorted(unexpected_indices),
                "all_round_trips_exact": (
                    complete if check_performed else None
                ),
            }
            verification = (
                rows[0].get("exactness_verification")
                if rows
                else plan.get("exactness_verification")
            )
            if not isinstance(verification, dict):
                raise base.BenchmarkError(
                    f"missing exactness provenance for {key}"
                )
            result.update(
                {
                    "exactness_scope": verification.get("scope"),
                    "verification_algorithm": verification.get("algorithm"),
                    "source_reread_during_final_verification": verification.get(
                        "source_reread_during_final_verification"
                    ),
                    "restored_reread_during_final_verification": (
                        verification.get(
                            "restored_reread_during_final_verification"
                        )
                    ),
                }
            )
            resource = rows[0]["resource"] if rows else plan.get("resource")
            source_bytes = (
                rows[0]["logical_uncompressed_bytes"]
                if rows
                else plan.get("source_bytes")
            )
            result["source_bytes"] = source_bytes
            if resource:
                result.update(
                    {
                        "declared_cpu_slots": resource.get(
                            "declared_cpu_slots"
                        ),
                        "configured_shard_jobs": resource.get("shard_jobs"),
                        "effective_shard_jobs": resource.get(
                            "effective_shard_jobs"
                        ),
                        "processes_per_command": resource.get(
                            "processes_per_command"
                        ),
                        "cpu_slots_per_command": resource.get(
                            "cpu_slots_per_command"
                        ),
                        "codec_workers_per_process": resource.get(
                            "codec_workers"
                        ),
                        "expected_threads_per_process": resource.get(
                            "expected_threads_per_process"
                        ),
                        "execution_model": resource.get("execution_model"),
                        "cpu_affinity": resource.get("cpu_affinity"),
                        "torch_intra_op_threads": resource.get(
                            "runtime_controls",
                            {},
                        ).get("torch_intra_op_threads"),
                        "torch_inter_op_threads": resource.get(
                            "runtime_controls",
                            {},
                        ).get("torch_inter_op_threads"),
                    }
                )
            if complete:
                seconds = [row["phase_makespan_seconds"] for row in rows]
                gib_per_second = [
                    row["throughput_bytes_per_second"] / 1024**3
                    for row in rows
                ]
                result.update(
                    {
                        **prefixed_distribution("seconds", seconds),
                        **prefixed_distribution(
                            "gib_per_second",
                            gib_per_second,
                        ),
                    }
                )
                if operation == "compress":
                    ratios = [
                        row["logical_uncompressed_bytes"] / row["output_bytes"]
                        for row in rows
                    ]
                    result.update(
                        {
                            "median_archive_bytes": median(
                                row["output_bytes"] for row in rows
                            ),
                            **prefixed_distribution(
                                "compression_ratio_x",
                                ratios,
                            ),
                        }
                    )
            output.append(result)
    return output


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_environment_snapshots(
    results: Path,
    experiment_id: str | None,
) -> list[dict[str, Any]]:
    environments = []
    for path in sorted(results.glob("throughput-environment-*.json")):
        validate_results_output_path(results, path)
        try:
            environment = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise base.BenchmarkError(
                f"invalid throughput environment: {path}"
            ) from exc
        if (
            experiment_id is None
            or environment.get("experiment_id") == experiment_id
        ):
            environments.append(environment)
    return environments


def summarize(
    results: Path,
    experiment_id: str | None = None,
    *,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    validate_artifact_results_root(results)
    raw_path = results / "raw" / "throughput-runs.jsonl"
    validate_results_output_path(results, raw_path)
    records = latest_records(raw_path)
    environments = load_environment_snapshots(results, experiment_id)
    environment_ids = {
        environment.get("experiment_id") for environment in environments
    }
    raw_experiment_ids = {
        record.get("experiment_id")
        for record in records
        if record.get("experiment_id")
        and (
            experiment_id is None
            or record.get("experiment_id") == experiment_id
        )
    }
    missing_environment_ids = raw_experiment_ids - environment_ids
    if missing_environment_ids:
        if not allow_incomplete:
            raise base.BenchmarkError(
                "strict throughput summary is missing immutable environment "
                "snapshot(s): "
                + ", ".join(sorted(missing_environment_ids))
            )
        environments = []
    provisional_without_environment = not environments
    if provisional_without_environment and not allow_incomplete:
        raise base.BenchmarkError(
            "strict throughput summary requires a matching immutable "
            "environment snapshot; pass --allow-incomplete-summary only for "
            "provisional raw-log inspection"
        )
    try:
        expected = environment_expected_groups(environments)
    except base.BenchmarkError:
        if not allow_incomplete:
            raise
        environments = []
        provisional_without_environment = True
        expected = {}
    rows = summary_rows(records, experiment_id, expected)
    for row in rows:
        row["summary_mode"] = (
            "provisional_raw_without_environment"
            if provisional_without_environment
            else "formal_environment_validated"
        )
        row["publication_eligible"] = (
            not provisional_without_environment
            and row.get("summary_status") == "complete"
        )
    tables = results / "tables"
    validate_results_output_path(
        results,
        tables / "throughput-summary.csv",
    )
    validate_results_output_path(
        results,
        tables / "throughput-summary.json",
    )
    write_csv(tables / "throughput-summary.csv", rows)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id_filter": experiment_id,
        "summary_mode": (
            "provisional_raw_without_environment"
            if provisional_without_environment
            else "formal_environment_validated"
        ),
        "statistics": {
            "warmups_excluded": True,
            "center": "sample median",
            "quartiles": "R-7 linear interpolation",
            "iqr": "Q3 - Q1",
            "time_basis": (
                "whole-checkpoint wall-clock phase makespan, including command "
                "launch and output fsync; cache conditioning and exactness "
                "verification excluded"
            ),
            "throughput_numerator": "original uncompressed checkpoint bytes",
            "validity": (
                "a repetition is valid only when compress and decompress are "
                "both ok in the same latest attempt; when the round-trip check "
                "is performed, exactness must be true; explicit fast mode "
                "records exactness as not_checked/null"
            ),
            "incomplete_groups_are_publication_errors": True,
        },
        "rows": rows,
    }
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "throughput-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    incomplete = [
        row
        for row in rows
        if row.get("summary_status") != "complete"
    ]
    if incomplete and not allow_incomplete:
        labels = sorted(
            {
                (
                    f"{row['experiment_id']}/{row['checkpoint']}/"
                    f"{row['profile']}/{row['method']}"
                )
                for row in incomplete
            }
        )
        raise base.BenchmarkError(
            "throughput summary is incomplete: " + ", ".join(labels)
        )
    return rows


def method_provenance(
    method: str,
    brevis_bin: Path,
    repository_revision: str | None,
) -> dict[str, Any]:
    if method == "brevis":
        return {
            "kind": "local_brevis_binary",
            "binary_path": str(brevis_bin),
            "binary_sha256": (
                base.file_sha256(brevis_bin)
                if brevis_bin.is_file()
                else None
            ),
            "repository_revision": repository_revision,
            "version_command": None,
        }
    policy = base.CODEC_POLICIES[method]
    return {
        "kind": "external_codec",
        "version": base.command_version(list(policy.version_command)),
        "version_command": list(policy.version_command),
        "adapter_path": str(policy.adapter) if policy.adapter else None,
        "adapter_sha256": (
            base.file_sha256(policy.adapter)
            if policy.adapter and policy.adapter.is_file()
            else None
        ),
    }


def collect_environment(
    *,
    args: argparse.Namespace,
    checkpoints: Sequence[base.Checkpoint],
    resources: dict[tuple[str, str], ResourceSpec],
) -> tuple[str, dict[str, Any]]:
    cold_available = base.cache_control_available(args.drop_caches_command)
    host = base.host_context(
        SimpleNamespace(
            cold_available=cold_available,
            drop_caches_command=args.drop_caches_command,
        )
    )
    repository_revision = base.command_version(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"]
    )
    methods = {
        method: method_provenance(
            method,
            args.brevis_bin,
            repository_revision,
        )
        for method in args.methods
    }
    descriptors = [
        checkpoint_descriptor(checkpoint) for checkpoint in checkpoints
    ]
    host["scheduler_affinity_cpus"] = list(scheduler_cpu_set())
    immutable = {
        "harness_sha256": base.file_sha256(Path(__file__)),
        "paper_harness_sha256": base.file_sha256(
            ROOT / "scripts" / "run_benchmarks.py"
        ),
        "codec_adapter_sha256": base.file_sha256(
            ROOT / "scripts" / "benchmark_codecs.py"
        ),
        "repository_revision": repository_revision,
        "brevis_binary": str(args.brevis_bin),
        "brevis_binary_sha256": (
            base.file_sha256(args.brevis_bin)
            if args.brevis_bin.is_file()
            else None
        ),
        "method_provenance": methods,
        "host": host,
        "checkpoints": descriptors,
        "protocol": {
            "profiles": args.profiles,
            "methods": args.methods,
            "cache_mode": args.cache,
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "order_seed": args.order_seed,
            "brevis_execution": args.brevis_execution,
            "zipnn_execution": args.zipnn_execution,
            "resource_workers": args.resource_workers,
            "practical_workers": args.practical_workers,
            "brevis_workers": args.brevis_workers,
            "zipnn_workers": args.zipnn_workers,
            "requested_cpu_list": args.cpu_list,
            "scheduler_cpu_pool": list(args.cpu_pool),
            "cpu_slot_unit": "logical CPU IDs from sched_getaffinity",
            "keep_success_artifacts": args.keep_artifacts,
            "keep_failed_artifacts": args.keep_failed_artifacts,
            "round_trip_check": {
                "performed": not args.skip_round_trip_check,
                "strict_check_is_default": True,
                "skip_requested": args.skip_round_trip_check,
                "protocols": {
                    method: (
                        skipped_exactness_protocol()
                        if args.skip_round_trip_check
                        else exactness_protocol(method)
                    )
                    for method in args.methods
                },
            },
            "brevis": {
                "max_expansions": args.brevis_max_expansions,
                "tensors": args.brevis_tensors,
                "astar_heuristic": args.brevis_astar_heuristic,
            },
        },
        "resource_matrix": {
            f"{profile}/{method}": asdict(resource)
            for (profile, method), resource in sorted(resources.items())
        },
    }
    experiment_id = stable_hash(immutable, 20)
    return experiment_id, {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "generated_at": base.utc_now(),
        "argv": sys.argv,
        **immutable,
    }


def write_environment(results: Path, environment: dict[str, Any]) -> None:
    validate_artifact_results_root(results)
    encoded = json.dumps(environment, indent=2, sort_keys=True) + "\n"
    snapshot = (
        results
        / f"throughput-environment-{environment['experiment_id']}.json"
    )
    validate_results_output_path(results, snapshot)
    validate_results_output_path(
        results,
        results / "throughput-environment.json",
    )
    snapshot.write_text(encoded)
    (results / "throughput-environment.json").write_text(encoded)


def dry_run_matrix(
    *,
    checkpoints: Sequence[base.Checkpoint],
    profiles: Sequence[str],
    methods: Sequence[str],
    resources: dict[tuple[str, str], ResourceSpec],
    results: Path,
    brevis_bin: Path,
    brevis_config: base.BrevisConfig,
) -> None:
    for checkpoint in checkpoints:
        source = checkpoint.files[0]
        for profile in profiles:
            for method in methods:
                resource = resources[(profile, method)]
                archive = (
                    results
                    / "dry-run"
                    / safe_component(checkpoint.name)
                    / safe_component(profile)
                    / safe_component(method)
                    / f"archive{archive_suffix(method)}"
                )
                restored = archive.with_name("restored.safetensors")
                print(
                    json.dumps(
                        {
                            "checkpoint": checkpoint.name,
                            "method": method,
                            "profile": profile,
                            "resource": resource.provenance(
                                len(checkpoint.files)
                            ),
                        },
                        sort_keys=True,
                    )
                )
                print(
                    "  "
                    + shlex.join(
                        build_command(
                            method,
                            "compress",
                            source,
                            archive,
                            resource,
                            brevis_bin,
                            brevis_config,
                        )
                    )
                )
                print(
                    "  "
                    + shlex.join(
                        build_command(
                            method,
                            "decompress",
                            archive,
                            restored,
                            resource,
                            brevis_bin,
                            brevis_config,
                        )
                    )
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "summarize"))
    parser.add_argument("--checkpoint", type=Path, nargs="+")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "results" / "throughput",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=base.GENERIC_METHODS,
        default=list(base.GENERIC_METHODS),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=PROFILE_NAMES,
        default=["resource-matched"],
    )
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--cache",
        choices=("hot", "cold", "unconditioned"),
        default="hot",
    )
    parser.add_argument("--drop-caches-command")
    available = len(scheduler_cpu_set())
    parser.add_argument(
        "--resource-workers",
        type=int,
        default=min(16, available),
        help="CPU-slot cap shared by every method in resource-matched.",
    )
    parser.add_argument(
        "--practical-workers",
        type=int,
        default=min(32, available),
        help="Shard-process cap for practical single-thread baselines.",
    )
    parser.add_argument(
        "--brevis-workers",
        type=int,
        default=min(32, available),
        help=(
            "Brevis internal workers or shard processes in the practical "
            "profile, selected by --brevis-execution."
        ),
    )
    parser.add_argument(
        "--zipnn-workers",
        type=int,
        default=min(DEFAULT_ZIPNN_WORKERS, available),
        help="ZipNN threads or shard processes in the practical profile.",
    )
    parser.add_argument(
        "--brevis-execution",
        choices=BREVIS_EXECUTIONS,
        default="threads",
        help=(
            "'threads': one process using Brevis --workers; 'processes': one "
            "Brevis --workers 1 process per shard, up to the profile limit."
        ),
    )
    parser.add_argument(
        "--zipnn-execution",
        choices=ZIPNN_EXECUTIONS,
        default="threads",
        help=(
            "'threads': one process using ZipNN's thread setting; 'processes': "
            "one single-thread process per shard, up to the profile limit."
        ),
    )
    parser.add_argument(
        "--cpu-list",
        help=(
            "Optional subset of the current scheduler affinity. Formal runs "
            "are taskset-pinned to the first declared slots from this pool; "
            "without it, the current sched_getaffinity set is the pool."
        ),
    )
    parser.add_argument(
        "--brevis-bin",
        type=Path,
        default=base.DEFAULT_BREVIS_BIN,
    )
    parser.add_argument("--brevis-max-expansions", type=int, default=512)
    parser.add_argument("--brevis-tensors", type=int, default=256)
    parser.add_argument(
        "--brevis-astar-heuristic",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--keep-failed-artifacts",
        action="store_true",
        help=(
            "Preserve an owned failed-attempt workspace for diagnosis. "
            "By default failed artifacts are safely removed."
        ),
    )
    parser.add_argument(
        "--skip-round-trip-check",
        action="store_true",
        help=(
            "Throughput fast mode: time compression and decompression but skip "
            "the final exactness scan. Raw results are explicitly not checked."
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--experiment-id",
        help="Optional experiment filter for the summarize stage.",
    )
    parser.add_argument(
        "--allow-incomplete-summary",
        action="store_true",
        help="Write provisional incomplete rows instead of failing closed.",
    )
    args = parser.parse_args(argv)
    args.results = args.results.expanduser().absolute()
    args.brevis_bin = args.brevis_bin.expanduser().resolve()
    if args.stage == "run" and not args.checkpoint:
        parser.error("run requires --checkpoint")
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be non-negative; repetitions must be positive")
    if any(
        value < 1
        for value in (
            args.resource_workers,
            args.practical_workers,
            args.brevis_workers,
            args.zipnn_workers,
        )
    ):
        parser.error("all worker counts must be positive")
    if args.brevis_max_expansions < 0 or args.brevis_tensors < 0:
        parser.error("Brevis search limits must be non-negative")
    args.profiles = list(dict.fromkeys(args.profiles))
    args.methods = list(dict.fromkeys(args.methods))
    scheduler_cpus = set(scheduler_cpu_set())
    if args.cpu_list:
        try:
            args.cpu_list, args.cpu_list_count = parse_cpu_list(args.cpu_list)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
        if not shutil.which("taskset"):
            parser.error("--cpu-list requires taskset")
        requested = set(expand_cpu_list(args.cpu_list))
        outside = sorted(requested - scheduler_cpus)
        if outside:
            parser.error(
                "--cpu-list includes CPUs outside sched_getaffinity: "
                + ",".join(map(str, outside))
            )
        args.cpu_pool = tuple(sorted(requested))
    else:
        args.cpu_list_count = None
        args.cpu_pool = tuple(sorted(scheduler_cpus))
    if args.cache == "cold" and not base.cache_control_available(
        args.drop_caches_command
    ):
        parser.error(
            "cold cache requested but no working cache-drop mechanism is available"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "summarize":
        rows = summarize(
            args.results,
            args.experiment_id,
            allow_incomplete=args.allow_incomplete_summary,
        )
        print(f"wrote {len(rows)} summary rows under {args.results / 'tables'}")
        return 0

    checkpoints = [base.load_checkpoint(path) for path in args.checkpoint]
    names = [checkpoint.name for checkpoint in checkpoints]
    if len(names) != len(set(names)):
        raise base.BenchmarkError("checkpoint result labels must be unique")
    resources = {
        (profile, method): resolve_resource(
            method,
            profile,
            matched_workers=args.resource_workers,
            practical_workers=args.practical_workers,
            brevis_workers=args.brevis_workers,
            zipnn_workers=args.zipnn_workers,
            zipnn_execution=args.zipnn_execution,
            cpu_affinity=None,
            brevis_execution=args.brevis_execution,
        )
        for profile in args.profiles
        for method in args.methods
    }
    resources = bind_resource_affinity(
        resources,
        args.cpu_pool,
        taskset_available=bool(shutil.which("taskset")),
    )
    brevis_config = base.BrevisConfig(
        workers=args.brevis_workers,
        max_expansions=args.brevis_max_expansions,
        tensors=args.brevis_tensors,
        astar_heuristic=bool(args.brevis_astar_heuristic),
    )
    if args.dry_run:
        dry_run_matrix(
            checkpoints=checkpoints,
            profiles=args.profiles,
            methods=args.methods,
            resources=resources,
            results=args.results,
            brevis_bin=args.brevis_bin,
            brevis_config=brevis_config,
        )
        return 0

    validate_artifact_results_root(args.results)
    environment_id, environment = collect_environment(
        args=args,
        checkpoints=checkpoints,
        resources=resources,
    )
    write_environment(args.results, environment)
    descriptors = {
        descriptor["name"]: descriptor
        for descriptor in environment["checkpoints"]
    }
    execute_schedule(
        results=args.results,
        checkpoints=checkpoints,
        profiles=args.profiles,
        methods=args.methods,
        resources=resources,
        experiment_id=environment_id,
        warmups=args.warmups,
        repetitions=args.repetitions,
        cache_mode=args.cache,
        drop_caches_command=args.drop_caches_command,
        brevis_bin=args.brevis_bin,
        brevis_config=brevis_config,
        keep_artifacts=args.keep_artifacts,
        keep_failed_artifacts=args.keep_failed_artifacts,
        skip_round_trip_check=args.skip_round_trip_check,
        rerun=args.rerun,
        order_seed=args.order_seed,
        checkpoint_descriptors=descriptors,
    )
    rows = summarize(
        args.results,
        environment_id,
        allow_incomplete=args.allow_incomplete_summary,
    )
    print(
        f"completed experiment {environment_id}; "
        f"wrote {len(rows)} summary rows"
    )
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["_zipnn-adapter"]:
        raise SystemExit(zipnn_adapter_main(sys.argv[2:]))
    raise SystemExit(main())
