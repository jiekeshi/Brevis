#!/usr/bin/env python3
"""Run the Brevis paper benchmark as resumable, one-shot experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from benchmark_corpus import CHECKPOINTS as PAPER_CHECKPOINTS
from benchmark_corpus import CHECKPOINT_BY_NAME

ROOT = Path(__file__).resolve().parent.parent
CODEC_HELPER = ROOT / "scripts" / "benchmark_codecs.py"
DEFAULT_BREVIS_BIN = ROOT / "zig-out" / "bin" / "brevis"
GENERIC_METHODS = ("brevis", "zstd-9", "zipnn", "lz4-hc-9", "snappy")
SPECIALIZED_METHODS = ("dfloat11", "ecf8")
ALL_METHODS = GENERIC_METHODS + SPECIALIZED_METHODS
DEFAULT_BUDGETS = (0, 1, 8, 32, 128, 512)
DEFAULT_WORKERS = (1, 2, 4, 8, 16, 32)
READ_CHUNK = 64 * 1024 * 1024


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Checkpoint:
    name: str
    directory: Path
    files: tuple[Path, ...]
    repo_id: str | None = None
    revision: str | None = None

    @property
    def source_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.files)


@dataclass(frozen=True)
class BrevisConfig:
    workers: int
    max_expansions: int = 512
    tensors: int = 256
    astar_heuristic: bool = True

    @property
    def phog_enabled(self) -> bool:
        return self.max_expansions != 0 and self.tensors != 0


@dataclass
class Measurement:
    wall_seconds: float
    peak_rss_bytes: int | None
    stdout_path: str
    stderr_path: str
    stdout: str


class ResultLog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.records = read_jsonl(path)
        self.completed = {
            record["run_id"]: record
            for record in self.records
            if record.get("status") == "ok"
        }

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.completed.get(run_id)

    def append(self, record: dict[str, Any]) -> None:
        record = {"schema_version": 1, **record}
        line = json.dumps(record, sort_keys=True) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(line)
                output.flush()
                os.fsync(output.fileno())
            self.records.append(record)
            if record.get("status") == "ok":
                self.completed[record["run_id"]] = record


class Deadline:
    def __init__(self, hours: float | None):
        self.end = None if hours is None else time.monotonic() + hours * 3600

    def require_time(self) -> None:
        if self.end is not None and time.monotonic() >= self.end:
            raise BenchmarkError("experiment deadline reached")

    def seconds_left(self) -> float | None:
        return None if self.end is None else max(0.0, self.end - time.monotonic())

    def can_fit(self, seconds: float) -> bool:
        remaining = self.seconds_left()
        return remaining is None or seconds <= remaining


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        f"{path}:{line_number}: invalid JSONL"
                    ) from exc
    return records


def load_manifest_checkpoint(directory: Path) -> Checkpoint:
    manifest_path = directory / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = tuple(directory / item["path"] for item in manifest["weights"])
    validate_source_files(files, manifest["name"])
    return Checkpoint(
        manifest["name"],
        directory,
        files,
        manifest.get("repo_id"),
        manifest.get("revision"),
    )


def load_checkpoint(path: Path, name: str | None = None) -> Checkpoint:
    path = path.expanduser().resolve()
    if path.is_file():
        validate_source_files((path,), name or path.stem)
        return Checkpoint(name or path.stem, path.parent, (path,))
    if not path.is_dir():
        raise BenchmarkError(f"checkpoint path does not exist: {path}")
    if (path / "download-manifest.json").exists():
        checkpoint = load_manifest_checkpoint(path)
        return checkpoint if name is None else Checkpoint(
            name,
            checkpoint.directory,
            checkpoint.files,
            checkpoint.repo_id,
            checkpoint.revision,
        )
    files = tuple(
        sorted(
            item
            for item in path.glob("*.safetensors")
            if ".znn." not in item.name and not item.name.endswith(".brv")
        )
    )
    validate_source_files(files, name or path.name)
    return Checkpoint(name or path.name, path, files)


def load_corpus(
    root: Path,
    names: list[str] | None,
    allow_custom: bool = False,
) -> list[Checkpoint]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise BenchmarkError(f"models root does not exist: {root}")
    checkpoints = [
        load_manifest_checkpoint(path.parent)
        for path in root.glob("*/download-manifest.json")
    ]
    if not checkpoints and (root / "download-manifest.json").exists():
        checkpoints = [load_manifest_checkpoint(root)]
    if allow_custom:
        selected = set(names) if names else None
    else:
        unknown = {item.name for item in checkpoints} - CHECKPOINT_BY_NAME.keys()
        if unknown:
            raise BenchmarkError(
                f"non-paper checkpoint(s): {', '.join(sorted(unknown))}"
            )
        selected = set(names) if names else {
            checkpoint.name for checkpoint in PAPER_CHECKPOINTS
        }
        unknown_selection = selected - CHECKPOINT_BY_NAME.keys()
        if unknown_selection:
            raise BenchmarkError(
                f"non-paper checkpoint(s): {', '.join(sorted(unknown_selection))}"
            )
    if selected:
        checkpoints = [item for item in checkpoints if item.name in selected]
        missing = selected - {item.name for item in checkpoints}
        if missing:
            raise BenchmarkError(
                f"missing downloaded checkpoint(s): {', '.join(sorted(missing))}"
            )
    if not allow_custom:
        for checkpoint in checkpoints:
            expected = CHECKPOINT_BY_NAME[checkpoint.name]
            if (
                checkpoint.repo_id != expected.repo_id
                or checkpoint.revision != expected.revision
            ):
                raise BenchmarkError(
                    f"{checkpoint.name}: manifest is not the frozen paper revision"
                )
    return sorted(checkpoints, key=lambda item: item.source_bytes)


def validate_source_files(files: Iterable[Path], label: str) -> None:
    files = tuple(files)
    if not files:
        raise BenchmarkError(f"{label}: no canonical safetensors files")
    for path in files:
        if not path.is_file():
            raise BenchmarkError(f"{label}: missing {path}")
        if path.stat().st_size < 8:
            raise BenchmarkError(f"{label}: invalid safetensors file {path}")


def safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise BenchmarkError(f"{path}: truncated safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if size > 64 * 1024 * 1024:
            raise BenchmarkError(f"{path}: implausible safetensors header")
        encoded = source.read(size)
    try:
        header = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"{path}: invalid safetensors header") from exc
    return header, 8 + size


def compare_range(
    left_path: Path,
    left_offset: int,
    right_path: Path,
    right_offset: int,
    size: int,
) -> bool:
    with left_path.open("rb") as left, right_path.open("rb") as right:
        left.seek(left_offset)
        right.seek(right_offset)
        remaining = size
        while remaining:
            chunk_size = min(READ_CHUNK, remaining)
            if left.read(chunk_size) != right.read(chunk_size):
                return False
            remaining -= chunk_size
    return True


def tensor_exact(source: Path, restored: Path) -> bool:
    source_header, source_base = safetensors_header(source)
    restored_header, restored_base = safetensors_header(restored)
    if source_header.get("__metadata__") != restored_header.get("__metadata__"):
        return False
    source_names = set(source_header) - {"__metadata__"}
    restored_names = set(restored_header) - {"__metadata__"}
    if source_names != restored_names:
        return False
    for name in source_names:
        expected = source_header[name]
        actual = restored_header[name]
        if expected["dtype"] != actual["dtype"] or expected["shape"] != actual["shape"]:
            return False
        expected_start, expected_end = expected["data_offsets"]
        actual_start, actual_end = actual["data_offsets"]
        size = expected_end - expected_start
        if size != actual_end - actual_start:
            return False
        if not compare_range(
            source,
            source_base + expected_start,
            restored,
            restored_base + actual_start,
            size,
        ):
            return False
    return True


def byte_exact(source: Path, restored: Path) -> bool:
    if source.stat().st_size != restored.stat().st_size:
        return False
    return compare_range(source, 0, restored, 0, source.stat().st_size)


def warm_page_cache(path: Path) -> None:
    with path.open("rb", buffering=0) as source:
        while source.read(READ_CHUNK):
            pass


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise BenchmarkError(f"converter did not produce {path}")


def linux_tree_rss(root_pid: int) -> int | None:
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    found = False
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text()
            match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
            if match:
                total += int(match.group(1)) * 1024
                found = True
            children = Path(f"/proc/{pid}/task/{pid}/children").read_text()
            pending.extend(int(child) for child in children.split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total if found else None


def ps_tree_rss(root_pid: int) -> int | None:
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss="],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    rows = [tuple(map(int, line.split())) for line in output.splitlines()]
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for pid, parent, value in rows:
        children.setdefault(parent, []).append(pid)
        rss[pid] = value * 1024
    pending = [root_pid]
    seen: set[int] = set()
    total = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        pending.extend(children.get(pid, ()))
    return total or None


def process_tree_rss(root_pid: int) -> int | None:
    return (
        linux_tree_rss(root_pid)
        if sys.platform.startswith("linux")
        else ps_tree_rss(root_pid)
    )


def tail(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return data[-limit:].decode(errors="replace")


def fsync_output(path: Path | None) -> None:
    if path is None:
        return
    files = (path,) if path.is_file() else (
        item for item in path.rglob("*") if item.is_file()
    )
    for file in files:
        with file.open("rb", buffering=0) as output:
            os.fsync(output.fileno())


def measure(
    command: list[str],
    log_dir: Path,
    run_id: str,
    output: Path | None = None,
    cwd: Path | None = None,
) -> Measurement:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{run_id}.stdout"
    stderr_path = log_dir / f"{run_id}.stderr"
    started = time.perf_counter()
    peak_rss: int | None = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
        )
        while process.poll() is None:
            current = process_tree_rss(process.pid)
            if current is not None:
                peak_rss = max(peak_rss or 0, current)
            time.sleep(0.05)
        return_code = process.wait()
    fsync_output(output)
    wall_seconds = time.perf_counter() - started
    if return_code:
        raise BenchmarkError(
            f"command failed ({return_code}): {shlex.join(command)}\n"
            f"{tail(stderr_path)}"
        )
    return Measurement(
        wall_seconds,
        peak_rss,
        str(stdout_path),
        str(stderr_path),
        tail(stdout_path),
    )


def parse_brevis_search(stdout: str) -> dict[str, int]:
    match = re.search(
        r"search: expanded=(\d+), completed=(\d+), "
        r"budget_exhausted=(\d+), literal_fallback=(\d+)",
        stdout,
    )
    if not match:
        return {}
    keys = (
        "expanded_states",
        "completed_candidates",
        "budget_exhausted_tensors",
        "literal_fallback_tensors",
    )
    return dict(zip(keys, map(int, match.groups())))


def run_id(fields: dict[str, Any]) -> str:
    identity = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def brevis_limits(source: Path) -> list[str]:
    size = source.stat().st_size
    return [
        "--max-total-bytes",
        str(max(16 * 1024**3, size * 2)),
        "--max-tensor-bytes",
        str(max(4 * 1024**3, size)),
    ]


def command_for(
    method: str,
    operation: str,
    source: Path,
    output: Path,
    workers: int,
    brevis_bin: Path,
    brevis_config: BrevisConfig,
) -> list[str]:
    if method == "brevis":
        if operation == "compress":
            return [
                str(brevis_bin),
                "compress",
                str(source),
                str(output),
                "--max-expansions",
                str(brevis_config.max_expansions),
                "--tensors",
                str(brevis_config.tensors),
                "--astar-heuristic",
                str(int(brevis_config.astar_heuristic)),
                "--workers",
                str(workers),
                *brevis_limits(source),
            ]
        return [
            str(brevis_bin),
            "decompress",
            str(source),
            str(output),
            "--workers",
            str(workers),
            *brevis_limits(source),
        ]
    if method == "zstd-9":
        return (
            ["zstd", "-q", "-f", "-9", "-T1", str(source), "-o", str(output)]
            if operation == "compress"
            else ["zstd", "-q", "-d", "-f", str(source), "-o", str(output)]
        )
    if method == "lz4-hc-9":
        return (
            ["lz4", "-q", "-f", "-9", str(source), str(output)]
            if operation == "compress"
            else ["lz4", "-q", "-d", "-f", str(source), str(output)]
        )
    if method in ("zipnn", "snappy"):
        return [
            sys.executable,
            str(CODEC_HELPER),
            method,
            operation,
            str(source),
            str(output),
            "--threads",
            str(workers),
        ]
    raise BenchmarkError(f"unsupported generic method: {method}")


def verify_command(
    brevis_bin: Path,
    archive: Path,
    source: Path,
    workers: int,
) -> list[str]:
    return [
        str(brevis_bin),
        "verify",
        str(archive),
        str(source),
        "--workers",
        str(workers),
        *brevis_limits(source),
    ]


def drop_caches(command: str | None) -> bool:
    os.sync()
    if command:
        subprocess.run(shlex.split(command), check=True)
        return True
    if sys.platform.startswith("linux") and os.geteuid() == 0:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")
        return True
    if sys.platform == "darwin" and os.geteuid() == 0 and shutil.which("purge"):
        subprocess.run(["purge"], check=True)
        return True
    return False


def cache_control_available(command: str | None) -> bool:
    if command:
        words = shlex.split(command)
        return bool(words and shutil.which(words[0]))
    if sys.platform.startswith("linux"):
        return os.geteuid() == 0 and Path("/proc/sys/vm/drop_caches").exists()
    return sys.platform == "darwin" and os.geteuid() == 0 and bool(
        shutil.which("purge")
    )


def archive_path(results: Path, checkpoint: str, method: str, source: Path) -> Path:
    return (
        results
        / "archives"
        / slug(checkpoint)
        / slug(method)
        / f"{source.stem}.{slug(method)}.brv"
    )


def operation_identity(
    checkpoint: Checkpoint,
    source: Path,
    method: str,
    operation: str,
    cache: str,
    workers: int | None,
    stage: str,
    brevis_config: BrevisConfig | None = None,
    variant: str | None = None,
) -> dict[str, Any]:
    is_brevis = method == "brevis"
    return {
        "stage": stage,
        "checkpoint": checkpoint.name,
        "shard": source.name,
        "revision": checkpoint.revision,
        "source_size_bytes": source.stat().st_size,
        "method": method,
        "operation": operation,
        "cache": cache,
        "workers": workers,
        "max_expansions": (
            brevis_config.max_expansions
            if is_brevis and brevis_config
            else None
        ),
        "phog": (
            brevis_config.phog_enabled
            if is_brevis and brevis_config
            else None
        ),
        "astar_heuristic": (
            brevis_config.astar_heuristic
            if is_brevis and brevis_config
            else None
        ),
        "variant": variant,
    }


def append_failure(
    log: ResultLog,
    identity: dict[str, Any],
    command: list[str],
    error: Exception,
) -> None:
    log.append(
        {
            **identity,
            "run_id": run_id(identity),
            "status": "failed",
            "command": command,
            "finished_at": utc_now(),
            "error": str(error),
        }
    )


def append_not_run(
    log: ResultLog,
    checkpoint: Checkpoint,
    source: Path,
    method: str,
    reason: str,
) -> None:
    identity = operation_identity(
        checkpoint,
        source,
        method,
        "compress",
        "unconditioned",
        None,
        "corpus",
    )
    log.append(
        {
            **identity,
            "run_id": run_id(identity),
            "status": "not_run",
            "reason": reason,
            "finished_at": utc_now(),
        }
    )


def execute_operation(
    *,
    args: argparse.Namespace,
    log: ResultLog,
    checkpoint: Checkpoint,
    source: Path,
    method: str,
    operation: str,
    cache: str,
    workers: int,
    input_path: Path,
    output_path: Path,
    brevis_config: BrevisConfig,
    stage: str,
    variant: str | None = None,
) -> dict[str, Any] | None:
    identity = operation_identity(
        checkpoint,
        source,
        method,
        operation,
        cache,
        workers,
        stage,
        brevis_config,
        variant,
    )
    identifier = run_id(identity)
    previous = None if args.rerun else log.get(identifier)
    if previous and output_path.exists():
        print(f"  resume {checkpoint.name}/{source.name}/{method}/{operation}/{cache}")
        return previous

    command = command_for(
        method,
        operation,
        input_path,
        output_path,
        workers,
        args.brevis_bin,
        brevis_config,
    )
    if args.dry_run:
        print(f"  {shlex.join(command)}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.deadline.require_time()
    if cache == "cold" and not drop_caches(args.drop_caches_command):
        raise BenchmarkError("cold-cache control is unavailable")
    if cache == "hot":
        warm_page_cache(input_path)

    print(f"  run {checkpoint.name}/{source.name}/{method}/{operation}/{cache}")
    started_at = utc_now()
    attempt_id = uuid.uuid4().hex
    try:
        measurement = measure(
            command,
            args.results / "logs",
            identifier,
            output_path,
        )
        size = tree_size(output_path)
        source_bytes = source.stat().st_size
        record = {
            **identity,
            "run_id": identifier,
            "status": "ok",
            "attempt_id": attempt_id,
            "started_at": started_at,
            "finished_at": utc_now(),
            "command": command,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "source_bytes": source_bytes,
            "output_bytes": size,
            "wall_seconds": measurement.wall_seconds,
            "throughput_bytes_per_second": source_bytes / measurement.wall_seconds,
            "peak_rss_bytes": measurement.peak_rss_bytes,
            "stdout_path": measurement.stdout_path,
            "stderr_path": measurement.stderr_path,
            **(
                parse_brevis_search(measurement.stdout)
                if method == "brevis" and operation == "compress"
                else {}
            ),
        }
        log.append(record)
        return record
    except Exception as exc:
        append_failure(log, identity, command, exc)
        raise


def validate_restored(
    method: str,
    source: Path,
    restored: Path,
) -> bool:
    return tensor_exact(source, restored) if method == "zipnn" else byte_exact(
        source,
        restored,
    )


def record_exactness(
    log: ResultLog,
    record: dict[str, Any] | None,
    exact: bool,
    verified_records: Iterable[dict[str, Any] | None],
) -> None:
    if record is None:
        return
    log.append(
        {
            "run_id": f"{record['run_id']}-verify",
            "status": "ok" if exact else "failed",
            "stage": record["stage"],
            "checkpoint": record["checkpoint"],
            "shard": record["shard"],
            "method": record["method"],
            "operation": "verify",
            "cache": record["cache"],
            "workers": record["workers"],
            "exact": exact,
            "verified_attempts": [
                [item["run_id"], item.get("attempt_id")]
                for item in verified_records
                if item is not None
            ],
            "finished_at": utc_now(),
        }
    )
    if not exact:
        raise BenchmarkError(
            f"{record['checkpoint']}/{record['shard']}/{record['method']}: "
            "restored tensor data differs"
        )


def run_pair(
    args: argparse.Namespace,
    log: ResultLog,
    checkpoint: Checkpoint,
    source: Path,
    method: str,
    cache: str,
    workers: int,
    stage: str,
    archive: Path,
    brevis_config: BrevisConfig,
    keep_archive: bool = True,
) -> None:
    verify_identity = operation_identity(
        checkpoint,
        source,
        method,
        "decompress",
        cache,
        workers,
        stage,
        brevis_config,
    )
    verify_id = f"{run_id(verify_identity)}-verify"
    if not args.rerun and log.get(verify_id):
        print(f"  resume {checkpoint.name}/{source.name}/{method}/{cache} pair")
        return

    compress_record = execute_operation(
        args=args,
        log=log,
        checkpoint=checkpoint,
        source=source,
        method=method,
        operation="compress",
        cache=cache,
        workers=workers,
        input_path=source,
        output_path=archive,
        brevis_config=brevis_config,
        stage=stage,
    )
    if args.dry_run:
        restored = args.results / "tmp" / f"{slug(checkpoint.name)}-{source.name}.restore"
    else:
        if not archive.exists() and compress_record:
            archive = Path(compress_record["output_path"])
        restored = (
            args.results
            / "tmp"
            / slug(checkpoint.name)
            / slug(method)
            / f"{source.name}.{cache}.restored"
        )
    decompress_record = execute_operation(
        args=args,
        log=log,
        checkpoint=checkpoint,
        source=source,
        method=method,
        operation="decompress",
        cache=cache,
        workers=workers,
        input_path=archive,
        output_path=restored,
        brevis_config=brevis_config,
        stage=stage,
    )
    if not args.dry_run:
        exact = validate_restored(method, source, restored)
        record_exactness(
            log,
            decompress_record,
            exact,
            (compress_record, decompress_record),
        )
        restored.unlink(missing_ok=True)
        if not keep_archive:
            archive.unlink(missing_ok=True)


def method_workers(method: str, practical_workers: int) -> int:
    return practical_workers if method in ("brevis", "zipnn") else 1


def run_core(args: argparse.Namespace, log: ResultLog, core: Checkpoint) -> None:
    methods = [method for method in args.methods if method in GENERIC_METHODS]
    cache = "cold" if args.cold_available else "unconditioned"
    for method in methods:
        for source in core.files:
            kept = archive_path(args.results, core.name, method, source)
            run_pair(
                args,
                log,
                core,
                source,
                method,
                cache,
                1,
                "core",
                kept,
                BrevisConfig(1),
            )
            hot_archive = (
                args.results
                / "tmp"
                / slug(core.name)
                / slug(method)
                / f"{source.stem}.hot.brv"
            )
            run_pair(
                args,
                log,
                core,
                source,
                method,
                "hot",
                1,
                "core",
                hot_archive,
                BrevisConfig(1),
                keep_archive=False,
            )


def run_brevis_variant(
    args: argparse.Namespace,
    log: ResultLog,
    checkpoint: Checkpoint,
    stage: str,
    variant: str,
    config: BrevisConfig,
    keep: bool,
) -> None:
    for source in checkpoint.files:
        identity = operation_identity(
            checkpoint,
            source,
            "brevis",
            "compress",
            "hot",
            config.workers,
            stage,
            config,
            variant,
        )
        if not args.rerun and log.get(f"{run_id(identity)}-verify"):
            print(f"  resume {checkpoint.name}/{source.name}/{variant}")
            continue
        archive = (
            args.results
            / ("archives" if keep else "tmp")
            / slug(checkpoint.name)
            / slug(variant)
            / f"{source.stem}.{slug(variant)}.brv"
        )
        record = execute_operation(
            args=args,
            log=log,
            checkpoint=checkpoint,
            source=source,
            method="brevis",
            operation="compress",
            cache="hot",
            workers=config.workers,
            input_path=source,
            output_path=archive,
            brevis_config=config,
            stage=stage,
            variant=variant,
        )
        if args.dry_run:
            continue
        subprocess.run(
            verify_command(args.brevis_bin, archive, source, config.workers),
            check=True,
            stdout=subprocess.DEVNULL,
        )
        record_exactness(log, record, True, (record,))
        if not keep:
            archive.unlink(missing_ok=True)


def run_sweeps(args: argparse.Namespace, log: ResultLog, core: Checkpoint) -> None:
    for budget in args.search_budgets:
        if budget == 512:
            continue
        run_brevis_variant(
            args,
            log,
            core,
            "pareto",
            f"budget-{budget}",
            BrevisConfig(1, budget, 256, True),
            keep=True,
        )
    for workers in args.worker_sweep:
        if workers == 1:
            continue
        run_brevis_variant(
            args,
            log,
            core,
            "workers",
            f"workers-{workers}",
            BrevisConfig(workers),
            keep=False,
        )


def run_ablation(args: argparse.Namespace, log: ResultLog, core: Checkpoint) -> None:
    run_brevis_variant(
        args,
        log,
        core,
        "ablation",
        "no-phog",
        BrevisConfig(1, 512, 0, True),
        keep=True,
    )
    run_brevis_variant(
        args,
        log,
        core,
        "ablation",
        "no-astar-heuristic",
        BrevisConfig(1, 512, 256, False),
        keep=True,
    )


def size_shard(
    args: argparse.Namespace,
    log: ResultLog,
    checkpoint: Checkpoint,
    source: Path,
    method: str,
) -> None:
    workers = method_workers(method, args.workers)
    run_pair(
        args,
        log,
        checkpoint,
        source,
        method,
        "unconditioned",
        workers,
        "corpus",
        archive_path(args.results, checkpoint.name, method, source),
        BrevisConfig(workers),
    )


def run_generic_corpus(
    args: argparse.Namespace,
    log: ResultLog,
    checkpoints: list[Checkpoint],
) -> None:
    methods = [method for method in args.methods if method in GENERIC_METHODS]
    for checkpoint in checkpoints:
        print(f"\n== corpus: {checkpoint.name} ({checkpoint.source_bytes} bytes)")
        for method in methods:
            args.deadline.require_time()
            jobs = args.shard_jobs if method in ("zstd-9", "lz4-hc-9", "snappy") else 1
            first, *remaining = checkpoint.files
            size_shard(args, log, checkpoint, first, method)
            if not remaining:
                continue
            latest_pair = {}
            for row in log.records:
                if (
                    row.get("status") == "ok"
                    and row.get("stage") == "corpus"
                    and row.get("checkpoint") == checkpoint.name
                    and row.get("method") == method
                    and row.get("shard") == first.name
                    and row.get("operation") in ("compress", "decompress")
                ):
                    latest_pair[row["operation"]] = row
            measured_seconds = sum(
                row.get("wall_seconds", 0.0) for row in latest_pair.values()
            )
            remaining_bytes = sum(source.stat().st_size for source in remaining)
            estimated_seconds = (
                remaining_bytes * measured_seconds / first.stat().st_size / jobs
                if measured_seconds and first.stat().st_size
                else 0.0
            )
            if not args.deadline.can_fit(estimated_seconds):
                reason = (
                    f"time budget: estimated {estimated_seconds:.0f}s, "
                    f"{args.deadline.seconds_left():.0f}s left"
                )
                print(f"  stop {checkpoint.name}/{method}: {reason}")
                for source in remaining:
                    append_not_run(log, checkpoint, source, method, reason)
                continue
            if jobs == 1:
                for source in remaining:
                    size_shard(args, log, checkpoint, source, method)
                continue
            with ThreadPoolExecutor(max_workers=min(jobs, len(remaining))) as pool:
                futures = [
                    pool.submit(size_shard, args, log, checkpoint, source, method)
                    for source in remaining
                ]
                for future in as_completed(futures):
                    future.result()


def load_specialized_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config = json.loads(path.expanduser().read_text())
    unknown = set(config) - set(SPECIALIZED_METHODS)
    if unknown:
        raise BenchmarkError(f"unknown specialized method(s): {sorted(unknown)}")
    return config


def format_command(
    template: list[str],
    checkpoint: Checkpoint,
    archive: Path,
    workers: int,
) -> list[str]:
    values = {
        "source_dir": str(checkpoint.directory),
        "archive": str(archive),
        "workers": str(workers),
        "checkpoint": checkpoint.name,
        "repo_id": checkpoint.repo_id or str(checkpoint.directory),
        "revision": checkpoint.revision or "",
    }
    return [part.format_map(values) for part in template]


def specialized_identity(
    checkpoint: Checkpoint,
    method: str,
    workers: int,
) -> dict[str, Any]:
    return {
        "stage": "specialized",
        "checkpoint": checkpoint.name,
        "shard": None,
        "revision": checkpoint.revision,
        "source_size_bytes": checkpoint.source_bytes,
        "method": method,
        "operation": "conversion",
        "cache": "unconditioned",
        "workers": workers,
    }


def record_specialized_skip(
    log: ResultLog,
    checkpoint: Checkpoint,
    method: str,
    workers: int,
    reason: str,
) -> None:
    identity = specialized_identity(checkpoint, method, workers)
    log.append(
        {
            **identity,
            "run_id": run_id(identity),
            "status": "not_run",
            "reason": reason,
            "finished_at": utc_now(),
        }
    )


def run_specialized(
    args: argparse.Namespace,
    log: ResultLog,
    checkpoints: list[Checkpoint],
    config: dict[str, Any],
) -> None:
    for method in (item for item in args.methods if item in SPECIALIZED_METHODS):
        settings = config[method]
        pattern = re.compile(settings.get("checkpoint_pattern", ".*"))
        workers = int(settings.get("workers", 32))
        for checkpoint in checkpoints:
            if not pattern.search(checkpoint.name):
                if not args.dry_run:
                    record_specialized_skip(
                        log,
                        checkpoint,
                        method,
                        workers,
                        "unsupported checkpoint",
                    )
                continue
            args.deadline.require_time()
            default_archive = (
                args.results
                / "archives"
                / slug(checkpoint.name)
                / f"{slug(method)}.brv"
            )
            output_template = settings.get("output_path", "{archive}")
            archive = Path(
                format_command(
                    [output_template],
                    checkpoint,
                    default_archive,
                    workers,
                )[0]
            )
            identity = specialized_identity(checkpoint, method, workers)
            identifier = run_id(identity)
            if not args.rerun and log.get(identifier) and archive.exists():
                print(f"  resume {checkpoint.name}/{method}/conversion")
                continue
            command = format_command(
                settings["compress_command"],
                checkpoint,
                archive,
                workers,
            )
            if args.dry_run:
                print(f"  {shlex.join(command)}")
                continue
            archive.parent.mkdir(parents=True, exist_ok=True)
            cwd = Path(settings["cwd"]).expanduser() if settings.get("cwd") else None
            try:
                measured = measure(
                    command,
                    args.results / "logs",
                    identifier,
                    archive,
                    cwd,
                )
                validate = settings.get("validate_command")
                if validate:
                    subprocess.run(
                        format_command(validate, checkpoint, archive, workers),
                        cwd=cwd,
                        check=True,
                    )
                elif not settings.get("validates_during_compression"):
                    raise BenchmarkError(
                        f"{method}: validation is not configured"
                    )
                archive_bytes = tree_size(archive)
                log.append(
                    {
                        **identity,
                        "run_id": identifier,
                        "status": "ok",
                        "attempt_id": uuid.uuid4().hex,
                        "command": command,
                        "source_bytes": checkpoint.source_bytes,
                        "output_bytes": archive_bytes,
                        "wall_seconds": measured.wall_seconds,
                        "throughput_bytes_per_second": (
                            checkpoint.source_bytes / measured.wall_seconds
                        ),
                        "peak_rss_bytes": measured.peak_rss_bytes,
                        "output_path": str(archive),
                        "exact": True,
                        "finished_at": utc_now(),
                    }
                )
            except Exception as exc:
                append_failure(log, identity, command, exc)
                raise


def command_version(
    command: list[str],
    cwd: Path | None = None,
) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "available"


def method_versions(
    args: argparse.Namespace,
    specialized: dict[str, Any],
) -> dict[str, str | None]:
    commands = {
        "brevis": ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
        "zstd-9": ["zstd", "--version"],
        "lz4-hc-9": ["lz4", "--version"],
        "zipnn": [sys.executable, str(CODEC_HELPER), "zipnn", "version"],
        "snappy": [sys.executable, str(CODEC_HELPER), "snappy", "version"],
    }
    versions = {
        method: command_version(command)
        for method, command in commands.items()
        if method in args.methods
    }
    for method in SPECIALIZED_METHODS:
        if method not in args.methods:
            continue
        settings = specialized.get(method)
        versions[method] = (
            command_version(
                settings["version_command"],
                Path(settings["cwd"]).expanduser()
                if settings.get("cwd")
                else None,
            )
            if settings and settings.get("version_command")
            else None
        )
    return versions


def build_brevis(args: argparse.Namespace) -> None:
    if args.brevis_bin != DEFAULT_BREVIS_BIN:
        if not args.brevis_bin.is_file():
            raise BenchmarkError(f"Brevis binary does not exist: {args.brevis_bin}")
        return
    if args.dry_run:
        print("  zig build -Doptimize=ReleaseFast")
        return
    subprocess.run(
        ["zig", "build", "-Doptimize=ReleaseFast"],
        cwd=ROOT,
        check=True,
    )


def physical_cores() -> int | None:
    if sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"],
                text=True,
            ))
        except (OSError, subprocess.CalledProcessError, ValueError):
            return None
    if sys.platform.startswith("linux"):
        pairs = set()
        physical = core = None
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("physical id"):
                    physical = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line and physical is not None and core is not None:
                    pairs.add((physical, core))
                    physical = core = None
            if physical is not None and core is not None:
                pairs.add((physical, core))
        except OSError:
            return None
        return len(pairs) or None
    return None


def total_memory() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def write_environment(
    args: argparse.Namespace,
    versions: dict[str, str | None],
    corpus: list[Checkpoint],
) -> None:
    revision = command_version(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    path = args.results / "environment.json"
    existing = json.loads(path.read_text()) if path.exists() else {}
    method_versions = {
        **existing.get("method_versions", {}),
        **versions,
    }
    corpus_by_name = {
        checkpoint["name"]: checkpoint
        for checkpoint in existing.get("corpus", ())
    }
    corpus_by_name.update(
        {
            checkpoint.name: {
                "name": checkpoint.name,
                "repo_id": checkpoint.repo_id,
                "revision": checkpoint.revision,
                "source_bytes": checkpoint.source_bytes,
                "shards": [source.name for source in checkpoint.files],
            }
            for checkpoint in corpus
        }
    )
    environment = {
        "generated_at": utc_now(),
        "brevis_revision": revision,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
        "physical_cores": physical_cores(),
        "ram_bytes": total_memory(),
        "workers": args.workers,
        "shard_jobs": args.shard_jobs,
        "method_versions": method_versions,
        "corpus": list(corpus_by_name.values()),
        "frozen_corpus": (
            existing.get("frozen_corpus", True)
            and not args.allow_custom_corpus
        ),
        "cold_cache_available": args.cold_available,
        "drop_caches_command": args.drop_caches_command,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")


def preflight(
    args: argparse.Namespace,
    specialized: dict[str, Any],
) -> dict[str, str | None]:
    if "brevis" in args.methods:
        build_brevis(args)
    versions = method_versions(args, specialized)
    missing = [method for method, version in versions.items() if version is None]
    for method, version in versions.items():
        print(f"{method:12} {version or 'MISSING'}")
    if missing and not args.allow_missing:
        raise BenchmarkError(
            "missing method(s): "
            + ", ".join(missing)
            + "; install them or pass --allow-missing"
        )
    args.methods = [method for method in args.methods if method not in missing]
    return versions


def verified_attempts(records: list[dict[str, Any]]) -> set[tuple[str, str | None]]:
    return {
        (run_id, attempt_id)
        for record in records
        if record.get("operation") == "verify"
        and record.get("status") == "ok"
        and record.get("exact")
        for run_id, attempt_id in record.get("verified_attempts", ())
    }


def latest_records(path: Path) -> list[dict[str, Any]]:
    latest = {}
    for record in read_jsonl(path):
        latest[record["run_id"]] = record
    return list(latest.values())


def aggregate(
    records: Iterable[dict[str, Any]],
    keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(tuple(record.get(key) for key in keys), []).append(record)
    output = []
    for identity, rows in sorted(groups.items()):
        source_bytes = sum(row.get("source_bytes", 0) for row in rows)
        wall = sum(row.get("wall_seconds", 0.0) for row in rows)
        rss_values = [
            row["peak_rss_bytes"]
            for row in rows
            if row.get("peak_rss_bytes") is not None
        ]
        result = dict(zip(keys, identity))
        result.update(
            {
                "source_bytes": source_bytes,
                "output_bytes": sum(row.get("output_bytes", 0) for row in rows),
                "archive_percent": (
                    100
                    * sum(row.get("output_bytes", 0) for row in rows)
                    / source_bytes
                    if source_bytes
                    else None
                ),
                "wall_seconds": wall,
                "throughput_gib_s": source_bytes / wall / 1024**3 if wall else None,
                "peak_rss_gib": max(rss_values) / 1024**3 if rss_values else None,
            }
        )
        output.append(result)
    return output


def end_to_end_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = aggregate(
        (
            row
            for row in records
            if row.get("stage") == "core"
            and row.get("operation") in ("compress", "decompress")
        ),
        ("method", "operation", "cache", "workers"),
    )
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        by_method.setdefault(row["method"], []).append(row)

    table = []
    for method, rows in sorted(by_method.items()):
        lookup = {
            (row["operation"], row["cache"]): row
            for row in rows
        }
        compression = next(
            (
                lookup[key]
                for key in (
                    ("compress", "cold"),
                    ("compress", "unconditioned"),
                    ("compress", "hot"),
                )
                if key in lookup
            ),
            None,
        )
        cold_measured = ("compress", "cold") in lookup
        row: dict[str, Any] = {
            "method": method,
            "workers": rows[0]["workers"],
            "archive_percent": compression["archive_percent"] if compression else None,
            "cold_cache_status": "measured" if cold_measured else "not measured",
        }
        for operation in ("compress", "decompress"):
            for cache in ("cold", "hot"):
                value = lookup.get((operation, cache))
                row[f"{operation}_{cache}_seconds"] = (
                    value["wall_seconds"] if value else None
                )
                row[f"{operation}_{cache}_gib_s"] = (
                    value["throughput_gib_s"] if value else None
                )
        for operation in ("compress", "decompress"):
            values = [
                value["peak_rss_gib"]
                for (candidate_operation, _), value in lookup.items()
                if candidate_operation == operation
                and value["peak_rss_gib"] is not None
            ]
            row[f"peak_rss_{operation}_gib"] = max(values) if values else None
        table.append(row)
    return table


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compression_effectiveness_table(
    results: Path,
    latest: list[dict[str, Any]],
    verified_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    environment_path = results / "environment.json"
    if not environment_path.exists():
        return []
    environment = json.loads(environment_path.read_text())
    versions = environment.get("method_versions", {})
    table = []
    for checkpoint in environment.get("corpus", ()):
        expected_shards = set(checkpoint["shards"])
        for method in versions:
            rows = [
                row
                for row in verified_records
                if row.get("checkpoint") == checkpoint["name"]
                and row.get("revision") == checkpoint["revision"]
                and row.get("method") == method
                and (
                    (
                        row.get("stage") == "corpus"
                        and row.get("operation") == "compress"
                    )
                    or (
                        row.get("stage") == "specialized"
                        and row.get("operation") == "conversion"
                    )
                )
            ]
            complete = bool(rows)
            if method in GENERIC_METHODS:
                complete = {row["shard"] for row in rows} == expected_shards
            summary = (
                aggregate(rows, ("checkpoint", "method"))[0]
                if complete
                else None
            )
            attempts = [
                row
                for row in latest
                if row.get("checkpoint") == checkpoint["name"]
                and row.get("revision") == checkpoint["revision"]
                and row.get("method") == method
                and row.get("stage") in ("corpus", "specialized")
            ]
            if summary:
                status = "ok"
            elif versions.get(method) is None:
                status = "missing dependency/config"
            elif any(row.get("status") == "failed" for row in attempts):
                status = "failed"
            elif any(row.get("status") == "not_run" for row in attempts):
                reasons = {
                    row.get("reason", "not run")
                    for row in attempts
                    if row.get("status") == "not_run"
                }
                status = "; ".join(sorted(reasons))
            elif rows:
                status = "incomplete"
            else:
                status = "not run"
            table.append(
                {
                    "checkpoint": checkpoint["name"],
                    "method": method,
                    "status": status,
                    "source_bytes": checkpoint["source_bytes"],
                    "output_bytes": summary["output_bytes"] if summary else None,
                    "archive_percent": (
                        summary["archive_percent"] if summary else None
                    ),
                }
            )
    return table


def summarize(results: Path) -> None:
    latest = latest_records(results / "raw" / "runs.jsonl")
    records = [
        record
        for record in latest
        if record.get("status") == "ok"
    ]
    verified = verified_attempts(records)
    verified_records = [
        record
        for record in records
        if (record.get("run_id"), record.get("attempt_id")) in verified
        or (
            record.get("stage") == "specialized"
            and record.get("exact")
        )
    ]
    table2 = compression_effectiveness_table(
        results,
        latest,
        verified_records,
    )
    table3 = end_to_end_table(verified_records)
    core_full = [
        row
        for row in verified_records
        if row.get("stage") == "core"
        and row.get("method") == "brevis"
        and row.get("operation") == "compress"
        and row.get("cache") == "hot"
        and row.get("workers") == 1
        and row.get("max_expansions") == 512
    ]
    figure1_rows = [
        row
        for row in verified_records
        if row.get("stage") == "pareto"
        and row.get("operation") == "compress"
    ] + [{**row, "variant": "budget-512"} for row in core_full]
    figure1 = aggregate(
        figure1_rows,
        ("variant", "max_expansions", "workers"),
    )
    figure2_rows = [
        row
        for row in verified_records
        if row.get("stage") == "workers"
        and row.get("operation") == "compress"
    ] + core_full
    figure2 = aggregate(
        figure2_rows,
        ("workers", "max_expansions"),
    )
    table4_rows = [
        row
        for row in verified_records
        if row.get("stage") == "ablation"
        and row.get("operation") == "compress"
    ] + [
        {
            **row,
            "variant": "full",
            "phog": True,
            "astar_heuristic": True,
        }
        for row in core_full
    ]
    table4 = aggregate(
        table4_rows,
        ("variant", "phog", "astar_heuristic"),
    )
    tables = results / "tables"
    write_csv(tables / "table2-compression-effectiveness.csv", table2)
    write_csv(tables / "table3-end-to-end.csv", table3)
    write_csv(tables / "figure1-archive-size-vs-time.csv", figure1)
    write_csv(tables / "figure2-throughput-rss-vs-workers.csv", figure2)
    write_csv(tables / "table4-ablation.csv", table4)

    failed = [
        record
        for record in latest
        if record.get("status") == "failed"
    ]
    incomplete = sum(row["status"] != "ok" for row in table2)
    status = [
        "# Benchmark status",
        "",
        f"- Successful raw records: {len(records)}",
        f"- Failed records: {len(failed)}",
        f"- Table 2 rows: {len(table2)}",
        f"- Table 2 incomplete/missing cells: {incomplete}",
        f"- End-to-end rows: {len(table3)}",
        "",
        "Table 5 headroom and operator attribution require the read-only archive "
        "analyzer and are intentionally not inferred from aggregate CLI output.",
    ]
    (results / "status.md").write_text("\n".join(status) + "\n")
    print(f"wrote summaries under {tables}")


def parse_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(dict.fromkeys(int(item) for item in text.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("preflight", "core", "sweeps", "ablation", "corpus", "all", "summarize"),
    )
    parser.add_argument("--models-root", type=Path)
    parser.add_argument("--core-model", type=Path)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--brevis-bin",
        type=Path,
        default=DEFAULT_BREVIS_BIN,
    )
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(ALL_METHODS))
    parser.add_argument("--models", nargs="+")
    practical_default = min(32, physical_cores() or os.cpu_count() or 1)
    parser.add_argument("--workers", type=int, default=practical_default)
    parser.add_argument("--shard-jobs", type=int, default=practical_default)
    parser.add_argument("--search-budgets", type=parse_int_list, default=DEFAULT_BUDGETS)
    parser.add_argument("--worker-sweep", type=parse_int_list, default=DEFAULT_WORKERS)
    parser.add_argument("--deadline-hours", type=float)
    parser.add_argument("--drop-caches-command")
    parser.add_argument("--specialized-config", type=Path)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--allow-custom-corpus", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.shard_jobs < 1:
        parser.error("workers and shard-jobs must be positive")
    sweep_limit = max(
        os.cpu_count() or 1,
        2 * (physical_cores() or os.cpu_count() or 1),
    )
    args.worker_sweep = tuple(
        workers for workers in args.worker_sweep if 0 < workers <= sweep_limit
    )
    args.results = args.results.expanduser().resolve()
    args.brevis_bin = args.brevis_bin.expanduser().resolve()
    args.deadline = Deadline(args.deadline_hours)
    args.cold_available = cache_control_available(args.drop_caches_command)
    return args


def main() -> int:
    args = parse_args()
    if args.stage == "summarize":
        summarize(args.results)
        return 0

    specialized = load_specialized_config(args.specialized_config)
    versions = preflight(args, specialized)
    log = ResultLog(args.results / "raw" / "runs.jsonl")
    core = (
        load_checkpoint(args.core_model, "qwen2.5-7b-local")
        if args.core_model
        else None
    )
    corpus = (
        load_corpus(
            args.models_root,
            args.models,
            args.allow_custom_corpus,
        )
        if args.models_root
        else []
    )
    if not args.dry_run:
        write_environment(args, versions, corpus)

    if args.stage in ("core", "sweeps", "ablation", "all") and core is None:
        raise BenchmarkError("--core-model is required for this stage")
    if args.stage in ("corpus", "all") and not corpus:
        raise BenchmarkError("--models-root is required for this stage")

    try:
        if args.stage in ("core", "all"):
            run_core(args, log, core)
        if args.stage in ("sweeps", "all"):
            run_sweeps(args, log, core)
        if args.stage in ("ablation", "all"):
            run_ablation(args, log, core)
        if args.stage in ("corpus", "all"):
            run_generic_corpus(args, log, corpus)
            run_specialized(args, log, corpus, specialized)
    finally:
        if args.stage == "all":
            summarize(args.results)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        sys.exit(f"benchmark: {exc}")
