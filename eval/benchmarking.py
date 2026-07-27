#!/usr/bin/env python3
"""Reproducible, per-run benchmarks for generic lossless compressors.

The module deliberately keeps algorithm timing narrow: after both diagnostic log
files are open, a clock starts immediately before the compressor/decompressor
process is spawned and stops after a successful output has been fsynced. Input
staging, output inspection, hashing, and bit-for-bit verification all happen
outside the timed region. Every compressor writes an archive to disk and every
decompressor writes a restored file to disk.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Sequence
from typing import Any, Mapping


SCHEMA_ID = "brevis.generic-baseline-benchmark"
SCHEMA_VERSION = 3
PROFILES = ("speed", "default", "ratio")
DEFAULT_REPETITIONS = 6
DEFAULT_SCHEDULE_SEED = 2701
DEFAULT_OPERATION_TIMEOUT_SECONDS = 3600.0
DEFAULT_VERSION_PROBE_TIMEOUT_SECONDS = 5.0
PROCESS_TERMINATION_GRACE_SECONDS = 0.2
MODEL_METADATA_KEYS = ("model_tag", "model_repo", "model_revision", "manifest", "shard")
CODEC_ENVIRONMENT_VARIABLES = (
    "GZIP",
    "BZIP",
    "BZIP2",
    "XZ_DEFAULTS",
    "XZ_OPT",
    "ZSTD_CLEVEL",
    "ZSTD_NBTHREADS",
    "LZ4_CLEVEL",
    "BROTLI_PARAM_MODE",
    "BROTLI_PARAM_QUALITY",
    "BROTLI_PARAM_LGWIN",
    "OMP_NUM_THREADS",
)


try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _LIBC_PIDFD_OPEN = getattr(_LIBC, "pidfd_open")
    _LIBC_PIDFD_OPEN.argtypes = [ctypes.c_int, ctypes.c_uint]
    _LIBC_PIDFD_OPEN.restype = ctypes.c_int
except (OSError, AttributeError):  # pragma: no cover - Linux evaluation hosts provide it.
    _LIBC_PIDFD_OPEN = None


@dataclasses.dataclass(frozen=True)
class ThreadPolicy:
    """Declared threading behavior for a CLI configuration."""

    compression: str
    decompression: str
    requested_compression_threads: int | None
    requested_decompression_threads: int | None
    cli_args: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "compression": self.compression,
            "decompression": self.decompression,
            "requested_compression_threads": self.requested_compression_threads,
            "requested_decompression_threads": self.requested_decompression_threads,
            "cli_args": list(self.cli_args),
        }


@dataclasses.dataclass(frozen=True)
class BaselineSpec:
    """A fully specified generic compressor configuration.

    ``compress_args`` and ``decompress_args`` are argv templates. The only
    substitutions are ``{input}`` and ``{output}``; no shell is involved.
    Tools such as gzip that derive the output name from the input use
    ``implicit_suffix`` and omit ``{output}`` from their argv.
    """

    method: str
    profile: str
    executable: str
    version_args: tuple[str, ...]
    compress_args: tuple[str, ...]
    decompress_args: tuple[str, ...]
    thread_policy: ThreadPolicy
    implicit_suffix: str | None = None
    notes: str = ""

    @property
    def identifier(self) -> str:
        return f"{self.method}/{self.profile}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "method": self.method,
            "profile": self.profile,
            "executable": self.executable,
            "version_args": list(self.version_args),
            "compress_args": list(self.compress_args),
            "decompress_args": list(self.decompress_args),
            "implicit_suffix": self.implicit_suffix,
            "thread_policy": self.thread_policy.to_dict(),
            "notes": self.notes,
        }


SERIAL_CLI = ThreadPolicy(
    compression="CLI is single-threaded",
    decompression="CLI is single-threaded",
    requested_compression_threads=1,
    requested_decompression_threads=1,
)
RAW_COPY = ThreadPolicy(
    compression="one Python streaming-copy process; threading is not configurable",
    decompression="one Python streaming-copy process; threading is not configurable",
    requested_compression_threads=None,
    requested_decompression_threads=None,
)
RAW_COPY_PROGRAM = (
    "import sys\n"
    "with open(sys.argv[1], 'rb') as source, "
    "open(sys.argv[2], 'xb') as output:\n"
    "    while chunk := source.read(1048576):\n"
    "        output.write(chunk)\n"
)
XZ_SINGLE = ThreadPolicy(
    compression="one worker requested explicitly",
    decompression="single-threaded decoder",
    requested_compression_threads=1,
    requested_decompression_threads=1,
    cli_args=("-T1",),
)
ZSTD_SINGLE = ThreadPolicy(
    compression="single-thread mode and synchronous I/O requested explicitly",
    decompression="serial decoder with asynchronous I/O disabled explicitly",
    requested_compression_threads=1,
    requested_decompression_threads=1,
    cli_args=("--single-thread", "--no-asyncio"),
)


def _spec(
    method: str,
    profile: str,
    executable: str,
    version_args: tuple[str, ...],
    level_args: tuple[str, ...],
    common_compress: tuple[str, ...],
    decompress_args: tuple[str, ...],
    thread_policy: ThreadPolicy,
    *,
    implicit_suffix: str | None,
    notes: str,
) -> BaselineSpec:
    return BaselineSpec(
        method=method,
        profile=profile,
        executable=executable,
        version_args=version_args,
        compress_args=(*common_compress, *level_args, "{input}")
        if implicit_suffix
        else (*common_compress, *level_args, "{input}", "{output}"),
        decompress_args=decompress_args,
        thread_policy=thread_policy,
        implicit_suffix=implicit_suffix,
        notes=notes,
    )


def _build_registry() -> tuple[BaselineSpec, ...]:
    specs: list[BaselineSpec] = [
        BaselineSpec(
            method="raw",
            profile="copy",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=("-c", RAW_COPY_PROGRAM, "{input}", "{output}"),
            decompress_args=("-c", RAW_COPY_PROGRAM, "{input}", "{output}"),
            thread_policy=RAW_COPY,
            notes=(
                "Uncompressed regular-file logical-copy reference using explicit "
                "1 MiB Python reads and writes, without clone or sparse-seek APIs. "
                "No codec is applied."
            ),
        )
    ]

    for profile, level, note in (
        ("speed", ("-1",), "Speed-oriented gzip level 1."),
        ("default", ("-6",), "Default-oriented profile: explicit gzip default level 6."),
        ("ratio", ("-9",), "Ratio-oriented profile: gzip's highest numbered level 9."),
    ):
        specs.append(_spec(
            "gzip", profile, "gzip", ("--version",), level,
            ("-n", "-k", "-f"), ("-d", "-k", "-f", "{input}"), SERIAL_CLI,
            implicit_suffix=".gz", notes=note + " -n removes timestamps and names.",
        ))

    for profile, level, note in (
        ("speed", ("-1",), "Speed-oriented bzip2 profile with 100 KiB blocks."),
        (
            "default", ("-9",),
            "Default-oriented profile: bzip2 defaults to its ratio-oriented 900 KiB block setting.",
        ),
        (
            "ratio", ("--best",),
            "Ratio-oriented profile: --best is equivalent to -9, so it matches bzip2/default.",
        ),
    ):
        specs.append(_spec(
            "bzip2", profile, "bzip2", ("--version",), level,
            ("-k", "-f"), ("-d", "-k", "-f", "{input}"), SERIAL_CLI,
            implicit_suffix=".bz2", notes=note,
        ))

    for profile, level, note in (
        ("speed", ("-1",), "Speed-oriented xz preset 1."),
        ("default", ("-6",), "Default-oriented profile: explicit xz default preset 6."),
        (
            "ratio", ("-9e",),
            "Ratio-oriented xz preset 9 with extreme mode; this bounds the profile "
            "and does not search custom filters.",
        ),
    ):
        specs.append(_spec(
            "xz", profile, "xz", ("--version",), level,
            ("-k", "-f", "-T1"),
            ("-d", "-k", "-f", "--no-sparse", "{input}"), XZ_SINGLE,
            implicit_suffix=".xz", notes=note,
        ))

    for profile, level, note in (
        ("speed", ("-1",), "Speed-oriented Zstandard level 1."),
        (
            "default", ("-3",),
            "Default-oriented profile: Zstandard's default compression level 3 "
            "under the serial, synchronous-I/O policy.",
        ),
        (
            "ratio", ("-19",),
            "Ratio-oriented level 19, bounded to regular levels.",
        ),
        (
            "ultra", ("--ultra", "-22"),
            "Ceiling-oriented extreme profile: Zstandard ultra level 22.",
        ),
    ):
        specs.append(BaselineSpec(
            method="zstd",
            profile=profile,
            executable="zstd",
            version_args=("--version",),
            compress_args=(
                "-q", "-f", "--single-thread", "--no-asyncio", *level,
                "{input}", "-o", "{output}",
            ),
            decompress_args=(
                "-d", "-q", "-f", "--no-asyncio", "--no-sparse",
                "{input}", "-o", "{output}",
            ),
            thread_policy=ZSTD_SINGLE,
            notes=(
                note + " Compression disables asynchronous I/O in addition to "
                "requesting single-thread mode; decoding disables asynchronous I/O "
                "and sparse output."
            ),
        ))

    for profile, level, note in (
        ("speed", ("--fast=5",), "Speed-oriented LZ4 acceleration setting 5."),
        ("default", ("-1",), "Default-oriented profile: explicit LZ4 default level 1."),
        ("ratio", ("-12",), "Ratio-oriented LZ4 high-compression level 12 (--best)."),
    ):
        specs.append(BaselineSpec(
            method="lz4",
            profile=profile,
            executable="lz4",
            version_args=("--version",),
            compress_args=("-q", "-f", *level, "{input}", "{output}"),
            decompress_args=("-d", "-q", "-f", "--no-sparse", "{input}", "{output}"),
            thread_policy=SERIAL_CLI,
            notes=note,
        ))

    for profile, level, note in (
        ("speed", ("-q", "1"), "Speed-oriented Brotli quality 1."),
        (
            "default", ("-q", "11"),
            "Default-oriented profile: explicit Brotli CLI default quality 11 and default window.",
        ),
        (
            "ratio", ("-q", "11", "-w", "24"),
            "Ratio-oriented quality 11 with window 24; the profile is bounded and "
            "does not tune other Brotli parameters.",
        ),
    ):
        specs.append(BaselineSpec(
            method="brotli",
            profile=profile,
            executable="brotli",
            version_args=("--version",),
            compress_args=("-f", *level, "-o", "{output}", "{input}"),
            decompress_args=("-d", "-f", "-o", "{output}", "{input}"),
            thread_policy=SERIAL_CLI,
            notes=note,
        ))

    identifiers = [spec.identifier for spec in specs]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("duplicate generic baseline identifier")
    return tuple(specs)


BASELINE_SPECS = _build_registry()
SPEC_BY_ID = {spec.identifier: spec for spec in BASELINE_SPECS}


def _render_args(template: Sequence[str], input_path: pathlib.Path, output_path: pathlib.Path) -> list[str]:
    substitutions = {"input": str(input_path), "output": str(output_path)}
    return [argument.format_map(substitutions) for argument in template]


def _captured_bytes(data: bytes) -> dict[str, Any]:
    """Keep human-readable and byte-exact forms of diagnostic output."""

    return {
        "text": data.decode("utf-8", errors="replace"),
        "base64": base64.b64encode(data).decode("ascii"),
        "size_bytes": len(data),
    }


def _rss_bytes(ru_maxrss: int | float) -> int:
    # getrusage(2) reports bytes on macOS and KiB on Linux and other BSD-like
    # systems supported by this evaluation environment.
    if sys.platform == "darwin":
        return int(ru_maxrss)
    return int(ru_maxrss * 1024)


def _signal_process_group(process: subprocess.Popen[bytes], signum: int) -> None:
    """Signal the session/process group created for one benchmark operation."""

    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _set_reaped_returncode(process: subprocess.Popen[bytes], status: int) -> int:
    exit_code = os.waitstatus_to_exitcode(status)
    # wait4 reaps the process directly, so keep Popen's state consistent and
    # prevent its destructor from attempting a second wait.
    process.returncode = exit_code
    return exit_code


def _force_cleanup_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Kill the complete process group and reap its direct child."""

    _signal_process_group(process, signal.SIGKILL)
    if process.returncode is not None:
        return
    try:
        if hasattr(os, "wait4"):
            _, status, _ = os.wait4(process.pid, 0)
            _set_reaped_returncode(process, status)
        else:  # pragma: no cover - evaluation hosts are Linux.
            process.wait()
    except ChildProcessError:
        # The child was already reaped between the state check and wait.
        process.poll()


def _open_pidfd(pid: int) -> tuple[int, str]:
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid), "os.pidfd_open"
    if _LIBC_PIDFD_OPEN is not None:
        descriptor = _LIBC_PIDFD_OPEN(pid, 0)
        if descriptor >= 0:
            return descriptor, "libc.pidfd_open"
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    raise NotImplementedError("pidfd_open is unavailable")


def _wait4_poll_fallback(
    process: subprocess.Popen[bytes], timeout_seconds: float | None,
) -> tuple[int, resource.struct_rusage, bool, str, str]:
    """Fallback for Linux kernels/Python builds without pidfd support."""

    deadline = None if timeout_seconds is None else time.perf_counter() + timeout_seconds
    while True:
        waited_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if waited_pid == process.pid:
            return (
                _set_reaped_returncode(process, status), usage, False,
                "wait4_wnohang_1ms_poll", "pidfd unavailable; 1 ms active polling",
            )
        if deadline is not None and time.perf_counter() >= deadline:
            _signal_process_group(process, signal.SIGTERM)
            grace_deadline = time.perf_counter() + PROCESS_TERMINATION_GRACE_SECONDS
            while time.perf_counter() < grace_deadline:
                waited_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
                if waited_pid == process.pid:
                    _signal_process_group(process, signal.SIGKILL)
                    return (
                        _set_reaped_returncode(process, status), usage, True,
                        "wait4_wnohang_1ms_poll", "pidfd unavailable; 1 ms active polling",
                    )
                time.sleep(0.001)
            _signal_process_group(process, signal.SIGKILL)
            _, status, usage = os.wait4(process.pid, 0)
            exit_code = _set_reaped_returncode(process, status)
            # The direct child may exit before every descendant processes
            # SIGTERM, so enforce SIGKILL on the remaining group as well.
            _signal_process_group(process, signal.SIGKILL)
            return (
                exit_code, usage, True, "wait4_wnohang_1ms_poll",
                "pidfd unavailable; 1 ms active polling",
            )
        time.sleep(0.001)


def _wait_direct_child(
    process: subprocess.Popen[bytes], timeout_seconds: float | None,
) -> tuple[int, resource.struct_rusage | None, bool, str, str]:
    """Wait using pidfd readiness where available, preserving direct-child rusage."""

    if hasattr(os, "wait4"):
        try:
            pidfd, pidfd_backend = _open_pidfd(process.pid)
        except (OSError, NotImplementedError) as exc:
            exit_code, usage, timed_out, strategy, detail = _wait4_poll_fallback(
                process, timeout_seconds,
            )
            return (
                exit_code, usage, timed_out, strategy,
                f"{detail}; pidfd_open failed: {type(exc).__name__}: {exc}",
            )

        selector = selectors.DefaultSelector()
        try:
            selector.register(pidfd, selectors.EVENT_READ)
        except OSError as exc:
            selector.close()
            os.close(pidfd)
            exit_code, usage, timed_out, strategy, detail = _wait4_poll_fallback(
                process, timeout_seconds,
            )
            return (
                exit_code, usage, timed_out, strategy,
                f"{detail}; pidfd selector registration failed: {type(exc).__name__}: {exc}",
            )
        try:
            ready = selector.select(timeout_seconds)
            timed_out = not ready
            if timed_out:
                _signal_process_group(process, signal.SIGTERM)
                if not selector.select(PROCESS_TERMINATION_GRACE_SECONDS):
                    _signal_process_group(process, signal.SIGKILL)
            _, status, usage = os.wait4(process.pid, 0)
            exit_code = _set_reaped_returncode(process, status)
            if timed_out:
                # A parent can exit on SIGTERM before one of its descendants.
                # Kill any remaining member of the isolated process group.
                _signal_process_group(process, signal.SIGKILL)
            return (
                exit_code, usage, timed_out, "pidfd_selector_wait4",
                f"readiness notification via {pidfd_backend}; no active timeout polling",
            )
        finally:
            selector.close()
            os.close(pidfd)

    # Non-POSIX fallback cannot provide per-child rusage. It still isolates and
    # terminates the process group when the platform supports start_new_session.
    try:  # pragma: no cover - evaluation hosts are Linux.
        exit_code = process.wait(timeout=timeout_seconds)
        return exit_code, None, False, "popen_wait", "non-POSIX fallback"
    except subprocess.TimeoutExpired:  # pragma: no cover
        _signal_process_group(process, signal.SIGTERM)
        try:
            exit_code = process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            exit_code = process.wait()
        _signal_process_group(process, signal.SIGKILL)
        return exit_code, None, True, "popen_wait", "non-POSIX fallback"


def _fsync_file(path: pathlib.Path) -> None:
    with path.open("rb") as output:
        os.fsync(output.fileno())


def _run_process(
    command: Sequence[str],
    log_dir: pathlib.Path,
    label: str,
    timeout_seconds: float | None,
    *,
    durable_output: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Run one process and return non-cumulative timing and resource data."""

    stdout_path = log_dir / f"{label}.stdout"
    stderr_path = log_dir / f"{label}.stderr"
    argv = [str(part) for part in command]
    started_ns: int | None = None
    ended_ns: int | None = None
    process: subprocess.Popen[bytes] | None = None
    usage: resource.struct_rusage | None = None
    exit_code: int | None = None
    error: str | None = None
    timed_out = False
    wait_strategy: str | None = None
    wait_strategy_detail: str | None = None
    durability = {
        "policy": (
            "file_fsync_after_successful_exit_before_wall_clock_stop"
            if durable_output is not None
            else "none"
        ),
        "target_path": str(durable_output) if durable_output is not None else None,
        "attempted": False,
        "succeeded": None,
        "status_code": "not_requested" if durable_output is None else "not_attempted",
        "error": None,
        "included_in_wall_time": False,
    }
    operation_started_at_utc = _utc_now()
    try:
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            # Open both logs before entering the measured region. Popen is the
            # first operation after the clock read.
            started_ns = time.perf_counter_ns()
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                close_fds=True,
                start_new_session=True,
            )
            wait_timeout = timeout_seconds
            if timeout_seconds is not None:
                spawn_elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
                wait_timeout = max(0.0, timeout_seconds - spawn_elapsed_seconds)
            exit_code, usage, timed_out, wait_strategy, wait_strategy_detail = _wait_direct_child(
                process, wait_timeout,
            )
            if durable_output is not None and not timed_out and exit_code == 0:
                durability["attempted"] = True
                durability["included_in_wall_time"] = True
                try:
                    _fsync_file(durable_output)
                except OSError as exc:
                    durability["succeeded"] = False
                    durability["status_code"] = "failed"
                    durability["error"] = f"{type(exc).__name__}: {exc}"
                else:
                    durability["succeeded"] = True
                    durability["status_code"] = "ok"
            ended_ns = time.perf_counter_ns()
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
        if process is not None:
            _force_cleanup_process_tree(process)
        ended_ns = time.perf_counter_ns()
    except BaseException:
        if process is not None:
            _force_cleanup_process_tree(process)
        raise

    if durable_output is not None and durability["status_code"] == "not_attempted":
        durability["status_code"] = "not_attempted_process_failed"

    if started_ns is None:
        # Opening a log failed before the timed region could begin.
        wall_time_ns = None
    else:
        if ended_ns is None:
            ended_ns = time.perf_counter_ns()
        wall_time_ns = ended_ns - started_ns

    stdout = stdout_path.read_bytes() if stdout_path.exists() else b""
    stderr = stderr_path.read_bytes() if stderr_path.exists() else b""
    term_signal = -exit_code if exit_code is not None and exit_code < 0 else None
    if timed_out:
        status_code = "timed_out"
    elif error is not None:
        status_code = "launch_or_wait_error"
    elif durability["status_code"] == "failed":
        status_code = "durability_failed"
    elif exit_code == 0:
        status_code = "ok"
    elif exit_code is not None and exit_code < 0:
        status_code = "terminated_by_signal"
    else:
        status_code = "nonzero_exit"
    return {
        "status_code": status_code,
        "started_at_utc": operation_started_at_utc,
        "ended_at_utc": _utc_now(),
        "command": argv,
        "wall_time_ns": wall_time_ns,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "direct_child_max_rss_bytes": _rss_bytes(usage.ru_maxrss) if usage is not None else None,
        "direct_child_user_cpu_time_ns": (
            int(usage.ru_utime * 1_000_000_000) if usage is not None else None
        ),
        "direct_child_system_cpu_time_ns": (
            int(usage.ru_stime * 1_000_000_000) if usage is not None else None
        ),
        "exit_code": exit_code,
        "term_signal": term_signal,
        "process_group_isolated": process is not None,
        "started_new_session": process is not None,
        "process_group_id": process.pid if process is not None else None,
        "wait_strategy": wait_strategy,
        "wait_strategy_detail": wait_strategy_detail,
        "stdout": _captured_bytes(stdout),
        "stderr": _captured_bytes(stderr),
        "error": error,
        "durability": durability,
    }


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _file_allocation(path: pathlib.Path) -> dict[str, int | None]:
    status = path.stat()
    blocks = getattr(status, "st_blocks", None)
    return {
        "logical_size_bytes": status.st_size,
        "st_blocks_512b": int(blocks) if blocks is not None else None,
        "allocated_size_bytes": int(blocks * 512) if blocks is not None else None,
    }


def _verify_file(expected: pathlib.Path, actual: pathlib.Path, expected_sha256: str) -> dict[str, Any]:
    """Scan complete files after decompression timing has stopped."""

    if not actual.is_file():
        return {
            "attempted": False,
            "bit_exact": False,
            "source_sha256": expected_sha256,
            "restored_sha256": None,
            "restored_size_bytes": None,
            "restored_storage": None,
            "error": f"restored output is missing: {actual}",
        }

    digest = hashlib.sha256()
    bit_exact = True
    try:
        with expected.open("rb") as expected_file, actual.open("rb") as actual_file:
            while True:
                expected_chunk = expected_file.read(8 << 20)
                actual_chunk = actual_file.read(8 << 20)
                if actual_chunk:
                    digest.update(actual_chunk)
                if expected_chunk != actual_chunk:
                    bit_exact = False
                if not expected_chunk and not actual_chunk:
                    break
        restored_sha256 = digest.hexdigest()
        bit_exact = bit_exact and restored_sha256 == expected_sha256
        return {
            "attempted": True,
            "bit_exact": bit_exact,
            "source_sha256": expected_sha256,
            "restored_sha256": restored_sha256,
            "restored_size_bytes": actual.stat().st_size,
            "restored_storage": _file_allocation(actual),
            "error": None if bit_exact else "restored bytes differ from source",
        }
    except OSError as exc:
        return {
            "attempted": True,
            "bit_exact": False,
            "source_sha256": expected_sha256,
            "restored_sha256": None,
            "restored_size_bytes": None,
            "restored_storage": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _stage_file(source: pathlib.Path, destination: pathlib.Path) -> str:
    """Stage an independent inode outside the timed region.

    External codecs are allowed to overwrite or unlink their inputs. A hard link
    would therefore let a faulty codec corrupt the original source or the archive
    retained for hashing. ``copy2`` may use a kernel copy or CoW acceleration,
    but the destination is always a distinct inode.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if os.path.samefile(source, destination):  # Defensive check for unusual filesystems.
        raise OSError(f"staging did not create an independent inode: {destination}")
    return "independent_copy"


def _failure_for(
    compression: dict[str, Any] | None,
    decompression: dict[str, Any] | None,
    verification: dict[str, Any] | None,
    compressed_exists: bool,
    archive_hash_error: str | None,
) -> tuple[str, str | None]:
    if compression is None:
        return "compression_not_started", "compression was not started"
    if compression["timed_out"]:
        return (
            "compression_timed_out",
            f"compression timed out after {compression['timeout_seconds']} seconds",
        )
    if compression["error"]:
        return "compression_launch_error", f"compression launch failed: {compression['error']}"
    if compression["exit_code"] != 0:
        return (
            "compression_nonzero_exit",
            f"compression exited with status {compression['exit_code']}",
        )
    if compression["durability"]["status_code"] == "failed":
        return (
            "compression_durability_failed",
            f"compression output durability failed: {compression['durability']['error']}",
        )
    if not compressed_exists:
        return "archive_missing", "compression exited successfully but produced no archive"
    if archive_hash_error is not None:
        return "archive_hash_failed", f"archive hashing failed: {archive_hash_error}"
    if decompression is None:
        return "decompression_not_started", "decompression was not started"
    if decompression["timed_out"]:
        return (
            "decompression_timed_out",
            f"decompression timed out after {decompression['timeout_seconds']} seconds",
        )
    if decompression["error"]:
        return (
            "decompression_launch_error",
            f"decompression launch failed: {decompression['error']}",
        )
    if decompression["exit_code"] != 0:
        return (
            "decompression_nonzero_exit",
            f"decompression exited with status {decompression['exit_code']}",
        )
    if decompression["durability"]["status_code"] == "failed":
        return (
            "decompression_durability_failed",
            f"decompression output durability failed: {decompression['durability']['error']}",
        )
    if verification is None:
        return "verification_not_started", "verification was not started"
    if verification["error"] is not None:
        return "verification_failed", verification["error"]
    return "ok", None


def _run_iteration(
    spec: BaselineSpec,
    executable: str,
    source: pathlib.Path,
    source_sha256: str,
    phase: str,
    index: int,
    run_dir: pathlib.Path,
    timeout_seconds: float | None,
    execution_order: int,
    order_within_repetition: int,
) -> dict[str, Any]:
    iteration_started_at_utc = _utc_now()
    compress_dir = run_dir / "compress"
    decompress_dir = run_dir / "decompress"
    compress_dir.mkdir(parents=True)
    decompress_dir.mkdir(parents=True)

    staged_source = compress_dir / "input.bin"
    source_staging = _stage_file(source, staged_source)
    if spec.implicit_suffix:
        compressed = pathlib.Path(f"{staged_source}{spec.implicit_suffix}")
    else:
        compressed = compress_dir / f"archive{spec.implicit_suffix or '.bin'}"
    compress_command = [
        executable,
        *_render_args(spec.compress_args, staged_source, compressed),
    ]
    compression = _run_process(
        compress_command, run_dir, "compression", timeout_seconds,
        durable_output=compressed,
    )

    compressed_exists = compressed.is_file()
    compressed_size = compressed.stat().st_size if compressed_exists else None
    archive_storage = _file_allocation(compressed) if compressed_exists else None
    archive_sha256: str | None = None
    archive_hash_error: str | None = None
    if compressed_exists:
        try:
            # The archive scan is outside compression and decompression timing.
            # It also conditions the archive cache before decode, without
            # guaranteeing that every page remains resident.
            archive_sha256 = _sha256_file(compressed)
        except OSError as exc:
            archive_hash_error = f"{type(exc).__name__}: {exc}"
    compression["output_size_bytes"] = compressed_size
    compression["output_storage"] = archive_storage
    decompression: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    archive_staging: str | None = None
    restored = decompress_dir / "restored.bin"

    if compression["status_code"] == "ok" and compressed_exists:
        if spec.implicit_suffix:
            staged_archive = pathlib.Path(f"{restored}{spec.implicit_suffix}")
        else:
            staged_archive = decompress_dir / f"archive{compressed.suffix or '.bin'}"
        archive_staging = _stage_file(compressed, staged_archive)
        decompress_command = [
            executable,
            *_render_args(spec.decompress_args, staged_archive, restored),
        ]
        decompression = _run_process(
            decompress_command, run_dir, "decompression", timeout_seconds,
            durable_output=restored,
        )
        decompression["output_size_bytes"] = restored.stat().st_size if restored.is_file() else None
        decompression["output_storage"] = (
            _file_allocation(restored) if restored.is_file() else None
        )
        if decompression["status_code"] == "ok":
            # This full scan is intentionally after _run_process has stopped its clock.
            verification = _verify_file(source, restored, source_sha256)

    status_code, failure = _failure_for(
        compression, decompression, verification, compressed_exists, archive_hash_error,
    )
    return {
        "status_code": status_code,
        "started_at_utc": iteration_started_at_utc,
        "ended_at_utc": _utc_now(),
        "phase": phase,
        "index": index,
        "execution_order": execution_order,
        "order_within_repetition": order_within_repetition,
        "compression": compression,
        "compressed_size_bytes": compressed_size,
        "archive_storage": archive_storage,
        "archive_sha256": archive_sha256,
        "archive_hash_error": archive_hash_error,
        "decompression": decompression,
        "verification": verification,
        "bit_exact": verification["bit_exact"] if verification is not None else False,
        "success": failure is None,
        "failure": failure,
        "staging": {
            "source": source_staging,
            "archive": archive_staging,
            "excluded_from_timing": True,
        },
        "artifact_directory": str(run_dir),
    }


def _extract_tool_version(spec: BaselineSpec, probe: Mapping[str, Any]) -> str | None:
    """Extract a stable version line without treating binary stdout as text.

    In particular, ``bzip2 --version`` writes its version to stderr but also
    compresses stdin to stdout. Looking at combined output can therefore prefix
    the version with an empty bzip2 archive.
    """

    stderr_lines = [
        line.strip() for line in probe["stderr"]["text"].splitlines() if line.strip()
    ]
    stdout_lines = [
        line.strip() for line in probe["stdout"]["text"].splitlines() if line.strip()
    ]
    if spec.method == "bzip2":
        for line in stderr_lines:
            if "bzip2" in line.lower() and "version" in line.lower():
                return line
        return None
    return next(iter((*stdout_lines, *stderr_lines)), None)


def _probe_tool(
    spec: BaselineSpec,
    executable: str,
    probe_dir: pathlib.Path,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    probe = _run_process(
        [executable, *spec.version_args],
        probe_dir,
        f"version-{spec.method}",
        timeout_seconds,
    )
    failure: str | None = None
    if probe["timed_out"]:
        failure = f"version probe timed out after {timeout_seconds} seconds"
    elif probe["error"] is not None:
        failure = f"version probe launch failed: {probe['error']}"
    elif probe["exit_code"] != 0:
        failure = f"version probe exited with status {probe['exit_code']}"
    return {
        "version": _extract_tool_version(spec, probe) if failure is None else None,
        "probe": probe,
        "failure": failure,
    }


def _apply_archive_consistency(
    method: dict[str, Any], record: dict[str, Any], phase: str,
) -> None:
    consistency = method["measured_archive_consistency"]
    if phase != "measured":
        record["archive_consistency"] = {
            "status_code": "not_applicable_warmup",
            "matches_reference": None,
        }
        return
    if not record["success"]:
        record["archive_consistency"] = {
            "status_code": "not_checked_iteration_failed",
            "matches_reference": None,
        }
        return

    size = record["compressed_size_bytes"]
    archive_sha256 = record["archive_sha256"]
    if consistency["reference_sha256"] is None:
        consistency.update({
            "status_code": "consistent",
            "reference_size_bytes": size,
            "reference_sha256": archive_sha256,
            "consistent_runs": 1,
        })
        record["archive_consistency"] = {
            "status_code": "reference",
            "matches_reference": True,
            "reference_size_bytes": size,
            "reference_sha256": archive_sha256,
        }
        return

    matches = (
        size == consistency["reference_size_bytes"]
        and archive_sha256 == consistency["reference_sha256"]
    )
    record["archive_consistency"] = {
        "status_code": "consistent" if matches else "archive_inconsistent",
        "matches_reference": matches,
        "reference_size_bytes": consistency["reference_size_bytes"],
        "reference_sha256": consistency["reference_sha256"],
    }
    if matches:
        consistency["consistent_runs"] += 1
        return

    consistency["status_code"] = "archive_inconsistent"
    consistency["inconsistent_runs"] += 1
    method["status_code"] = "archive_inconsistent"
    record["status_code"] = "archive_inconsistent"
    record["success"] = False
    record["failure"] = (
        "measured archive is not reproducible: size/SHA-256 differs from the "
        "first successful measured repetition"
    )


def select_specs(identifiers: Iterable[str] | None = None) -> tuple[BaselineSpec, ...]:
    if identifiers is None:
        return BASELINE_SPECS
    selected = []
    for identifier in identifiers:
        try:
            selected.append(SPEC_BY_ID[identifier])
        except KeyError as exc:
            choices = ", ".join(sorted(SPEC_BY_ID))
            raise ValueError(f"unknown baseline {identifier!r}; choose from: {choices}") from exc
    if not selected:
        raise ValueError("at least one baseline must be selected")
    return tuple(selected)


def _read_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cpu_quota() -> dict[str, Any]:
    """Read the effective cgroup CPU quota without modifying cgroup state."""

    cpu_max = _read_text(pathlib.Path("/sys/fs/cgroup/cpu.max"))
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2:
            quota_text, period_text = parts
            period = int(period_text)
            quota = None if quota_text == "max" else int(quota_text)
            return {
                "source": "/sys/fs/cgroup/cpu.max",
                "quota_microseconds": quota,
                "period_microseconds": period,
                "quota_cores": quota / period if quota is not None and period else None,
            }

    quota_text = _read_text(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
    period_text = _read_text(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
    if quota_text is not None and period_text is not None:
        quota_value = int(quota_text)
        period = int(period_text)
        quota = None if quota_value < 0 else quota_value
        return {
            "source": "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
            "quota_microseconds": quota,
            "period_microseconds": period,
            "quota_cores": quota / period if quota is not None and period else None,
        }
    return {
        "source": None,
        "quota_microseconds": None,
        "period_microseconds": None,
        "quota_cores": None,
    }


def _cpu_info() -> dict[str, Any]:
    affinity_count: int | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            pass

    model_name: str | None = None
    physical_cores: int | None = None
    cpuinfo = _read_text(pathlib.Path("/proc/cpuinfo"))
    if cpuinfo:
        core_pairs: set[tuple[str, str]] = set()
        for record in cpuinfo.split("\n\n"):
            fields: dict[str, str] = {}
            for line in record.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            model_name = model_name or fields.get("model name") or fields.get("Processor")
            if "physical id" in fields and "core id" in fields:
                core_pairs.add((fields["physical id"], fields["core id"]))
        if core_pairs:
            physical_cores = len(core_pairs)

    return {
        "model": model_name,
        "logical_cores_host": os.cpu_count(),
        "physical_cores_host": physical_cores,
        "affinity_logical_cores": affinity_count,
        "quota": _cpu_quota(),
    }


def _memory_info() -> dict[str, Any]:
    total: int | None = None
    available: int | None = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = int(page_size * os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError):
        pass

    meminfo = _read_text(pathlib.Path("/proc/meminfo"))
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                available = int(line.split()[1]) * 1024
                break
    cgroup_source: str | None = None
    cgroup_limit: int | None = None
    memory_max = _read_text(pathlib.Path("/sys/fs/cgroup/memory.max"))
    if memory_max is not None:
        cgroup_source = "/sys/fs/cgroup/memory.max"
        if memory_max != "max":
            cgroup_limit = int(memory_max)
    else:
        memory_limit = _read_text(
            pathlib.Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        )
        if memory_limit is not None:
            cgroup_source = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            cgroup_limit = int(memory_limit)
    return {
        "total_bytes": total,
        "available_bytes_at_start": available,
        "cgroup_limit_bytes": cgroup_limit,
        "cgroup_limit_source": cgroup_source,
    }


def _filesystem_info(path: pathlib.Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    status = os.stat(path)
    result: dict[str, Any] = {
        "path": str(path),
        "device": status.st_dev,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "mount_point": None,
        "filesystem_type": None,
        "source": None,
        "mount_option_names": None,
    }
    findmnt = shutil.which("findmnt")
    if findmnt is None:
        return result
    try:
        completed = subprocess.run(
            [findmnt, "--json", "--target", str(path), "--output", "TARGET,FSTYPE,SOURCE,OPTIONS"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if completed.returncode == 0:
            payload = json.loads(completed.stdout)
            filesystems = payload.get("filesystems", [])
            if filesystems:
                filesystem = filesystems[0]
                options = filesystem.get("options")
                result.update({
                    "mount_point": filesystem.get("target"),
                    "filesystem_type": filesystem.get("fstype"),
                    "source": filesystem.get("source"),
                    # Record option names, not values such as overlay layer paths
                    # or network-mount parameters.
                    "mount_option_names": (
                        [item.split("=", 1)[0] for item in options.split(",")]
                        if options else None
                    ),
                })
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return result


def _git_provenance() -> dict[str, Any]:
    repository = pathlib.Path(__file__).resolve().parent.parent
    git = shutil.which("git")
    result: dict[str, Any] = {
        "repository": str(repository),
        "commit": None,
        "dirty": None,
        "status_porcelain": None,
    }
    if git is None:
        return result
    try:
        commit = subprocess.run(
            [git, "-C", str(repository), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            text=True,
        )
        status = subprocess.run(
            [git, "-C", str(repository), "status", "--porcelain", "--untracked-files=normal"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            text=True,
        )
        if commit.returncode == 0:
            result["commit"] = commit.stdout.strip()
        if status.returncode == 0:
            status_text = status.stdout.rstrip("\n")
            result["dirty"] = bool(status_text)
            result["status_porcelain"] = status_text
    except (OSError, subprocess.TimeoutExpired):
        pass
    return result


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _paths_conflict(source: pathlib.Path, output: pathlib.Path) -> bool:
    source_resolved = source.resolve(strict=True)
    output_resolved = output.resolve(strict=False)
    if source_resolved == output_resolved:
        return True
    if output.exists():
        try:
            return os.path.samefile(source_resolved, output)
        except OSError:
            pass
    return False


def _validate_timeout(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, or None")


def _normalize_sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be a 64-character hexadecimal SHA-256")
    return normalized


def _metadata_value(
    explicit: Any, metadata: Mapping[str, Any], key: str, label: str,
) -> Any:
    declared = metadata.get(key)
    if explicit is not None and declared is not None and str(explicit) != str(declared):
        raise ValueError(f"conflicting {label} values in API arguments and input_metadata")
    return explicit if explicit is not None else declared


def _verify_declared_integrity(
    *,
    source_path: pathlib.Path,
    source_size: int,
    source_sha256: str,
    metadata: Mapping[str, Any],
    expected_source_size_bytes: int | None,
    expected_source_sha256: str | None,
    manifest_path: os.PathLike[str] | str | None,
    expected_manifest_sha256: str | None,
) -> dict[str, Any]:
    expected_size = _metadata_value(
        expected_source_size_bytes, metadata, "expected_source_size_bytes",
        "expected source size",
    )
    expected_source_hash = _metadata_value(
        expected_source_sha256, metadata, "expected_source_sha256",
        "expected source SHA-256",
    )
    metadata_manifest = metadata.get("manifest")
    if manifest_path is not None and metadata_manifest is not None:
        explicit_resolved = pathlib.Path(manifest_path).resolve(strict=False)
        metadata_resolved = pathlib.Path(metadata_manifest).resolve(strict=False)
        if explicit_resolved != metadata_resolved:
            raise ValueError(
                "conflicting manifest path values in API arguments and input_metadata"
            )
    declared_manifest = manifest_path if manifest_path is not None else metadata_manifest
    expected_manifest_hash = _metadata_value(
        expected_manifest_sha256, metadata, "expected_manifest_sha256",
        "expected manifest SHA-256",
    )
    model_metadata_declared = any(metadata.get(key) is not None for key in MODEL_METADATA_KEYS)

    if (expected_size is None) != (expected_source_hash is None):
        raise ValueError(
            "expected_source_size_bytes and expected_source_sha256 must be provided together"
        )
    if model_metadata_declared and expected_size is None:
        raise ValueError(
            "declared model metadata requires expected source size and SHA-256"
        )
    if model_metadata_declared and declared_manifest is None:
        raise ValueError("declared model metadata requires a manifest path")
    if (declared_manifest is None) != (expected_manifest_hash is None):
        raise ValueError(
            "manifest_path and expected_manifest_sha256 must be provided together"
        )

    normalized_source_hash: str | None = None
    if expected_size is not None:
        if isinstance(expected_size, bool):
            raise ValueError("expected_source_size_bytes must be a non-negative integer")
        if isinstance(expected_size, str):
            if not expected_size.isdecimal():
                raise ValueError("expected_source_size_bytes must be a non-negative integer")
            expected_size = int(expected_size)
        elif not isinstance(expected_size, int):
            raise ValueError("expected_source_size_bytes must be a non-negative integer")
        if expected_size < 0:
            raise ValueError("expected_source_size_bytes must be a non-negative integer")
        normalized_source_hash = _normalize_sha256(
            str(expected_source_hash), "expected_source_sha256",
        )
        if source_size != expected_size:
            raise ValueError(
                f"source size mismatch: expected {expected_size}, observed {source_size} for {source_path}"
            )
        if source_sha256 != normalized_source_hash:
            raise ValueError(
                "source SHA-256 mismatch: "
                f"expected {normalized_source_hash}, observed {source_sha256} for {source_path}"
            )

    manifest_record: dict[str, Any] | None = None
    manifest_binding: dict[str, Any] | None = None
    if declared_manifest is not None:
        manifest = pathlib.Path(declared_manifest).resolve(strict=True)
        if not manifest.is_file():
            raise ValueError(f"manifest is not a regular file: {manifest}")
        normalized_manifest_hash = _normalize_sha256(
            str(expected_manifest_hash), "expected_manifest_sha256",
        )
        actual_manifest_hash = _sha256_file(manifest)
        if actual_manifest_hash != normalized_manifest_hash:
            raise ValueError(
                "manifest SHA-256 mismatch: "
                f"expected {normalized_manifest_hash}, observed {actual_manifest_hash} for {manifest}"
            )
        manifest_record = {
            "path": str(manifest),
            "size_bytes": manifest.stat().st_size,
            "expected_sha256": normalized_manifest_hash,
            "actual_sha256": actual_manifest_hash,
            "verified": True,
        }
        if model_metadata_declared:
            required_metadata = ("model_tag", "model_repo", "model_revision", "shard")
            missing = [key for key in required_metadata if not metadata.get(key)]
            if missing:
                raise ValueError(
                    "declared model metadata requires nonempty " + ", ".join(missing)
                )
            try:
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot parse model manifest {manifest}: {exc}") from exc
            entries = (
                manifest_payload.get("models")
                if isinstance(manifest_payload, dict)
                else manifest_payload
            )
            if not isinstance(entries, list):
                raise ValueError("model manifest must be a JSON array or contain a models array")
            tag = str(metadata["model_tag"])
            matches = [
                entry for entry in entries
                if isinstance(entry, dict) and entry.get("tag") == tag
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"model manifest must contain exactly one entry for tag {tag!r}; "
                    f"found {len(matches)}"
                )
            model_entry = matches[0]
            for metadata_key, manifest_key in (
                ("model_repo", "repo"),
                ("model_revision", "revision"),
            ):
                if str(metadata[metadata_key]) != str(model_entry.get(manifest_key)):
                    raise ValueError(
                        f"declared {metadata_key} does not match manifest entry {tag!r}"
                    )
            shard = str(metadata["shard"])
            files = model_entry.get("files")
            if not isinstance(files, list):
                raise ValueError(f"manifest entry {tag!r} has no files array")
            shard_matches = [
                entry for entry in files
                if isinstance(entry, dict) and entry.get("file") == shard
            ]
            if len(shard_matches) != 1:
                raise ValueError(
                    f"manifest entry {tag!r} must contain exactly one file {shard!r}; "
                    f"found {len(shard_matches)}"
                )
            shard_entry = shard_matches[0]
            manifest_size = shard_entry.get("bytes")
            manifest_source_hash = shard_entry.get("sha256")
            if manifest_size != expected_size:
                raise ValueError(
                    f"expected source size does not match manifest entry {tag!r}/{shard!r}"
                )
            if not isinstance(manifest_source_hash, str) or (
                _normalize_sha256(manifest_source_hash, "manifest shard sha256")
                != normalized_source_hash
            ):
                raise ValueError(
                    f"expected source SHA-256 does not match manifest entry {tag!r}/{shard!r}"
                )
            manifest_binding = {
                "tag": tag,
                "repo": model_entry.get("repo"),
                "revision": model_entry.get("revision"),
                "shard": shard,
                "bytes": manifest_size,
                "sha256": normalized_source_hash,
                "entry_sha256": hashlib.sha256(json.dumps(
                    model_entry, sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
                "verified": True,
            }

    return {
        "model_metadata_declared": model_metadata_declared,
        "source": {
            "expected_size_bytes": expected_size,
            "actual_size_bytes": source_size,
            "expected_sha256": normalized_source_hash,
            "actual_sha256": source_sha256,
            "verified": expected_size is not None,
        },
        "manifest": manifest_record,
        "manifest_binding": manifest_binding,
    }


def _balanced_measured_orders(
    selected: Sequence[BaselineSpec], repetitions: int, seed: int,
) -> list[tuple[BaselineSpec, ...]]:
    """Generate deterministic forward/reverse permutation pairs."""

    generator = random.Random(seed)
    orders: list[tuple[BaselineSpec, ...]] = []
    for pair_start in range(0, repetitions, 2):
        forward = list(selected)
        generator.shuffle(forward)
        orders.append(tuple(forward))
        if pair_start + 1 < repetitions:
            orders.append(tuple(reversed(forward)))
    return orders


def benchmark_file(
    source: os.PathLike[str] | str,
    *,
    warmups: int = 1,
    repetitions: int = DEFAULT_REPETITIONS,
    specs: Sequence[BaselineSpec] | None = None,
    work_dir: os.PathLike[str] | str | None = None,
    keep_artifacts: bool = False,
    timeout_seconds: float | None = DEFAULT_OPERATION_TIMEOUT_SECONDS,
    version_probe_timeout_seconds: float | None = DEFAULT_VERSION_PROBE_TIMEOUT_SECONDS,
    schedule_seed: int = DEFAULT_SCHEDULE_SEED,
    input_metadata: Mapping[str, Any] | None = None,
    evidence_policy: Mapping[str, Any] | None = None,
    expected_source_size_bytes: int | None = None,
    expected_source_sha256: str | None = None,
    manifest_path: os.PathLike[str] | str | None = None,
    expected_manifest_sha256: str | None = None,
    checkpoint_path: os.PathLike[str] | str | None = None,
    force_checkpoint: bool = False,
) -> dict[str, Any]:
    """Benchmark ``source`` and return a versioned JSON-serializable document."""

    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    _validate_timeout("timeout_seconds", timeout_seconds)
    _validate_timeout("version_probe_timeout_seconds", version_probe_timeout_seconds)
    if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
        raise ValueError("schedule_seed must be an integer")
    selected = tuple(specs) if specs is not None else BASELINE_SPECS
    if not selected:
        raise ValueError("at least one baseline spec is required")
    identifiers = [spec.identifier for spec in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("selected baseline identifiers must be unique")

    source_path = pathlib.Path(source).resolve(strict=True)
    if not source_path.is_file():
        raise ValueError(f"source is not a regular file: {source_path}")
    checkpoint = pathlib.Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None:
        if _paths_conflict(source_path, checkpoint):
            raise ValueError("source and checkpoint/output paths must differ")
        if checkpoint.exists() and not force_checkpoint:
            raise FileExistsError(f"checkpoint/output already exists: {checkpoint}")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

    metadata = dict(input_metadata or {})
    policy = dict(evidence_policy or {})
    try:
        json.dumps(metadata)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"input_metadata must be JSON serializable: {exc}") from exc
    try:
        json.dumps(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evidence_policy must be JSON serializable: {exc}") from exc

    started_at_utc = _utc_now()
    source_size = source_path.stat().st_size
    # The source scan is provenance, not compression time.
    source_sha256 = _sha256_file(source_path)
    integrity = _verify_declared_integrity(
        source_path=source_path,
        source_size=source_size,
        source_sha256=source_sha256,
        metadata=metadata,
        expected_source_size_bytes=expected_source_size_bytes,
        expected_source_sha256=expected_source_sha256,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    parent = pathlib.Path(work_dir).resolve() if work_dir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    artifact_root = pathlib.Path(tempfile.mkdtemp(prefix="brevis-baselines-", dir=parent))
    method_results: list[dict[str, Any]] = []
    method_by_id: dict[str, dict[str, Any]] = {}
    probe_cache: dict[tuple[str, str, tuple[str, ...]], dict[str, Any]] = {}
    executable_cache: dict[str, tuple[str | None, str | None]] = {}
    execution_schedule: list[dict[str, Any]] = []
    execution_order = 0
    checkpoint_written = False
    measured_orders = _balanced_measured_orders(selected, repetitions, schedule_seed)

    environment = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "clock": "time.perf_counter_ns",
        "rss_source": (
            "os.wait4 direct-child ru_maxrss; descendant RSS is not aggregated into this field"
        ),
        "cpu": _cpu_info(),
        "memory": _memory_info(),
        "filesystems": {
            "source": _filesystem_info(source_path),
            "work": _filesystem_info(artifact_root),
            "checkpoint": (
                _filesystem_info(checkpoint.parent) if checkpoint is not None else None
            ),
        },
        "codec_environment_variables": {
            name: os.environ.get(name) for name in CODEC_ENVIRONMENT_VARIABLES
        },
    }
    provenance = {
        "git": _git_provenance(),
        "benchmark_script": {
            "path": str(pathlib.Path(__file__).resolve()),
            "sha256": _sha256_file(pathlib.Path(__file__).resolve()),
        },
    }

    def build_document(*, complete: bool) -> dict[str, Any]:
        return {
            "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
            "status": "complete" if complete else "in_progress",
            "started_at_utc": started_at_utc,
            "ended_at_utc": _utc_now(),
            "source": {
                "path": str(source_path),
                "size_bytes": source_size,
                "sha256": source_sha256,
            },
            "integrity": integrity,
            "input_metadata": metadata,
            "evidence_policy": policy or None,
            "configuration": {
                "warmups": warmups,
                "repetitions": repetitions,
                "timeout_seconds": timeout_seconds,
                "version_probe_timeout_seconds": version_probe_timeout_seconds,
                "termination_grace_seconds": PROCESS_TERMINATION_GRACE_SECONDS,
                "schedule_seed": schedule_seed,
                "measured_orders": [
                    {
                        "repetition": index,
                        "pair": index // 2,
                        "direction": "forward" if index % 2 == 0 else "reverse",
                        "methods": [spec.identifier for spec in order],
                    }
                    for index, order in enumerate(measured_orders)
                ],
                "schedule_balance": (
                    "paired" if repetitions % 2 == 0 else "unpaired_final_repetition"
                ),
                "keep_artifacts": keep_artifacts,
                "artifact_root": str(artifact_root) if keep_artifacts else None,
                "checkpoint_path": str(checkpoint) if checkpoint is not None else None,
                "cache_policy": (
                    "best-effort buffered I/O: source hashing and independent copy staging can "
                    "warm the page cache, but the harness neither flushes the cache nor guarantees "
                    "page residency; warmups precede measurements; every archive is hashed and "
                    "independently copied before decoding; raw/copy uses explicit 1 MiB reads "
                    "and writes without clone or sparse-seek APIs"
                ),
                "io_policy": (
                    "regular on-disk files; successful compression archives and successful "
                    "decompression outputs receive file fsync before their wall clocks stop; "
                    "source and staging files are not fsynced; no directory fsync or "
                    "drop_caches; xz, Zstandard, and LZ4 decoders use --no-sparse; "
                    "Zstandard compression and decoding use --no-asyncio"
                ),
                "timing_scope": (
                    "after stdout/stderr log open, immediately before process spawn, through "
                    "process reap and successful output file fsync"
                ),
                "verification_scope": (
                    "complete post-timing byte scan on every successful decompression"
                ),
                "scheduling_policy": (
                    "serial execution; measured repetitions use deterministic randomized "
                    "permutations in paired forward/reverse order"
                ),
                "short_process_timing_limitations": (
                    "Linux pidfd readiness avoids active polling when available, but wall times "
                    "still include process creation, scheduler latency, and direct-child reap; "
                    "fallback hosts use a recorded 1 ms wait4 polling interval"
                ),
                "execution_schedule": list(execution_schedule),
            },
            "provenance": provenance,
            "environment": environment,
            "methods": method_results,
        }

    def save_checkpoint(*, complete: bool) -> None:
        nonlocal checkpoint_written
        if checkpoint is None:
            return
        write_json(
            checkpoint,
            build_document(complete=complete),
            force=force_checkpoint or checkpoint_written,
        )
        checkpoint_written = True

    try:
        for spec in selected:
            resolved = shutil.which(spec.executable)
            method: dict[str, Any] = {
                "id": spec.identifier,
                "spec": spec.to_dict(),
                "available": resolved is not None,
                "resolved_executable": resolved,
                "executable_realpath": None,
                "executable_sha256": None,
                "executable_sha256_after_runs": None,
                "executable_unchanged_during_benchmark": None,
                "version": None,
                "version_probe": None,
                "warmups": [],
                "runs": [],
                "failure": None,
                "status_code": "pending",
                "measured_archive_consistency": {
                    "status_code": "not_checked",
                    "reference_size_bytes": None,
                    "reference_sha256": None,
                    "consistent_runs": 0,
                    "inconsistent_runs": 0,
                },
            }
            method_results.append(method)
            method_by_id[spec.identifier] = method
            if resolved is None:
                method["failure"] = f"executable not found on PATH: {spec.executable}"
                method["status_code"] = "executable_missing"
                continue

            if resolved not in executable_cache:
                try:
                    realpath = pathlib.Path(resolved).resolve(strict=True)
                    executable_cache[resolved] = (str(realpath), _sha256_file(realpath))
                except OSError:
                    executable_cache[resolved] = (None, None)
            realpath, executable_sha256 = executable_cache[resolved]
            method["executable_realpath"] = realpath
            method["executable_sha256"] = executable_sha256
            if realpath is None or executable_sha256 is None:
                method["failure"] = f"cannot resolve or hash executable: {resolved}"
                method["status_code"] = "executable_provenance_failed"
                continue

            probe_key = (spec.method, resolved, spec.version_args)
            if probe_key not in probe_cache:
                probe_cache[probe_key] = _probe_tool(
                    spec, resolved, artifact_root, version_probe_timeout_seconds,
                )
            tool = probe_cache[probe_key]
            method["version"] = tool["version"]
            method["version_probe"] = tool["probe"]
            if tool["failure"] is not None:
                method["failure"] = tool["failure"]
                method["status_code"] = "version_probe_failed"
            else:
                method["status_code"] = "ready"

        warmup_orders = [
            (*selected[index % len(selected):], *selected[:index % len(selected)])
            for index in range(warmups)
        ]
        for phase, phase_orders in (("warmup", warmup_orders), ("measured", measured_orders)):
            for index, ordered_specs in enumerate(phase_orders):
                for order_within_repetition, spec in enumerate(ordered_specs):
                    method = method_by_id[spec.identifier]
                    if not method["available"] or method["failure"] is not None:
                        continue
                    destination = method["warmups"] if phase == "warmup" else method["runs"]
                    resolved = method["resolved_executable"]
                    assert resolved is not None
                    run_dir = pathlib.Path(tempfile.mkdtemp(
                        prefix=f"{spec.method}-{spec.profile}-{phase}-{index}-",
                        dir=artifact_root,
                    ))
                    execution_schedule.append({
                        "execution_order": execution_order,
                        "phase": phase,
                        "repetition": index,
                        "order_within_repetition": order_within_repetition,
                        "method": spec.identifier,
                    })
                    iteration_started_at_utc = _utc_now()
                    try:
                        record = _run_iteration(
                            spec, resolved, source_path, source_sha256, phase, index, run_dir,
                            timeout_seconds, execution_order, order_within_repetition,
                        )
                        _apply_archive_consistency(method, record, phase)
                        record["ended_at_utc"] = _utc_now()
                        destination.append(record)
                    except Exception as exc:  # Preserve a harness failure without losing other methods.
                        destination.append({
                            "status_code": "harness_error",
                            "started_at_utc": iteration_started_at_utc,
                            "ended_at_utc": _utc_now(),
                            "phase": phase,
                            "index": index,
                            "execution_order": execution_order,
                            "order_within_repetition": order_within_repetition,
                            "compression": None,
                            "compressed_size_bytes": None,
                            "archive_storage": None,
                            "archive_sha256": None,
                            "archive_hash_error": None,
                            "archive_consistency": {
                                "status_code": "not_checked_harness_error",
                                "matches_reference": None,
                            },
                            "decompression": None,
                            "verification": None,
                            "bit_exact": False,
                            "success": False,
                            "failure": f"benchmark harness error: {type(exc).__name__}: {exc}",
                            "staging": None,
                            "artifact_directory": str(run_dir),
                        })
                    finally:
                        if not keep_artifacts:
                            shutil.rmtree(run_dir, ignore_errors=True)
                    execution_order += 1
                    save_checkpoint(complete=False)

        for method in method_results:
            realpath = method["executable_realpath"]
            if realpath is not None and method["executable_sha256"] is not None:
                try:
                    after_hash = _sha256_file(pathlib.Path(realpath))
                except OSError as exc:
                    method["status_code"] = "executable_provenance_failed"
                    method["failure"] = (
                        "cannot hash executable after benchmark: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    method["executable_sha256_after_runs"] = after_hash
                    unchanged = after_hash == method["executable_sha256"]
                    method["executable_unchanged_during_benchmark"] = unchanged
                    if not unchanged:
                        method["status_code"] = "executable_changed"
                        method["failure"] = (
                            "executable SHA-256 changed during benchmark"
                        )
            failures = [
                run["failure"]
                for run in (*method["warmups"], *method["runs"])
                if not run["success"]
            ]
            if failures:
                method["failure"] = f"{len(failures)} iteration(s) failed; see per-iteration records"
                if method["status_code"] == "ready":
                    method["status_code"] = "iteration_failed"
            elif method["status_code"] == "ready":
                method["status_code"] = "ok"

        document = build_document(complete=True)
        save_checkpoint(complete=True)
    finally:
        if not keep_artifacts:
            shutil.rmtree(artifact_root, ignore_errors=True)

    return document


def write_json(
    path: os.PathLike[str] | str,
    document: dict[str, Any],
    *,
    force: bool = False,
) -> None:
    """Publish JSON atomically using an exclusive, same-directory temp file.

    Without ``force``, a same-directory reservation prevents cooperating writers
    from racing past the no-clobber check. ``os.replace`` then swaps the complete
    temporary file into place atomically. Forced writers explicitly accept
    last-writer-wins behavior, while their unique temporary files remain safe.
    """

    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reservation = destination.with_name(f".{destination.name}.lock")
    reservation_owned = False
    temporary: pathlib.Path | None = None
    try:
        if not force:
            try:
                reservation_fd = os.open(
                    reservation, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600,
                )
            except FileExistsError as exc:
                raise FileExistsError(
                    f"output is being written concurrently: {destination}"
                ) from exc
            else:
                os.close(reservation_fd)
                reservation_owned = True
            if destination.exists():
                raise FileExistsError(f"output already exists: {destination}")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
        )
        temporary = pathlib.Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(document, output, indent=2, sort_keys=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        if reservation_owned:
            try:
                reservation.unlink()
            except FileNotFoundError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, help="versioned JSON result path")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_OPERATION_TIMEOUT_SECONDS,
        help="timeout applied independently to compression and decompression",
    )
    parser.add_argument(
        "--version-probe-timeout-seconds",
        type=float,
        default=DEFAULT_VERSION_PROBE_TIMEOUT_SECONDS,
        help="short timeout applied independently to executable version probes",
    )
    parser.add_argument("--schedule-seed", type=int, default=DEFAULT_SCHEDULE_SEED)
    parser.add_argument("--force", action="store_true", help="atomically replace an existing output")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="do not fail the CLI solely because a selected executable is unavailable",
    )
    parser.add_argument("--model-tag")
    parser.add_argument("--model-repo")
    parser.add_argument("--model-revision")
    parser.add_argument("--manifest")
    parser.add_argument("--shard")
    parser.add_argument(
        "--engineering-evidence",
        action="store_true",
        help="mark the standalone result as engineering-only evidence",
    )
    parser.add_argument("--expected-source-size", type=int)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        metavar="METHOD/PROFILE",
        help="run one registry entry; repeat this option to select several",
    )
    parser.add_argument("--list", action="store_true", help="print the registry as JSON and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps([spec.to_dict() for spec in BASELINE_SPECS], indent=2))
        return 0
    if args.source is None or args.output is None:
        parser.error("source and --output are required unless --list is used")
    try:
        source_path = args.source.resolve(strict=True)
        if _paths_conflict(source_path, args.output):
            raise ValueError("source and --output must refer to different files")
        if args.output.exists() and not args.force:
            raise FileExistsError(
                f"output already exists (pass --force to replace it): {args.output}"
            )
        specs = select_specs(args.methods)
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
        document = benchmark_file(
            source_path,
            warmups=args.warmups,
            repetitions=args.repetitions,
            specs=specs,
            work_dir=args.work_dir,
            keep_artifacts=args.keep_artifacts,
            timeout_seconds=args.timeout_seconds,
            version_probe_timeout_seconds=args.version_probe_timeout_seconds,
            schedule_seed=args.schedule_seed,
            input_metadata=metadata,
            evidence_policy=(
                {
                    "run_class": "engineering",
                    "paper_eligible": False,
                    "reason": "standalone engineering run outside a frozen formal campaign",
                }
                if args.engineering_evidence
                else None
            ),
            expected_source_size_bytes=args.expected_source_size,
            expected_source_sha256=args.expected_source_sha256,
            manifest_path=args.manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            checkpoint_path=args.output,
            force_checkpoint=args.force,
        )
    except (OSError, ValueError) as exc:
        print(f"benchmarking: {exc}", file=sys.stderr)
        return 2

    missing = any(not method["available"] for method in document["methods"])
    failed_available = any(
        method["available"] and method["failure"] is not None
        for method in document["methods"]
    )
    return 1 if failed_available or (missing and not args.allow_missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
