#!/usr/bin/env python3
"""Read-only preflight for the frozen DFloat11 and ECF8 protocols."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_benchmarks as harness
from benchmark_corpus import CHECKPOINT_BY_NAME


ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_MODEL_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json")
DFLOAT11_ARCHITECTURES = {"LlamaForCausalLM", "Qwen3ForCausalLM"}
ECF8_ARCHITECTURES = {"Qwen3ForCausalLM"}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    method: str | None = None
    checkpoint: str | None = None

    def render(self) -> str:
        scope = "/".join(
            item for item in (self.method, self.checkpoint) if item
        )
        label = f"[{scope}] " if scope else ""
        return f"{self.severity:7} {self.code}: {label}{self.message}"


def finding(
    findings: list[Finding],
    severity: str,
    code: str,
    message: str,
    *,
    method: str | None = None,
    checkpoint: str | None = None,
) -> None:
    findings.append(Finding(severity, code, message, method, checkpoint))


def read_json(
    path: Path,
    findings: list[Finding],
    *,
    method: str | None = None,
    checkpoint: str | None = None,
) -> Any | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        finding(
            findings,
            "ERROR",
            "missing_file",
            f"missing {path}",
            method=method,
            checkpoint=checkpoint,
        )
    except (OSError, json.JSONDecodeError) as exc:
        finding(
            findings,
            "ERROR",
            "invalid_json",
            f"cannot read {path}: {exc}",
            method=method,
            checkpoint=checkpoint,
        )
    return None


def discover_manifests(
    models_root: Path,
    findings: list[Finding],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    if not models_root.is_dir():
        finding(
            findings,
            "ERROR",
            "missing_models_root",
            f"models root does not exist: {models_root}",
        )
        return {}
    paths = sorted(models_root.glob("*/download-manifest.json"))
    if (models_root / "download-manifest.json").is_file():
        paths.insert(0, models_root / "download-manifest.json")
    manifests: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        manifest = read_json(path, findings)
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("name"), str
        ):
            finding(
                findings,
                "ERROR",
                "invalid_manifest",
                f"{path} must contain a string name",
            )
            continue
        name = manifest["name"]
        if name in manifests:
            finding(
                findings,
                "ERROR",
                "duplicate_checkpoint",
                f"multiple manifests declare {name}",
                checkpoint=name,
            )
            continue
        manifests[name] = (path.parent, manifest)
    return manifests


def resolve_executable(executable: str, cwd: Path | None) -> Path | None:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    if "/" in executable:
        candidate = (cwd or Path.cwd()) / candidate
        return candidate if candidate.is_file() else None
    located = shutil.which(executable)
    return Path(located) if located else None


def check_runtime(
    method: str,
    settings: dict[str, Any],
    findings: list[Finding],
) -> None:
    cwd = (
        Path(settings["cwd"]).expanduser()
        if settings.get("cwd")
        else None
    )
    if cwd is not None and not cwd.is_dir():
        finding(
            findings,
            "ERROR",
            "missing_cwd",
            f"configured cwd does not exist: {cwd}",
            method=method,
        )
    checkout_value = settings.get("upstream_checkout")
    checkout = (
        Path(checkout_value).expanduser()
        if isinstance(checkout_value, str) and checkout_value
        else None
    )
    if checkout is None or not checkout.is_dir():
        finding(
            findings,
            "ERROR",
            "missing_checkout",
            f"official checkout does not exist: {checkout_value!r}",
            method=method,
        )
    for command_name in ("version_command", "compress_command"):
        command = settings[command_name]
        executable = resolve_executable(command[0], cwd)
        if executable is None:
            finding(
                findings,
                "ERROR",
                "missing_executable",
                f"{command_name} executable is unavailable: {command[0]}",
                method=method,
            )
        if (
            command_name == "compress_command"
            and len(command) > 1
            and command[1].endswith(".py")
        ):
            script = Path(command[1])
            if not script.is_absolute():
                script = (cwd or Path.cwd()) / script
            if not script.is_file():
                finding(
                    findings,
                    "ERROR",
                    "missing_adapter",
                    f"adapter script does not exist: {script}",
                    method=method,
                )

    try:
        completed = subprocess.run(
            settings["version_command"],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
        )
        actual_commit = completed.stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        finding(
            findings,
            "ERROR",
            "version_command_failed",
            f"cannot identify official checkout: {exc}",
            method=method,
        )
    else:
        if actual_commit != settings["expected_commit"]:
            finding(
                findings,
                "ERROR",
                "commit_mismatch",
                f"expected {settings['expected_commit']}, got {actual_commit}",
                method=method,
            )

    modules = settings.get("runtime_modules", ())
    environment = os.environ.copy()
    if checkout is not None:
        environment["PYTHONPATH"] = str(checkout) + (
            os.pathsep + environment["PYTHONPATH"]
            if environment.get("PYTHONPATH")
            else ""
        )
    if modules:
        python = resolve_executable(settings["compress_command"][0], cwd)
        if python is not None:
            probe = (
                "import importlib.util,json,sys;"
                "names=json.loads(sys.argv[1]);"
                "print(json.dumps([n for n in names "
                "if importlib.util.find_spec(n) is None]))"
            )
            try:
                completed = subprocess.run(
                    [str(python), "-c", probe, json.dumps(modules)],
                    cwd=cwd,
                    env=environment,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=20,
                )
                missing = json.loads(completed.stdout.strip())
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
                json.JSONDecodeError,
            ) as exc:
                finding(
                    findings,
                    "ERROR",
                    "runtime_probe_failed",
                    f"cannot inspect isolated Python environment: {exc}",
                    method=method,
                )
            else:
                if missing:
                    finding(
                        findings,
                        "ERROR",
                        "missing_python_modules",
                        f"missing from isolated environment: {missing}",
                        method=method,
                    )
    imports = settings.get("runtime_import_modules", ())
    if imports:
        python = resolve_executable(settings["compress_command"][0], cwd)
        if python is not None:
            probe = (
                "import importlib,json,sys;"
                "[importlib.import_module(n) "
                "for n in json.loads(sys.argv[1])];"
                "print('ok')"
            )
            try:
                subprocess.run(
                    [str(python), "-c", probe, json.dumps(imports)],
                    cwd=cwd,
                    env=environment,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=30,
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as exc:
                output = (
                    exc.stdout.strip()
                    if isinstance(exc, subprocess.CalledProcessError)
                    and isinstance(exc.stdout, str)
                    else str(exc)
                )
                finding(
                    findings,
                    "ERROR",
                    "runtime_import_failed",
                    f"cannot import configured runtime module(s): {output}",
                    method=method,
                )


def check_manifest_files(
    method: str,
    checkpoint: str,
    directory: Path,
    manifest: dict[str, Any],
    findings: list[Finding],
) -> tuple[list[Path], set[str]]:
    weights = manifest.get("weights")
    if not isinstance(weights, list) or not weights:
        finding(
            findings,
            "ERROR",
            "invalid_manifest_weights",
            "manifest weights must be a non-empty list",
            method=method,
            checkpoint=checkpoint,
        )
        return [], set()
    paths: list[Path] = []
    total_size = 0
    dtypes: set[str] = set()
    for entry in weights:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            finding(
                findings,
                "ERROR",
                "invalid_manifest_weight",
                f"invalid weight entry: {entry!r}",
                method=method,
                checkpoint=checkpoint,
            )
            continue
        path = directory / entry["path"]
        paths.append(path)
        if not path.is_file():
            finding(
                findings,
                "ERROR",
                "missing_weight",
                f"missing declared weight: {path}",
                method=method,
                checkpoint=checkpoint,
            )
            continue
        actual_size = path.stat().st_size
        total_size += actual_size
        expected_size = entry.get("size")
        if expected_size is not None and actual_size != expected_size:
            finding(
                findings,
                "ERROR",
                "weight_size_mismatch",
                f"{path.name}: expected {expected_size}, got {actual_size}",
                method=method,
                checkpoint=checkpoint,
            )
        try:
            header, _ = harness.safetensors_header(path)
        except harness.BenchmarkError as exc:
            finding(
                findings,
                "ERROR",
                "invalid_safetensors",
                str(exc),
                method=method,
                checkpoint=checkpoint,
            )
            continue
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            dtype = tensor.get("dtype") if isinstance(tensor, dict) else None
            if not isinstance(dtype, str):
                finding(
                    findings,
                    "ERROR",
                    "invalid_tensor_header",
                    f"{path.name}:{name} has no dtype",
                    method=method,
                    checkpoint=checkpoint,
                )
            else:
                dtypes.add(dtype)
    expected_total = manifest.get("source_bytes")
    if expected_total is not None and total_size != expected_total:
        finding(
            findings,
            "ERROR",
            "source_size_mismatch",
            f"manifest source_bytes={expected_total}, files total {total_size}",
            method=method,
            checkpoint=checkpoint,
        )
    index_name = manifest.get("index_file")
    if not isinstance(index_name, str) or not (directory / index_name).is_file():
        finding(
            findings,
            "ERROR",
            "missing_weight_index",
            "manifest index_file is missing",
            method=method,
            checkpoint=checkpoint,
        )
    if manifest.get("sha256_verified") is not True:
        finding(
            findings,
            "WARNING",
            "hashes_not_verified",
            "manifest does not attest that weight SHA-256 values were verified; "
            "this preflight intentionally does not stream large payloads",
            method=method,
            checkpoint=checkpoint,
        )
    return paths, dtypes


def check_model(
    method: str,
    checkpoint: str,
    directory: Path,
    manifest: dict[str, Any],
    findings: list[Finding],
) -> None:
    frozen = CHECKPOINT_BY_NAME.get(checkpoint)
    if frozen is None:
        finding(
            findings,
            "ERROR",
            "checkpoint_not_frozen",
            "checkpoint is absent from the frozen benchmark corpus",
            method=method,
            checkpoint=checkpoint,
        )
    elif (
        manifest.get("repo_id") != frozen.repo_id
        or manifest.get("revision") != frozen.revision
    ):
        finding(
            findings,
            "ERROR",
            "checkpoint_revision_mismatch",
            "manifest repo_id/revision differs from the frozen corpus",
            method=method,
            checkpoint=checkpoint,
        )

    _, dtypes = check_manifest_files(
        method,
        checkpoint,
        directory,
        manifest,
        findings,
    )
    config = read_json(
        directory / "config.json",
        findings,
        method=method,
        checkpoint=checkpoint,
    )
    if config is None:
        return
    if not isinstance(config, dict):
        finding(
            findings,
            "ERROR",
            "invalid_model_config",
            "config.json must contain a JSON object",
            method=method,
            checkpoint=checkpoint,
        )
        return
    architecture_values = config.get("architectures")
    architectures = (
        set(architecture_values)
        if isinstance(architecture_values, list)
        and all(isinstance(item, str) for item in architecture_values)
        else set()
    )
    if method == "dfloat11":
        if not architectures.intersection(DFLOAT11_ARCHITECTURES):
            finding(
                findings,
                "ERROR",
                "unsupported_architecture",
                f"DFloat11 wrapper does not support {sorted(architectures)}",
                method=method,
                checkpoint=checkpoint,
            )
        if dtypes != {"BF16"}:
            finding(
                findings,
                "ERROR",
                "unsupported_source_dtype",
                f"DFloat11 frozen protocol requires BF16-only tensors, got "
                f"{sorted(dtypes)}",
                method=method,
                checkpoint=checkpoint,
            )
    else:
        if not architectures.intersection(ECF8_ARCHITECTURES):
            finding(
                findings,
                "ERROR",
                "unsupported_architecture",
                f"ECF8 wrapper does not support {sorted(architectures)}",
                method=method,
                checkpoint=checkpoint,
            )
        if "F8_E4M3" not in dtypes or not dtypes.issubset(
            {"BF16", "F8_E4M3"}
        ):
            finding(
                findings,
                "ERROR",
                "unsupported_source_dtype",
                "ECF8 frozen protocol requires Qwen3 E4M3 FP8/BF16 tensors, "
                f"got {sorted(dtypes)}",
                method=method,
                checkpoint=checkpoint,
            )
        quantization = config.get("quantization_config")
        if not isinstance(quantization, dict) or (
            str(quantization.get("quant_method", "")).lower() != "fp8"
            or "e4m3" not in str(quantization.get("fmt", "")).lower()
        ):
            finding(
                findings,
                "ERROR",
                "invalid_fp8_config",
                "config.json must declare quant_method=fp8 and an E4M3 fmt",
                method=method,
                checkpoint=checkpoint,
            )
        if not (directory / "tokenizer_config.json").is_file() or not any(
            (directory / name).is_file() for name in TOKENIZER_MODEL_FILES
        ):
            finding(
                findings,
                "ERROR",
                "missing_tokenizer",
                "ECF8 official converter requires tokenizer_config.json and "
                "tokenizer model assets",
                method=method,
                checkpoint=checkpoint,
            )


def audit(
    config: dict[str, Any],
    models_root: Path,
    methods: list[str],
    *,
    skip_runtime: bool,
    publication_ready: bool,
) -> list[Finding]:
    findings: list[Finding] = []
    manifests = discover_manifests(models_root, findings)
    for method in methods:
        settings = config.get(method)
        if settings is None:
            finding(
                findings,
                "ERROR",
                "missing_method_config",
                "method is not configured",
                method=method,
            )
            continue
        if not skip_runtime:
            check_runtime(method, settings, findings)
        protocol = settings["protocol"]
        if protocol["independent_bitwise_validation_performed"] is not True:
            finding(
                findings,
                "ERROR" if publication_ready else "WARNING",
                "independent_bitwise_validation_missing",
                "official elementwise value-equality validation is configured, "
                "but an independent tensor payload bit comparison has not been "
                "performed",
                method=method,
            )
        if settings.get("upstream_license") is None:
            finding(
                findings,
                "WARNING",
                "upstream_license_missing",
                "the pinned upstream checkout has no declared license in this "
                "protocol; resolve redistribution/publication implications",
                method=method,
            )
        pattern = re.compile(settings["checkpoint_pattern"])
        supported = set(protocol["supported_checkpoints"])
        unexpected = sorted(
            name
            for name in manifests
            if pattern.search(name) and name not in supported
        )
        if unexpected:
            finding(
                findings,
                "ERROR",
                "pattern_overmatch",
                f"checkpoint pattern also matches undeclared models: {unexpected}",
                method=method,
            )
        for checkpoint in protocol["supported_checkpoints"]:
            discovered = manifests.get(checkpoint)
            if discovered is None:
                finding(
                    findings,
                    "ERROR",
                    "missing_checkpoint",
                    f"required checkpoint is absent under {models_root}",
                    method=method,
                    checkpoint=checkpoint,
                )
                continue
            check_model(
                method,
                checkpoint,
                discovered[0],
                discovered[1],
                findings,
            )
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate specialized baseline config, fixed commits, lightweight "
            "model metadata, and publication protocol without loading weights."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "specialized-baselines.example.json",
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=harness.SPECIALIZED_METHODS,
        default=list(harness.SPECIALIZED_METHODS),
    )
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Skip checkout, executable, commit, and isolated-module checks.",
    )
    parser.add_argument(
        "--publication-ready",
        action="store_true",
        help=(
            "Treat missing independent tensor-bit comparison as an error. "
            "The current official validators alone do not satisfy this mode."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    findings: list[Finding] = []
    try:
        config = harness.load_specialized_config(args.config)
    except (OSError, json.JSONDecodeError, harness.BenchmarkError) as exc:
        finding(
            findings,
            "ERROR",
            "invalid_specialized_config",
            str(exc),
        )
    else:
        findings.extend(
            audit(
                config,
                args.models_root.expanduser().resolve(),
                args.methods,
                skip_runtime=args.skip_runtime,
                publication_ready=args.publication_ready,
            )
        )
    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2))
    else:
        for item in findings:
            print(item.render())
        errors = sum(item.severity == "ERROR" for item in findings)
        warnings = sum(item.severity == "WARNING" for item in findings)
        print(f"preflight: {errors} error(s), {warnings} warning(s)")
    return int(any(item.severity == "ERROR" for item in findings))


if __name__ == "__main__":
    raise SystemExit(main())
