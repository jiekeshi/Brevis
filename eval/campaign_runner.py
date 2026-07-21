#!/usr/bin/env python3
"""Expand the frozen experiment matrix and execute at most one audited task.

Planning is the default.  The planner never treats an unsupported campaign as
completed: it emits an explicit ``unsupported_kind`` task.  Execution requires
the semantic SHA-256 of exactly one supported task and dispatches to the audited
Brevis-system or exact-registry generic-codec harness.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import benchmarking as common
import brevis_benchmarking as system


ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_EXPERIMENTS = ROOT / "eval" / "experiments-v1.json"
DEFAULT_CACHE_ROOT = ROOT / "eval" / "cache"
DEFAULT_STAGING_ROOT = DEFAULT_CACHE_ROOT / "formal-staging"
DEFAULT_EXECUTION_LOCK = pathlib.Path("/tmp/brevis-machine-benchmark.lock")
PLAN_SCHEMA_ID = "brevis.campaign-plan"
PLAN_SCHEMA_VERSION = 1
TASK_SEMANTIC_SPEC_ID = "brevis.campaign-task-semantics.v1"
EXCLUSIVE_EXECUTION_POLICY_ID = "brevis.machine-advisory-execution-lock.v1"
SUPPORTED_BREVIS_KINDS = {
    "brevis_system",
    "brevis_search_sweep",
    "brevis_search_grid",
    "brevis_reranking_ablation",
    "brevis_operator_ablation",
    "brevis_terminal_ablation",
    "brevis_calibration_sweep",
    "brevis_parallel_scaling",
}
SUPPORTED_GENERIC_KINDS = {"generic_codecs"}
UNSUPPORTED_KINDS = {
    "specialized_baselines",
    "specialized_codecs",
}
BREVIS_FORMAL_GATES = {
    "require_declared_integrity": True,
    "require_clean_git": True,
    "build_binary_releasefast": True,
    "force_checkpoint": False,
}
GENERIC_FORMAL_GATES = {
    "require_declared_integrity": True,
    "require_clean_git": True,
    "require_codec_environment_unset": True,
    "require_exact_registry": True,
    "require_bit_exact_all_iterations": True,
    "require_stable_measured_archives": True,
    "require_unchanged_executables": True,
    "force_checkpoint": False,
}
GENERIC_REGISTRY_ID = "all_19_pinned_configurations"
GENERIC_METHOD_COUNT = 19
GENERIC_WARMUPS = 1
GENERIC_REPETITIONS = 6
GENERIC_SCHEDULE_SEED = 2701
GENERIC_DISK_RESERVE_FRACTION = 0.30
GENERIC_DISK_SOURCE_MULTIPLIER = 5
GENERIC_DISK_FIXED_BYTES = 512 << 20
GENERIC_CHECKPOINT_BYTES_PER_ITERATION = 128 << 10
SEARCH_OPTION_ORDER = (
    ("max_expansions", "--max-expansions"),
    ("max_nodes", "--max-nodes"),
    ("max_depth", "--max-depth"),
    ("sample_elems", "--sample-elems"),
    ("rerank_candidates", "--rerank-candidates"),
    ("rerank_blocks", "--rerank-blocks"),
)
SAFE_ID = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,126}[a-z0-9])?")
SHA256 = re.compile(r"[0-9a-f]{64}")


class CampaignError(RuntimeError):
    """The frozen campaign matrix is malformed or cannot be executed safely."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CampaignError(f"value is not canonical-JSON serializable: {exc}") from exc


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _implementation_binding() -> dict[str, Any]:
    git = common._git_provenance()
    commit = git.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise CampaignError("cannot bind campaign tasks to a canonical Git commit")
    runner_path = pathlib.Path(__file__).resolve()
    harness_path = pathlib.Path(system.__file__).resolve()
    return {
        "git_commit": commit,
        "campaign_runner_sha256": common._sha256_file(runner_path),
        "brevis_benchmarking_sha256": common._sha256_file(harness_path),
        "generic_benchmarking_sha256": common._sha256_file(
            pathlib.Path(common.__file__).resolve()
        ),
    }


def _machine_binding() -> dict[str, Any]:
    memory = common._memory_info()
    payload = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "cpu": common._cpu_info(),
        "memory": {
            "total_bytes": memory.get("total_bytes"),
            "cgroup_limit_bytes": memory.get("cgroup_limit_bytes"),
            "cgroup_limit_source": memory.get("cgroup_limit_source"),
        },
        "gpu": system._gpu_info(),
    }
    return {
        "spec_id": "brevis.execution-machine-fingerprint.v1",
        "sha256": _semantic_sha256(payload),
        "payload": payload,
    }


@contextlib.contextmanager
def _exclusive_execution_lock(path: pathlib.Path | None = None):
    """Acquire the runner's nonblocking machine-local advisory lock."""

    path = DEFAULT_EXECUTION_LOCK if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CampaignError(f"cannot safely open machine advisory lock {path}: {exc}") from exc
    acquired = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise CampaignError(f"machine advisory lock is not a regular file: {path}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            raise CampaignError(
                f"another campaign execution holds the machine advisory lock {path}"
            ) from exc
        os.fchmod(descriptor, 0o600)
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            f"pid={os.getpid()} task-runner={pathlib.Path(__file__).resolve()}\n".encode(),
        )
        yield path
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{label} must be an object")
    return value


def _array(value: Any, label: str, *, nonempty: bool = True) -> list[Any]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a nonempty" if nonempty else "an"
        raise CampaignError(f"{label} must be {qualifier} array")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CampaignError(f"{label} must be an integer >= {minimum}")
    return value


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or SAFE_ID.fullmatch(value) is None:
        raise CampaignError(f"{label} is not a filename-safe campaign identifier")
    return value


def _unique_strings(value: Any, label: str) -> tuple[str, ...]:
    values = _array(value, label)
    if not all(isinstance(item, str) and item for item in values):
        raise CampaignError(f"{label} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise CampaignError(f"{label} must not contain duplicates")
    return tuple(values)


def _read_json_object(path: pathlib.Path, label: str) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read {label} {path}: {exc}") from exc
    return _object(value, label), hashlib.sha256(raw).hexdigest()


def _resolve_declared_path(root: pathlib.Path, value: Any, label: str) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise CampaignError(f"{label} must be a nonempty path string")
    declared = pathlib.PurePosixPath(value)
    if declared.is_absolute() or ".." in declared.parts:
        raise CampaignError(f"{label} must be a repository-relative path")
    return (root / pathlib.Path(*declared.parts)).resolve(strict=False)


def _load_inputs(
    experiments_path: pathlib.Path, repository_root: pathlib.Path,
) -> tuple[Mapping[str, Any], str, pathlib.Path, list[Mapping[str, Any]], str]:
    matrix, matrix_sha256 = _read_json_object(experiments_path, "experiment matrix")
    schema = _object(matrix.get("schema"), "experiment matrix schema")
    if schema.get("id") != "brevis.experiment-campaigns" or schema.get("version") != 1:
        raise CampaignError("only brevis.experiment-campaigns schema version 1 is supported")
    if matrix.get("status") != "preregistered_configuration_not_results":
        raise CampaignError("experiment matrix is not labelled as preregistered configuration")
    manifest_binding = _object(matrix.get("model_manifest"), "model_manifest")
    manifest_path = _resolve_declared_path(
        repository_root, manifest_binding.get("path"), "model_manifest.path",
    )
    expected_sha256 = manifest_binding.get("sha256")
    if not isinstance(expected_sha256, str) or SHA256.fullmatch(expected_sha256) is None:
        raise CampaignError("model_manifest.sha256 is not a canonical SHA-256")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest_value = json.loads(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read model manifest {manifest_path}: {exc}") from exc
    actual_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise CampaignError(
            "model manifest SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    if not isinstance(manifest_value, list) or not manifest_value:
        raise CampaignError("model manifest must be a nonempty JSON array")
    models: list[Mapping[str, Any]] = []
    for index, value in enumerate(manifest_value):
        models.append(_object(value, f"model manifest entry {index}"))
    return matrix, matrix_sha256, manifest_path, models, actual_sha256


def _validate_models(models: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    by_tag: dict[str, Mapping[str, Any]] = {}
    for index, model in enumerate(models):
        tag = _safe_id(model.get("tag"), f"model[{index}].tag")
        if tag in by_tag:
            raise CampaignError(f"duplicate model tag {tag!r}")
        for field in ("repo", "revision"):
            if not isinstance(model.get(field), str) or not model[field]:
                raise CampaignError(f"model {tag!r} has no nonempty {field}")
        files = _array(model.get("files"), f"model {tag!r}.files")
        seen_files: set[str] = set()
        selected_bytes = 0
        for file_index, file_value in enumerate(files):
            file_record = _object(file_value, f"model {tag!r} file {file_index}")
            filename = file_record.get("file")
            if not isinstance(filename, str) or not filename:
                raise CampaignError(f"model {tag!r} contains an empty filename")
            relative = pathlib.PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts or filename in seen_files:
                raise CampaignError(f"model {tag!r} has an unsafe or duplicate file path")
            seen_files.add(filename)
            selected_bytes += _integer(
                file_record.get("bytes"), f"model {tag!r}/{filename} bytes",
            )
            digest = file_record.get("sha256")
            if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
                raise CampaignError(f"model {tag!r}/{filename} has an invalid SHA-256")
        if model.get("selected_bytes") != selected_bytes:
            raise CampaignError(f"model {tag!r} selected_bytes does not match its files")
        by_tag[tag] = model
    return by_tag


def _validate_inventory(matrix: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    inventory = _object(matrix.get("operator_inventory"), "operator_inventory")
    if inventory.get("mandatory_raw_fallback") != "raw":
        raise CampaignError("operator inventory must declare raw as mandatory fallback")
    terminals = _unique_strings(inventory.get("terminal_codecs"), "terminal_codecs")
    transforms = _unique_strings(
        inventory.get("reversible_operators"), "reversible_operators",
    )
    if "raw" in terminals or "raw" in transforms or set(terminals) & set(transforms):
        raise CampaignError("raw cannot be disabled and operator inventories must be disjoint")
    return terminals, transforms


def _search_defaults(matrix: Mapping[str, Any]) -> dict[str, int]:
    common = _object(matrix.get("common"), "common")
    defaults = _object(common.get("search_defaults"), "common.search_defaults")
    result: dict[str, int] = {}
    for key, _ in SEARCH_OPTION_ORDER:
        minimum = 1 if key in {"max_expansions", "max_nodes", "sample_elems"} else 0
        result[key] = _integer(defaults.get(key), f"search_defaults.{key}", minimum=minimum)
    if defaults.get("max_realizations_cli_exposed") is not False:
        raise CampaignError("planner assumes max_realizations is not CLI exposed")
    _integer(defaults.get("max_realizations"), "search_defaults.max_realizations", minimum=1)
    return result


def _settings_args(settings: Mapping[str, int], disabled: Iterable[str] = ()) -> tuple[str, ...]:
    arguments: list[str] = []
    for key, option in SEARCH_OPTION_ORDER:
        arguments.extend((option, str(settings[key])))
    for operator in disabled:
        if operator == "raw":
            raise CampaignError("raw is mandatory and cannot be disabled")
        arguments.extend(("--disable-op", operator))
    return tuple(arguments)


def _spec(
    identifier: str,
    guidance: str,
    settings: Mapping[str, int],
    all_operators: Sequence[str],
    *,
    disabled: Iterable[str] = (),
    notes: str,
    require_raw_only: bool = False,
) -> system.BrevisSpec:
    disabled_tuple = tuple(disabled)
    disabled_set = set(disabled_tuple)
    if "raw" in disabled_set:
        raise CampaignError("raw is mandatory and cannot be disabled")
    enabled_ops = tuple(operator for operator in all_operators if operator not in disabled_set)
    expected = (
        ("max_expansions", settings["max_expansions"]),
        ("max_nodes", settings["max_nodes"]),
        ("max_depth", settings["max_depth"]),
        ("sample_elems", settings["sample_elems"]),
        ("rerank_candidates", settings["rerank_candidates"]),
        ("rerank_blocks", settings["rerank_blocks"]),
        ("enabled_ops", enabled_ops),
        (
            "rerank_enabled",
            settings["rerank_candidates"] > 0 and settings["rerank_blocks"] > 0,
        ),
        ("max_realizations", system.FROZEN_MAX_REALIZATIONS),
        ("max_realizations_scope", system.FROZEN_MAX_REALIZATIONS_SCOPE),
        (
            "tensor_search_uses_max_realizations",
            system.FROZEN_TENSOR_SEARCH_USES_MAX_REALIZATIONS,
        ),
        ("target_block_bytes", system.FROZEN_TARGET_BLOCK_BYTES),
    )
    if guidance == "fixed":
        if disabled_tuple:
            raise CampaignError("fixed configurations cannot disable search operators")
        return system.BrevisSpec(
            identifier, "fixed", "none", notes=notes,
            expected_effective_config=expected,
        )
    if guidance not in {"uniform", "phog"}:
        raise CampaignError(f"unsupported guidance mode {guidance!r}")
    return system.BrevisSpec(
        identifier=identifier,
        plan="search",
        prior_policy="canonical" if guidance == "phog" else "none",
        search_args=_settings_args(settings, disabled_tuple),
        notes=notes,
        require_raw_only=require_raw_only,
        expected_effective_config=expected,
    )


def _campaign_specs(
    campaign: Mapping[str, Any],
    matrix: Mapping[str, Any],
    terminals: Sequence[str],
    transforms: Sequence[str],
) -> list[dict[str, Any]]:
    """Return deterministic comparison groups for one campaign."""

    kind = campaign.get("kind")
    if kind in SUPPORTED_GENERIC_KINDS:
        if campaign.get("registry") != GENERIC_REGISTRY_ID:
            raise CampaignError(
                f"generic codec campaign must declare registry={GENERIC_REGISTRY_ID!r}"
            )
        specs = tuple(common.BASELINE_SPECS)
        identifiers = [spec.identifier for spec in specs]
        if len(specs) != GENERIC_METHOD_COUNT or len(identifiers) != len(set(identifiers)):
            raise CampaignError("generic baseline registry is not the frozen 19-entry set")
        return [{
            "id": "generic-codec-comparison",
            "specs": specs,
            "dimension": {
                "registry": GENERIC_REGISTRY_ID,
                "method_ids": identifiers,
                "comparison_semantics": (
                    "all 19 pinned registry entries share one balanced harness invocation"
                ),
            },
            "schedule": {
                "status_code": "within_group_balanced_outer_axis_balanced",
                "within_group_balanced": True,
                "outer_axis_balanced": True,
                "limitation": None,
            },
        }]
    if kind in UNSUPPORTED_KINDS:
        return [{
            "id": str(kind).replace("_", "-"),
            "specs": (),
            "dimension": {
                "unsupported_kind": kind,
                "registry": campaign.get("registry"),
                "placeholder_semantics": (
                    "one coordination placeholder per model file; this is not one row per "
                    "baseline configuration and is not an experimental result"
                ),
            },
            "schedule": {
                "status_code": "unsupported_kind",
                "within_group_balanced": False,
                "outer_axis_balanced": False,
            },
        }]
    if kind not in SUPPORTED_BREVIS_KINDS:
        raise CampaignError(f"unknown campaign kind {kind!r}")
    defaults = _search_defaults(matrix)
    all_operators = ("raw", *terminals, *transforms)
    groups: list[dict[str, Any]] = []

    def make_spec(
        identifier: str,
        guidance: str,
        settings: Mapping[str, int],
        *,
        disabled: Iterable[str] = (),
        notes: str,
        require_raw_only: bool = False,
    ) -> system.BrevisSpec:
        return _spec(
            identifier, guidance, settings, all_operators,
            disabled=disabled, notes=notes, require_raw_only=require_raw_only,
        )

    def add_group(
        identifier: str,
        specs: Sequence[system.BrevisSpec],
        dimension: Mapping[str, Any],
        *,
        outer_axis_balanced: bool,
        limitation: str | None = None,
    ) -> None:
        selected = tuple(specs)
        if not selected:
            raise CampaignError("supported comparison groups must contain Brevis specs")
        canonical_args = {
            spec.search_args for spec in selected if spec.prior_policy == "canonical"
        }
        if len(canonical_args) > 1:
            raise CampaignError(
                "comparison group would require multiple incompatible canonical priors"
            )
        groups.append({
            "id": _safe_id(identifier, "comparison group id"),
            "specs": selected,
            "dimension": dict(dimension),
            "schedule": {
                "status_code": (
                    (
                        "within_group_balanced_outer_axis_balanced"
                        if outer_axis_balanced
                        else "within_group_balanced_outer_axis_unpaired"
                    )
                    if len(selected) > 1
                    else (
                        "singleton_outer_axis_balanced"
                        if outer_axis_balanced else "singleton_outer_axis_unpaired"
                    )
                ),
                "within_group_balanced": len(selected) > 1,
                "outer_axis_balanced": outer_axis_balanced,
                "limitation": limitation,
            },
        })

    if kind in {"brevis_system", "brevis_parallel_scaling"}:
        configurations = _unique_strings(
            campaign.get("configurations"), f"campaign {campaign['id']} configurations",
        )
        required_configurations = (
            ("raw-terminal", "fixed", "uniform", "phog")
            if kind == "brevis_system" else ("fixed", "uniform", "phog")
        )
        if configurations != required_configurations:
            raise CampaignError(
                f"{kind} requires configurations {required_configurations!r} in frozen order"
            )
        selected_specs: list[system.BrevisSpec] = []
        for configuration in configurations:
            if configuration == "raw-terminal":
                selected_specs.append(make_spec(
                    configuration, "uniform", defaults,
                    disabled=(*terminals, *transforms),
                    notes="Framed Brevis archive with every non-raw production disabled.",
                    require_raw_only=True,
                ))
            elif configuration == "fixed":
                selected_specs.append(make_spec(
                    configuration, "fixed", defaults,
                    notes="Fixed dtype-specific DSL template with raw fallback.",
                ))
            elif configuration in {"uniform", "phog"}:
                selected_specs.append(make_spec(
                    configuration, configuration, defaults,
                    notes=f"Frozen default {configuration} search configuration.",
                ))
            else:
                raise CampaignError(f"unknown core configuration {configuration!r}")
        add_group(
            "core-comparison" if kind == "brevis_system" else "scaling-method-comparison",
            selected_specs,
            {"configurations": list(configurations)},
            outer_axis_balanced=kind == "brevis_system",
            limitation=(
                None if kind == "brevis_system" else
                "method order is balanced within each jobs point; jobs points are separate "
                "outer tasks and require the bound machine fingerprint"
            ),
        )
    elif kind == "brevis_search_sweep":
        guidance_modes = _unique_strings(campaign.get("guidance_modes"), "guidance_modes")
        sweep = _object(campaign.get("independent_sweep"), "independent_sweep")
        held = dict(_object(sweep.get("held_constant"), "independent_sweep.held_constant"))
        varying = [key for key, value in sweep.items() if key != "held_constant" and isinstance(value, list)]
        if len(varying) != 1:
            raise CampaignError("an independent search sweep must vary exactly one option")
        key = varying[0]
        if key not in dict(SEARCH_OPTION_ORDER):
            raise CampaignError(f"search sweep varies unsupported option {key!r}")
        values = _array(sweep[key], f"independent_sweep.{key}")
        if set(guidance_modes) != {"uniform", "phog"}:
            raise CampaignError("search sweeps require exactly uniform and phog guidance")
        for value in values:
            settings = dict(defaults)
            settings.update(held)
            settings[key] = _integer(value, f"{key} sweep value")
            _validate_settings(settings)
            specs = [
                make_spec(
                    f"{guidance}-{key.replace('_', '-')}-{value}", guidance, settings,
                    notes=f"Independent {key} sweep at {value} with held constants recorded.",
                )
                for guidance in guidance_modes
            ]
            add_group(
                f"{key.replace('_', '-')}-{value}", specs,
                {"axis": key, "value": value, "guidance_modes": list(guidance_modes)},
                outer_axis_balanced=False,
                limitation=f"guidance is paired within this point; {key} values are outer tasks",
            )
    elif kind == "brevis_search_grid":
        guidance_modes = _unique_strings(campaign.get("guidance_modes"), "guidance_modes")
        grid = _object(campaign.get("grid"), "grid")
        expansions = _array(grid.get("max_expansions"), "grid.max_expansions")
        depths = _array(grid.get("max_depth"), "grid.max_depth")
        held = dict(_object(campaign.get("held_constant"), "held_constant"))
        if set(guidance_modes) != {"uniform", "phog"}:
            raise CampaignError("search grids require exactly uniform and phog guidance")
        for expansion in expansions:
            for depth in depths:
                settings = dict(defaults)
                settings.update(held)
                settings["max_expansions"] = _integer(expansion, "grid.max_expansions value", minimum=1)
                settings["max_depth"] = _integer(depth, "grid.max_depth value")
                _validate_settings(settings)
                specs = [
                    make_spec(
                        f"{guidance}-budget-{expansion}-depth-{depth}", guidance, settings,
                        notes="Secondary budget-by-depth interaction grid point.",
                    )
                    for guidance in guidance_modes
                ]
                add_group(
                    f"budget-{expansion}-depth-{depth}", specs,
                    {
                        "grid": {"max_expansions": expansion, "max_depth": depth},
                        "guidance_modes": list(guidance_modes),
                    },
                    outer_axis_balanced=False,
                    limitation="guidance is paired within this point; grid points are outer tasks",
                )
    elif kind == "brevis_reranking_ablation":
        guidance_modes = _unique_strings(campaign.get("guidance_modes"), "guidance_modes")
        candidate = _object(campaign.get("candidate_sweep"), "candidate_sweep")
        blocks = _object(campaign.get("block_sweep"), "block_sweep")
        seen: set[tuple[tuple[str, int], ...]] = set()
        if set(guidance_modes) != {"uniform", "phog"}:
            raise CampaignError("reranking ablations require exactly uniform and phog guidance")
        rerank_points = [
            ("candidates", value, candidate.get("rerank_blocks"))
            for value in _array(candidate.get("rerank_candidates"), "candidate_sweep.rerank_candidates")
        ] + [
            ("blocks", blocks.get("rerank_candidates"), value)
            for value in _array(blocks.get("rerank_blocks"), "block_sweep.rerank_blocks")
        ]
        for axis, candidates, block_count in rerank_points:
            settings = dict(defaults)
            settings["rerank_candidates"] = _integer(candidates, "rerank_candidates")
            settings["rerank_blocks"] = _integer(block_count, "rerank_blocks")
            _validate_settings(settings)
            semantic = tuple(sorted(settings.items()))
            if semantic in seen:
                continue
            seen.add(semantic)
            specs = [
                make_spec(
                    f"{guidance}-rerank-c{candidates}-b{block_count}", guidance, settings,
                    notes="Reranking width/block probe ablation; zero disables the corresponding stage.",
                )
                for guidance in guidance_modes
            ]
            add_group(
                f"rerank-c{candidates}-b{block_count}", specs,
                {
                    "axis": axis,
                    "rerank_candidates": candidates,
                    "rerank_blocks": block_count,
                    "guidance_modes": list(guidance_modes),
                },
                outer_axis_balanced=False,
                limitation="guidance is paired within this point; reranking points are outer tasks",
            )
    elif kind == "brevis_operator_ablation":
        guidance_modes = _unique_strings(campaign.get("guidance_modes"), "guidance_modes")
        disabled_values = _unique_strings(
            campaign.get("one_disabled_operator_per_configuration"),
            "one_disabled_operator_per_configuration",
        )
        if set(disabled_values) != set(transforms) or "raw" in disabled_values:
            raise CampaignError("operator leave-one-out list must equal the frozen transform inventory")
        if guidance_modes != ("uniform",):
            raise CampaignError("operator leave-one-out requires uniform guidance only")
        specs = [make_spec(
            "uniform-default", "uniform", defaults,
            notes="Reference search configuration for paired operator ablations.",
        )]
        specs.extend(
            make_spec(
                f"uniform-without-{operator.replace('_', '-')}", "uniform", defaults,
                disabled=(operator,),
                notes=f"Leave-one-out transform ablation disabling {operator}; raw remains enabled.",
            )
            for operator in disabled_values
        )
        add_group(
            "operator-leave-one-out", specs,
            {"reference": "uniform-default", "disabled_operators": list(disabled_values)},
            outer_axis_balanced=True,
        )
    elif kind == "brevis_terminal_ablation":
        guidance_modes = _unique_strings(campaign.get("guidance_modes"), "guidance_modes")
        disabled_values = _unique_strings(
            campaign.get("one_disabled_terminal_per_configuration"),
            "one_disabled_terminal_per_configuration",
        )
        if set(disabled_values) != set(terminals) or "raw" in disabled_values:
            raise CampaignError("terminal leave-one-out list must equal terminal codecs and exclude raw")
        if campaign.get("mandatory_raw_fallback_remains_enabled") is not True:
            raise CampaignError("terminal ablation must keep raw fallback enabled")
        if guidance_modes != ("uniform",):
            raise CampaignError("terminal leave-one-out requires uniform guidance only")
        specs = [make_spec(
            "uniform-default", "uniform", defaults,
            notes="Reference search configuration for paired terminal ablations.",
        )]
        specs.extend(
            make_spec(
                f"uniform-without-{terminal}", "uniform", defaults,
                disabled=(terminal,),
                notes=f"Leave-one-out terminal ablation disabling {terminal}; raw remains enabled.",
            )
            for terminal in disabled_values
        )
        add_group(
            "terminal-leave-one-out", specs,
            {"reference": "uniform-default", "disabled_terminals": list(disabled_values)},
            outer_axis_balanced=True,
        )
    elif kind == "brevis_calibration_sweep":
        guidance_modes = _unique_strings(campaign.get("guidance_modes"), "guidance_modes")
        if set(guidance_modes) != {"phog"}:
            raise CampaignError("calibration sweep must use PHOG guidance")
        add_group(
            "phog-calibration", [make_spec(
                "phog", "phog", defaults,
                notes="Frozen downstream PHOG search for the calibration-size sweep.",
            )],
            {"guidance": "phog"},
            outer_axis_balanced=False,
            limitation=(
                "calibration_tensors is global to one harness invocation; calibration-size "
                "points are separate outer-unpaired tasks and timing contrasts are not paired"
            ),
        )
    else:  # pragma: no cover - exhaustive over SUPPORTED_BREVIS_KINDS.
        raise AssertionError(kind)

    group_ids = [group["id"] for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise CampaignError(f"campaign {campaign['id']!r} generated duplicate comparison groups")
    for group in groups:
        identifiers = [spec.identifier for spec in group["specs"]]
        if len(identifiers) != len(set(identifiers)):
            raise CampaignError(
                f"comparison group {group['id']!r} generated duplicate configuration IDs"
            )
    return groups


def _validate_settings(settings: Mapping[str, Any]) -> None:
    if set(settings) != {key for key, _ in SEARCH_OPTION_ORDER}:
        raise CampaignError("search settings must specify every exposed frozen option")
    for key, _ in SEARCH_OPTION_ORDER:
        minimum = 1 if key in {"max_expansions", "max_nodes", "sample_elems"} else 0
        _integer(settings[key], f"search setting {key}", minimum=minimum)


def _shard_output_label(filename: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "-", filename).strip("-").lower()
    suffix = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:70]}-{suffix}" if stem else suffix


def _command_templates(
    spec: system.BrevisSpec, jobs: int, calibration_tensors: int,
) -> Mapping[str, Any]:
    binary = str(system.DEFAULT_BINARY.resolve(strict=False))
    search = list(spec.search_args)
    plan_and_search = ["--plan", spec.plan, "--jobs", str(jobs), *search]
    prior = ["--prior", "<CANONICAL_PRIOR>"] if spec.prior_policy == "canonical" else []
    return {
        "configuration_probe": [binary, "config", *search],
        "calibration": (
            [
                binary, "calibrate", "<INPUT>", "<PRIOR>", "--format", "json",
                "--tensors", str(calibration_tensors), "--jobs", str(jobs), *search,
            ]
            if spec.prior_policy == "canonical" else None
        ),
        "compression": [
            binary, "compress", "<INPUT>", "<ARCHIVE>",
            *plan_and_search, *prior,
        ],
        "diagnostic": [
            binary, "bench", "<INPUT>", "--format", "json",
            *plan_and_search, *prior,
        ],
        "decompression": [
            binary, "decompress", "<ARCHIVE>", "<RESTORED>",
            "--jobs", str(jobs),
        ],
        "placeholder_policy": "angle-bracket paths are harness-created runtime paths",
    }


def _cache_input_path(cache_root: pathlib.Path, model: Mapping[str, Any], filename: str) -> pathlib.Path:
    repo_directory = str(model["repo"]).replace("/", "__")
    relative = pathlib.PurePosixPath(filename)
    return cache_root / repo_directory / str(model["revision"]) / pathlib.Path(*relative.parts)


def _task(
    *,
    matrix_sha256: str,
    manifest_sha256: str,
    manifest_path: pathlib.Path,
    cache_root: pathlib.Path,
    staging_root: pathlib.Path,
    campaign: Mapping[str, Any],
    model: Mapping[str, Any],
    file_record: Mapping[str, Any],
    group: Mapping[str, Any],
    jobs: int,
    calibration_tensors: int,
    common_configuration: Mapping[str, Any],
    implementation_binding: Mapping[str, Any],
    scaling_machine_binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    filename = str(file_record["file"])
    input_path = _cache_input_path(cache_root, model, filename).resolve(strict=False)
    execution_lock_path = DEFAULT_EXECUTION_LOCK.resolve(strict=False)
    specs = tuple(group["specs"])
    supported = bool(specs)
    is_brevis = campaign["kind"] in SUPPORTED_BREVIS_KINDS
    is_generic = campaign["kind"] in SUPPORTED_GENERIC_KINDS
    if supported and is_brevis == is_generic:
        raise CampaignError("supported campaign must select exactly one harness family")
    brevis_specs = specs if is_brevis else ()
    baseline_specs = specs if is_generic else ()
    schedule = dict(group["schedule"])
    formal_gates = (
        dict(BREVIS_FORMAL_GATES) if is_brevis
        else dict(GENERIC_FORMAL_GATES) if is_generic
        else None
    )
    executor_kind = (
        "brevis_system_harness" if is_brevis
        else "generic_baseline_harness" if is_generic
        else None
    )
    executor_implementation = (
        "brevis_benchmarking.benchmark_file" if is_brevis
        else "benchmarking.benchmark_file" if is_generic
        else None
    )
    exclusive_policy = {
        "policy_id": EXCLUSIVE_EXECUTION_POLICY_ID,
        "required": True,
        "backend": "fcntl.flock",
        "acquisition": "exclusive_nonblocking",
        "lock_path": str(execution_lock_path),
        "scope": (
            "all cooperating campaign_runner executions in this OS filesystem namespace, "
            "independent of repository, worktree, and cache root"
        ),
        "limitations": (
            "advisory only; it cannot block manual brevis_benchmarking invocations, "
            "other workloads, or executions on another host/container filesystem namespace; "
            "the shared temporary pathname is not a security boundary against a hostile user"
        ),
    }
    semantic = {
        "spec_id": TASK_SEMANTIC_SPEC_ID,
        "experiment_matrix_sha256": matrix_sha256,
        "model_manifest_sha256": manifest_sha256,
        "implementation_binding": dict(implementation_binding),
        "machine_binding": (
            dict(scaling_machine_binding) if scaling_machine_binding is not None else None
        ),
        "campaign_id": campaign["id"],
        "campaign_kind": campaign["kind"],
        "campaign_stage": campaign.get("stage"),
        "campaign_resource_gate": campaign.get("resource_gate"),
        "model": {
            "tag": model["tag"],
            "repo": model["repo"],
            "revision": model["revision"],
            "scope": model.get("scope"),
        },
        "input": {
            "file": filename,
            "bytes": file_record["bytes"],
            "sha256": file_record["sha256"],
        },
        "comparison_group_id": group["id"],
        "dimension": dict(group["dimension"]),
        "comparison_schedule": schedule,
        "brevis_specs": [spec.to_dict() for spec in brevis_specs],
        "baseline_specs": [spec.to_dict() for spec in baseline_specs],
        "jobs": jobs,
        "calibration_tensors": calibration_tensors,
        "warmups": campaign["warmups"],
        "repetitions": campaign["repetitions"],
        "timeout_seconds_per_process": common_configuration["timeout_seconds"],
        "version_probe_timeout_seconds": (
            common.DEFAULT_VERSION_PROBE_TIMEOUT_SECONDS if is_generic else None
        ),
        "schedule_seed": common_configuration["schedule_seed"],
        "disk_reserve_fraction": common_configuration["disk_reserve_fraction"],
        "executor": {
            "kind": executor_kind,
            "supported": supported,
            "status_code": "ready" if supported else "unsupported_kind",
            "implementation": executor_implementation,
            "formal_gates": formal_gates,
            "exclusive_execution_policy": exclusive_policy,
        },
    }
    task_sha256 = _semantic_sha256(semantic)
    configuration_label = str(group["id"])
    output_path = (
        staging_root
        / str(campaign["id"])
        / str(model["tag"])
        / _shard_output_label(filename)
        / configuration_label
        / f"jobs-{jobs}"
        / f"calibration-{calibration_tensors}"
        / f"{task_sha256}.json"
    )
    if is_brevis:
        executor = {
            "supported": True,
            "status_code": "ready",
            "implementation": executor_implementation,
            "formal_gates": formal_gates,
            "exclusive_execution_policy": exclusive_policy,
            "brevis_command_templates": [
                {
                    "configuration_id": spec.identifier,
                    "commands": _command_templates(spec, jobs, calibration_tensors),
                }
                for spec in specs
            ],
        }
    elif is_generic:
        executor = {
            "supported": True,
            "status_code": "ready",
            "implementation": executor_implementation,
            "formal_gates": formal_gates,
            "exclusive_execution_policy": exclusive_policy,
            "brevis_command_templates": None,
            "generic_registry": [spec.to_dict() for spec in baseline_specs],
        }
    else:
        executor = {
            "supported": False,
            "status_code": "unsupported_kind",
            "implementation": None,
            "formal_gates": None,
            "reason": (
                f"campaign_runner does not execute {campaign['kind']!r}; use its "
                "dedicated audited harness and do not interpret this plan row as a run"
            ),
            "exclusive_execution_policy": exclusive_policy,
            "brevis_command_templates": None,
            "generic_registry": None,
        }
    input_status = {
        "exists_at_plan_time": input_path.is_file(),
        "size_matches_at_plan_time": (
            input_path.stat().st_size == file_record["bytes"]
            if input_path.is_file() else None
        ),
        "sha256_checked_at_plan_time": False,
        "note": "execution rehashes the complete input through the formal harness",
    }
    return {
        "task_semantic_sha256": task_sha256,
        "task_semantics": semantic,
        "campaign": {
            "id": campaign["id"],
            "stage": campaign.get("stage"),
            "kind": campaign["kind"],
            "resource_gate": campaign.get("resource_gate"),
        },
        "model": {
            "tag": model["tag"],
            "repo": model["repo"],
            "revision": model["revision"],
            "scope": model.get("scope"),
        },
        "input": {
            "path": str(input_path),
            "file": filename,
            "expected_size_bytes": file_record["bytes"],
            "expected_sha256": file_record["sha256"],
            "plan_time_status": input_status,
        },
        "manifest": {
            "path": str(manifest_path),
            "expected_sha256": manifest_sha256,
        },
        "comparison_group": {
            "id": group["id"],
            "schedule": schedule,
        },
        "dimension": dict(group["dimension"]),
        "brevis_specs": [spec.to_dict() for spec in brevis_specs],
        "baseline_specs": [spec.to_dict() for spec in baseline_specs],
        "implementation_binding": dict(implementation_binding),
        "machine_binding": (
            dict(scaling_machine_binding) if scaling_machine_binding is not None else None
        ),
        "configuration": {
            "jobs": jobs,
            "calibration_tensors": calibration_tensors,
            "warmups": campaign["warmups"],
            "repetitions": campaign["repetitions"],
            "timeout_seconds_per_process": common_configuration["timeout_seconds"],
            "version_probe_timeout_seconds": (
                common.DEFAULT_VERSION_PROBE_TIMEOUT_SECONDS if is_generic else None
            ),
            "schedule_seed": common_configuration["schedule_seed"],
            "disk_reserve_fraction": common_configuration["disk_reserve_fraction"],
        },
        "executor": executor,
        "suggested_output_path": str(output_path),
        "suggested_receipt_path": (
            str(output_path.with_suffix(".validation.json")) if is_generic else None
        ),
        "output_policy": {
            "under_configured_staging_root": output_path.is_relative_to(
                staging_root.resolve(strict=False)
            ),
            "default_staging_root_is_git_ignored": (
                staging_root.resolve(strict=False)
                == DEFAULT_STAGING_ROOT.resolve(strict=False)
            ),
            "overwrite": False,
            "one_task_per_result": True,
        },
        "limitations": [
            value for value in (
                schedule.get("limitation"), exclusive_policy["limitations"],
            ) if value
        ],
    }


def plan_campaigns(
    experiments_path: os.PathLike[str] | str = DEFAULT_EXPERIMENTS,
    *,
    repository_root: os.PathLike[str] | str = ROOT,
    cache_root: os.PathLike[str] | str = DEFAULT_CACHE_ROOT,
    staging_root: os.PathLike[str] | str = DEFAULT_STAGING_ROOT,
    campaign_ids: Iterable[str] | None = None,
    model_tags: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Strictly expand schema-v1 campaigns into deterministic atomic tasks."""

    root = pathlib.Path(repository_root).resolve(strict=False)
    experiments = pathlib.Path(experiments_path).resolve(strict=True)
    cache = pathlib.Path(cache_root).resolve(strict=False)
    staging = pathlib.Path(staging_root).resolve(strict=False)
    matrix, matrix_sha256, manifest_path, models, manifest_sha256 = _load_inputs(
        experiments, root,
    )
    models_by_tag = _validate_models(models)
    terminals, transforms = _validate_inventory(matrix)
    common_payload = _object(matrix.get("common"), "common")
    if common_payload.get("binary_build") != "ReleaseFast":
        raise CampaignError("formal campaign runner requires binary_build=ReleaseFast")
    archive_defaults = _object(
        common_payload.get("archive_defaults"), "common.archive_defaults",
    )
    if archive_defaults.get("raw_fallback_enabled") is not True:
        raise CampaignError("formal campaign matrix must keep raw fallback enabled")
    if archive_defaults.get("target_block_bytes") != system.FROZEN_TARGET_BLOCK_BYTES:
        raise CampaignError("campaign target_block_bytes disagrees with the harness contract")
    schedule_seed = _integer(common_payload.get("schedule_seed"), "common.schedule_seed")
    calibration_seed = _integer(
        common_payload.get("calibration_seed"), "common.calibration_seed",
    )
    if calibration_seed != system.DEFAULT_CALIBRATION_SEED:
        raise CampaignError(
            "campaign calibration_seed disagrees with the harness's pinned seed"
        )
    default_calibration = _integer(
        common_payload.get("calibration_tensors"), "common.calibration_tensors", minimum=1,
    )
    default_timeout = _integer(
        common_payload.get("timeout_seconds_per_process_default"),
        "common.timeout_seconds_per_process_default", minimum=1,
    )
    reserve = common_payload.get("disk_reserve_fraction")
    if isinstance(reserve, bool) or not isinstance(reserve, (int, float)) or not 0 <= reserve < 1:
        raise CampaignError("common.disk_reserve_fraction must be in [0, 1)")
    _search_defaults(matrix)
    search_defaults_payload = _object(
        common_payload.get("search_defaults"), "common.search_defaults",
    )
    if search_defaults_payload.get("max_realizations") != system.FROZEN_MAX_REALIZATIONS:
        raise CampaignError("campaign max_realizations disagrees with the harness contract")

    campaigns = _array(matrix.get("campaigns"), "campaigns")
    campaign_by_id: dict[str, Mapping[str, Any]] = {}
    for index, campaign_value in enumerate(campaigns):
        campaign = _object(campaign_value, f"campaign {index}")
        identifier = _safe_id(campaign.get("id"), f"campaign {index}.id")
        if identifier in campaign_by_id:
            raise CampaignError(f"duplicate campaign id {identifier!r}")
        campaign_by_id[identifier] = campaign
    selected_campaign_ids = (
        tuple(campaign_ids) if campaign_ids is not None else tuple(campaign_by_id)
    )
    if not selected_campaign_ids or len(set(selected_campaign_ids)) != len(selected_campaign_ids):
        raise CampaignError("selected campaign IDs must be nonempty and unique")
    unknown_campaigns = set(selected_campaign_ids) - set(campaign_by_id)
    if unknown_campaigns:
        raise CampaignError("unknown campaign ID(s): " + ", ".join(sorted(unknown_campaigns)))
    selected_model_filter = tuple(model_tags) if model_tags is not None else None
    if selected_model_filter is not None:
        if not selected_model_filter or len(set(selected_model_filter)) != len(selected_model_filter):
            raise CampaignError("selected model tags must be nonempty and unique")
        unknown_models = set(selected_model_filter) - set(models_by_tag)
        if unknown_models:
            raise CampaignError("unknown model tag(s): " + ", ".join(sorted(unknown_models)))

    implementation_binding = _implementation_binding()
    machine_binding: Mapping[str, Any] | None = None
    tasks: list[dict[str, Any]] = []
    for campaign_id in selected_campaign_ids:
        campaign = campaign_by_id[campaign_id]
        model_ids = _unique_strings(campaign.get("model_tags"), f"campaign {campaign_id}.model_tags")
        missing_models = set(model_ids) - set(models_by_tag)
        if missing_models:
            raise CampaignError(
                f"campaign {campaign_id!r} references unknown models: "
                + ", ".join(sorted(missing_models))
            )
        if selected_model_filter is not None:
            model_ids = tuple(tag for tag in model_ids if tag in selected_model_filter)
        warmups = _integer(campaign.get("warmups"), f"campaign {campaign_id}.warmups")
        repetitions = _integer(
            campaign.get("repetitions"), f"campaign {campaign_id}.repetitions", minimum=1,
        )
        jobs_values = [
            _integer(value, f"campaign {campaign_id}.jobs", minimum=1)
            for value in _array(campaign.get("jobs"), f"campaign {campaign_id}.jobs")
        ]
        if len(jobs_values) != len(set(jobs_values)):
            raise CampaignError(f"campaign {campaign_id!r} contains duplicate jobs")
        calibration_value = campaign.get("calibration_tensors", [default_calibration])
        calibration_values = [
            _integer(value, f"campaign {campaign_id}.calibration_tensors", minimum=1)
            for value in _array(calibration_value, f"campaign {campaign_id}.calibration_tensors")
        ]
        if len(calibration_values) != len(set(calibration_values)):
            raise CampaignError(f"campaign {campaign_id!r} contains duplicate calibration sizes")
        timeout = _integer(
            campaign.get("timeout_seconds_per_process", default_timeout),
            f"campaign {campaign_id}.timeout_seconds_per_process", minimum=1,
        )
        stage = _integer(campaign.get("stage"), f"campaign {campaign_id}.stage", minimum=1)
        if campaign.get("kind") in SUPPORTED_GENERIC_KINDS:
            if warmups != GENERIC_WARMUPS or repetitions != GENERIC_REPETITIONS:
                raise CampaignError("generic codec campaign requires exactly 1 warmup and 6 repetitions")
            if jobs_values != [1]:
                raise CampaignError("generic codec campaign requires the single jobs=1 task")
            if calibration_values != [default_calibration]:
                raise CampaignError(
                    "generic codec campaign cannot expand the irrelevant calibration axis"
                )
            if schedule_seed != GENERIC_SCHEDULE_SEED:
                raise CampaignError("generic codec campaign schedule seed must be 2701")
            if float(reserve) != GENERIC_DISK_RESERVE_FRACTION:
                raise CampaignError("generic codec campaign requires a 30% disk reserve")
        normalized_campaign = dict(campaign)
        normalized_campaign.update({
            "stage": stage, "warmups": warmups, "repetitions": repetitions,
        })
        groups = _campaign_specs(normalized_campaign, matrix, terminals, transforms)
        if campaign.get("kind") == "brevis_parallel_scaling" and machine_binding is None:
            machine_binding = _machine_binding()
        for model_id in model_ids:
            model = models_by_tag[model_id]
            for file_value in _array(model.get("files"), f"model {model_id}.files"):
                file_record = _object(file_value, f"model {model_id} file")
                for group in groups:
                    for jobs in jobs_values:
                        for calibration_tensors in calibration_values:
                            tasks.append(_task(
                                matrix_sha256=matrix_sha256,
                                manifest_sha256=manifest_sha256,
                                manifest_path=manifest_path,
                                cache_root=cache,
                                staging_root=staging,
                                campaign=normalized_campaign,
                                model=model,
                                file_record=file_record,
                                group=group,
                                jobs=jobs,
                                calibration_tensors=calibration_tensors,
                                common_configuration={
                                    "timeout_seconds": timeout,
                                    "schedule_seed": schedule_seed,
                                    "disk_reserve_fraction": float(reserve),
                                },
                                implementation_binding=implementation_binding,
                                scaling_machine_binding=(
                                    machine_binding
                                    if campaign.get("kind") == "brevis_parallel_scaling"
                                    else None
                                ),
                            ))
    if not tasks:
        raise CampaignError("campaign/model selection produced no atomic tasks")
    task_hashes = [task["task_semantic_sha256"] for task in tasks]
    if len(task_hashes) != len(set(task_hashes)):
        duplicates = sorted({value for value in task_hashes if task_hashes.count(value) > 1})
        raise CampaignError(
            "campaign expansion produced duplicate atomic task semantics: "
            + ", ".join(duplicates)
        )
    return {
        "schema": {"id": PLAN_SCHEMA_ID, "version": PLAN_SCHEMA_VERSION},
        "status": "plan_only_not_results",
        "execution_default": "dry_run",
        "planner_configuration": {
            "repository_root": str(root),
            "cache_root": str(cache),
            "staging_root": str(staging),
        },
        "implementation_binding": implementation_binding,
        "exclusive_execution": {
            "policy_id": EXCLUSIVE_EXECUTION_POLICY_ID,
            "lock_path": str(DEFAULT_EXECUTION_LOCK.resolve(strict=False)),
            "advisory_limit": (
                "does not block manual harness calls, unrelated workloads, or executions in "
                "another host/container filesystem namespace; it is not a multi-user security "
                "boundary"
            ),
        },
        "experiment_matrix": {
            "path": str(experiments),
            "sha256": matrix_sha256,
            "schema": matrix["schema"],
            "status": matrix["status"],
        },
        "model_manifest": {
            "path": str(manifest_path),
            "expected_sha256": manifest_sha256,
            "actual_sha256": manifest_sha256,
            "verified": True,
        },
        "selection": {
            "campaign_ids": list(selected_campaign_ids),
            "model_tags": list(selected_model_filter) if selected_model_filter is not None else None,
        },
        "task_count": len(tasks),
        "supported_task_count": sum(task["executor"]["supported"] for task in tasks),
        "unsupported_task_count": sum(not task["executor"]["supported"] for task in tasks),
        "tasks": tasks,
    }


def _reconstruct_spec(value: Mapping[str, Any]) -> system.BrevisSpec:
    allowed = {
        "id", "plan", "prior_policy", "search_args", "notes", "require_raw_only",
        "expected_effective_config",
    }
    if set(value) != allowed:
        raise CampaignError("planned BrevisSpec fields are incomplete or unknown")
    search_args = value.get("search_args")
    if not isinstance(search_args, list):
        raise CampaignError("planned BrevisSpec search_args must be an array")
    expected = value.get("expected_effective_config")
    if not isinstance(expected, Mapping):
        raise CampaignError("planned expected_effective_config must be an object")
    expected_pairs: list[tuple[str, Any]] = []
    for key in system.EXPECTED_EFFECTIVE_CONFIG_KEYS:
        if key not in expected:
            raise CampaignError(f"planned expected_effective_config lacks {key}")
        item = expected[key]
        if key == "enabled_ops":
            if not isinstance(item, list):
                raise CampaignError("planned expected enabled_ops must be an array")
            item = tuple(item)
        expected_pairs.append((key, item))
    if set(expected) != set(system.EXPECTED_EFFECTIVE_CONFIG_KEYS):
        raise CampaignError("planned expected_effective_config has unknown keys")
    try:
        return system.BrevisSpec(
            identifier=value["id"],
            plan=value["plan"],
            prior_policy=value["prior_policy"],
            search_args=tuple(search_args),
            notes=value["notes"],
            require_raw_only=value["require_raw_only"],
            expected_effective_config=tuple(expected_pairs),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CampaignError(f"invalid planned BrevisSpec: {exc}") from exc


def _generic_disk_gate(
    source: pathlib.Path,
    output: pathlib.Path,
    *,
    reserve_fraction: float,
    method_count: int,
    iterations_per_method: int,
) -> dict[str, Any]:
    """Conservatively reserve one filesystem for generic work and checkpoints."""

    if not math.isfinite(reserve_fraction) or reserve_fraction != GENERIC_DISK_RESERVE_FRACTION:
        raise CampaignError("generic formal execution requires exactly a 30% disk reserve")
    output.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output.parent)
    source_bytes = source.stat().st_size
    estimated_work_bytes = (
        source_bytes * GENERIC_DISK_SOURCE_MULTIPLIER
        + GENERIC_DISK_FIXED_BYTES
        + method_count * iterations_per_method * GENERIC_CHECKPOINT_BYTES_PER_ITERATION
    )
    reserved_bytes = math.ceil(usage.total * reserve_fraction)
    projected_free_bytes = usage.free - estimated_work_bytes
    gate = {
        "spec_id": "brevis.generic-disk-gate.v1",
        "path": str(output.parent),
        "filesystem_total_bytes": usage.total,
        "filesystem_free_bytes": usage.free,
        "reserve_fraction": reserve_fraction,
        "reserved_bytes": reserved_bytes,
        "source_bytes": source_bytes,
        "source_multiplier": GENERIC_DISK_SOURCE_MULTIPLIER,
        "fixed_bytes": GENERIC_DISK_FIXED_BYTES,
        "checkpoint_bytes_per_iteration": GENERIC_CHECKPOINT_BYTES_PER_ITERATION,
        "method_count": method_count,
        "iterations_per_method": iterations_per_method,
        "estimated_work_and_checkpoint_bytes": estimated_work_bytes,
        "projected_free_bytes": projected_free_bytes,
        "passes": projected_free_bytes >= reserved_bytes,
        "limitation": (
            "preflight estimate on the shared work/checkpoint filesystem; it cannot reserve "
            "space against unrelated concurrent writers"
        ),
    }
    return gate


def _generic_post_run_disk_gate(
    preflight_gate: Mapping[str, Any], output: pathlib.Path,
) -> dict[str, Any]:
    """Record whether the actual post-run free space still satisfies the reserve."""

    gate = dict(preflight_gate)
    try:
        usage = shutil.disk_usage(output.parent)
    except OSError as exc:
        gate.update({
            "post_run_free_bytes": None,
            "post_run_reserved_bytes": None,
            "post_run_passes": False,
            "post_run_error": f"{type(exc).__name__}: {exc}",
        })
        return gate
    reserve_fraction = preflight_gate.get("reserve_fraction")
    reserved_bytes = (
        math.ceil(usage.total * reserve_fraction)
        if isinstance(reserve_fraction, (int, float))
        and not isinstance(reserve_fraction, bool)
        and math.isfinite(reserve_fraction)
        and 0 <= reserve_fraction < 1
        else None
    )
    passes = (
        isinstance(reserved_bytes, int)
        and not isinstance(reserved_bytes, bool)
        and usage.free >= reserved_bytes
    )
    gate.update({
        "post_run_filesystem_total_bytes": usage.total,
        "post_run_free_bytes": usage.free,
        "post_run_reserved_bytes": reserved_bytes,
        "post_run_passes": passes,
        "post_run_error": None,
    })
    return gate


def _generic_validation_failure(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _path_entry_exists(path: pathlib.Path) -> bool:
    """Return true for files, directories, and dangling symbolic links."""

    return os.path.lexists(os.fspath(path))


def _validate_generic_result(
    result: Mapping[str, Any],
    *,
    task_sha256: str,
    specs: Sequence[common.BaselineSpec],
    configuration: Mapping[str, Any],
    input_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    model: Mapping[str, Any],
    campaign_record: Mapping[str, Any],
    group_record: Mapping[str, Any],
    implementation: Mapping[str, Any],
    matrix_path: pathlib.Path,
    expected_matrix_sha256: str,
    source: pathlib.Path,
    output: pathlib.Path,
    disk_gate: Mapping[str, Any],
    stage_override_reasons: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Validate the persisted generic result without discarding partial evidence."""

    failures: list[dict[str, str]] = []
    checks: dict[str, Any] = {}

    def require(code: str, condition: bool, message: str) -> None:
        checks[code] = bool(condition)
        if not condition:
            failures.append(_generic_validation_failure(code, message))

    expected_spec_dicts = [spec.to_dict() for spec in specs]
    expected_ids = [spec.identifier for spec in specs]
    schema = result.get("schema")
    require(
        "top_schema",
        schema == {"id": common.SCHEMA_ID, "version": common.SCHEMA_VERSION},
        "raw result schema is not the frozen generic benchmark schema",
    )
    require("top_status", result.get("status") == "complete", "raw result is not complete")

    metadata = result.get("input_metadata")
    metadata_ok = isinstance(metadata, Mapping) and all((
        metadata.get("campaign_task_semantic_sha256") == task_sha256,
        metadata.get("campaign_id") == campaign_record.get("id"),
        metadata.get("campaign_comparison_group_id") == group_record.get("id"),
        metadata.get("campaign_experiment_matrix_sha256") == expected_matrix_sha256,
        metadata.get("campaign_runner_sha256")
        == implementation.get("campaign_runner_sha256"),
        metadata.get("generic_benchmarking_sha256")
        == implementation.get("generic_benchmarking_sha256"),
        metadata.get("generic_registry_sha256") == _semantic_sha256(expected_spec_dicts),
        metadata.get("model_tag") == model.get("tag"),
        metadata.get("model_repo") == model.get("repo"),
        metadata.get("model_revision") == model.get("revision"),
        metadata.get("manifest") == manifest_record.get("path"),
        metadata.get("shard") == input_record.get("file"),
        metadata.get("campaign_disk_gate") == dict(disk_gate),
    ))
    require("task_and_input_metadata", metadata_ok, "raw result lost task/input bindings")

    observed_configuration = result.get("configuration")
    configuration_ok = isinstance(observed_configuration, Mapping) and all((
        observed_configuration.get("warmups") == GENERIC_WARMUPS,
        observed_configuration.get("repetitions") == GENERIC_REPETITIONS,
        observed_configuration.get("schedule_seed") == GENERIC_SCHEDULE_SEED,
        observed_configuration.get("timeout_seconds")
        == configuration.get("timeout_seconds_per_process"),
        observed_configuration.get("version_probe_timeout_seconds")
        == configuration.get("version_probe_timeout_seconds"),
        observed_configuration.get("schedule_balance") == "paired",
        observed_configuration.get("keep_artifacts") is False,
        pathlib.Path(str(observed_configuration.get("checkpoint_path"))).resolve(strict=False)
        == output.resolve(strict=False),
    ))
    require("configuration", configuration_ok, "generic benchmark configuration drifted")

    measured_orders = (
        observed_configuration.get("measured_orders")
        if isinstance(observed_configuration, Mapping) else None
    )
    frozen_measured_ids = [
        [spec.identifier for spec in order]
        for order in common._balanced_measured_orders(
            specs, GENERIC_REPETITIONS, GENERIC_SCHEDULE_SEED,
        )
    ]
    orders_ok = isinstance(measured_orders, list) and len(measured_orders) == GENERIC_REPETITIONS
    if orders_ok:
        for index, order in enumerate(measured_orders):
            if not isinstance(order, Mapping):
                orders_ok = False
                break
            methods = order.get("methods")
            if not (
                isinstance(methods, list)
                and methods == frozen_measured_ids[index]
                and order.get("repetition") == index
                and order.get("pair") == index // 2
                and order.get("direction") == (
                    "forward" if index % 2 == 0 else "reverse"
                )
            ):
                orders_ok = False
                break
            if index % 2 == 1 and methods != measured_orders[index - 1].get("methods", [])[::-1]:
                orders_ok = False
                break
    require("balanced_measured_orders", orders_ok, "measured method orders are incomplete/unbalanced")

    methods = result.get("methods")
    pre_run_failure_codes = {
        "executable_missing", "executable_provenance_failed", "version_probe_failed",
    }
    runnable_ids: set[str] = set()
    structured_pre_run_failure_ids: set[str] = set()
    for method_index, identifier in enumerate(expected_ids):
        method = (
            methods[method_index]
            if isinstance(methods, list) and method_index < len(methods) else None
        )
        status_code = method.get("status_code") if isinstance(method, Mapping) else None
        consistency = (
            method.get("measured_archive_consistency")
            if isinstance(method, Mapping) else None
        )
        common_failure_shape = (
            isinstance(method, Mapping)
            and status_code in pre_run_failure_codes
            and isinstance(method.get("failure"), str)
            and bool(method.get("failure"))
            and method.get("warmups") == []
            and method.get("runs") == []
            and isinstance(consistency, Mapping)
            and consistency.get("status_code") == "not_checked"
            and consistency.get("reference_size_bytes") is None
            and consistency.get("reference_sha256") is None
            and consistency.get("consistent_runs") == 0
            and consistency.get("inconsistent_runs") == 0
        )
        if status_code == "executable_missing":
            failure_specific_shape = (
                method.get("available") is False
                and method.get("resolved_executable") is None
                and method.get("executable_realpath") is None
                and method.get("executable_sha256") is None
                and method.get("version") is None
                and method.get("version_probe") is None
            )
        elif status_code == "executable_provenance_failed":
            failure_specific_shape = (
                method.get("available") is True
                and isinstance(method.get("resolved_executable"), str)
                and bool(method.get("resolved_executable"))
                and (
                    method.get("executable_realpath") is None
                    or method.get("executable_sha256") is None
                )
                and method.get("version") is None
                and method.get("version_probe") is None
            )
        elif status_code == "version_probe_failed":
            probe = method.get("version_probe")
            failure_specific_shape = (
                method.get("available") is True
                and isinstance(method.get("resolved_executable"), str)
                and isinstance(method.get("executable_realpath"), str)
                and isinstance(method.get("executable_sha256"), str)
                and SHA256.fullmatch(method["executable_sha256"]) is not None
                and method.get("version") is None
                and isinstance(probe, Mapping)
                and (
                    probe.get("timed_out") is True
                    or probe.get("error") is not None
                    or probe.get("exit_code") != 0
                )
            )
        else:
            failure_specific_shape = False
        is_structured_pre_run_failure = (
            common_failure_shape and failure_specific_shape
        )
        if is_structured_pre_run_failure:
            structured_pre_run_failure_ids.add(identifier)
        else:
            runnable_ids.add(identifier)

    expected_execution_schedule: list[dict[str, Any]] = []
    execution_order = 0
    for warmup_index in range(GENERIC_WARMUPS):
        rotation = warmup_index % len(expected_ids)
        warmup_ids = (*expected_ids[rotation:], *expected_ids[:rotation])
        for order_within_repetition, identifier in enumerate(warmup_ids):
            if identifier not in runnable_ids:
                continue
            expected_execution_schedule.append({
                "execution_order": execution_order,
                "phase": "warmup",
                "repetition": warmup_index,
                "order_within_repetition": order_within_repetition,
                "method": identifier,
            })
            execution_order += 1
    for measured_index, identifiers in enumerate(frozen_measured_ids):
        for order_within_repetition, identifier in enumerate(identifiers):
            if identifier not in runnable_ids:
                continue
            expected_execution_schedule.append({
                "execution_order": execution_order,
                "phase": "measured",
                "repetition": measured_index,
                "order_within_repetition": order_within_repetition,
                "method": identifier,
            })
            execution_order += 1
    observed_execution_schedule = (
        observed_configuration.get("execution_schedule")
        if isinstance(observed_configuration, Mapping) else None
    )
    schedule_ok = orders_ok and observed_execution_schedule == expected_execution_schedule
    require(
        "execution_schedule",
        schedule_ok,
        "execution schedule does not exactly match the warmup and paired measured orders",
    )
    schedule_by_iteration = {
        (entry["phase"], entry["repetition"], entry["method"]): entry
        for entry in expected_execution_schedule
    }

    method_order_ok = (
        isinstance(methods, list)
        and [method.get("id") if isinstance(method, Mapping) else None for method in methods]
        == expected_ids
    )
    require("method_order", method_order_ok, "raw result does not contain the exact 19 methods in order")
    method_checks: list[dict[str, Any]] = []
    if isinstance(methods, list):
        for index, expected_spec in enumerate(specs):
            method = methods[index] if index < len(methods) else None
            identifier = expected_spec.identifier
            record: dict[str, Any] = {"id": identifier}
            if not isinstance(method, Mapping):
                failures.append(_generic_validation_failure(
                    "method_record_missing", f"method record {identifier} is missing",
                ))
                record["valid"] = False
                method_checks.append(record)
                continue
            record.update({
                "spec_matches": method.get("spec") == expected_spec_dicts[index],
                "available": method.get("available") is True,
                "status_ok": method.get("status_code") == "ok" and method.get("failure") is None,
                "tool_provenance": (
                    isinstance(method.get("resolved_executable"), str)
                    and bool(method.get("resolved_executable"))
                    and isinstance(method.get("executable_realpath"), str)
                    and pathlib.Path(method["executable_realpath"]).is_absolute()
                    and isinstance(method.get("version"), str)
                    and bool(method.get("version"))
                    and isinstance(method.get("version_probe"), Mapping)
                    and method["version_probe"].get("status_code") == "ok"
                    and method["version_probe"].get("exit_code") == 0
                    and method["version_probe"].get("timed_out") is False
                    and method["version_probe"].get("error") is None
                ),
                "executable_unchanged": (
                    method.get("executable_unchanged_during_benchmark") is True
                    and isinstance(method.get("executable_sha256"), str)
                    and SHA256.fullmatch(method["executable_sha256"]) is not None
                    and method.get("executable_sha256") == method.get("executable_sha256_after_runs")
                ),
            })
            warmups = method.get("warmups")
            runs = method.get("runs")
            record["warmup_count"] = len(warmups) if isinstance(warmups, list) else None
            record["measured_count"] = len(runs) if isinstance(runs, list) else None
            iterations_ok = (
                isinstance(warmups, list) and len(warmups) == GENERIC_WARMUPS
                and isinstance(runs, list) and len(runs) == GENERIC_REPETITIONS
            )
            if iterations_ok:
                iteration_slots = (
                    *(("warmup", slot, iteration) for slot, iteration in enumerate(warmups)),
                    *(("measured", slot, iteration) for slot, iteration in enumerate(runs)),
                )
                for expected_phase, expected_repetition, iteration in iteration_slots:
                    if not isinstance(iteration, Mapping):
                        iterations_ok = False
                        break
                    phase = iteration.get("phase")
                    repetition = iteration.get("index")
                    schedule_entry = schedule_by_iteration.get(
                        (expected_phase, expected_repetition, identifier)
                    )
                    compression = iteration.get("compression")
                    decompression = iteration.get("decompression")
                    verification = iteration.get("verification")
                    archive_storage = iteration.get("archive_storage")
                    archive_sha256 = iteration.get("archive_sha256")
                    compressed_size = iteration.get("compressed_size_bytes")
                    archive_consistency = iteration.get("archive_consistency")
                    expected_consistency_status = (
                        "not_applicable_warmup" if phase == "warmup"
                        else "reference" if repetition == 0 else "consistent"
                    )
                    if not all((
                        iteration.get("success") is True,
                        iteration.get("failure") is None,
                        iteration.get("bit_exact") is True,
                        iteration.get("status_code") == "ok",
                        phase == expected_phase,
                        repetition == expected_repetition,
                        schedule_entry is not None,
                        iteration.get("execution_order")
                        == (schedule_entry or {}).get("execution_order"),
                        iteration.get("order_within_repetition")
                        == (schedule_entry or {}).get("order_within_repetition"),
                        isinstance(compression, Mapping),
                        (compression or {}).get("status_code") == "ok",
                        (compression or {}).get("exit_code") == 0,
                        (compression or {}).get("timed_out") is False,
                        (compression or {}).get("error") is None,
                        isinstance(decompression, Mapping),
                        (decompression or {}).get("status_code") == "ok",
                        (decompression or {}).get("exit_code") == 0,
                        (decompression or {}).get("timed_out") is False,
                        (decompression or {}).get("error") is None,
                        isinstance(verification, Mapping),
                        (verification or {}).get("attempted") is True,
                        (verification or {}).get("bit_exact") is True,
                        (verification or {}).get("source_sha256")
                        == input_record.get("expected_sha256"),
                        (verification or {}).get("restored_sha256")
                        == input_record.get("expected_sha256"),
                        (verification or {}).get("restored_size_bytes")
                        == input_record.get("expected_size_bytes"),
                        (verification or {}).get("error") is None,
                        isinstance(compressed_size, int) and not isinstance(compressed_size, bool)
                        and compressed_size >= 0,
                        isinstance(archive_storage, Mapping),
                        (archive_storage or {}).get("logical_size_bytes") == compressed_size,
                        isinstance(archive_sha256, str)
                        and SHA256.fullmatch(archive_sha256) is not None,
                        isinstance(archive_consistency, Mapping),
                        (archive_consistency or {}).get("status_code")
                        == expected_consistency_status,
                        (
                            (archive_consistency or {}).get("matches_reference") is None
                            if phase == "warmup"
                            else (archive_consistency or {}).get("matches_reference") is True
                        ),
                    )):
                        iterations_ok = False
                        break
            record["all_iterations_bit_exact"] = iterations_ok
            consistency = method.get("measured_archive_consistency")
            measured_reference_size = (
                runs[0].get("compressed_size_bytes")
                if isinstance(runs, list) and runs and isinstance(runs[0], Mapping)
                else None
            )
            measured_reference_sha256 = (
                runs[0].get("archive_sha256")
                if isinstance(runs, list) and runs and isinstance(runs[0], Mapping)
                else None
            )
            measured_rows_consistent = (
                isinstance(runs, list)
                and len(runs) == GENERIC_REPETITIONS
                and all(
                    isinstance(run, Mapping)
                    and run.get("compressed_size_bytes") == measured_reference_size
                    and run.get("archive_sha256") == measured_reference_sha256
                    for run in runs
                )
            )
            record["measured_archive_consistent"] = (
                isinstance(consistency, Mapping)
                and consistency.get("status_code") == "consistent"
                and consistency.get("consistent_runs") == GENERIC_REPETITIONS
                and consistency.get("inconsistent_runs") == 0
                and consistency.get("reference_sha256") == measured_reference_sha256
                and isinstance(measured_reference_sha256, str)
                and SHA256.fullmatch(measured_reference_sha256) is not None
                and consistency.get("reference_size_bytes") == measured_reference_size
                and isinstance(measured_reference_size, int)
                and not isinstance(measured_reference_size, bool)
                and measured_rows_consistent
            )
            record["valid"] = all(
                value is True for key, value in record.items()
                if key in {
                    "spec_matches", "available", "status_ok", "tool_provenance",
                    "executable_unchanged",
                    "all_iterations_bit_exact", "measured_archive_consistent",
                }
            )
            if not record["valid"]:
                failure_code = (
                    "method_pre_run_failure"
                    if (
                        identifier in structured_pre_run_failure_ids
                        and record.get("spec_matches") is True
                    )
                    else "method_validation_failed"
                )
                failures.append(_generic_validation_failure(
                    failure_code, f"generic method {identifier} failed formal validation",
                ))
            method_checks.append(record)
    checks["methods"] = method_checks

    source_record = result.get("source")
    require(
        "source_provenance",
        isinstance(source_record, Mapping)
        and pathlib.Path(str(source_record.get("path"))).resolve(strict=False)
        == source.resolve(strict=False)
        and source_record.get("size_bytes") == input_record.get("expected_size_bytes")
        and source_record.get("sha256") == input_record.get("expected_sha256"),
        "raw result source provenance disagrees with the task",
    )
    integrity = result.get("integrity")
    source_integrity = integrity.get("source") if isinstance(integrity, Mapping) else None
    manifest_integrity = integrity.get("manifest") if isinstance(integrity, Mapping) else None
    manifest_binding = integrity.get("manifest_binding") if isinstance(integrity, Mapping) else None
    require(
        "integrity",
        isinstance(integrity, Mapping)
        and integrity.get("model_metadata_declared") is True
        and isinstance(source_integrity, Mapping)
        and source_integrity.get("verified") is True
        and source_integrity.get("expected_size_bytes")
        == input_record.get("expected_size_bytes")
        and source_integrity.get("actual_size_bytes")
        == input_record.get("expected_size_bytes")
        and source_integrity.get("expected_sha256") == input_record.get("expected_sha256")
        and source_integrity.get("actual_sha256") == input_record.get("expected_sha256")
        and isinstance(manifest_integrity, Mapping)
        and manifest_integrity.get("verified") is True
        and pathlib.Path(str(manifest_integrity.get("path"))).resolve(strict=False)
        == pathlib.Path(str(manifest_record.get("path"))).resolve(strict=False)
        and manifest_integrity.get("expected_sha256")
        == manifest_record.get("expected_sha256")
        and manifest_integrity.get("actual_sha256") == manifest_record.get("expected_sha256")
        and isinstance(manifest_binding, Mapping)
        and manifest_binding.get("verified") is True
        and manifest_binding.get("tag") == model.get("tag")
        and manifest_binding.get("repo") == model.get("repo")
        and manifest_binding.get("revision") == model.get("revision")
        and manifest_binding.get("shard") == input_record.get("file")
        and manifest_binding.get("bytes") == input_record.get("expected_size_bytes")
        and manifest_binding.get("sha256") == input_record.get("expected_sha256"),
        "source or manifest integrity binding is incomplete",
    )

    provenance = result.get("provenance")
    result_git = provenance.get("git") if isinstance(provenance, Mapping) else None
    result_script = provenance.get("benchmark_script") if isinstance(provenance, Mapping) else None
    require(
        "result_git_provenance",
        isinstance(result_git, Mapping)
        and result_git.get("commit") == implementation.get("git_commit")
        and result_git.get("dirty") is False,
        "raw result Git provenance is not the clean bound commit",
    )
    require(
        "result_script_provenance",
        isinstance(result_script, Mapping)
        and pathlib.Path(str(result_script.get("path"))).resolve(strict=False)
        == pathlib.Path(common.__file__).resolve(strict=False)
        and result_script.get("sha256") == implementation.get("generic_benchmarking_sha256"),
        "raw result benchmarking.py hash disagrees with the task",
    )
    environment = result.get("environment")
    codec_environment = (
        environment.get("codec_environment_variables")
        if isinstance(environment, Mapping) else None
    )
    filesystems = environment.get("filesystems") if isinstance(environment, Mapping) else None
    require(
        "environment_provenance",
        isinstance(environment, Mapping)
        and isinstance(environment.get("cpu"), Mapping)
        and isinstance(environment.get("memory"), Mapping)
        and isinstance(filesystems, Mapping)
        and isinstance(filesystems.get("source"), Mapping)
        and isinstance(filesystems.get("work"), Mapping)
        and isinstance(filesystems.get("checkpoint"), Mapping),
        "raw result lacks CPU, memory, or source/work/checkpoint filesystem provenance",
    )
    require(
        "codec_environment_unset",
        isinstance(codec_environment, Mapping)
        and set(codec_environment) == set(common.CODEC_ENVIRONMENT_VARIABLES)
        and all(codec_environment.get(name) is None for name in common.CODEC_ENVIRONMENT_VARIABLES),
        "codec-affecting environment variables were present in the raw result",
    )

    try:
        current_source_sha256 = common._sha256_file(source)
    except OSError:
        current_source_sha256 = None
    require(
        "source_unchanged_after_run",
        current_source_sha256 == input_record.get("expected_sha256"),
        "source SHA-256 changed during generic benchmark execution",
    )
    manifest_path = pathlib.Path(str(manifest_record.get("path")))
    try:
        current_manifest_sha256 = common._sha256_file(manifest_path)
    except OSError:
        current_manifest_sha256 = None
    require(
        "manifest_unchanged_after_run",
        current_manifest_sha256 == manifest_record.get("expected_sha256"),
        "manifest SHA-256 changed during generic benchmark execution",
    )
    try:
        current_matrix_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    except OSError:
        current_matrix_sha256 = None
    require(
        "matrix_unchanged_after_run",
        current_matrix_sha256 == expected_matrix_sha256,
        "experiment matrix changed during generic benchmark execution",
    )
    current_git = common._git_provenance()
    require(
        "clean_bound_git_after_run",
        current_git.get("commit") == implementation.get("git_commit")
        and current_git.get("dirty") is False,
        "Git commit/tree changed during generic benchmark execution",
    )
    current_implementation = _implementation_binding()
    require(
        "scripts_unchanged_after_run",
        all(current_implementation.get(key) == implementation.get(key) for key in (
            "git_commit", "campaign_runner_sha256", "brevis_benchmarking_sha256",
            "generic_benchmarking_sha256",
        )),
        "campaign runner or generic benchmark script changed during execution",
    )
    for reason in stage_override_reasons:
        failures.append(_generic_validation_failure(
            reason, "campaign stage/resource gate was explicitly overridden",
        ))
    checks["disk_gate"] = dict(disk_gate)
    return checks, failures


def _generic_exception_document(
    *, task_sha256: str, source: pathlib.Path, metadata: Mapping[str, Any], exc: BaseException,
) -> dict[str, Any]:
    """Preserve a truthful raw artifact when the generic harness raises pre-checkpoint."""

    return {
        "schema": {"id": "brevis.generic-campaign-harness-failure", "version": 1},
        "status": "harness_exception",
        "success": False,
        "failure": {"type": type(exc).__name__, "message": str(exc)},
        "source": {"path": str(source)},
        "input_metadata": dict(metadata),
        "campaign_task_semantic_sha256": task_sha256,
    }


def _execute_generic_task(
    *,
    task: Mapping[str, Any],
    task_sha256: str,
    specs: Sequence[common.BaselineSpec],
    configuration: Mapping[str, Any],
    input_record: Mapping[str, Any],
    manifest_record: Mapping[str, Any],
    model: Mapping[str, Any],
    campaign_record: Mapping[str, Any],
    group_record: Mapping[str, Any],
    implementation: Mapping[str, Any],
    matrix_path: pathlib.Path,
    expected_matrix_sha256: str,
    source: pathlib.Path,
    output: pathlib.Path,
    receipt_path: pathlib.Path,
    repository_root: pathlib.Path,
    lock_path: pathlib.Path,
    stage_override_reasons: Sequence[str],
) -> dict[str, Any]:
    """Run one exact generic comparison group and persist its validation receipt."""

    if not _plan_output_safe_during_execution(receipt_path, repository_root):
        raise CampaignError(
            "suggested generic validation receipt would dirty the Brevis repository"
        )
    for path, label in ((output, "raw result"), (receipt_path, "validation receipt")):
        if _path_entry_exists(path):
            raise CampaignError(f"refusing to overwrite existing generic {label}: {path}")

    present_codec_environment = [
        name for name in common.CODEC_ENVIRONMENT_VARIABLES if name in os.environ
    ]
    if present_codec_environment:
        raise CampaignError(
            "generic formal execution requires codec environment variables to be truly "
            "unset; present: " + ", ".join(present_codec_environment)
        )

    expected_registry = [spec.to_dict() for spec in common.BASELINE_SPECS]
    if len(specs) != GENERIC_METHOD_COUNT or [spec.to_dict() for spec in specs] != expected_registry:
        raise CampaignError("generic execution requires the exact ordered 19-method registry")
    registry_sha256 = _semantic_sha256(expected_registry)
    metadata = {
        "model_tag": model["tag"],
        "model_repo": model["repo"],
        "model_revision": model["revision"],
        "manifest": manifest_record["path"],
        "shard": input_record["file"],
        "campaign_id": campaign_record["id"],
        "campaign_comparison_group_id": group_record["id"],
        "campaign_task_semantic_sha256": task_sha256,
        "campaign_experiment_matrix_sha256": expected_matrix_sha256,
        "campaign_runner_sha256": implementation["campaign_runner_sha256"],
        "generic_benchmarking_sha256": implementation["generic_benchmarking_sha256"],
        "generic_registry_sha256": registry_sha256,
    }

    with _exclusive_execution_lock(lock_path) as acquired_lock_path:
        for path, label in ((output, "raw result"), (receipt_path, "validation receipt")):
            if _path_entry_exists(path):
                raise CampaignError(f"refusing to overwrite existing generic {label}: {path}")
        present_codec_environment = [
            name for name in common.CODEC_ENVIRONMENT_VARIABLES if name in os.environ
        ]
        if present_codec_environment:
            raise CampaignError(
                "codec environment changed before the locked generic execution: "
                + ", ".join(present_codec_environment)
            )
        preflight_git = common._git_provenance()
        if not (
            preflight_git.get("commit") == implementation.get("git_commit")
            and preflight_git.get("dirty") is False
        ):
            raise CampaignError(
                "generic formal execution requires the clean Git commit bound by the task"
            )
        locked_implementation = _implementation_binding()
        for key in (
            "git_commit", "campaign_runner_sha256", "brevis_benchmarking_sha256",
            "generic_benchmarking_sha256",
        ):
            if locked_implementation.get(key) != implementation.get(key):
                raise CampaignError(
                    f"current implementation {key} drifted before generic execution"
                )

        disk_gate = _generic_disk_gate(
            source,
            output,
            reserve_fraction=configuration["disk_reserve_fraction"],
            method_count=GENERIC_METHOD_COUNT,
            iterations_per_method=GENERIC_WARMUPS + GENERIC_REPETITIONS,
        )
        if disk_gate.get("passes") is not True:
            raise CampaignError(
                "generic disk-reserve gate failed: projected free space would fall below 30%"
            )
        metadata["campaign_disk_gate"] = disk_gate

        benchmark_exception: BaseException | None = None
        checkpoint_missing_after_return = False
        returned_result: Any = None
        try:
            returned_result = common.benchmark_file(
                source,
                specs=tuple(common.BASELINE_SPECS),
                warmups=GENERIC_WARMUPS,
                repetitions=GENERIC_REPETITIONS,
                work_dir=output.parent,
                keep_artifacts=False,
                timeout_seconds=configuration["timeout_seconds_per_process"],
                version_probe_timeout_seconds=configuration[
                    "version_probe_timeout_seconds"
                ],
                schedule_seed=GENERIC_SCHEDULE_SEED,
                input_metadata=metadata,
                expected_source_size_bytes=input_record["expected_size_bytes"],
                expected_source_sha256=input_record["expected_sha256"],
                manifest_path=manifest_record["path"],
                expected_manifest_sha256=manifest_record["expected_sha256"],
                checkpoint_path=output,
                force_checkpoint=False,
            )
        except BaseException as exc:  # Preserve evidence even on interruption.
            benchmark_exception = exc
            if not _path_entry_exists(output):
                common.write_json(
                    output,
                    _generic_exception_document(
                        task_sha256=task_sha256,
                        source=source,
                        metadata=metadata,
                        exc=exc,
                    ),
                    force=False,
                )
        else:
            if not _path_entry_exists(output):
                checkpoint_missing_after_return = True
                if isinstance(returned_result, Mapping):
                    common.write_json(output, dict(returned_result), force=False)
                else:
                    common.write_json(
                        output,
                        _generic_exception_document(
                            task_sha256=task_sha256,
                            source=source,
                            metadata=metadata,
                            exc=CampaignError(
                                "generic harness returned no mapping and wrote no checkpoint"
                            ),
                        ),
                        force=False,
                    )

        raw_bytes: bytes | None = None
        raw_result: Mapping[str, Any] | None = None
        failures: list[dict[str, str]] = []
        checks: dict[str, Any] = {}
        postvalidation_exception: BaseException | None = None
        post_run_disk_gate = _generic_post_run_disk_gate(disk_gate, output)
        if post_run_disk_gate.get("post_run_passes") is not True:
            failures.append(_generic_validation_failure(
                "post_run_disk_reserve",
                "actual post-run free space fell below the bound 30% reserve or could not "
                "be measured",
            ))
        post_run_codec_environment = {
            name: {
                "present": name in os.environ,
                "value": os.environ.get(name),
            }
            for name in common.CODEC_ENVIRONMENT_VARIABLES
        }
        if any(
            record["present"] for record in post_run_codec_environment.values()
        ):
            failures.append(_generic_validation_failure(
                "codec_environment_changed_during_run",
                "a codec-affecting environment variable became present during execution",
            ))
        try:
            raw_bytes = output.read_bytes()
            decoded = json.loads(raw_bytes)
            if not isinstance(decoded, Mapping):
                raise ValueError("top-level JSON value is not an object")
            raw_result = decoded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            failures.append(_generic_validation_failure(
                "raw_result_unreadable", f"cannot validate persisted raw result: {exc}",
            ))

        if raw_result is not None:
            try:
                checks, validation_failures = _validate_generic_result(
                    raw_result,
                    task_sha256=task_sha256,
                    specs=specs,
                    configuration=configuration,
                    input_record=input_record,
                    manifest_record=manifest_record,
                    model=model,
                    campaign_record=campaign_record,
                    group_record=group_record,
                    implementation=implementation,
                    matrix_path=matrix_path,
                    expected_matrix_sha256=expected_matrix_sha256,
                    source=source,
                    output=output,
                    disk_gate=disk_gate,
                    stage_override_reasons=stage_override_reasons,
                )
                failures.extend(validation_failures)
            except BaseException as exc:
                postvalidation_exception = exc
                failures.append(_generic_validation_failure(
                    "postvalidation_exception",
                    f"generic result postvalidation raised {type(exc).__name__}: {exc}",
                ))
        if benchmark_exception is not None:
            failures.append(_generic_validation_failure(
                "benchmark_exception",
                f"generic harness raised {type(benchmark_exception).__name__}: "
                f"{benchmark_exception}",
            ))
        if checkpoint_missing_after_return:
            failures.append(_generic_validation_failure(
                "harness_checkpoint_missing",
                "generic harness returned without publishing its required checkpoint",
            ))
        checks["post_run_disk_gate"] = post_run_disk_gate
        checks["post_run_codec_environment"] = post_run_codec_environment

        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest() if raw_bytes is not None else None
        result_only_failure_codes = {"method_pre_run_failure"}
        formal_gate_only_failure_codes = {
            "post_run_disk_reserve",
            "codec_environment_changed_during_run",
            *stage_override_reasons,
        }
        contract_failures = [
            failure for failure in failures
            if failure.get("code") not in (
                result_only_failure_codes | formal_gate_only_failure_codes
            )
        ]
        method_outcomes = checks.get("methods")
        result_success = (
            checks.get("top_schema") is True
            and checks.get("top_status") is True
            and checks.get("method_order") is True
            and isinstance(method_outcomes, list)
            and len(method_outcomes) == GENERIC_METHOD_COUNT
            and all(
                isinstance(outcome, Mapping) and outcome.get("valid") is True
                for outcome in method_outcomes
            )
        )
        attempt_contract_valid = not contract_failures
        formal_eligible = (
            attempt_contract_valid
            and result_success
            and not any(
                failure.get("code") in formal_gate_only_failure_codes
                for failure in failures
            )
        )
        receipt = {
            "schema": {"id": "brevis.generic-campaign-validation", "version": 1},
            "status": "complete",
            "task_semantic_sha256": task_sha256,
            "task_semantics": dict(_object(task.get("task_semantics"), "task semantics")),
            "raw_result": {
                "path": str(output),
                "present": raw_bytes is not None,
                "size_bytes": len(raw_bytes) if raw_bytes is not None else None,
                "sha256": raw_sha256,
                "schema": raw_result.get("schema") if raw_result is not None else None,
                "status": raw_result.get("status") if raw_result is not None else None,
                "preserved_without_selective_omission": True,
            },
            "receipt_path": str(receipt_path),
            "attempt_contract_valid": attempt_contract_valid,
            "result_success": result_success,
            "formal_eligible": formal_eligible,
            "postvalidation": {
                "status_code": "passed" if formal_eligible else "failed",
                "checks": checks,
                "failures": failures,
            },
            "method_outcomes": method_outcomes if isinstance(method_outcomes, list) else [],
            "disk_gate": post_run_disk_gate,
            "implementation_binding": dict(implementation),
            "generic_registry": {
                "id": GENERIC_REGISTRY_ID,
                "count": len(specs),
                "sha256": registry_sha256,
            },
            "exclusive_lock_path": str(acquired_lock_path),
            "stage_override_reasons": list(stage_override_reasons),
            "benchmark_exception": (
                {
                    "type": type(benchmark_exception).__name__,
                    "message": str(benchmark_exception),
                }
                if benchmark_exception is not None else None
            ),
        }
        common.write_json(receipt_path, receipt, force=False)
        if benchmark_exception is not None and not isinstance(benchmark_exception, Exception):
            raise benchmark_exception
        if (
            postvalidation_exception is not None
            and not isinstance(postvalidation_exception, Exception)
        ):
            raise postvalidation_exception
        return receipt


def execute_one(
    plan: Mapping[str, Any], task_sha256: str, *, allow_unmet_stage_gate: bool = False,
) -> dict[str, Any]:
    """Execute exactly one selected task through every formal harness gate."""

    schema = _object(plan.get("schema"), "plan schema")
    if schema.get("id") != PLAN_SCHEMA_ID or schema.get("version") != PLAN_SCHEMA_VERSION:
        raise CampaignError("execute_one requires a campaign-plan schema version 1 document")
    if plan.get("status") != "plan_only_not_results":
        raise CampaignError("execute_one requires an unexecuted plan document")
    if not isinstance(allow_unmet_stage_gate, bool):
        raise CampaignError("allow_unmet_stage_gate must be boolean")
    matrix_record = _object(plan.get("experiment_matrix"), "plan experiment_matrix")
    matrix_path = pathlib.Path(str(matrix_record.get("path")))
    expected_matrix_sha256 = matrix_record.get("sha256")
    if not isinstance(expected_matrix_sha256, str) or SHA256.fullmatch(expected_matrix_sha256) is None:
        raise CampaignError("plan experiment matrix SHA-256 is invalid")
    try:
        actual_matrix_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CampaignError(f"cannot rehash planned experiment matrix {matrix_path}: {exc}") from exc
    if actual_matrix_sha256 != expected_matrix_sha256:
        raise CampaignError("experiment matrix changed after task planning")
    if not isinstance(task_sha256, str) or SHA256.fullmatch(task_sha256) is None:
        raise CampaignError("--execute-one requires one canonical task SHA-256")
    tasks = _array(plan.get("tasks"), "plan.tasks", nonempty=False)
    matches = [task for task in tasks if task.get("task_semantic_sha256") == task_sha256]
    if len(matches) != 1:
        raise CampaignError(
            f"--execute-one must resolve exactly one task; found {len(matches)} for {task_sha256}"
        )
    task = _object(matches[0], "selected task")
    semantics = _object(task.get("task_semantics"), "selected task semantics")
    if _semantic_sha256(semantics) != task_sha256:
        raise CampaignError("selected task semantics do not match its SHA-256")
    if semantics.get("spec_id") != TASK_SEMANTIC_SPEC_ID:
        raise CampaignError("selected task has an unsupported semantic specification")
    if semantics.get("experiment_matrix_sha256") != expected_matrix_sha256:
        raise CampaignError("selected task is bound to a different experiment matrix")

    planner_configuration = _object(
        plan.get("planner_configuration"), "plan planner_configuration",
    )
    selection = _object(plan.get("selection"), "plan selection")
    planner_paths: dict[str, str] = {}
    for key in ("repository_root", "cache_root", "staging_root"):
        value = planner_configuration.get(key)
        if not isinstance(value, str) or not value:
            raise CampaignError(f"plan planner_configuration.{key} must be a nonempty path")
        planner_paths[key] = value
    selected_campaign_ids = _array(selection.get("campaign_ids"), "selection.campaign_ids")
    if not all(isinstance(value, str) and value for value in selected_campaign_ids):
        raise CampaignError("selection.campaign_ids must contain nonempty strings")
    selected_model_tags = selection.get("model_tags")
    if selected_model_tags is not None:
        selected_model_tags = _array(selected_model_tags, "selection.model_tags")
        if not all(isinstance(value, str) and value for value in selected_model_tags):
            raise CampaignError("selection.model_tags must contain nonempty strings")
    regenerated = plan_campaigns(
        matrix_path,
        repository_root=planner_paths["repository_root"],
        cache_root=planner_paths["cache_root"],
        staging_root=planner_paths["staging_root"],
        campaign_ids=selected_campaign_ids,
        model_tags=selected_model_tags,
    )
    regenerated_matches = [
        candidate for candidate in regenerated["tasks"]
        if candidate.get("task_semantic_sha256") == task_sha256
    ]
    if len(regenerated_matches) != 1:
        raise CampaignError(
            "selected task is not generated by the current bound matrix and implementation"
        )
    regenerated_task = _object(regenerated_matches[0], "regenerated selected task")
    if regenerated_task.get("task_semantics") != semantics:
        raise CampaignError("regenerated task semantics disagree with the supplied plan")
    task = regenerated_task
    semantics = _object(task["task_semantics"], "regenerated task semantics")

    implementation = _object(
        semantics.get("implementation_binding"), "task implementation binding",
    )
    plan_implementation = _object(
        plan.get("implementation_binding"), "plan implementation binding",
    )
    if dict(plan_implementation) != dict(implementation):
        raise CampaignError("plan implementation binding disagrees with the selected task")
    current_implementation = _implementation_binding()
    for key in (
        "git_commit", "campaign_runner_sha256", "brevis_benchmarking_sha256",
        "generic_benchmarking_sha256",
    ):
        if implementation.get(key) != current_implementation.get(key):
            raise CampaignError(f"current implementation {key} drifted from the task binding")
    if task.get("implementation_binding") != dict(implementation):
        raise CampaignError("task implementation declaration disagrees with its semantics")

    expected_machine = semantics.get("machine_binding")
    if expected_machine is not None:
        expected_machine = _object(expected_machine, "task machine binding")
        if _machine_binding() != dict(expected_machine):
            raise CampaignError("current machine drifted from the scaling task binding")
        if task.get("machine_binding") != dict(expected_machine):
            raise CampaignError("task machine declaration disagrees with its semantics")
    executor = _object(task.get("executor"), "selected task executor")
    semantic_executor = _object(
        semantics.get("executor"), "selected task semantic executor",
    )
    if (
        executor.get("supported") != semantic_executor.get("supported")
        or executor.get("status_code") != semantic_executor.get("status_code")
        or executor.get("implementation") != semantic_executor.get("implementation")
        or executor.get("formal_gates") != semantic_executor.get("formal_gates")
        or executor.get("exclusive_execution_policy")
        != semantic_executor.get("exclusive_execution_policy")
    ):
        raise CampaignError("selected task executor disagrees with its hashed semantics")
    policy = _object(
        semantic_executor.get("exclusive_execution_policy"),
        "selected task exclusive execution policy",
    )
    if (
        policy.get("policy_id") != EXCLUSIVE_EXECUTION_POLICY_ID
        or policy.get("required") is not True
        or policy.get("acquisition") != "exclusive_nonblocking"
        or not isinstance(policy.get("lock_path"), str)
    ):
        raise CampaignError("selected task has an unsupported exclusive execution policy")
    if executor.get("supported") is not True or executor.get("status_code") != "ready":
        raise CampaignError(
            "selected task is machine-labelled unsupported and cannot be executed: "
            + str(executor.get("reason"))
        )
    campaign_kind = semantics.get("campaign_kind")
    is_brevis = campaign_kind in SUPPORTED_BREVIS_KINDS
    is_generic = campaign_kind in SUPPORTED_GENERIC_KINDS
    if is_brevis == is_generic:
        raise CampaignError("selected task does not identify exactly one supported harness family")
    expected_formal_gates = BREVIS_FORMAL_GATES if is_brevis else GENERIC_FORMAL_GATES
    expected_executor_kind = (
        "brevis_system_harness" if is_brevis else "generic_baseline_harness"
    )
    expected_executor_implementation = (
        "brevis_benchmarking.benchmark_file"
        if is_brevis else "benchmarking.benchmark_file"
    )
    if (
        executor.get("formal_gates") != expected_formal_gates
        or executor.get("implementation") != expected_executor_implementation
        or semantic_executor.get("kind") != expected_executor_kind
    ):
        raise CampaignError("selected task does not require every formal execution gate")

    brevis_spec_values = task.get("brevis_specs")
    semantic_brevis_spec_values = semantics.get("brevis_specs")
    baseline_spec_values = task.get("baseline_specs")
    semantic_baseline_spec_values = semantics.get("baseline_specs")
    if (
        not isinstance(brevis_spec_values, list)
        or brevis_spec_values != semantic_brevis_spec_values
        or not isinstance(baseline_spec_values, list)
        or baseline_spec_values != semantic_baseline_spec_values
    ):
        raise CampaignError("selected comparison-group specs disagree with hashed semantics")
    if is_brevis:
        if not brevis_spec_values or baseline_spec_values:
            raise CampaignError("Brevis task must contain only nonempty Brevis specs")
        brevis_specs = tuple(
            _reconstruct_spec(_object(value, "selected task BrevisSpec"))
            for value in brevis_spec_values
        )
        generic_specs: tuple[common.BaselineSpec, ...] = ()
    else:
        expected_baseline_values = [spec.to_dict() for spec in common.BASELINE_SPECS]
        if (
            brevis_spec_values
            or baseline_spec_values != expected_baseline_values
            or len(baseline_spec_values) != GENERIC_METHOD_COUNT
            or executor.get("generic_registry") != expected_baseline_values
        ):
            raise CampaignError("generic task must contain the exact ordered 19-method registry")
        brevis_specs = ()
        generic_specs = tuple(common.BASELINE_SPECS)
    input_record = _object(task.get("input"), "selected task input")
    manifest_record = _object(task.get("manifest"), "selected task manifest")
    configuration = _object(task.get("configuration"), "selected task configuration")
    model = _object(task.get("model"), "selected task model")
    semantic_input = _object(semantics.get("input"), "selected task semantic input")
    if (
        input_record.get("file") != semantic_input.get("file")
        or input_record.get("expected_size_bytes") != semantic_input.get("bytes")
        or input_record.get("expected_sha256") != semantic_input.get("sha256")
    ):
        raise CampaignError("selected task input declaration disagrees with its hashed semantics")
    semantic_model = _object(semantics.get("model"), "selected task semantic model")
    if any(model.get(key) != semantic_model.get(key) for key in ("tag", "repo", "revision", "scope")):
        raise CampaignError("selected task model declaration disagrees with its hashed semantics")
    for key in (
        "jobs", "calibration_tensors", "warmups", "repetitions",
        "timeout_seconds_per_process", "version_probe_timeout_seconds",
        "schedule_seed", "disk_reserve_fraction",
    ):
        if configuration.get(key) != semantics.get(key):
            raise CampaignError(
                f"selected task configuration {key} disagrees with its hashed semantics"
            )
    manifest_plan_record = _object(plan.get("model_manifest"), "plan model_manifest")
    if (
        manifest_record.get("expected_sha256") != semantics.get("model_manifest_sha256")
        or manifest_record.get("expected_sha256") != manifest_plan_record.get("actual_sha256")
    ):
        raise CampaignError("selected task manifest binding disagrees with its hashed semantics")
    campaign_record = _object(task.get("campaign"), "selected task campaign")
    group_record = _object(task.get("comparison_group"), "selected comparison group")
    if (
        campaign_record.get("id") != semantics.get("campaign_id")
        or campaign_record.get("kind") != semantics.get("campaign_kind")
        or campaign_record.get("stage") != semantics.get("campaign_stage")
        or campaign_record.get("resource_gate") != semantics.get("campaign_resource_gate")
        or group_record.get("id") != semantics.get("comparison_group_id")
        or group_record.get("schedule") != semantics.get("comparison_schedule")
        or task.get("dimension") != semantics.get("dimension")
    ):
        raise CampaignError("selected task dimensions disagree with its hashed semantics")
    stage = _integer(campaign_record.get("stage"), "selected campaign stage", minimum=1)
    extra_ineligibility: tuple[str, ...] = ()
    if stage > 1:
        if not allow_unmet_stage_gate:
            raise CampaignError(
                "campaign stage/resource gate has no completion ledger; stage >1 is rejected "
                "unless --allow-unmet-stage-gate is explicitly supplied"
            )
        extra_ineligibility = ("campaign_stage_gate_not_proven",)
    output = pathlib.Path(str(task.get("suggested_output_path")))
    repository_root = pathlib.Path(planner_paths["repository_root"])
    if not _plan_output_safe_during_execution(
        output, repository_root,
    ):
        raise CampaignError(
            "suggested task output would dirty the Brevis repository during formal "
            "execution; use an ignored staging root or a path outside the repository"
        )
    if _path_entry_exists(output):
        raise CampaignError(f"refusing to overwrite existing task result: {output}")
    source = pathlib.Path(str(input_record.get("path")))
    if not source.is_file():
        raise CampaignError(f"selected task input is unavailable: {source}")

    lock_path = pathlib.Path(policy["lock_path"])
    if is_generic:
        if any((
            configuration.get("jobs") != 1,
            configuration.get("warmups") != GENERIC_WARMUPS,
            configuration.get("repetitions") != GENERIC_REPETITIONS,
            configuration.get("schedule_seed") != GENERIC_SCHEDULE_SEED,
            configuration.get("disk_reserve_fraction")
            != GENERIC_DISK_RESERVE_FRACTION,
            configuration.get("version_probe_timeout_seconds")
            != common.DEFAULT_VERSION_PROBE_TIMEOUT_SECONDS,
        )):
            raise CampaignError("generic task configuration drifted from the formal contract")
        declared_receipt_path = task.get("suggested_receipt_path")
        expected_receipt_path = output.with_suffix(".validation.json")
        if (
            not isinstance(declared_receipt_path, str)
            or pathlib.Path(declared_receipt_path) != expected_receipt_path
        ):
            raise CampaignError("generic task validation receipt path is missing or inconsistent")
        return _execute_generic_task(
            task=task,
            task_sha256=task_sha256,
            specs=generic_specs,
            configuration=configuration,
            input_record=input_record,
            manifest_record=manifest_record,
            model=model,
            campaign_record=campaign_record,
            group_record=group_record,
            implementation=implementation,
            matrix_path=matrix_path,
            expected_matrix_sha256=expected_matrix_sha256,
            source=source,
            output=output,
            receipt_path=expected_receipt_path,
            repository_root=repository_root,
            lock_path=lock_path,
            stage_override_reasons=extra_ineligibility,
        )

    with _exclusive_execution_lock(lock_path) as lock_path:
        result = system.benchmark_file(
            source,
            specs=brevis_specs,
            warmups=configuration["warmups"],
            repetitions=configuration["repetitions"],
            jobs=configuration["jobs"],
            calibration_tensors=configuration["calibration_tensors"],
            timeout_seconds=configuration["timeout_seconds_per_process"],
            schedule_seed=configuration["schedule_seed"],
            input_metadata={
                "model_tag": model["tag"],
                "model_repo": model["repo"],
                "model_revision": model["revision"],
                "manifest": manifest_record["path"],
                "shard": input_record["file"],
                "campaign_id": campaign_record["id"],
                "campaign_comparison_group_id": group_record["id"],
                "campaign_task_semantic_sha256": task_sha256,
            },
            expected_source_size_bytes=input_record["expected_size_bytes"],
            expected_source_sha256=input_record["expected_sha256"],
            manifest_path=manifest_record["path"],
            expected_manifest_sha256=manifest_record["expected_sha256"],
            checkpoint_path=output,
            force_checkpoint=False,
            require_declared_integrity=True,
            require_clean_git=True,
            build_binary=True,
            additional_formal_ineligibility_reasons=extra_ineligibility,
            disk_reserve_fraction=configuration["disk_reserve_fraction"],
        )

    end_implementation = _implementation_binding()
    for key in (
        "git_commit", "campaign_runner_sha256", "brevis_benchmarking_sha256",
    ):
        if end_implementation.get(key) != implementation.get(key):
            raise CampaignError(f"current implementation {key} changed during execution")

    provenance = _object(result.get("provenance"), "benchmark result provenance")
    result_git = _object(provenance.get("git"), "benchmark result Git provenance")
    result_harness = _object(provenance.get("harness"), "benchmark result harness provenance")
    if result_git.get("commit") != implementation.get("git_commit"):
        raise CampaignError("benchmark result commit disagrees with the task binding")
    if result_harness.get("sha256") != implementation.get("brevis_benchmarking_sha256"):
        raise CampaignError("benchmark result harness hash disagrees with the task binding")
    result_metadata = _object(result.get("input_metadata"), "benchmark result input metadata")
    if result_metadata.get("campaign_task_semantic_sha256") != task_sha256:
        raise CampaignError("benchmark result lost its task semantic binding")
    classification = _object(
        result.get("run_classification"), "benchmark result run classification",
    )
    reasons = classification.get("formal_ineligibility_reasons")
    if not isinstance(reasons, list):
        raise CampaignError("benchmark result has no formal ineligibility reasons array")
    if extra_ineligibility:
        if classification.get("formal_eligible") is not False or not set(extra_ineligibility) <= set(reasons):
            raise CampaignError("stage-gate override did not downgrade the result from formal")
    elif classification.get("formal_eligible") is not True:
        raise CampaignError("stage-1 execution unexpectedly failed formal eligibility gates")
    return {
        "schema": {"id": "brevis.campaign-execution-receipt", "version": 1},
        "task_semantic_sha256": task_sha256,
        "result_path": str(output),
        "result_status": result.get("status"),
        "result_success": result.get("success"),
        "formal_eligible": classification.get("formal_eligible"),
        "exclusive_lock_path": str(lock_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=pathlib.Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--repository-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--cache-root", type=pathlib.Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--staging-root", type=pathlib.Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--campaign", action="append")
    parser.add_argument("--model-tag", action="append")
    parser.add_argument("--plan-output", type=pathlib.Path)
    parser.add_argument("--execute-one", metavar="TASK_SHA256")
    parser.add_argument(
        "--allow-unmet-stage-gate", action="store_true",
        help="run stage >1 without a completion ledger and mark the result nonformal",
    )
    return parser


def _plan_output_safe_during_execution(
    path: pathlib.Path, repository_root: pathlib.Path = ROOT,
) -> bool:
    """Return whether writing this path cannot dirty the declared Git repository."""

    resolved = path.resolve(strict=False)
    repository = repository_root.resolve(strict=False)
    try:
        resolved.relative_to(repository)
    except ValueError:
        return True
    git = shutil.which("git")
    if git is None:
        return False
    completed = subprocess.run(
        [git, "-C", str(repository), "check-ignore", "-q", "--", str(resolved)],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = plan_campaigns(
            args.experiments,
            repository_root=args.repository_root,
            cache_root=args.cache_root,
            staging_root=args.staging_root,
            campaign_ids=args.campaign,
            model_tags=args.model_tag,
        )
        if (
            args.execute_one is not None
            and args.plan_output is not None
            and not _plan_output_safe_during_execution(
                args.plan_output, args.repository_root,
            )
        ):
            raise CampaignError(
                "--plan-output would dirty the Brevis repository before its formal clean-tree "
                "gate; omit it, write outside the repository, or use an ignored cache path"
            )
        if args.plan_output is not None:
            common.write_json(args.plan_output, plan, force=False)
        if args.execute_one is None:
            if args.plan_output is None:
                print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        receipt = execute_one(
            plan, args.execute_one,
            allow_unmet_stage_gate=args.allow_unmet_stage_gate,
        )
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 0 if receipt["formal_eligible"] else 1
    except (OSError, ValueError, CampaignError, system.BenchmarkError) as exc:
        print(f"campaign runner: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
