#!/usr/bin/env python3
"""Download the canonical model files for the Brevis benchmark corpus."""

from __future__ import annotations

import os

# huggingface_hub reads Xet settings at import time.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from benchmark_corpus import CHECKPOINTS, CheckpointSpec as Checkpoint

MODEL_SUPPORT_FILES = (
    "config.json",
    "generation_config.json",
    "model_index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)

try:
    from huggingface_hub import HfApi, get_token, hf_hub_download
except ImportError:
    sys.exit(
        "Install the downloader first:\n"
        "  python3 -m pip install -U huggingface_hub hf-xet"
    )


@dataclass(frozen=True)
class WeightFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class DownloadPlan:
    checkpoint: Checkpoint
    revision: str
    directory: Path
    index_file: str | None
    support_files: tuple[str, ...]
    weights: tuple[WeightFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(file.size for file in self.weights)


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    names = ", ".join(item.name for item in CHECKPOINTS)
    parser = argparse.ArgumentParser(
        description=(
            "Download the canonical safetensors shards and model configuration "
            "needed by the benchmark."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination directory on a large local filesystem.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAME",
        help=f"Subset to download. Available names: {names}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent files per checkpoint (default: 8).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Resolve current main instead of using the pinned revisions.",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Read every downloaded byte once more and verify SHA-256.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve revisions, files, and required space without downloading weights.",
    )
    return parser.parse_args()


def select_checkpoints(names: list[str] | None) -> tuple[Checkpoint, ...]:
    if not names:
        return CHECKPOINTS
    by_name = {item.name: item for item in CHECKPOINTS}
    unknown = sorted(set(names) - by_name.keys())
    if unknown:
        sys.exit(f"Unknown model name(s): {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def file_metadata(sibling: object) -> WeightFile:
    lfs = getattr(sibling, "lfs", None)
    sha256 = getattr(lfs, "sha256", None)
    size = getattr(sibling, "size", None)
    if size is None or sha256 is None:
        raise RuntimeError(f"Missing LFS metadata for {sibling.rfilename}")
    return WeightFile(sibling.rfilename, size, sha256)


def resolve_plan(
    api: HfApi,
    checkpoint: Checkpoint,
    output: Path,
    token: str | None,
    latest: bool,
) -> DownloadPlan:
    requested_revision = "main" if latest else checkpoint.revision
    info = api.model_info(
        checkpoint.repo_id,
        revision=requested_revision,
        files_metadata=True,
        token=token,
    )
    if info.gated and not token:
        raise PermissionError(
            f"{checkpoint.repo_id} is gated and no Hugging Face token is configured"
        )
    revision = info.sha
    directory = output / checkpoint.name
    siblings = {item.rfilename: item for item in info.siblings or ()}

    if checkpoint.single_file:
        names = [checkpoint.single_file]
        index_file = None
    else:
        index_file = checkpoint.index_file
        if index_file not in siblings:
            raise RuntimeError(f"{checkpoint.repo_id} has no {index_file}")
        index_path = hf_hub_download(
            checkpoint.repo_id,
            index_file,
            revision=revision,
            local_dir=directory,
            token=token,
        )
        index = json.loads(Path(index_path).read_text())
        names = sorted(set(index["weight_map"].values()))

    support_files = tuple(
        name
        for name in MODEL_SUPPORT_FILES
        if name in siblings
    )
    required_support = set(checkpoint.required_support_files)
    if not checkpoint.single_file:
        required_support.add("config.json")
    missing_support = sorted(required_support - set(support_files))
    if missing_support:
        raise RuntimeError(
            f"{checkpoint.repo_id} is missing support files: {missing_support}"
        )
    missing = [name for name in names if name not in siblings]
    if missing:
        raise RuntimeError(
            f"{checkpoint.repo_id} index references missing files: {missing}"
        )
    weights = tuple(file_metadata(siblings[name]) for name in names)
    return DownloadPlan(
        checkpoint,
        revision,
        directory,
        index_file,
        support_files,
        weights,
    )


def missing_bytes(plan: DownloadPlan) -> int:
    return sum(
        item.size
        for item in plan.weights
        if not (plan.directory / item.path).is_file()
        or (plan.directory / item.path).stat().st_size != item.size
    )


def download_weight(
    plan: DownloadPlan,
    item: WeightFile,
    token: str | None,
    retries: int = 3,
) -> Path:
    for attempt in range(1, retries + 1):
        try:
            path = Path(
                hf_hub_download(
                    plan.checkpoint.repo_id,
                    item.path,
                    revision=plan.revision,
                    local_dir=plan.directory,
                    token=token,
                )
            )
            if path.stat().st_size != item.size:
                raise RuntimeError(
                    f"{item.path}: expected {item.size} bytes, got {path.stat().st_size}"
                )
            return path
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def download_support_files(
    plan: DownloadPlan,
    token: str | None,
) -> None:
    for name in plan.support_files:
        hf_hub_download(
            plan.checkpoint.repo_id,
            name,
            revision=plan.revision,
            local_dir=plan.directory,
            token=token,
        )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    buffer = bytearray(16 * 1024 * 1024)
    view = memoryview(buffer)
    with path.open("rb", buffering=0) as source:
        while count := source.readinto(buffer):
            digest.update(view[:count])
    return digest.hexdigest()


def verify_hashes(plan: DownloadPlan, workers: int) -> None:
    def verify(item: WeightFile) -> None:
        actual = hash_file(plan.directory / item.path)
        if actual != item.sha256:
            raise RuntimeError(
                f"{plan.checkpoint.name}/{item.path}: SHA-256 mismatch "
                f"(expected {item.sha256}, got {actual})"
            )

    with ThreadPoolExecutor(max_workers=min(workers, 4)) as pool:
        futures = [pool.submit(verify, item) for item in plan.weights]
        for future in as_completed(futures):
            future.result()


def write_manifest(plan: DownloadPlan, sha256_verified: bool) -> dict[str, object]:
    manifest = {
        "schema_version": 1,
        "name": plan.checkpoint.name,
        "repo_id": plan.checkpoint.repo_id,
        "revision": plan.revision,
        "index_file": plan.index_file,
        "support_files": list(plan.support_files),
        "source_bytes": plan.total_bytes,
        "sha256_verified": sha256_verified,
        "weights": [asdict(item) for item in plan.weights],
    }
    path = plan.directory / "download-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        sys.exit("--workers must be at least 1")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    token = get_token()
    checkpoints = select_checkpoints(args.models)
    api = HfApi(token=token)

    plans = []
    for checkpoint in checkpoints:
        print(f"Resolving {checkpoint.name} ({checkpoint.repo_id})...")
        try:
            plans.append(
                resolve_plan(api, checkpoint, output, token, args.latest)
            )
        except Exception as exc:
            sys.exit(
                f"Cannot resolve {checkpoint.repo_id}: {exc}\n"
                "For gated models, accept the license and run `hf auth login` "
                "or set HF_TOKEN."
            )

    required = sum(missing_bytes(plan) for plan in plans)
    total = sum(plan.total_bytes for plan in plans)
    free = shutil.disk_usage(output).free
    print(
        f"\nCorpus: {len(plans)} checkpoints, {human_bytes(total)} source weights\n"
        f"Missing locally: {human_bytes(required)}\n"
        f"Free space: {human_bytes(free)}"
    )
    for plan in plans:
        print(
            f"  {plan.checkpoint.name:30} "
            f"{len(plan.weights):3} file(s)  {human_bytes(plan.total_bytes):>10}  "
            f"{plan.revision}"
        )

    if args.dry_run:
        return 0
    if free < required:
        sys.exit(
            f"Not enough free space: need {human_bytes(required)}, "
            f"have {human_bytes(free)}"
        )

    corpus = []
    for plan in plans:
        print(
            f"\nDownloading {plan.checkpoint.name}: "
            f"{len(plan.weights)} file(s), {human_bytes(plan.total_bytes)}"
        )
        download_support_files(plan, token)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(download_weight, plan, item, token)
                for item in plan.weights
            ]
            for future in as_completed(futures):
                future.result()

        if args.verify_sha256:
            print(f"Verifying SHA-256 for {plan.checkpoint.name}...")
            verify_hashes(plan, args.workers)
        corpus.append(write_manifest(plan, args.verify_sha256))

    root_manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_bytes": sum(item["source_bytes"] for item in corpus),
        "checkpoints": corpus,
    }
    (output / "corpus-manifest.json").write_text(
        json.dumps(root_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nDone. Manifest: {output / 'corpus-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
