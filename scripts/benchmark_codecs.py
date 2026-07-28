#!/usr/bin/env python3
"""File adapters for benchmark codecs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024


def snappy_file(
    source: Path,
    output: Path,
    _threads: int,
    *,
    decompress: bool,
) -> None:
    try:
        import snappy
    except ImportError as exc:
        raise SystemExit("install python-snappy") from exc

    codec = snappy.StreamDecompressor() if decompress else snappy.StreamCompressor()
    transform = codec.decompress if decompress else codec.add_chunk
    with source.open("rb") as reader, output.open("wb") as writer:
        while chunk := reader.read(CHUNK_BYTES):
            writer.write(transform(chunk))


def libdeflate_file(
    source: Path,
    output: Path,
    _threads: int,
    *,
    compression_level: int,
    decompress: bool,
) -> None:
    options = ("-d",) if decompress else (f"-{compression_level}",)
    with output.open("wb") as writer:
        subprocess.run(
            ["libdeflate-gzip", "-q", *options, "-c", str(source)],
            stdout=writer,
            check=True,
        )


def zipnn_compress(source: Path, output: Path, threads: int) -> None:
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
        from zipnn import ZipNN
        from zipnn.util_header import EnumFormat
        from zipnn.util_safetensors import (
            COMPRESSED_DTYPE,
            METADATA_KEY,
            build_compressed_tensor_info,
        )
        from zipnn.util_torch import zipnn_is_floating_point
    except ImportError as exc:
        raise SystemExit("install zipnn, torch, and safetensors") from exc

    tensors = {}
    compressed_info = {}
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for name in checkpoint.keys():
            tensor = checkpoint.get_tensor(name)
            if not zipnn_is_floating_point(
                EnumFormat.TORCH.value,
                tensor,
                tensor.dtype,
            ):
                tensors[name] = tensor
                continue

            codec = ZipNN(
                input_format="torch",
                bytearray_dtype=tensor.dtype,
                method="HUFFMAN",
                threads=threads,
            )
            encoded = codec.compress(tensor)
            if len(encoded) >= tensor.element_size() * tensor.nelement():
                tensors[name] = tensor
                continue
            tensors[name] = torch.frombuffer(
                bytearray(encoded),
                dtype=COMPRESSED_DTYPE,
            )
            compressed_info[name] = build_compressed_tensor_info(tensor)

        metadata = dict(checkpoint.metadata() or {})

    metadata[METADATA_KEY] = json.dumps(compressed_info)
    save_file(tensors, output, metadata)


def zipnn_decompress(source: Path, output: Path, threads: int) -> None:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
        from zipnn import ZipNN
        from zipnn.util_safetensors import (
            COMPRESSED_DTYPE,
            COMPRESSION_METHOD,
            METADATA_KEY,
            get_compressed_tensors_metadata,
        )
    except ImportError as exc:
        raise SystemExit("install zipnn, torch, and safetensors") from exc

    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        metadata = checkpoint.metadata()
        compressed_info = get_compressed_tensors_metadata(metadata)
        codec = ZipNN(
            input_format="torch",
            bytearray_dtype=COMPRESSED_DTYPE,
            method=COMPRESSION_METHOD,
            threads=threads,
        )
        for name in checkpoint.keys():
            tensor = checkpoint.get_tensor(name)
            tensors[name] = (
                codec.decompress(tensor.contiguous().numpy())
                if name in compressed_info
                else tensor
            )
        if metadata:
            metadata.pop(METADATA_KEY, None)

    save_file(tensors, output, metadata)


def package_version(package: str) -> None:
    print(importlib.metadata.version(package))


def libdeflate_version() -> None:
    subprocess.run(["libdeflate-gzip", "-V"], check=True)


@dataclass(frozen=True)
class CodecAdapter:
    compress: Callable[[Path, Path, int], None]
    decompress: Callable[[Path, Path, int], None]
    version: Callable[[], None]


CODECS = {
    "libdeflate-1": CodecAdapter(
        partial(libdeflate_file, compression_level=1, decompress=False),
        partial(libdeflate_file, compression_level=1, decompress=True),
        libdeflate_version,
    ),
    "snappy": CodecAdapter(
        partial(snappy_file, decompress=False),
        partial(snappy_file, decompress=True),
        partial(package_version, "python-snappy"),
    ),
    "zipnn": CodecAdapter(
        zipnn_compress,
        zipnn_decompress,
        partial(package_version, "zipnn"),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("codec", choices=CODECS)
    parser.add_argument("operation", choices=("compress", "decompress", "version"))
    parser.add_argument("source", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    codec = CODECS[args.codec]

    if args.operation == "version":
        codec.version()
        return
    if args.source is None or args.output is None or args.threads < 1:
        parser.error("compress/decompress require source, output, and positive threads")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    operation = codec.compress if args.operation == "compress" else codec.decompress
    operation(args.source, args.output, args.threads)


if __name__ == "__main__":
    main()
