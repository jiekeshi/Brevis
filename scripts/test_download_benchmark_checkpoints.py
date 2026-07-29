import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_corpus as corpus
import download_benchmark_checkpoints as downloader
import run_benchmarks as benchmark


def sibling(path: str, size: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        rfilename=path,
        size=size,
        lfs=SimpleNamespace(sha256=(path.encode().hex() + "0" * 64)[:64]),
    )


class FakeApi:
    def __init__(self, revision: str, siblings: list[SimpleNamespace]):
        self.info = SimpleNamespace(
            gated=False,
            sha=revision,
            siblings=siblings,
        )

    def model_info(self, *args, **kwargs) -> SimpleNamespace:
        return self.info


class CheckpointDownloaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_corpus_v2_appends_without_replacing_paper_rows(self):
        paper_names = tuple(item.name for item in corpus.PAPER_V1_CHECKPOINTS)
        v2_names = tuple(item.name for item in corpus.CORPUS_V2_CHECKPOINTS)

        self.assertEqual(10, len(paper_names))
        self.assertEqual(paper_names, v2_names[: len(paper_names)])
        self.assertIn("whisper-large-v3-f16", v2_names)
        self.assertIn("sdxl-base-1.0-f16", v2_names)
        self.assertEqual(
            (
                "voxtral-mini-3b-2507-bf16",
                "qwen-image-bf16",
            ),
            v2_names[len(paper_names) :],
        )
        self.assertEqual(
            corpus.PAPER_V1_CHECKPOINTS,
            downloader.select_checkpoints(None, "paper-v1"),
        )
        self.assertEqual(
            corpus.CORPUS_V2_EXTENSIONS,
            downloader.select_checkpoints(None, "extensions"),
        )

    def test_harness_corpus_v2_selects_all_twelve_frozen_manifests(self):
        models = self.root / "models"
        for checkpoint in corpus.CORPUS_V2_CHECKPOINTS:
            directory = models / checkpoint.name
            directory.mkdir(parents=True)
            weight = directory / "fixture.safetensors"
            weight.write_bytes(b"\0" * 8)
            (directory / "download-manifest.json").write_text(
                json.dumps(
                    {
                        "name": checkpoint.name,
                        "repo_id": checkpoint.repo_id,
                        "revision": checkpoint.revision,
                        "weights": [{"path": weight.name}],
                    }
                )
            )

        loaded = benchmark.load_corpus(
            models,
            names=None,
            corpus_preset="corpus-v2",
        )

        self.assertEqual(
            {item.name for item in corpus.CORPUS_V2_CHECKPOINTS},
            {item.name for item in loaded},
        )

    def test_component_index_paths_are_relative_to_the_index(self):
        names = downloader.indexed_weight_names(
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            {
                "weight_map": {
                    "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                    "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
                }
            },
            {
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            },
        )

        self.assertEqual(
            {
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
                "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
            },
            names,
        )

    def test_component_index_rejects_parent_traversal(self):
        with self.assertRaisesRegex(RuntimeError, "unsafe shard path"):
            downloader.indexed_weight_names(
                "transformer/model.safetensors.index.json",
                {"weight_map": {"a": "../outside.safetensors"}},
                set(),
            )

    def test_qwen_image_plan_combines_two_indexes_and_vae(self):
        checkpoint = corpus.CHECKPOINT_BY_NAME["qwen-image-bf16"]
        text_shard = "text_encoder/model-00001-of-00001.safetensors"
        transformer_shard = (
            "transformer/diffusion_pytorch_model-00001-of-00001.safetensors"
        )
        vae = "vae/diffusion_pytorch_model.safetensors"
        files = {
            checkpoint.index_files[0]: {
                "weight_map": {"text": Path(text_shard).name}
            },
            checkpoint.index_files[1]: {
                "weight_map": {"transformer": Path(transformer_shard).name}
            },
        }
        siblings = [
            sibling(checkpoint.index_files[0]),
            sibling(checkpoint.index_files[1]),
            sibling(text_shard, 11),
            sibling(transformer_shard, 13),
            sibling(vae, 17),
            *[sibling(path) for path in checkpoint.required_support_files],
        ]

        def fake_download(repo_id, path, **kwargs):
            destination = self.root / "indexes" / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(files[path]))
            return str(destination)

        with patch.object(
            downloader,
            "hf_hub_download",
            side_effect=fake_download,
        ):
            plan = downloader.resolve_plan(
                FakeApi(checkpoint.revision, siblings),
                checkpoint,
                self.root / "models",
                token=None,
                latest=False,
            )

        self.assertEqual(checkpoint.index_files, plan.index_files)
        self.assertEqual(
            (text_shard, transformer_shard, vae),
            tuple(item.path for item in plan.weights),
        )
        self.assertEqual(41, plan.total_bytes)

        plan.directory.mkdir(parents=True)
        manifest = downloader.write_manifest(plan, sha256_verified=False)
        self.assertEqual(2, manifest["schema_version"])
        self.assertIsNone(manifest["index_file"])
        self.assertEqual(list(checkpoint.index_files), manifest["index_files"])
        self.assertEqual(
            ["corpus-v2", "extensions"],
            manifest["corpus_membership"],
        )
        self.assertEqual(
            [text_shard, transformer_shard, vae],
            [item["path"] for item in manifest["weights"]],
        )

    def test_root_manifest_merge_keeps_preexisting_paper_checkpoint(self):
        old = {
            "schema_version": 1,
            "name": "whisper-large-v3-f16",
            "repo_id": "openai/whisper-large-v3",
            "revision": corpus.CHECKPOINT_BY_NAME[
                "whisper-large-v3-f16"
            ].revision,
            "source_bytes": 31,
            "weights": [{"path": "model.safetensors"}],
        }
        path = self.root / "corpus-manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_bytes": 31,
                    "checkpoints": [old],
                }
            )
        )
        extension = {
            "schema_version": 2,
            "name": "qwen-image-bf16",
            "repo_id": "Qwen/Qwen-Image",
            "revision": corpus.CHECKPOINT_BY_NAME["qwen-image-bf16"].revision,
            "source_bytes": 57,
            "weights": [{"path": "vae/diffusion_pytorch_model.safetensors"}],
        }

        downloader.write_root_manifest(
            self.root,
            [extension],
            requested_corpus="extensions",
            explicit_models=False,
        )

        merged = json.loads(path.read_text())
        self.assertEqual(
            ["whisper-large-v3-f16", "qwen-image-bf16"],
            [item["name"] for item in merged["checkpoints"]],
        )
        self.assertEqual(88, merged["source_bytes"])
        self.assertEqual(["qwen-image-bf16"], merged["last_requested_models"])

    def test_voxtral_index_excludes_duplicate_consolidated_file(self):
        checkpoint = corpus.CHECKPOINT_BY_NAME[
            "voxtral-mini-3b-2507-bf16"
        ]
        first = "model-00001-of-00002.safetensors"
        second = "model-00002-of-00002.safetensors"
        index = {
            "weight_map": {
                "encoder": first,
                "decoder": second,
            }
        }
        siblings = [
            sibling(checkpoint.index_files[0]),
            sibling(first, 19),
            sibling(second, 23),
            sibling("consolidated.safetensors", 42),
            *[sibling(path) for path in checkpoint.required_support_files],
        ]
        index_path = self.root / "model.safetensors.index.json"
        index_path.write_text(json.dumps(index))

        with patch.object(
            downloader,
            "hf_hub_download",
            return_value=str(index_path),
        ):
            plan = downloader.resolve_plan(
                FakeApi(checkpoint.revision, siblings),
                checkpoint,
                self.root / "models",
                token=None,
                latest=False,
            )

        self.assertEqual((first, second), tuple(item.path for item in plan.weights))
        self.assertEqual(42, plan.total_bytes)

    def test_pinned_revision_must_resolve_exactly(self):
        checkpoint = corpus.CHECKPOINT_BY_NAME["bert-fp32"]
        api = FakeApi("f" * 40, [sibling("model.safetensors")])

        with self.assertRaisesRegex(RuntimeError, "expected frozen revision"):
            downloader.resolve_plan(
                api,
                checkpoint,
                self.root / "models",
                token=None,
                latest=False,
            )


if __name__ == "__main__":
    unittest.main()
