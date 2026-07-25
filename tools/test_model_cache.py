"""Tests for tools/model_cache.py.

`drop` deletes files, so its guard conditions are covered explicitly: a wrong
digest must never be reported as usable, and dropping must not reach outside the
model it was asked for.
"""

from __future__ import annotations

import hashlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import model_cache  # noqa: E402


PAYLOAD = b"weights" * 32
OTHER = b"different" * 16


def manifest_entry(tag: str, repo: str, payload: bytes) -> dict:
    return {
        "tag": tag,
        "repo": repo,
        "revision": "0123456789abcdef",
        "execution_stage": 1,
        "selected_bytes": len(payload),
        "files": [{
            "file": "model.safetensors",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }


class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

        self.manifest = self.root / "models.json"
        self.manifest.write_text(json.dumps([
            manifest_entry("alpha", "org/alpha", PAYLOAD),
            manifest_entry("beta", "org/beta", OTHER),
        ]))
        self.cache = self.root / "cache"

    def run_cli(self, *argv: str) -> tuple[int, str]:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = model_cache.main([
                "--manifest", str(self.manifest),
                "--cache-root", str(self.cache), *argv,
            ])
        return code, buffer.getvalue()

    def place(self, tag: str, payload: bytes) -> pathlib.Path:
        index = model_cache.models(self.manifest)
        path, _ = model_cache.paths(index[tag], self.cache)[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path

    # ---- check() ----

    def test_check_reports_each_distinct_failure(self):
        index = model_cache.models(self.manifest)
        path, spec = model_cache.paths(index["alpha"], self.cache)[0]
        self.assertEqual("missing", model_cache.check(path, spec))

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PAYLOAD[:-1])
        self.assertEqual("wrong-size", model_cache.check(path, spec))

        # Right length, wrong content: only hashing can catch this.
        path.write_bytes(b"x" * len(PAYLOAD))
        self.assertEqual("wrong-sha256", model_cache.check(path, spec))

        path.write_bytes(PAYLOAD)
        self.assertEqual("ok", model_cache.check(path, spec))

    def test_cache_path_derives_from_repo_and_revision(self):
        index = model_cache.models(self.manifest)
        path, _ = model_cache.paths(index["alpha"], self.cache)[0]
        self.assertEqual(
            self.cache / "org__alpha" / "0123456789abcdef" / "model.safetensors",
            path,
        )

    # ---- verify ----

    def test_verify_fails_closed_on_corruption(self):
        self.place("alpha", PAYLOAD)
        self.assertEqual(0, self.run_cli("verify", "alpha")[0])

        self.place("alpha", b"x" * len(PAYLOAD))
        code, out = self.run_cli("verify", "alpha")
        self.assertEqual(1, code)
        self.assertIn("wrong-sha256", out)

    def test_verify_treats_absent_model_as_not_a_failure(self):
        code, out = self.run_cli("verify")
        self.assertEqual(0, code)
        self.assertEqual("", out.strip())

    # ---- fetch ----

    def test_fetch_skips_files_already_correct(self):
        self.place("alpha", PAYLOAD)
        with mock.patch.object(model_cache.subprocess, "run") as run:
            code, out = self.run_cli("fetch", "alpha")
        run.assert_not_called()
        self.assertEqual(0, code)
        self.assertIn("have model.safetensors", out)

    def test_fetch_rejects_a_download_that_fails_integrity(self):
        def fake_run(argv, check):
            pathlib.Path(argv[argv.index("-o") + 1]).write_bytes(b"x" * len(PAYLOAD))

        with mock.patch.object(model_cache.subprocess, "run", fake_run):
            code, _ = self.run_cli("fetch", "alpha")
        self.assertEqual(1, code)

    def test_fetch_promotes_a_verified_download_into_place(self):
        def fake_run(argv, check):
            pathlib.Path(argv[argv.index("-o") + 1]).write_bytes(PAYLOAD)

        with mock.patch.object(model_cache.subprocess, "run", fake_run):
            code, out = self.run_cli("fetch", "alpha")
        self.assertEqual(0, code)
        self.assertIn("complete and verified", out)
        index = model_cache.models(self.manifest)
        path, spec = model_cache.paths(index["alpha"], self.cache)[0]
        self.assertEqual("ok", model_cache.check(path, spec))
        self.assertFalse(list(path.parent.glob("*.part")))

    # ---- drop ----

    def test_drop_removes_only_the_named_model(self):
        kept = self.place("beta", OTHER)
        dropped = self.place("alpha", PAYLOAD)
        code, out = self.run_cli("drop", "alpha")
        self.assertEqual(0, code)
        self.assertFalse(dropped.exists())
        self.assertTrue(kept.exists())
        self.assertIn("freed", out)

    def test_drop_prunes_emptied_directories_but_not_shared_ones(self):
        self.place("alpha", PAYLOAD)
        sibling = self.cache / "org__alpha" / "other-revision"
        sibling.mkdir(parents=True)
        self.run_cli("drop", "alpha")
        self.assertFalse((self.cache / "org__alpha" / "0123456789abcdef").exists())
        self.assertTrue(sibling.exists(), "a second revision must survive")

    def test_drop_is_idempotent(self):
        self.assertEqual(0, self.run_cli("drop", "alpha")[0])
        self.assertEqual(0, self.run_cli("drop", "alpha")[0])

    # ---- status and argument handling ----

    def test_status_reports_state_per_model(self):
        self.place("alpha", PAYLOAD)
        _, out = self.run_cli("status")
        self.assertRegex(out, r"alpha\s+complete")
        self.assertRegex(out, r"beta\s+absent")

    def test_status_verify_downgrades_a_corrupt_file(self):
        self.place("alpha", b"x" * len(PAYLOAD))
        self.assertRegex(self.run_cli("status")[1], r"alpha\s+complete")
        self.assertRegex(self.run_cli("status", "--verify")[1], r"alpha\s+partial")

    def test_unknown_tag_is_rejected_before_any_filesystem_work(self):
        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()):
            model_cache.main(["--manifest", str(self.manifest),
                              "--cache-root", str(self.cache), "drop", "nope"])
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
