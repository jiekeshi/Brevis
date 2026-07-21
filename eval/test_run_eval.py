import pathlib
import tempfile
import unittest
from unittest import mock

import run_eval


class ModelEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.sources = {}
        for name, size in (("model-00001-of-00002.safetensors", 10), ("model-00002-of-00002.safetensors", 14)):
            path = self.root / name
            path.write_bytes(bytes(range(size)))
            self.sources[name] = path
        self.model = {
            "tag": "mock-2shard",
            "repo": "test/model",
            "revision": "a" * 40,
            "files": list(self.sources),
            "note": "offline fixture",
        }
        self.archive_source = {}

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self, repo, revision, name, url=None):
        self.assertEqual("test/model", repo)
        self.assertEqual("a" * 40, revision)
        self.assertIsNone(url)
        return self.sources[name]

    def command(self, *cmd):
        action = str(cmd[1])
        if action == "calibrate":
            pathlib.Path(cmd[3]).write_bytes(b"prior")
        elif action == "compress":
            src, out = pathlib.Path(cmd[2]), pathlib.Path(cmd[3])
            divisor = 2 if ".uniform." in out.name else 3
            out.write_bytes(b"x" * (src.stat().st_size // divisor + 1))
            self.archive_source[out] = src
        elif action == "decompress":
            archive, out = pathlib.Path(cmd[2]), pathlib.Path(cmd[3])
            out.write_bytes(self.archive_source[archive].read_bytes())
        return 1.0

    @staticmethod
    def baseline(path, tool, cargs, dargs, out):
        divisors = {"gzip": 2, "zstd": 3, "xz": 4}
        return path.stat().st_size // divisors[tool], 0.25, 0.125, True

    @staticmethod
    def openzl(path, out):
        return path.stat().st_size // 5, 0.25, 0.125, True

    def patches(self):
        return (
            mock.patch.object(run_eval, "fetch", side_effect=self.fetch),
            mock.patch.object(run_eval, "timed", side_effect=self.command),
            mock.patch.object(run_eval, "sh"),
            mock.patch.object(run_eval, "baseline", side_effect=self.baseline),
            mock.patch.object(run_eval, "openzl", side_effect=self.openzl),
        )

    def test_multishard_pipeline_aggregates_and_cleans_artifacts(self):
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            row = run_eval.evaluate_model(self.model, self.work)

        self.assertEqual(24, row["raw"])
        self.assertEqual(14, row["uniform"])
        self.assertEqual(9, row["brevis"])
        self.assertEqual(12, row["gzip"])
        self.assertEqual(7, row["zstd"])
        self.assertEqual(5, row["xz"])
        self.assertEqual(4, row["openzl"])
        self.assertEqual(2.0, row["t_cal"])
        self.assertEqual(2.0, row["t_cmp"])
        self.assertEqual(2.0, row["t_dec_jobs1"])
        self.assertEqual(2.0, row["t_dec"])
        self.assertTrue(row["bitexact_jobs1"])
        self.assertTrue(row["bitexact"])
        self.assertEqual(list(self.sources), [shard["file"] for shard in row["shards"]])
        self.assertFalse([path for path in self.work.rglob("*") if path.is_file()])

    def test_failed_shard_raises_and_cleans_every_temporary_file(self):
        calls = 0

        def corrupt_second_shard(*cmd):
            nonlocal calls
            elapsed = self.command(*cmd)
            if str(cmd[1]) == "decompress":
                calls += 1
                if calls == 3:
                    pathlib.Path(cmd[3]).write_bytes(b"corrupt")
            return elapsed

        patches = self.patches()
        patches = (patches[0], mock.patch.object(run_eval, "timed", side_effect=corrupt_second_shard), *patches[2:])
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaises(run_eval.EvalError):
                run_eval.evaluate_model(self.model, self.work)

        self.assertFalse([path for path in self.work.rglob("*") if path.is_file()])


if __name__ == "__main__":
    unittest.main()
