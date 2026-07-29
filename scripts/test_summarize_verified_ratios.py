import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_verified_ratios as summary


def compression(
    run_id: str,
    attempt_id: str,
    output_bytes: int,
) -> dict:
    return {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": "ok",
        "stage": "corpus",
        "checkpoint": "fixture",
        "revision": "revision",
        "shard": "model.safetensors",
        "method": "brevis",
        "operation": "compress",
        "source_bytes": 100,
        "output_bytes": output_bytes,
    }


def verification(
    run_id: str,
    attempt_id: str,
    *,
    exact: bool,
    status: str,
) -> dict:
    return {
        "run_id": f"{run_id}-verify",
        "status": status,
        "stage": "corpus",
        "checkpoint": "fixture",
        "shard": "model.safetensors",
        "method": "brevis",
        "operation": "verify",
        "exact": exact,
        "verified_attempts": [[run_id, attempt_id]],
    }


class VerifiedRatioSummaryTests(unittest.TestCase):
    def test_only_last_successful_exact_verification_is_selected(self):
        records = [
            compression("old", "old-attempt", 80),
            verification(
                "old",
                "old-attempt",
                exact=True,
                status="ok",
            ),
            compression("failed", "failed-attempt", 10),
            verification(
                "failed",
                "failed-attempt",
                exact=False,
                status="failed",
            ),
            compression("new", "new-attempt", 60),
            verification(
                "new",
                "new-attempt",
                exact=True,
                status="ok",
            ),
        ]

        selected, diagnostics = summary.select_verified_compressions(records)

        self.assertEqual([], diagnostics)
        verified = selected["fixture", "brevis", "model.safetensors"]
        self.assertEqual(60, verified.compression["output_bytes"])

    def test_cell_requires_every_expected_shard(self):
        expected = [
            summary.ExpectedCheckpoint(
                "fixture",
                "revision",
                {
                    "model.safetensors": 100,
                    "second.safetensors": 200,
                },
            )
        ]
        verified = summary.VerifiedCompression(
            compression("run", "attempt", 60),
            verification("run", "attempt", exact=True, status="ok"),
            2,
        )

        cells = summary.summarize_cells(
            expected,
            ("brevis",),
            {("fixture", "brevis", "model.safetensors"): verified},
        )

        self.assertEqual("incomplete", cells[0]["status"])
        self.assertEqual(1, cells[0]["verified_shards"])
        self.assertEqual(2, cells[0]["expected_shards"])
        self.assertEqual("second.safetensors", cells[0]["missing_shards"])


if __name__ == "__main__":
    unittest.main()
