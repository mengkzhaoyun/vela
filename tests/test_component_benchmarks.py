"""CLI and reporting contract checks; these tests do not measure GPU performance."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = ("benchmark_ltx_trial.py", "benchmark_vela_synthetic.py")


def invoke(script: str, *args: str) -> subprocess.CompletedProcess:
    # -S prevents reliance on installed GPU libraries for configuration/reporting.
    return subprocess.run(
        [sys.executable, "-S", str(ROOT / "scripts" / script), *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20,
    )


def fixture(kind: str = "vela_synthetic_sr") -> dict:
    return {
        "schema_version": 1, "benchmark": kind, "measured": True,
        "device": {"type": "cuda", "memory_total_bytes": 24 * 2**30},
        "config": {"steps": 2, "weights": "random-placeholder", "quant": "int8",
                   "synthetic_input": True, "actual_backend": "pytorch-eager"},
        "timings": {"setup_seconds": 1, "warmup_seconds": 2,
                    "repeats_seconds": [2, 1, 3], "median_seconds": 2},
        "workload": {"batch_size": 4, "windows": 3, "canvas": 512,
                     "input_width": 854, "input_height": 480, "target_width": 1920, "target_height": 1080},
        "metrics": {"units_per_second": 2 if kind == "vela_synthetic_sr" else 1.5,
                    "unit": "frames" if kind == "vela_synthetic_sr" else "windows"},
        "memory": {"peak_allocated_bytes": 2**30, "peak_reserved_bytes": 2 * 2**30},
    }


class BenchmarkContracts(unittest.TestCase):
    def test_dry_run_needs_no_gpu_dependencies_and_writes_no_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            for script in RUNNERS:
                with self.subTest(script=script):
                    result = invoke(script, "--dry-run", "--output", output)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIs(json.loads(result.stdout)["measured"], False)
                    self.assertFalse(output.exists())

    def test_invalid_workloads_are_rejected(self):
        for script in RUNNERS:
            for flag, value in (("--repeats", "0"), ("--warmup", "-1"), ("--gpu", "-1"), ("--seed", str(2**64))):
                with self.subTest(script=script, flag=flag):
                    self.assertNotEqual(invoke(script, "--dry-run", flag, value).returncode, 0)

    def test_measured_mode_requires_explicit_destinations(self):
        for script in RUNNERS:
            with self.subTest(script=script):
                self.assertNotEqual(invoke(script).returncode, 0)

    def test_runners_do_not_accept_media_or_learned_ltx_weights(self):
        for script in RUNNERS:
            self.assertNotEqual(invoke(script, "--dry-run", "--input", "video.mp4").returncode, 0)
        self.assertNotEqual(invoke(RUNNERS[0], "--dry-run", "--transformer", "weights.bin").returncode, 0)

    def test_missing_sr_weights_do_not_produce_a_success_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            result = invoke(RUNNERS[1], "--model-path", Path(temporary) / "missing.bin", "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_existing_result_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "result.json"
            output.write_text("original", encoding="utf-8")
            result = invoke(RUNNERS[0], "--output", output)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")

    def summarize(self, report: dict) -> tuple[subprocess.CompletedProcess, str | None]:
        with tempfile.TemporaryDirectory() as temporary:
            source, output = Path(temporary) / "input.json", Path(temporary) / "review.md"
            source.write_text(json.dumps(report), encoding="utf-8")
            result = invoke("summarize_component_benchmarks.py", source, "--output", output)
            return result, output.read_text(encoding="utf-8") if output.exists() else None

    def test_summary_keeps_component_units_and_excludes_arbitrary_text(self):
        for kind, unit in (("vela_synthetic_sr", "frames/s"), ("ltx_trial_denoise", "windows/s")):
            report = fixture(kind)
            report["private_note"] = "DO_NOT_COPY_THIS_FIELD"
            result, content = self.summarize(report)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(unit, content)
            self.assertNotIn("DO_NOT_COPY_THIS_FIELD", content)
            self.assertIn("not be ranked", content)
            self.assertIn("512 px" if kind == "ltx_trial_denoise" else "854 × 480", content)

    def test_summary_rejects_dry_run_and_inconsistent_measurements(self):
        invalid = []
        row = fixture(); row["measured"] = False; invalid.append(row)
        row = fixture(); row["metrics"]["unit"] = "windows"; invalid.append(row)
        row = fixture(); row["metrics"]["units_per_second"] = 900; invalid.append(row)
        row = fixture(); row["timings"]["median_seconds"] = 7; invalid.append(row)
        row = fixture(); row["timings"]["repeats_seconds"] = []; invalid.append(row)
        row = fixture(); row["timings"]["repeats_seconds"] = [float("nan")]; invalid.append(row)
        row = fixture(); row["device"]["type"] = "cpu"; invalid.append(row)
        row = fixture(); row["benchmark"] = []; invalid.append(row)
        row = fixture(); row["schema_version"] = True; invalid.append(row)
        for index, report in enumerate(invalid):
            with self.subTest(index=index):
                result, content = self.summarize(report)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(content)


if __name__ == "__main__":
    unittest.main()
