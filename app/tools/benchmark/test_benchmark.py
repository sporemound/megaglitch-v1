"""Dependency-light checks for the framebuffer benchmark harness."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.benchmark.check_regressions import evaluate, evaluate_report
from tools.benchmark.run_baseline import _percentile, _summarize_times


HERE = Path(__file__).resolve().parent


def _regression_fixture():
    thresholds = json.loads(
        (HERE / "regression_thresholds.json").read_text(encoding="utf-8")
    )
    temporary = tempfile.TemporaryDirectory(prefix="megaglitch-regression-")
    results_dir = Path(temporary.name)
    for gate_name in (
        "pframe_segment",
        "shared_pframe_worker",
        "worker_lifecycle",
        "rust_graph_ir",
    ):
        artifact = thresholds[gate_name]["artifact"]
        payload = json.loads((HERE / "results" / artifact).read_text(encoding="utf-8"))
        (results_dir / artifact).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return temporary, results_dir, thresholds


def _case(payload, name):
    return next(case for case in payload["cases"] if case.get("name") == name)


def _set_rust_timing_regression(results_dir, thresholds):
    gate = thresholds["rust_graph_ir"]
    path = results_dir / gate["artifact"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    native = _case(payload, gate["native_case"])
    native["mean_frame_ms"] = float(gate["native_mean_ms_max"]) + 3.0
    native["p99_frame_ms"] = float(gate["native_p99_ms_max"]) + 3.0
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, payload


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_percentile_interpolates_sorted_samples(self):
        self.assertEqual(_percentile([4.0, 1.0, 3.0, 2.0], 0), 1.0)
        self.assertEqual(_percentile([4.0, 1.0, 3.0, 2.0], 1), 4.0)
        self.assertAlmostEqual(_percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_summary_reports_latency_and_rate(self):
        result = _summarize_times([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["mean_frame_ms"], 2.5)
        self.assertEqual(result["median_frame_ms"], 2.5)
        self.assertEqual(result["max_frame_ms"], 4.0)
        self.assertEqual(result["average_fps"], 400.0)
        self.assertGreater(result["one_percent_low_fps"], 0.0)

    def test_summary_represents_empty_sample(self):
        result = _summarize_times([])
        self.assertEqual(result["frames"], 0)
        self.assertEqual(result["mean_frame_ms"], 0.0)
        self.assertEqual(result["average_fps"], 0.0)

    def test_committed_native_regression_artifacts_pass(self):
        thresholds = json.loads(
            (HERE / "regression_thresholds.json").read_text(encoding="utf-8")
        )
        self.assertEqual(evaluate(HERE / "results", thresholds), [])

    def test_rust_timing_is_strict_by_default_and_advisory_only_when_requested(self):
        temporary, results_dir, thresholds = _regression_fixture()
        self.addCleanup(temporary.cleanup)
        _set_rust_timing_regression(results_dir, thresholds)

        strict_failures = evaluate(results_dir, thresholds)
        self.assertTrue(any("Rust graph IR mean_frame_ms" in item for item in strict_failures))

        failures, advisories = evaluate_report(
            results_dir,
            thresholds,
            rust_performance_advisory=True,
        )
        self.assertEqual(failures, [])
        self.assertTrue(any("Rust graph IR mean_frame_ms" in item for item in advisories))
        self.assertTrue(any("Rust graph IR p99_frame_ms" in item for item in advisories))

    def test_rust_correctness_remains_hard_in_advisory_timing_mode(self):
        mutations = {
            "pixel exact": lambda payload, _python, _native: payload.update(
                graph_ir_worker_pixel_exact=False
            ),
            "hash": lambda _payload, _python, native: native.update(
                output_sha256="0" * 64
            ),
            "backend": lambda _payload, _python, native: native.update(
                backend="python_process"
            ),
            "drops": lambda _payload, _python, native: native.update(dropped_frames=1),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                temporary, results_dir, thresholds = _regression_fixture()
                try:
                    gate = thresholds["rust_graph_ir"]
                    path, payload = _set_rust_timing_regression(
                        results_dir, thresholds
                    )
                    python_case = _case(payload, gate["python_case"])
                    native_case = _case(payload, gate["native_case"])
                    mutate(payload, python_case, native_case)
                    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

                    failures, advisories = evaluate_report(
                        results_dir,
                        thresholds,
                        rust_performance_advisory=True,
                    )
                    self.assertTrue(failures)
                    self.assertTrue(advisories)
                finally:
                    temporary.cleanup()

    def test_production_regressions_remain_hard_with_rust_timing_advisory(self):
        temporary, results_dir, thresholds = _regression_fixture()
        self.addCleanup(temporary.cleanup)
        _set_rust_timing_regression(results_dir, thresholds)
        shared_gate = thresholds["shared_pframe_worker"]
        shared_path = results_dir / shared_gate["artifact"]
        shared_payload = json.loads(shared_path.read_text(encoding="utf-8"))
        shared_case = next(
            case
            for case in shared_payload["cases"]
            if case.get("name") == "worker_representative_pframe"
            and case.get("transport") == "shared"
        )
        shared_case["mean_frame_ms"] = float(shared_gate["mean_ms_max"]) + 1.0
        shared_path.write_text(json.dumps(shared_payload, indent=2), encoding="utf-8")

        failures, advisories = evaluate_report(
            results_dir,
            thresholds,
            rust_performance_advisory=True,
        )
        self.assertTrue(any("shared P-frame mean_frame_ms" in item for item in failures))
        self.assertTrue(any("Rust graph IR mean_frame_ms" in item for item in advisories))

    def test_cli_reports_advisories_but_keeps_strict_default(self):
        temporary, results_dir, thresholds = _regression_fixture()
        self.addCleanup(temporary.cleanup)
        _set_rust_timing_regression(results_dir, thresholds)
        checker = HERE / "check_regressions.py"
        thresholds_path = HERE / "regression_thresholds.json"

        strict = subprocess.run(
            [
                sys.executable,
                str(checker),
                "--results-dir",
                str(results_dir),
                "--thresholds",
                str(thresholds_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)
        strict_report = json.loads(strict.stdout)
        self.assertFalse(strict_report["passed"])
        self.assertTrue(strict_report["failures"])

        advisory = subprocess.run(
            [
                sys.executable,
                str(checker),
                "--results-dir",
                str(results_dir),
                "--thresholds",
                str(thresholds_path),
                "--rust-performance-advisory",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(advisory.returncode, 0, advisory.stdout + advisory.stderr)
        advisory_report = json.loads(advisory.stdout)
        self.assertTrue(advisory_report["passed"])
        self.assertEqual(advisory_report["failures"], [])
        self.assertTrue(advisory_report["advisories"])
        self.assertEqual(advisory_report["rust_performance_policy"], "advisory")

    def test_python_host_release_gate_requests_rust_timing_advisory(self):
        release_script = (HERE.parent / "build_and_run.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("--rust-performance-advisory", release_script)
        thresholds = json.loads(
            (HERE / "regression_thresholds.json").read_text(encoding="utf-8")
        )
        self.assertEqual(thresholds["rust_graph_ir"]["native_mean_ms_max"], 14.0)
        self.assertEqual(thresholds["rust_graph_ir"]["native_p99_ms_max"], 16.0)


if __name__ == "__main__":
    unittest.main()
