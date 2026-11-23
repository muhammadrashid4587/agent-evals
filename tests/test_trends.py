"""Tests for the reports-directory trend comparison."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_evals.trends import (
    compare_history,
    format_trend,
    load_reports_dir,
)


def _report(started_at, results, suite="calc", weighted_score=1.0):
    """Build a run-report dict in the shape ``load_report`` returns."""
    passed = sum(1 for r in results if r["passed"])
    return {
        "suite": suite,
        "started_at": started_at,
        "total": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "weighted_score": weighted_score,
        "total_latency_ms": sum(r.get("latency_ms", 0.0) for r in results),
        "results": [
            {
                "case_id": r["case_id"],
                "passed": r["passed"],
                "score": r.get("score", 1.0 if r["passed"] else 0.0),
                "latency_ms": r.get("latency_ms", 1.0),
            }
            for r in results
        ],
    }


def _history():
    """Four runs: a steady case, a case that regresses and stays
    broken, a case fixed in the last run, and a case added later."""
    return [
        _report(
            "2026-11-20T10:00:00+00:00",
            [
                {"case_id": "steady", "passed": True},
                {"case_id": "regressor", "passed": True},
                {"case_id": "fixed", "passed": False, "score": 0.0},
            ],
            weighted_score=0.67,
        ),
        _report(
            "2026-11-20T11:00:00+00:00",
            [
                {"case_id": "steady", "passed": True},
                {"case_id": "regressor", "passed": True},
                {"case_id": "fixed", "passed": False, "score": 0.5},
            ],
            weighted_score=0.83,
        ),
        _report(
            "2026-11-20T12:00:00+00:00",
            [
                {"case_id": "steady", "passed": True},
                {"case_id": "regressor", "passed": False},
                {"case_id": "fixed", "passed": False, "score": 0.5},
                {"case_id": "late", "passed": True},
            ],
            weighted_score=0.5,
        ),
        _report(
            "2026-11-20T13:00:00+00:00",
            [
                {"case_id": "steady", "passed": True},
                {"case_id": "regressor", "passed": False},
                {"case_id": "fixed", "passed": True},
                {"case_id": "late", "passed": False},
            ],
            weighted_score=0.5,
        ),
    ]


class TestCaseTrend(unittest.TestCase):
    def setUp(self):
        self.trend = compare_history(_history())
        self.by_id = {c.case_id: c for c in self.trend.cases}

    def test_per_case_pass_rates(self):
        self.assertEqual(self.by_id["steady"].pass_rate, 1.0)
        self.assertEqual(self.by_id["regressor"].pass_rate, 0.5)
        self.assertEqual(self.by_id["fixed"].pass_rate, 0.25)
        self.assertEqual(self.by_id["late"].pass_rate, 0.5)

    def test_runs_present_ignores_runs_before_case_existed(self):
        self.assertEqual(self.by_id["late"].runs_present, 2)
        self.assertEqual(self.by_id["late"].passed_history, [None, None, True, False])

    def test_newly_failing_and_newly_passing(self):
        # "late" failed in the last run after passing the run before.
        self.assertEqual(
            [c.case_id for c in self.trend.newly_failing], ["late"]
        )
        # "fixed" passed in the last run after failing the run before.
        self.assertEqual(
            [c.case_id for c in self.trend.newly_passing], ["fixed"]
        )
        # "regressor" failed in the last two runs: not newly failing.
        self.assertFalse(self.by_id["regressor"].newly_failing)

    def test_worst_regressions_rank_by_reliability(self):
        # Both "regressor" and "late" passed once and fail now, with
        # the same pass rate (0.5) and the same worst score drop
        # (1.0), so the tie breaks on case_id.
        self.assertEqual(
            [c.case_id for c in self.trend.worst_regressions],
            ["late", "regressor"],
        )

    def test_worst_drop_is_largest_consecutive_score_drop(self):
        # "fixed" went 0.0 -> 0.5 -> 0.5 -> 1.0: never dropped.
        self.assertEqual(self.by_id["fixed"].worst_drop, 0.0)
        # "regressor" went 1.0 -> 1.0 -> 0.0: a drop of 1.0.
        self.assertEqual(self.by_id["regressor"].worst_drop, 1.0)

    def test_score_delta_first_to_last(self):
        self.assertEqual(self.by_id["fixed"].score_delta, 1.0)
        self.assertEqual(self.by_id["regressor"].score_delta, -1.0)

    def test_failing_streak_and_flaky(self):
        self.assertEqual(self.by_id["regressor"].current_failing_streak, 2)
        self.assertEqual(self.by_id["steady"].current_failing_streak, 0)
        self.assertEqual(
            {c.case_id for c in self.trend.flaky_cases},
            {"regressor", "fixed", "late"},
        )
        self.assertNotIn("steady", {c.case_id for c in self.trend.flaky_cases})

    def test_headline_deltas(self):
        # Pass rates: 2/3 -> 2/3 -> 2/4 -> 2/4; weighted score 0.67 -> 0.5.
        self.assertAlmostEqual(self.trend.pass_rate_delta, 0.5 - 2 / 3, places=4)
        self.assertAlmostEqual(self.trend.weighted_score_delta, -0.17, places=4)


def _latency_history():
    """Four runs with explicit per-case latencies: one case getting
    steadily slower, one getting faster, one flat, one with a single
    sample, and one absent from a middle run."""
    return [
        _report(
            "2026-11-21T10:00:00+00:00",
            [
                {"case_id": "slowing", "passed": True, "latency_ms": 10.0},
                {"case_id": "improving", "passed": True, "latency_ms": 20.0},
                {"case_id": "flat", "passed": True, "latency_ms": 5.0},
                {"case_id": "gappy", "passed": True, "latency_ms": 8.0},
            ],
        ),
        _report(
            "2026-11-21T11:00:00+00:00",
            [
                {"case_id": "slowing", "passed": True, "latency_ms": 12.0},
                {"case_id": "improving", "passed": True, "latency_ms": 15.0},
                {"case_id": "flat", "passed": True, "latency_ms": 5.0},
            ],
        ),
        _report(
            "2026-11-21T12:00:00+00:00",
            [
                {"case_id": "slowing", "passed": True, "latency_ms": 15.0},
                {"case_id": "improving", "passed": True, "latency_ms": 12.0},
                {"case_id": "flat", "passed": True, "latency_ms": 5.0},
                {"case_id": "gappy", "passed": True, "latency_ms": 9.0},
            ],
        ),
        _report(
            "2026-11-21T13:00:00+00:00",
            [
                {"case_id": "slowing", "passed": True, "latency_ms": 20.0},
                {"case_id": "improving", "passed": True, "latency_ms": 10.0},
                {"case_id": "flat", "passed": True, "latency_ms": 5.0},
                {"case_id": "gappy", "passed": True, "latency_ms": 9.0},
                {"case_id": "single", "passed": True, "latency_ms": 3.0},
            ],
        ),
    ]


class TestLatencyTrends(unittest.TestCase):
    def setUp(self):
        self.trend = compare_history(_latency_history())
        self.by_id = {c.case_id: c for c in self.trend.cases}

    def test_first_last_mean_min_max(self):
        slowing = self.by_id["slowing"]
        self.assertEqual(slowing.first_latency_ms, 10.0)
        self.assertEqual(slowing.last_latency_ms, 20.0)
        self.assertEqual(slowing.mean_latency_ms, 14.25)
        self.assertEqual(slowing.min_latency_ms, 10.0)
        self.assertEqual(slowing.max_latency_ms, 20.0)

    def test_latency_delta_ms_and_pct(self):
        slowing = self.by_id["slowing"]
        self.assertEqual(slowing.latency_delta_ms, 10.0)
        self.assertEqual(slowing.latency_delta_pct, 100.0)
        improving = self.by_id["improving"]
        self.assertEqual(improving.latency_delta_ms, -10.0)
        self.assertEqual(improving.latency_delta_pct, -50.0)
        self.assertEqual(self.by_id["flat"].latency_delta_pct, 0.0)

    def test_latency_slope_uses_run_index(self):
        # slowing: 10, 12, 15, 20 over runs 0..3 -> slope 3.3 ms/run.
        self.assertAlmostEqual(
            self.by_id["slowing"].latency_slope_ms_per_run, 3.3, places=3
        )
        # improving: 20, 15, 12, 10 -> slope -3.3 ms/run.
        self.assertAlmostEqual(
            self.by_id["improving"].latency_slope_ms_per_run, -3.3, places=3
        )
        self.assertEqual(self.by_id["flat"].latency_slope_ms_per_run, 0.0)
        # gappy: 8 at run 0, absent run 1, 9 at run 2, 9 at run 3.
        # The gap keeps its spacing: slope over x = 0, 2, 3 is 5/14.
        self.assertAlmostEqual(
            self.by_id["gappy"].latency_slope_ms_per_run, 5 / 14, places=3
        )

    def test_latency_trend_labels(self):
        self.assertEqual(self.by_id["slowing"].latency_trend, "slowing")
        self.assertEqual(self.by_id["improving"].latency_trend, "improving")
        self.assertEqual(self.by_id["flat"].latency_trend, "flat")
        # A single sample has no slope and no trend.
        self.assertIsNone(self.by_id["single"].latency_slope_ms_per_run)
        self.assertIsNone(self.by_id["single"].latency_trend)
        self.assertEqual(self.by_id["single"].latency_delta_pct, 0.0)

    def test_latency_history_skips_absent_runs(self):
        self.assertEqual(
            self.by_id["gappy"].latency_history, [8.0, None, 9.0, 9.0]
        )
        self.assertEqual(self.by_id["gappy"].latencies, [8.0, 9.0, 9.0])

    def test_slowest_growing_ranks_by_relative_slope(self):
        # Only "slowing" trends as slowing; "gappy" grows too but
        # its slope is small relative to its mean, so it is flat.
        self.assertEqual(
            [c.case_id for c in self.trend.slowest_growing], ["slowing"]
        )
        self.assertEqual(self.by_id["gappy"].latency_trend, "flat")

    def test_latency_fields_in_dict(self):
        data = self.by_id["slowing"].to_dict()
        self.assertEqual(data["first_latency_ms"], 10.0)
        self.assertEqual(data["last_latency_ms"], 20.0)
        self.assertEqual(data["mean_latency_ms"], 14.25)
        self.assertEqual(data["latency_delta_pct"], 100.0)
        self.assertAlmostEqual(
            data["latency_slope_ms_per_run"], 3.3, places=3
        )
        self.assertEqual(data["latency_trend"], "slowing")
        self.assertEqual(data["latency_history"], [10.0, 12.0, 15.0, 20.0])
        self.assertEqual(
            self.trend.to_dict()["slowest_growing"], ["slowing"]
        )

    def test_format_shows_slowest_growing_and_latency_column(self):
        text = format_trend(self.trend)
        self.assertIn("Slowest growing latency (1):", text)
        self.assertIn("slowing (10.0 -> 20.0 ms, +100.0%", text)
        self.assertIn("10.0 -> 20.0 ms (+100%)", text)
        self.assertIn("latency", text)

    def test_format_single_report_has_no_slowest_growing_section(self):
        text = format_trend(compare_history(_latency_history()[:1]))
        self.assertNotIn("Slowest growing latency", text)


class TestLatencyGate(unittest.TestCase):
    def setUp(self):
        self.trend = compare_history(_latency_history())
        self.by_id = {c.case_id: c for c in self.trend.cases}

    def test_gate_passes_on_flat_and_improving_trends(self):
        # Slopes: slowing +3.3, improving -3.3, flat 0.0, gappy ~0.36
        # ms/run. A 5.0 threshold lets every case through.
        self.assertEqual(self.trend.latency_gate_failures(5.0), [])
        self.assertTrue(self.trend.latency_gate_passed(5.0))

    def test_gate_fails_on_steep_slowing_slope(self):
        # A 3.0 threshold catches only the slowing case (+3.3).
        failures = self.trend.latency_gate_failures(3.0)
        self.assertEqual([c.case_id for c in failures], ["slowing"])
        self.assertFalse(self.trend.latency_gate_passed(3.0))

    def test_threshold_is_strict_and_respected(self):
        # A slope exactly at the threshold passes: the threshold is
        # a budget, not a tripwire.
        self.assertEqual(self.trend.latency_gate_failures(3.3), [])
        # Just below it, the same case trips the gate.
        self.assertEqual(
            [c.case_id for c in self.trend.latency_gate_failures(3.29)],
            ["slowing"],
        )

    def test_gate_ranks_failures_by_slope(self):
        # With a 0.0 threshold both growing cases trip, worst first.
        failures = self.trend.latency_gate_failures(0.0)
        self.assertEqual(
            [c.case_id for c in failures], ["slowing", "gappy"]
        )

    def test_single_sample_case_never_trips_gate(self):
        # "single" has one latency sample, so no slope at all.
        self.assertIsNone(self.by_id["single"].latency_slope_ms_per_run)
        self.assertNotIn(
            "single",
            [c.case_id for c in self.trend.latency_gate_failures(0.0)],
        )

    def test_slowing_cases_property(self):
        self.assertEqual(
            [c.case_id for c in self.trend.slowing_cases], ["slowing"]
        )

    def test_gate_fields_in_dict(self):
        data = self.trend.to_dict(latency_gate_threshold_ms_per_run=3.0)
        self.assertEqual(data["latency_gate_threshold_ms_per_run"], 3.0)
        self.assertEqual(data["latency_gate_failures"], ["slowing"])
        self.assertFalse(data["latency_gate_passed"])
        # Without a threshold, the gate keys are absent.
        plain = self.trend.to_dict()
        self.assertNotIn("latency_gate_failures", plain)
        self.assertNotIn("latency_gate_passed", plain)

    def test_format_shows_gate_section(self):
        text = format_trend(self.trend, latency_gate_threshold_ms_per_run=3.0)
        self.assertIn("Latency gate FAILED", text)
        self.assertIn("slowing (+3.30 ms/run", text)
        passing = format_trend(
            self.trend, latency_gate_threshold_ms_per_run=5.0
        )
        self.assertIn("Latency gate passed", passing)
        self.assertNotIn("Latency gate FAILED", passing)


class TestLatencyGateCli(unittest.TestCase):
    def _write_reports(self, tmp, reports):
        for i, report in enumerate(reports):
            with open(os.path.join(tmp, f"run-{i}.json"), "w") as f:
                json.dump(report, f)

    def _run_cli(self, tmp, *extra):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "agent_evals.trends",
                "--reports-dir",
                tmp,
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=os.path.join(os.path.dirname(__file__), ".."),
        )

    def test_cli_fail_on_trend_trips_on_steep_slope(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, _latency_history())
            proc = self._run_cli(tmp, "--fail-on-trend", "3.0")
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Latency gate FAILED", proc.stdout)
            self.assertIn("slowing", proc.stdout)

    def test_cli_fail_on_trend_passes_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            # All cases pass every run, so with a generous threshold
            # the trend gates clean.
            self._write_reports(tmp, _latency_history())
            proc = self._run_cli(tmp, "--fail-on-trend", "5.0")
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Latency gate passed", proc.stdout)

    def test_cli_fail_on_slowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, _latency_history())
            proc = self._run_cli(tmp, "--fail-on-slowing")
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Slowing gate FAILED", proc.stderr)
            self.assertIn("slowing", proc.stderr)

    def test_cli_fail_on_slowing_passes_when_nothing_slowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Flat and improving cases only: no case is slowing.
            flat = [
                _report(
                    "2026-11-21T10:00:00+00:00",
                    [
                        {"case_id": "flat", "passed": True, "latency_ms": 5.0},
                        {"case_id": "improving", "passed": True, "latency_ms": 20.0},
                    ],
                ),
                _report(
                    "2026-11-21T11:00:00+00:00",
                    [
                        {"case_id": "flat", "passed": True, "latency_ms": 5.0},
                        {"case_id": "improving", "passed": True, "latency_ms": 10.0},
                    ],
                ),
            ]
            self._write_reports(tmp, flat)
            proc = self._run_cli(tmp, "--fail-on-slowing")
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Slowing cases (0)", proc.stdout)

    def test_cli_json_carries_gate_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, _latency_history())
            out = os.path.join(tmp, "trend.json")
            proc = self._run_cli(
                tmp, "--fail-on-trend", "3.0", "--json", out
            )
            self.assertEqual(proc.returncode, 1)
            with open(out) as f:
                data = json.load(f)
            self.assertEqual(data["latency_gate_failures"], ["slowing"])
            self.assertFalse(data["latency_gate_passed"])

    def test_cli_rejects_negative_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, _latency_history())
            proc = self._run_cli(tmp, "--fail-on-trend", "-1")
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--fail-on-trend must not be negative", proc.stderr)


class TestSingleReport(unittest.TestCase):
    def test_single_report_has_no_newly_failing_and_no_deltas(self):
        trend = compare_history(_history()[:1])
        self.assertEqual(trend.n_reports, 1)
        self.assertEqual(trend.newly_failing, [])
        self.assertEqual(trend.newly_passing, [])
        self.assertIsNone(trend.pass_rate_delta)
        self.assertIsNone(trend.weighted_score_delta)
        self.assertEqual(trend.worst_regressions, [])


class TestFormatTrend(unittest.TestCase):
    def test_format_mentions_the_sections(self):
        text = format_trend(compare_history(_history()))
        for fragment in (
            "Reports: 4",
            "Newly failing (1):",
            "late",
            "Newly passing (1):",
            "fixed",
            "Worst regressions (2):",
            "Flaky across history (3):",
            "Per-case trend:",
            "steady",
        ):
            self.assertIn(fragment, text)

    def test_format_single_report_has_no_trend_line(self):
        text = format_trend(compare_history(_history()[:1]))
        self.assertIn("Reports: 1", text)
        self.assertNotIn("Pass rate trend:", text)


class TestLoadReportsDir(unittest.TestCase):
    def test_loads_sorted_by_started_at_and_skips_bad_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Out-of-order filenames: started_at decides the order.
            for name, report in (
                ("b.json", _history()[3]),
                ("a.json", _history()[0]),
            ):
                with open(os.path.join(tmp, name), "w") as f:
                    json.dump(report, f)
            # A file that is not a report is skipped, not fatal.
            with open(os.path.join(tmp, "junk.json"), "w") as f:
                json.dump({"not": "a report"}, f)
            with open(os.path.join(tmp, "broken.json"), "w") as f:
                f.write("{not json")
            reports = load_reports_dir(tmp)
            self.assertEqual(len(reports), 2)
            self.assertEqual(reports[0]["_path"], "a.json")
            self.assertEqual(reports[1]["_path"], "b.json")

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_reports_dir("/nonexistent/reports/dir")


class TestTrendsCli(unittest.TestCase):
    def _write_reports(self, tmp, reports):
        for i, report in enumerate(reports):
            with open(os.path.join(tmp, f"run-{i}.json"), "w") as f:
                json.dump(report, f)

    def test_cli_exits_1_on_new_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, _history())
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_evals.trends",
                    "--reports-dir",
                    tmp,
                ],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("Newly failing (1):", proc.stdout)
            self.assertIn("late", proc.stdout)

    def test_cli_exits_0_when_nothing_newly_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            # First two runs only: nothing newly failed in the latest
            # run (the "fixed" case stayed failing, "regressor" still
            # passed), so the trend gates clean.
            self._write_reports(tmp, _history()[:2])
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_evals.trends",
                    "--reports-dir",
                    tmp,
                ],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Reports: 2", proc.stdout)

    def test_cli_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, _history())
            out = os.path.join(tmp, "trend.json")
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_evals.trends",
                    "--reports-dir",
                    tmp,
                    "--json",
                    out,
                ],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            self.assertEqual(proc.returncode, 1)
            with open(out) as f:
                data = json.load(f)
            self.assertEqual(data["n_reports"], 4)
            self.assertEqual(data["newly_failing"], ["late"])
            self.assertEqual(data["newly_passing"], ["fixed"])

    def test_cli_empty_directory_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agent_evals.trends",
                    "--reports-dir",
                    tmp,
                ],
                capture_output=True,
                text=True,
                cwd=os.path.join(os.path.dirname(__file__), ".."),
            )
            self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
