"""Tests for agent-evals report comparison (compare.py)."""

import json
import os
import tempfile
import unittest

from agent_evals.compare import compare_reports, format_comparison, main


def _report(suite="s", results=None, weighted_score=1.0, total_latency_ms=10.0):
    results = results or []
    return {
        "suite": suite,
        "started_at": "2026-11-22T00:00:00+00:00",
        "ended_at": "2026-11-22T00:00:01+00:00",
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "pass_rate": 1.0,
        "total_weight": len(results),
        "earned_weight": float(sum(1 for r in results if r["passed"])),
        "weighted_score": weighted_score,
        "total_latency_ms": total_latency_ms,
        "total_tokens": None,
        "results": results,
    }


def _result(case_id, passed, score=1.0, latency_ms=1.0):
    return {
        "case_id": case_id,
        "passed": passed,
        "detail": "ok" if passed else "fail",
        "latency_ms": latency_ms,
        "tokens": None,
        "output": "",
        "timed_out": False,
        "score": score,
        "weight": 1.0,
    }


class TestCompareReports(unittest.TestCase):
    def test_regression_detected_and_fails(self):
        comp = compare_reports(
            _report(results=[_result("a", True), _result("b", True)]),
            _report(results=[_result("a", True), _result("b", False)]),
        )
        self.assertTrue(comp.has_regression)
        self.assertEqual([c.case_id for c in comp.regressions], ["b"])
        self.assertEqual(comp.verdict, "REGRESSION")
        self.assertEqual(comp.pass_rate_delta, -0.5)

    def test_fix_detected_and_verdict_improved(self):
        comp = compare_reports(
            _report(results=[_result("a", False)]),
            _report(results=[_result("a", True)]),
        )
        self.assertFalse(comp.has_regression)
        self.assertEqual([c.case_id for c in comp.fixes], ["a"])
        self.assertEqual(comp.verdict, "IMPROVED")

    def test_new_and_removed_cases(self):
        comp = compare_reports(
            _report(results=[_result("a", True), _result("old", True)]),
            _report(results=[_result("a", True), _result("fresh", True)]),
        )
        self.assertEqual([c.case_id for c in comp.new_cases], ["fresh"])
        self.assertEqual([c.case_id for c in comp.removed_cases], ["old"])
        self.assertEqual(comp.verdict, "IMPROVED")

    def test_score_change_on_still_failing_case(self):
        comp = compare_reports(
            _report(results=[_result("a", False, score=0.25)]),
            _report(results=[_result("a", False, score=0.75)]),
        )
        self.assertFalse(comp.has_regression)
        self.assertEqual(len(comp.score_changes), 1)
        self.assertEqual(comp.score_changes[0].baseline_score, 0.25)
        self.assertEqual(comp.score_changes[0].current_score, 0.75)
        self.assertEqual(comp.verdict, "IMPROVED")

    def test_identical_reports_are_unchanged(self):
        report = _report(results=[_result("a", True), _result("b", False)])
        comp = compare_reports(report, report)
        self.assertFalse(comp.has_regression)
        self.assertEqual(comp.verdict, "UNCHANGED")
        self.assertEqual(comp.pass_rate_delta, 0.0)
        self.assertEqual(comp.weighted_score_delta, 0.0)
        self.assertEqual(comp.latency_delta_ms, 0.0)

    def test_timed_out_case_that_used_to_pass_is_a_regression(self):
        # A pass that becomes a timeout is still passed -> not passed.
        base = _report(results=[_result("a", True)])
        cur = _report(results=[_result("a", False, score=0.0)])
        comp = compare_reports(base, cur)
        self.assertEqual([c.case_id for c in comp.regressions], ["a"])

    def test_headline_deltas(self):
        comp = compare_reports(
            _report(results=[_result("a", True)], weighted_score=0.5,
                    total_latency_ms=100.0),
            _report(results=[_result("a", True)], weighted_score=1.0,
                    total_latency_ms=150.0),
        )
        self.assertEqual(comp.weighted_score_delta, 0.5)
        self.assertEqual(comp.latency_delta_ms, 50.0)

    def test_per_case_latency_delta(self):
        comp = compare_reports(
            _report(results=[_result("a", True, latency_ms=10.0)]),
            _report(results=[_result("a", True, latency_ms=25.0)]),
        )
        self.assertEqual(comp.changes[0].latency_delta_ms, 15.0)

    def test_suite_name_mismatch_recorded(self):
        comp = compare_reports(
            _report(suite="old-suite", results=[_result("a", True)]),
            _report(suite="new-suite", results=[_result("a", True)]),
        )
        self.assertEqual(comp.baseline_suite, "old-suite")
        self.assertEqual(comp.current_suite, "new-suite")
        text = format_comparison(comp)
        self.assertIn("suite names differ", text)

    def test_to_dict_round_trips_through_json(self):
        comp = compare_reports(
            _report(results=[_result("a", True)]),
            _report(results=[_result("a", False)]),
        )
        data = json.loads(json.dumps(comp.to_dict()))
        self.assertEqual(data["verdict"], "REGRESSION")
        self.assertEqual(data["regressions"], ["a"])
        self.assertEqual(len(data["changes"]), 1)
        self.assertEqual(data["changes"][0]["kind"], "regression")

    def test_format_lists_regressions_and_fixes(self):
        comp = compare_reports(
            _report(results=[_result("reg", True), _result("fix", False)]),
            _report(results=[_result("reg", False), _result("fix", True)]),
        )
        text = format_comparison(comp)
        self.assertIn("Regressions (1):", text)
        self.assertIn("reg", text)
        self.assertIn("Fixes (1):", text)
        self.assertIn("fix", text)
        self.assertIn("Verdict: REGRESSION", text)

    def test_empty_reports_compare_cleanly(self):
        comp = compare_reports(_report(results=[]), _report(results=[]))
        self.assertEqual(comp.verdict, "UNCHANGED")
        self.assertEqual(comp.pass_rate_delta, 0.0)


class TestCompareCli(unittest.TestCase):
    def _write(self, tmp, name, report):
        path = os.path.join(tmp, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f)
        return path

    def test_cli_exits_one_on_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._write(
                tmp, "base.json", _report(results=[_result("a", True)])
            )
            cur = self._write(
                tmp, "cur.json", _report(results=[_result("a", False)])
            )
            self.assertEqual(
                main(["--baseline", base, "--current", cur]), 1
            )

    def test_cli_exits_zero_without_regression(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._write(
                tmp, "base.json", _report(results=[_result("a", False)])
            )
            cur = self._write(
                tmp, "cur.json", _report(results=[_result("a", True)])
            )
            self.assertEqual(
                main(["--baseline", base, "--current", cur]), 0
            )

    def test_cli_json_output_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self._write(
                tmp, "base.json", _report(results=[_result("a", True)])
            )
            cur = self._write(
                tmp, "cur.json", _report(results=[_result("a", True)])
            )
            out = os.path.join(tmp, "diff.json")
            self.assertEqual(
                main(["--baseline", base, "--current", cur, "--json", out]),
                0,
            )
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["verdict"], "UNCHANGED")


if __name__ == "__main__":
    unittest.main()
