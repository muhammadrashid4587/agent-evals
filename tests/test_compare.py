"""Tests for agent-evals report comparison (compare.py)."""

import json
import os
import tempfile
import unittest

from agent_evals.compare import compare_reports, format_comparison, main


def _report(suite="s", results=None, weighted_score=1.0, total_latency_ms=10.0,
            total_tokens=None):
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
        "total_tokens": total_tokens,
        "results": results,
    }


def _result(case_id, passed, score=1.0, latency_ms=1.0, tokens=None):
    return {
        "case_id": case_id,
        "passed": passed,
        "detail": "ok" if passed else "fail",
        "latency_ms": latency_ms,
        "tokens": tokens,
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


class TestTokenDeltas(unittest.TestCase):
    def test_per_case_token_delta(self):
        comp = compare_reports(
            _report(results=[_result("a", True, tokens=100)]),
            _report(results=[_result("a", True, tokens=150)]),
        )
        self.assertEqual(comp.changes[0].token_delta, 50)
        self.assertEqual(comp.changes[0].score_delta, 0.0)

    def test_per_case_score_delta(self):
        comp = compare_reports(
            _report(results=[_result("a", False, score=0.25)]),
            _report(results=[_result("a", False, score=0.75)]),
        )
        self.assertEqual(comp.changes[0].score_delta, 0.5)

    def test_total_token_delta(self):
        comp = compare_reports(
            _report(results=[_result("a", True, tokens=100)], total_tokens=100),
            _report(results=[_result("a", True, tokens=80)], total_tokens=80),
        )
        self.assertEqual(comp.baseline_total_tokens, 100)
        self.assertEqual(comp.current_total_tokens, 80)
        self.assertEqual(comp.token_delta, -20)

    def test_total_tokens_summed_from_results_when_report_total_missing(self):
        # A hand-built report may carry per-result tokens but no
        # report-level total; the comparison still finds the delta.
        base = _report(results=[_result("a", True, tokens=40)])
        cur = _report(results=[_result("a", True, tokens=60)])
        del base["total_tokens"]
        del cur["total_tokens"]
        comp = compare_reports(base, cur)
        self.assertEqual(comp.token_delta, 20)

    def test_token_delta_none_when_a_run_reports_no_tokens(self):
        comp = compare_reports(
            _report(results=[_result("a", True, tokens=100)]),
            _report(results=[_result("a", True, tokens=None)]),
        )
        self.assertIsNone(comp.changes[0].token_delta)
        self.assertIsNone(comp.token_delta)
        self.assertNotIn("Tokens:", format_comparison(comp))

    def test_new_and_removed_cases_have_no_token_delta(self):
        comp = compare_reports(
            _report(results=[_result("old", True, tokens=10)]),
            _report(results=[_result("fresh", True, tokens=20)]),
        )
        for change in comp.changes:
            self.assertIsNone(change.token_delta)
            self.assertIsNone(change.score_delta)


class TestWinner(unittest.TestCase):
    def test_current_wins_on_higher_weighted_score(self):
        comp = compare_reports(
            _report(results=[_result("a", True)], weighted_score=0.5),
            _report(results=[_result("a", True)], weighted_score=1.0),
        )
        self.assertEqual(comp.winner, "current")

    def test_baseline_wins_on_higher_weighted_score(self):
        comp = compare_reports(
            _report(results=[_result("a", True)], weighted_score=1.0),
            _report(results=[_result("a", True)], weighted_score=0.5),
        )
        self.assertEqual(comp.winner, "baseline")

    def test_pass_rate_breaks_a_score_tie(self):
        # Same weighted score, but the current run passed more cases
        # (heavier cases failed in the baseline, lighter ones passed).
        comp = compare_reports(
            _report(
                results=[_result("a", True), _result("b", False)],
                weighted_score=0.5,
            ),
            _report(
                results=[_result("a", True), _result("b", True)],
                weighted_score=0.5,
            ),
        )
        self.assertEqual(comp.winner, "current")

    def test_latency_breaks_a_score_and_pass_rate_tie(self):
        comp = compare_reports(
            _report(results=[_result("a", True)], total_latency_ms=200.0),
            _report(results=[_result("a", True)], total_latency_ms=100.0),
        )
        self.assertEqual(comp.winner, "current")

    def test_tokens_break_the_final_tie(self):
        comp = compare_reports(
            _report(
                results=[_result("a", True, tokens=300)], total_tokens=300
            ),
            _report(
                results=[_result("a", True, tokens=200)], total_tokens=200
            ),
        )
        self.assertEqual(comp.winner, "current")

    def test_tokens_do_not_break_a_tie_when_one_run_reports_none(self):
        # The baseline reported tokens and the current run did not:
        # cost cannot decide, so this is a tie, not a current win.
        comp = compare_reports(
            _report(
                results=[_result("a", True, tokens=300)], total_tokens=300
            ),
            _report(results=[_result("a", True, tokens=None)]),
        )
        self.assertIsNone(comp.winner)

    def test_identical_reports_have_no_winner(self):
        report = _report(results=[_result("a", True)])
        comp = compare_reports(report, report)
        self.assertIsNone(comp.winner)
        self.assertIn("Winner: tie", format_comparison(comp))

    def test_winner_is_in_json(self):
        comp = compare_reports(
            _report(results=[_result("a", True)], weighted_score=0.5),
            _report(results=[_result("a", True)], weighted_score=1.0),
        )
        data = json.loads(json.dumps(comp.to_dict()))
        self.assertEqual(data["winner"], "current")
        self.assertEqual(data["baseline_total_tokens"], None)
        self.assertEqual(data["token_delta"], None)
        self.assertEqual(data["changes"][0]["score_delta"], 0.0)


class TestSideBySideTable(unittest.TestCase):
    def test_table_shows_score_latency_and_token_deltas(self):
        comp = compare_reports(
            _report(results=[_result("a", True, score=0.5, latency_ms=25.0,
                                     tokens=100)],
                    total_latency_ms=25.0, total_tokens=100),
            _report(results=[_result("a", True, score=1.0, latency_ms=10.0,
                                     tokens=120)],
                    total_latency_ms=10.0, total_tokens=120),
        )
        text = format_comparison(comp)
        self.assertIn("Per-case (side by side):", text)
        self.assertIn("Winner: current", text)
        self.assertIn("Tokens: +20 (100 -> 120)", text)
        lines = [line for line in text.splitlines() if line.startswith("a ")]
        self.assertEqual(len(lines), 1)
        row = lines[0]
        self.assertIn("0.50", row)
        self.assertIn("1.00", row)
        self.assertIn("+0.50", row)
        self.assertIn("-15.0", row)
        self.assertIn("+20", row)

    def test_table_hides_token_columns_when_no_tokens_reported(self):
        comp = compare_reports(
            _report(results=[_result("a", True)]),
            _report(results=[_result("a", True)]),
        )
        text = format_comparison(comp)
        self.assertIn("Per-case (side by side):", text)
        self.assertNotIn("base_tok", text)
        self.assertNotIn("Tokens:", text)

    def test_table_marks_movement_and_missing_sides(self):
        comp = compare_reports(
            _report(results=[_result("a", True), _result("old", True)]),
            _report(results=[_result("a", False), _result("fresh", True)]),
        )
        text = format_comparison(comp)
        # Table rows start at column 0; the Regressions/Fixes sections
        # below are indented, so only unindented lines are table rows.
        rows = {
            line.split()[0]: line
            for line in text.splitlines()
            if line
            and not line.startswith(" ")
            and line.split()[0] in ("a", "old", "fresh")
        }
        self.assertIn("regression", rows["a"])
        self.assertIn("removed", rows["old"])
        self.assertIn("new", rows["fresh"])
        self.assertIn("-", rows["fresh"])  # no baseline side
        self.assertIn("-", rows["old"])  # no current side

    def test_empty_reports_table_says_no_cases(self):
        comp = compare_reports(_report(results=[]), _report(results=[]))
        self.assertIn("(no cases in either run)", format_comparison(comp))


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
            self.assertIsNone(data["winner"])
            self.assertIn("token_delta", data)


if __name__ == "__main__":
    unittest.main()
