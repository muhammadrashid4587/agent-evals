"""Tests for agent-evals flaky-case history (v0.17.0).

A single stability report flags flaky cases in one burst of
repeated runs. The flaky history compares stability reports across
time: which cases keep flipping, which just started, and which
stopped.
"""

import json
import os
import tempfile
import unittest

from agent_evals.evals import Case, Suite
from agent_evals.flaky_history import (
    compare_flaky_history,
    format_flaky_history,
    load_stability_reports_dir,
)
from agent_evals.stability import format_stability, repeat_suite


def _suite():
    return Suite(
        name="s",
        cases=[
            Case(id="stable-pass", prompt="p", scorer={"type": "exact", "expected": "19"}),
            Case(id="stable-fail", prompt="p", scorer={"type": "exact", "expected": "nope"}),
            # Distinct prompt so the test agent can flip only this case.
            Case(id="flaky", prompt="flip", scorer={"type": "exact", "expected": "19"}),
        ],
    )


def _alternating_agent():
    """Answers '19' always, except the 'flip' prompt, which it
    alternates between '19' and '20' on successive calls."""
    calls = {"n": 0}

    def agent(prompt):
        if prompt == "flip":
            calls["n"] += 1
            return "19" if calls["n"] % 2 else "20"
        return "19"

    return agent


def _stability_dict(suite="s", started_at="2026-11-22T00:00:00+00:00",
                    repetitions=4, cases=None, path=None):
    """A stability report dict, as StabilityReport.to_dict returns."""
    cases = cases if cases is not None else []
    data = {
        "suite": suite,
        "repetitions": repetitions,
        "started_at": started_at,
        "ended_at": started_at,
        "tag_filter": None,
        "flaky": [c["case_id"] for c in cases if c["flaky"]],
        "always_passing": [c["case_id"] for c in cases
                           if c["passes"] == c["runs"]],
        "always_failing": [c["case_id"] for c in cases
                           if c["passes"] == 0],
        "tag_breakdown": {},
        "cases": cases,
    }
    if path is not None:
        data["_path"] = path
    return data


def _case(case_id, runs, passes, flaky=None):
    if flaky is None:
        flaky = 0 < passes < runs
    return {
        "case_id": case_id,
        "runs": runs,
        "passes": passes,
        "fails": runs - passes,
        "pass_rate": round(passes / runs, 4) if runs else 0.0,
        "flaky": flaky,
        "mean_latency_ms": 1.0,
        "mean_score": round(passes / runs, 4) if runs else 0.0,
        "timeouts": 0,
        "tags": [],
    }


class TestStabilityReportFails(unittest.TestCase):
    def test_fails_count_in_dict(self):
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=4)
        by_id = {c["case_id"]: c for c in report.to_dict()["cases"]}
        self.assertEqual(by_id["flaky"]["runs"], 4)
        self.assertEqual(by_id["flaky"]["passes"], 2)
        self.assertEqual(by_id["flaky"]["fails"], 2)
        self.assertTrue(by_id["flaky"]["flaky"])
        self.assertEqual(by_id["stable-pass"]["fails"], 0)
        self.assertEqual(by_id["stable-fail"]["fails"], 4)

    def test_format_stability_lists_flaky_cases(self):
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=2)
        text = format_stability(report)
        self.assertIn("Flaky cases (1):", text)
        self.assertIn("flaky  1/2 (50.0%)", text)

    def test_format_stability_omits_section_without_flaky(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertNotIn("Flaky cases", format_stability(report))


class TestCompareFlakyHistory(unittest.TestCase):
    def _three_reports(self):
        # "flaky" flips in the first two reports, stable in the third;
        # "new" is stable in the first two, flips only in the third;
        # "late" is added in the third report and flips on arrival;
        # "stable-pass" never flips.
        return [
            _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[
                    _case("stable-pass", 4, 4),
                    _case("flaky", 4, 2),
                    _case("new", 4, 4),
                ],
                path="r1.json",
            ),
            _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[
                    _case("stable-pass", 4, 4),
                    _case("flaky", 4, 3),
                    _case("new", 4, 4),
                ],
                path="r2.json",
            ),
            _stability_dict(
                started_at="2026-11-22T00:00:00+00:00",
                cases=[
                    _case("stable-pass", 4, 4),
                    _case("flaky", 4, 4),
                    _case("new", 4, 1),
                    _case("late", 4, 2),
                ],
                path="r3.json",
            ),
        ]

    def test_persistently_flaky_needs_two_flaky_reports(self):
        history = compare_flaky_history(self._three_reports())
        self.assertEqual(
            [c.case_id for c in history.persistently_flaky_cases],
            ["flaky"],
        )
        by_id = {c.case_id: c for c in history.cases}
        self.assertEqual(by_id["flaky"].flaky_reports, 2)
        self.assertTrue(by_id["flaky"].persistently_flaky)
        # "new" flipped once: flaky, but not yet persistent.
        self.assertTrue(by_id["new"].flaky)
        self.assertFalse(by_id["new"].persistently_flaky)
        self.assertEqual(by_id["new"].flaky_reports, 1)
        # "late" flipped on arrival: flaky, but it was never stable
        # before, so it is not newly flaky either.
        self.assertTrue(by_id["late"].flaky)
        self.assertFalse(by_id["late"].newly_flaky)
        self.assertFalse(by_id["late"].persistently_flaky)

    def test_newly_flaky_and_stopped_flaky(self):
        history = compare_flaky_history(self._three_reports())
        self.assertEqual(
            [c.case_id for c in history.newly_flaky], ["new"]
        )
        self.assertEqual(
            [c.case_id for c in history.stopped_flaky], ["flaky"]
        )
        by_id = {c.case_id: c for c in history.cases}
        self.assertTrue(by_id["new"].newly_flaky)
        self.assertFalse(by_id["new"].stopped_flaky)
        self.assertTrue(by_id["flaky"].stopped_flaky)
        self.assertFalse(by_id["flaky"].newly_flaky)

    def test_flaky_in_latest_is_the_live_signal(self):
        history = compare_flaky_history(self._three_reports())
        self.assertEqual(
            [c.case_id for c in history.flaky_in_latest], ["new", "late"]
        )
        self.assertTrue(history.has_flaky_in_latest)
        by_id = {c.case_id: c for c in history.cases}
        # "flaky" was flaky twice but is stable now.
        self.assertFalse(by_id["flaky"].flaky_in_latest)
        self.assertTrue(by_id["new"].flaky_in_latest)

    def test_case_added_mid_history_gets_none_backfill(self):
        history = compare_flaky_history(self._three_reports())
        by_id = {c.case_id: c for c in history.cases}
        self.assertEqual(by_id["late"].pass_rate_history, [None, None, 0.5])
        self.assertEqual(by_id["late"].flaky_history, [None, None, True])
        self.assertEqual(by_id["late"].reports_present, 1)
        self.assertEqual(by_id["flaky"].pass_rate_history, [0.5, 0.75, 1.0])
        self.assertEqual(by_id["flaky"].flaky_history, [True, True, False])

    def test_totals_and_pass_rate_delta(self):
        history = compare_flaky_history(self._three_reports())
        by_id = {c.case_id: c for c in history.cases}
        self.assertEqual(by_id["flaky"].total_runs, 12)
        self.assertEqual(by_id["flaky"].total_passes, 9)
        self.assertEqual(by_id["flaky"].total_fails, 3)
        self.assertEqual(by_id["flaky"].pass_rate_delta, 0.5)
        self.assertEqual(by_id["new"].pass_rate_delta, -0.75)
        self.assertEqual(by_id["late"].pass_rate_delta, None)

    def test_single_report_never_persistent_or_newly(self):
        history = compare_flaky_history(self._three_reports()[:1])
        self.assertEqual(history.persistently_flaky_cases, [])
        self.assertEqual(history.newly_flaky, [])
        self.assertEqual(history.stopped_flaky, [])
        self.assertEqual(
            [c.case_id for c in history.flaky_in_latest], ["flaky"]
        )

    def test_flaky_derived_from_passes_when_flag_absent(self):
        # An old or hand-built report may carry no "flaky" flag.
        report = _stability_dict(
            cases=[dict(_case("flip", 4, 2), flaky=None)],
        )
        report["cases"][0].pop("flaky")
        history = compare_flaky_history([report])
        self.assertEqual(
            [c.case_id for c in history.flaky_cases], ["flip"]
        )

    def test_to_dict_round_trip(self):
        history = compare_flaky_history(self._three_reports())
        data = json.loads(json.dumps(history.to_dict()))
        self.assertEqual(data["n_reports"], 3)
        self.assertEqual(data["flaky"], ["flaky", "new", "late"])
        self.assertEqual(data["persistently_flaky"], ["flaky"])
        self.assertEqual(data["flaky_in_latest"], ["new", "late"])
        self.assertEqual(data["newly_flaky"], ["new"])
        self.assertEqual(data["stopped_flaky"], ["flaky"])
        by_id = {c["case_id"]: c for c in data["cases"]}
        self.assertEqual(by_id["flaky"]["flaky_reports"], 2)
        self.assertEqual(by_id["flaky"]["total_fails"], 3)
        self.assertEqual(
            by_id["flaky"]["pass_rate_history"], [0.5, 0.75, 1.0]
        )


class TestLoadStabilityReportsDir(unittest.TestCase):
    def test_loads_only_stability_reports_sorted_by_started_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A run report (results list) and a bad file are skipped.
            with open(os.path.join(tmp, "run.json"), "w") as f:
                json.dump({"suite": "s", "results": []}, f)
            with open(os.path.join(tmp, "bad.json"), "w") as f:
                f.write("{not json")
            first = _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[_case("a", 2, 2)],
            )
            second = _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[_case("a", 2, 1)],
            )
            # Written out of order on disk; started_at decides.
            with open(os.path.join(tmp, "b.json"), "w") as f:
                json.dump(second, f)
            with open(os.path.join(tmp, "a.json"), "w") as f:
                json.dump(first, f)
            reports = load_stability_reports_dir(tmp)
            self.assertEqual(
                [r["_path"] for r in reports], ["a.json", "b.json"]
            )
            self.assertEqual(
                [r["started_at"] for r in reports],
                ["2026-11-20T00:00:00+00:00",
                 "2026-11-21T00:00:00+00:00"],
            )

    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_stability_reports_dir("/nonexistent/dir/xyz")


class TestFormatFlakyHistory(unittest.TestCase):
    def test_sections_and_history_marks(self):
        reports = [
            _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[_case("flaky", 4, 2), _case("ok", 4, 4)],
                path="r1.json",
            ),
            _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[_case("flaky", 4, 3), _case("ok", 4, 4)],
                path="r2.json",
            ),
        ]
        text = format_flaky_history(compare_flaky_history(reports))
        self.assertIn("Stability reports: 2", text)
        self.assertIn("Persistently flaky (1):", text)
        self.assertIn("flaky (flaky in 2 of 2 reports: 50%F 75%F)", text)
        self.assertIn("Flaky in latest report (1):", text)
        self.assertIn("Per-case flaky history:", text)
        self.assertIn("FLAKY", text)

    def test_no_flaky_reports_clean_summary(self):
        reports = [
            _stability_dict(
                cases=[_case("ok", 4, 4)],
                path="r1.json",
            ),
        ]
        text = format_flaky_history(compare_flaky_history(reports))
        self.assertIn("Flaky in latest report: 0", text)
        # The header counts the groups, but no section is printed.
        self.assertNotIn("Persistently flaky (", text)
        self.assertNotIn("Flaky in latest report (", text)


class TestFlakyHistoryCLI(unittest.TestCase):
    def _write_reports(self, tmp, reports):
        for i, report in enumerate(reports):
            with open(os.path.join(tmp, f"r{i}.json"), "w") as f:
                json.dump(report, f)

    def test_cli_exits_1_when_flaky_in_latest(self):
        from agent_evals.flaky_history import main

        reports = [
            _stability_dict(cases=[_case("flaky", 4, 2)]),
            _stability_dict(cases=[_case("flaky", 4, 3)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(main(["--reports-dir", tmp]), 1)

    def test_cli_exits_0_when_stable_and_writes_json(self):
        from agent_evals.flaky_history import main

        reports = [
            _stability_dict(cases=[_case("ok", 4, 4)]),
            _stability_dict(cases=[_case("ok", 4, 4)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            out = os.path.join(tmp, "history.json")
            self.assertEqual(
                main(["--reports-dir", tmp, "--json", out]), 0
            )
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["n_reports"], 2)
            self.assertEqual(data["flaky"], [])

    def test_cli_exits_2_with_no_stability_reports(self):
        from agent_evals.flaky_history import main

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "run.json"), "w") as f:
                json.dump({"suite": "s", "results": []}, f)
            self.assertEqual(main(["--reports-dir", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
