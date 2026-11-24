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


def _case(case_id, runs, passes, flaky=None, threshold=None,
          mean_score=None):
    if flaky is None:
        flaky = 0 < passes < runs
    if mean_score is None:
        mean_score = round(passes / runs, 4) if runs else 0.0
    return {
        "case_id": case_id,
        "runs": runs,
        "passes": passes,
        "fails": runs - passes,
        "pass_rate": round(passes / runs, 4) if runs else 0.0,
        "flaky": flaky,
        "mean_latency_ms": 1.0,
        "mean_score": mean_score,
        "timeouts": 0,
        "tags": [],
        "threshold": threshold,
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


class TestFlakyStreak(unittest.TestCase):
    """Consecutive-flaky streaks (v0.48.0).

    ``flaky_streak`` is the trailing run of flaky reports: a stable
    report, or a report the case was absent from, breaks it. Only
    an unbroken run up to the latest report counts — that is what
    makes a case stuck-flaky rather than merely flaky before.
    """

    def _streak_reports(self):
        return [
            _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[
                    _case("stuck", 4, 2),
                    _case("broken", 4, 2),
                    _case("gap", 4, 2),
                    _case("fresh", 4, 4),
                    _case("stopped", 4, 2),
                    _case("gone", 4, 2),
                ],
                path="r1.json",
            ),
            _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[
                    _case("stuck", 4, 3),
                    _case("broken", 4, 4),
                    _case("fresh", 4, 4),
                    _case("stopped", 4, 3),
                    _case("gone", 4, 3),
                ],
                path="r2.json",
            ),
            _stability_dict(
                started_at="2026-11-22T00:00:00+00:00",
                cases=[
                    _case("stuck", 4, 1),
                    _case("broken", 4, 2),
                    _case("gap", 4, 2),
                    _case("fresh", 4, 2),
                    _case("stopped", 4, 4),
                ],
                path="r3.json",
            ),
        ]

    def test_streak_counts_trailing_consecutive_flaky_reports(self):
        history = compare_flaky_history(self._streak_reports())
        by_id = {c.case_id: c for c in history.cases}
        self.assertEqual(by_id["stuck"].flaky_streak, 3)
        # Flaky, then stable, then flaky again: the run is 1.
        self.assertEqual(by_id["broken"].flaky_streak, 1)
        # Absent from the middle report: the gap breaks the run.
        self.assertEqual(by_id["gap"].flaky_streak, 1)
        # Just started flipping: streak 1, however stable before.
        self.assertEqual(by_id["fresh"].flaky_streak, 1)
        # Stable in the latest report: no streak at all.
        self.assertEqual(by_id["stopped"].flaky_streak, 0)
        # Absent from the latest report: no streak at all.
        self.assertEqual(by_id["gone"].flaky_streak, 0)

    def test_is_stuck_flaky_needs_n_consecutive_reports(self):
        history = compare_flaky_history(self._streak_reports())
        by_id = {c.case_id: c for c in history.cases}
        self.assertTrue(by_id["stuck"].is_stuck_flaky(2))
        self.assertTrue(by_id["stuck"].is_stuck_flaky(3))
        self.assertFalse(by_id["stuck"].is_stuck_flaky(4))
        self.assertFalse(by_id["broken"].is_stuck_flaky(2))
        self.assertFalse(by_id["fresh"].is_stuck_flaky(2))
        self.assertFalse(by_id["stopped"].is_stuck_flaky(2))

    def test_stuck_flaky_cases_longest_streak_first(self):
        reports = [
            _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[_case("a", 4, 4), _case("b", 4, 2)],
                path="r1.json",
            ),
            _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[_case("a", 4, 3), _case("b", 4, 2)],
                path="r2.json",
            ),
            _stability_dict(
                started_at="2026-11-22T00:00:00+00:00",
                cases=[_case("a", 4, 1), _case("b", 4, 2)],
                path="r3.json",
            ),
            _stability_dict(
                started_at="2026-11-23T00:00:00+00:00",
                cases=[_case("a", 4, 3), _case("b", 4, 3)],
                path="r4.json",
            ),
        ]
        history = compare_flaky_history(reports)
        # "a" has a 3-report streak, "b" a 4-report streak: the
        # longer streak sorts first, and a higher N drops "a".
        self.assertEqual(
            [c.case_id for c in history.stuck_flaky_cases(2)],
            ["b", "a"],
        )
        self.assertEqual(
            [c.case_id for c in history.stuck_flaky_cases(4)], ["b"]
        )
        self.assertEqual(history.stuck_flaky_cases(5), [])

    def test_streak_in_to_dict(self):
        history = compare_flaky_history(self._streak_reports())
        data = json.loads(json.dumps(history.to_dict()))
        by_id = {c["case_id"]: c for c in data["cases"]}
        self.assertEqual(by_id["stuck"]["flaky_streak"], 3)
        self.assertEqual(by_id["broken"]["flaky_streak"], 1)
        self.assertEqual(by_id["stopped"]["flaky_streak"], 0)


class TestThresholdAwareFlakyHistory(unittest.TestCase):
    """Threshold-aware flaky history (v0.45.0).

    The history records the pass threshold in effect per report,
    so a move between the flaky and stable groups is attributed:
    same scores with a different threshold means the bar moved,
    not the agent; a genuine score move means the agent did.
    """

    def _two_reports(self, first, second):
        return [
            _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[first],
                path="r1.json",
            ),
            _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[second],
                path="r2.json",
            ),
        ]

    def test_threshold_history_recorded_per_report(self):
        reports = self._two_reports(
            _case("a", 4, 4, threshold=0.75, mean_score=0.8125),
            _case("a", 4, 1, threshold=0.9, mean_score=0.8125),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertEqual(case.threshold_history, [0.75, 0.9])
        self.assertEqual(case.mean_score_history, [0.8125, 0.8125])
        self.assertTrue(case.threshold_changed)

    def test_threshold_history_backfills_none_for_absent_case(self):
        reports = [
            _stability_dict(
                started_at="2026-11-20T00:00:00+00:00",
                cases=[_case("a", 4, 4)],
                path="r1.json",
            ),
            _stability_dict(
                started_at="2026-11-21T00:00:00+00:00",
                cases=[
                    _case("a", 4, 4),
                    _case("late", 4, 2, threshold=0.8),
                ],
                path="r2.json",
            ),
        ]
        history = compare_flaky_history(reports)
        by_id = {c.case_id: c for c in history.cases}
        self.assertEqual(by_id["late"].threshold_history, [None, 0.8])
        self.assertEqual(
            by_id["late"].mean_score_history, [None, 0.5]
        )
        self.assertFalse(by_id["late"].threshold_changed)

    def test_newly_flaky_explained_by_threshold_change(self):
        # Same mean score in both reports (0.8125), but the bar
        # moved 0.75 -> 0.9, so the case went from passing every
        # run to flipping: the threshold moved, not the agent.
        reports = self._two_reports(
            _case("a", 4, 4, threshold=0.75, mean_score=0.8125),
            _case("a", 4, 1, threshold=0.9, mean_score=0.8125),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.newly_flaky)
        self.assertEqual(case.transition_cause, "threshold")
        self.assertTrue(case.threshold_explained_transition)
        self.assertEqual(
            case.transition_boundary,
            {
                "prev_threshold": 0.75,
                "latest_threshold": 0.9,
                "prev_mean_score": 0.8125,
                "latest_mean_score": 0.8125,
            },
        )
        self.assertEqual(
            [c.case_id for c in history.threshold_explained_newly_flaky],
            ["a"],
        )
        self.assertEqual(
            [c.case_id for c in history.threshold_explained_transitions],
            ["a"],
        )

    def test_stopped_flaky_explained_by_threshold_change(self):
        # Lowering the bar 0.9 -> 0.75 with the same mean score
        # takes the case from flipping to passing every run.
        reports = self._two_reports(
            _case("a", 4, 1, threshold=0.9, mean_score=0.8125),
            _case("a", 4, 4, threshold=0.75, mean_score=0.8125),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.stopped_flaky)
        self.assertEqual(case.transition_cause, "threshold")
        self.assertEqual(
            [c.case_id for c in history.threshold_explained_stopped_flaky],
            ["a"],
        )

    def test_newly_flaky_from_genuine_score_move(self):
        # Same threshold throughout; the mean score fell, so the
        # case started flipping because the agent got worse.
        reports = self._two_reports(
            _case("a", 4, 4, threshold=0.75, mean_score=0.8125),
            _case("a", 4, 1, threshold=0.75, mean_score=0.5),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.newly_flaky)
        self.assertEqual(case.transition_cause, "score")
        self.assertFalse(case.threshold_explained_transition)
        self.assertEqual(history.threshold_explained_transitions, [])

    def test_threshold_change_plus_score_move_is_score(self):
        # The bar moved AND the scores moved: the move cannot be
        # blamed on the threshold alone.
        reports = self._two_reports(
            _case("a", 4, 4, threshold=0.75, mean_score=0.8125),
            _case("a", 4, 1, threshold=0.9, mean_score=0.5),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.newly_flaky)
        self.assertEqual(case.transition_cause, "score")
        self.assertFalse(case.threshold_explained_transition)

    def test_threshold_added_from_none_counts_as_change(self):
        # No threshold (a full score is required) -> 0.75, with
        # the same mean score: the bar moved.
        reports = self._two_reports(
            _case("a", 4, 4, threshold=None, mean_score=0.8125),
            _case("a", 4, 1, threshold=0.75, mean_score=0.8125),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.threshold_changed)
        self.assertEqual(case.transition_cause, "threshold")

    def test_no_transition_means_no_cause(self):
        # Threshold changed, but the case never moved between
        # groups: there is nothing to attribute.
        reports = self._two_reports(
            _case("a", 4, 4, threshold=0.75, mean_score=1.0),
            _case("a", 4, 4, threshold=0.9, mean_score=1.0),
        )
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.threshold_changed)
        self.assertIsNone(case.transition_cause)
        self.assertIsNone(case.transition_boundary)
        self.assertFalse(case.threshold_explained_transition)

    def test_old_report_without_threshold_or_score_is_indeterminate(self):
        # A report predating thresholds carries neither field:
        # the move cannot be attributed, and the history must
        # not crash or claim a cause.
        first = _case("a", 4, 4)
        first.pop("threshold")
        first.pop("mean_score")
        second = _case("a", 4, 1, threshold=0.9, mean_score=0.8125)
        reports = self._two_reports(first, second)
        history = compare_flaky_history(reports)
        case = history.cases[0]
        self.assertTrue(case.newly_flaky)
        self.assertIsNone(case.transition_cause)
        self.assertEqual(case.threshold_history, [None, 0.9])
        self.assertEqual(case.mean_score_history, [None, 0.8125])

    def test_to_dict_carries_threshold_fields(self):
        reports = self._two_reports(
            _case("a", 4, 4, threshold=0.75, mean_score=0.8125),
            _case("a", 4, 1, threshold=0.9, mean_score=0.8125),
        )
        data = json.loads(
            json.dumps(compare_flaky_history(reports).to_dict())
        )
        self.assertEqual(data["threshold_explained_newly_flaky"], ["a"])
        self.assertEqual(data["threshold_explained_transitions"], ["a"])
        self.assertEqual(data["threshold_explained_stopped_flaky"], [])
        by_id = {c["case_id"]: c for c in data["cases"]}
        self.assertEqual(by_id["a"]["threshold_history"], [0.75, 0.9])
        self.assertEqual(
            by_id["a"]["mean_score_history"], [0.8125, 0.8125]
        )
        self.assertTrue(by_id["a"]["threshold_changed"])
        self.assertEqual(by_id["a"]["transition_cause"], "threshold")
        self.assertTrue(by_id["a"]["threshold_explained_transition"])

    def test_format_annotates_threshold_and_score_moves(self):
        reports = self._two_reports(
            _case("bar-moved", 4, 4, threshold=0.75, mean_score=0.8125),
            _case("bar-moved", 4, 1, threshold=0.9, mean_score=0.8125),
        )
        # A second pair in the same history, with a genuine move.
        reports[0]["cases"].append(
            _case("agent-moved", 4, 4, threshold=0.75, mean_score=0.8125)
        )
        reports[1]["cases"].append(
            _case("agent-moved", 4, 1, threshold=0.75, mean_score=0.5)
        )
        text = format_flaky_history(compare_flaky_history(reports))
        self.assertIn("Threshold-explained transitions: 1", text)
        self.assertIn("Threshold-explained transitions (1):", text)
        self.assertIn(
            "bar-moved (stable before, flaky in latest report: "
            "100% 25%F) — threshold change 0.75 -> 0.9, "
            "mean score unchanged at 0.81",
            text,
        )
        self.assertIn(
            "agent-moved (stable before, flaky in latest report: "
            "100% 25%F) — mean score 0.81 -> 0.50",
            text,
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

    def test_stuck_flaky_section_only_with_stuck_n(self):
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
        history = compare_flaky_history(reports)
        # No N given: no stuck section at all.
        self.assertNotIn(
            "Stuck flaky", format_flaky_history(history)
        )
        text = format_flaky_history(history, stuck_n=2)
        self.assertIn(
            "Stuck flaky (flaky in at least 2 consecutive reports) (1):",
            text,
        )
        self.assertIn(
            "flaky (flaky in the last 2 consecutive reports: 50%F 75%F)",
            text,
        )
        # At N=3 the streak is too short: the section is omitted.
        self.assertNotIn(
            "Stuck flaky",
            format_flaky_history(history, stuck_n=3),
        )


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

    def test_cli_fail_on_newly_flaky_exits_1_when_newly_flaky(self):
        from agent_evals.flaky_history import main

        reports = [
            _stability_dict(cases=[_case("fresh", 4, 4)]),
            _stability_dict(cases=[_case("fresh", 4, 2)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(["--reports-dir", tmp, "--fail-on-newly-flaky"]), 1
            )

    def test_cli_fail_on_newly_flaky_exits_0_when_already_flaky(self):
        from agent_evals.flaky_history import main

        # Flaky in both reports: known flakiness, not a fresh
        # regression, so the narrow gate passes (the default gate
        # would still fail this history).
        reports = [
            _stability_dict(cases=[_case("known", 4, 2)]),
            _stability_dict(cases=[_case("known", 4, 3)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(["--reports-dir", tmp, "--fail-on-newly-flaky"]), 0
            )

    def test_cli_fail_on_newly_flaky_exits_0_when_stable(self):
        from agent_evals.flaky_history import main

        reports = [
            _stability_dict(cases=[_case("ok", 4, 4)]),
            _stability_dict(cases=[_case("ok", 4, 4)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(["--reports-dir", tmp, "--fail-on-newly-flaky"]), 0
            )

    def test_cli_fail_on_newly_flaky_exits_0_when_first_seen_flaky(self):
        from agent_evals.flaky_history import main

        # One report, flaky in it: the case was never stable before,
        # so it is flaky in the latest report but not newly flaky.
        reports = [_stability_dict(cases=[_case("new", 4, 2)])]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(["--reports-dir", tmp, "--fail-on-newly-flaky"]), 0
            )

    def test_cli_fail_on_persistent_flaky_exits_1_when_stuck(self):
        from agent_evals.flaky_history import main

        reports = [
            _stability_dict(cases=[_case("stuck", 4, 2)]),
            _stability_dict(cases=[_case("stuck", 4, 3)]),
            _stability_dict(cases=[_case("stuck", 4, 1)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    ["--reports-dir", tmp, "--fail-on-persistent-flaky", "3"]
                ),
                1,
            )

    def test_cli_fail_on_persistent_flaky_exits_0_when_newly_flaky(self):
        from agent_evals.flaky_history import main

        # Flaky only in the latest report (streak 1): a fresh
        # regression, not a stuck case, so the persistent gate
        # passes — the default gate would still fail this history.
        reports = [
            _stability_dict(cases=[_case("fresh", 4, 4)]),
            _stability_dict(cases=[_case("fresh", 4, 2)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    ["--reports-dir", tmp, "--fail-on-persistent-flaky", "2"]
                ),
                0,
            )

    def test_cli_fail_on_persistent_flaky_exits_0_when_first_seen(self):
        from agent_evals.flaky_history import main

        # One report, flaky in it: streak 1, below any N >= 2.
        reports = [_stability_dict(cases=[_case("new", 4, 2)])]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    ["--reports-dir", tmp, "--fail-on-persistent-flaky", "2"]
                ),
                0,
            )

    def test_cli_fail_on_persistent_flaky_exits_0_when_streak_broken(self):
        from agent_evals.flaky_history import main

        # Flaky, stable, flaky: flaky in two reports, but not
        # consecutively, so the streak is 1 and the gate passes.
        reports = [
            _stability_dict(cases=[_case("flip", 4, 2)]),
            _stability_dict(cases=[_case("flip", 4, 4)]),
            _stability_dict(cases=[_case("flip", 4, 3)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    ["--reports-dir", tmp, "--fail-on-persistent-flaky", "2"]
                ),
                0,
            )

    def test_cli_fail_on_persistent_flaky_exits_0_when_streak_short(self):
        from agent_evals.flaky_history import main

        # Streak 2, but the gate asks for 3 consecutive reports.
        reports = [
            _stability_dict(cases=[_case("stuck", 4, 4)]),
            _stability_dict(cases=[_case("stuck", 4, 2)]),
            _stability_dict(cases=[_case("stuck", 4, 3)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    ["--reports-dir", tmp, "--fail-on-persistent-flaky", "3"]
                ),
                0,
            )

    def test_cli_fail_on_persistent_flaky_rejects_n_below_2(self):
        from agent_evals.flaky_history import main

        reports = [_stability_dict(cases=[_case("ok", 4, 4)])]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            with self.assertRaises(SystemExit) as ctx:
                main(
                    ["--reports-dir", tmp, "--fail-on-persistent-flaky", "1"]
                )
            self.assertEqual(ctx.exception.code, 2)

    def test_cli_both_narrow_gates_either_can_fail(self):
        from agent_evals.flaky_history import main

        # "fresh" just started flipping (newly-flaky gate trips);
        # "stuck" is flaky in both reports (persistent gate trips).
        reports = [
            _stability_dict(
                cases=[_case("fresh", 4, 4), _case("stuck", 4, 2)]
            ),
            _stability_dict(
                cases=[_case("fresh", 4, 2), _case("stuck", 4, 3)]
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    [
                        "--reports-dir",
                        tmp,
                        "--fail-on-newly-flaky",
                        "--fail-on-persistent-flaky",
                        "2",
                    ]
                ),
                1,
            )
        # Neither trips when the history is clean of both.
        reports = [
            _stability_dict(cases=[_case("ok", 4, 4)]),
            _stability_dict(cases=[_case("ok", 4, 4)]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            self._write_reports(tmp, reports)
            self.assertEqual(
                main(
                    [
                        "--reports-dir",
                        tmp,
                        "--fail-on-newly-flaky",
                        "--fail-on-persistent-flaky",
                        "2",
                    ]
                ),
                0,
            )

    def test_cli_default_gate_still_fails_on_newly_flaky(self):
        from agent_evals.flaky_history import main

        reports = [
            _stability_dict(cases=[_case("fresh", 4, 4)]),
            _stability_dict(cases=[_case("fresh", 4, 2)]),
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
