"""Tests for agent-evals stability: repeated runs and flaky detection."""

import json
import os
import tempfile
import unittest

from agent_evals.evals import Case, Suite
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


class TestRepeatSuite(unittest.TestCase):
    def test_deterministic_agent_never_flaky(self):
        def agent(prompt):
            return "19"

        report = repeat_suite(_suite(), agent, repetitions=3)
        by_id = {c.case_id: c for c in report.cases}
        self.assertEqual(by_id["stable-pass"].pass_rate, 1.0)
        self.assertFalse(by_id["stable-pass"].flaky)
        self.assertEqual(by_id["stable-fail"].pass_rate, 0.0)
        self.assertFalse(by_id["stable-fail"].flaky)
        self.assertEqual(report.flaky_cases, [])
        self.assertEqual(
            [c.case_id for c in report.always_passing],
            ["stable-pass", "flaky"],
        )
        self.assertEqual(
            [c.case_id for c in report.always_failing], ["stable-fail"]
        )

    def test_alternating_agent_is_flaky(self):
        # The "flaky" case flips its answer every call; the other two
        # cases are stable, so only it is flagged.
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=4)
        by_id = {c.case_id: c for c in report.cases}
        flaky = by_id["flaky"]
        self.assertTrue(flaky.flaky)
        self.assertEqual(flaky.passes, 2)
        self.assertEqual(flaky.pass_rate, 0.5)
        self.assertEqual([c.case_id for c in report.flaky_cases], ["flaky"])
        # The stable cases are unaffected by the alternation.
        self.assertEqual(by_id["stable-pass"].pass_rate, 1.0)
        self.assertEqual(by_id["stable-fail"].pass_rate, 0.0)

    def test_single_repetition_flags_nothing_flaky(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=1)
        self.assertEqual(report.flaky_cases, [])

    def test_invalid_repetitions_rejected(self):
        for bad in (0, -2, 2.5, "3", True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                repeat_suite(_suite(), lambda prompt: "19", repetitions=bad)

    def test_cases_reported_in_suite_order(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertEqual(
            [c.case_id for c in report.cases],
            ["stable-pass", "stable-fail", "flaky"],
        )

    def test_to_dict_round_trip(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        data = report.to_dict()
        self.assertEqual(data["suite"], "s")
        self.assertEqual(data["repetitions"], 2)
        self.assertEqual(data["flaky"], [])
        self.assertEqual(data["always_passing"], ["stable-pass", "flaky"])
        self.assertEqual(data["always_failing"], ["stable-fail"])
        flaky_case = next(c for c in data["cases"] if c["case_id"] == "flaky")
        self.assertEqual(flaky_case["runs"], 2)
        self.assertEqual(flaky_case["passes"], 2)
        self.assertEqual(flaky_case["pass_rate"], 1.0)
        self.assertFalse(flaky_case["flaky"])

    def test_format_stability_marks_flaky(self):
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=2)
        text = format_stability(report)
        self.assertIn("Flaky: 1", text)
        self.assertIn("FLAKY", text)
        self.assertIn("repeated 2x", text)


class TestStabilityTagFilter(unittest.TestCase):
    def test_tag_filter_recorded_in_dict(self):
        report = repeat_suite(
            _suite(),
            lambda prompt: "19",
            repetitions=2,
            tags=["smoke", "fast"],
            tag_mode="all",
            exclude_tags=["slow"],
        )
        self.assertEqual(
            report.to_dict()["tag_filter"],
            {"tags": ["smoke", "fast"], "mode": "all",
             "exclude_tags": ["slow"]},
        )

    def test_no_tag_filter_is_null_in_dict(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertIsNone(report.to_dict()["tag_filter"])

    def test_exclude_only_filter_recorded(self):
        report = repeat_suite(
            _suite(), lambda prompt: "19", repetitions=2,
            exclude_tags=["slow"],
        )
        self.assertEqual(
            report.to_dict()["tag_filter"],
            {"tags": [], "mode": "any", "exclude_tags": ["slow"]},
        )

    def test_invalid_tag_mode_rejected(self):
        with self.assertRaises(ValueError):
            repeat_suite(
                _suite(), lambda prompt: "19", repetitions=2,
                tags=["smoke"], tag_mode="bogus",
            )

    def test_format_stability_shows_tag_filter(self):
        report = repeat_suite(
            _suite(),
            lambda prompt: "19",
            repetitions=2,
            tags=["smoke", "fast"],
            tag_mode="all",
            exclude_tags=["slow"],
        )
        text = format_stability(report)
        self.assertIn("Tags: smoke, fast (mode all)", text)
        self.assertIn("Excluded tags: slow", text)

    def test_format_stability_omits_tag_line_without_filter(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertNotIn("Tags:", format_stability(report))

    def test_cli_repeat_records_tag_filter_in_report(self):
        from agent_evals.cli import main

        suite = Suite(
            name="tagged",
            cases=[
                Case.from_dict({"id": "smoke-fast", "prompt": "2 + 3",
                                "scorer": {"type": "exact", "expected": "5"},
                                "tags": ["fast", "smoke"]}),
                Case.from_dict({"id": "smoke-slow", "prompt": "2 + 3",
                                "scorer": {"type": "exact", "expected": "5"},
                                "tags": ["slow", "smoke"]}),
                Case.from_dict({"id": "fast-only", "prompt": "2 + 3",
                                "scorer": {"type": "exact", "expected": "5"},
                                "tags": ["fast"]}),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            report_path = os.path.join(tmp, "stability.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(suite.to_dict(), f)
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "smoke",
                "--tag-mode", "all",
                "--exclude-tag", "slow",
                "--repeat", "2",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                stability = json.load(f)
            # Only the smoke case that is not slow was repeated.
            self.assertEqual(
                [c["case_id"] for c in stability["cases"]], ["smoke-fast"]
            )
            self.assertEqual(
                stability["tag_filter"],
                {"tags": ["smoke"], "mode": "all",
                 "exclude_tags": ["slow"]},
            )


class TestStabilityTagBreakdown(unittest.TestCase):
    def _tagged_suite(self):
        def case(case_id, prompt, expected, tags=None):
            data = {"id": case_id, "prompt": prompt,
                    "scorer": {"type": "exact", "expected": expected}}
            if tags is not None:
                data["tags"] = tags
            return Case.from_dict(data)

        return Suite(
            name="tagged",
            cases=[
                case("smoke-pass", "p", "19", ["smoke"]),
                # Flips every call, so 2 of 4 repetitions pass.
                case("smoke-flaky", "flip", "19", ["fast", "smoke"]),
                case("fast-fail", "p", "nope", ["fast"]),
                case("untagged", "p", "19"),
            ],
        )

    def test_breakdown_values(self):
        report = repeat_suite(
            self._tagged_suite(), _alternating_agent(), repetitions=4
        )
        self.assertEqual(
            report.tag_breakdown,
            {
                "fast": {"cases": 2, "passes": 2, "runs": 8,
                         "pass_rate": 0.25, "mean_score": 0.25,
                         "flaky": 1},
                "smoke": {"cases": 2, "passes": 6, "runs": 8,
                          "pass_rate": 0.75, "mean_score": 0.75,
                          "flaky": 1},
            },
        )

    def test_multi_tag_case_counts_under_each_tag(self):
        report = repeat_suite(
            self._tagged_suite(), _alternating_agent(), repetitions=4
        )
        # smoke-flaky passes 2 of 4 under both of its tags.
        self.assertEqual(report.tag_breakdown["fast"]["passes"], 2)
        self.assertEqual(report.tag_breakdown["smoke"]["passes"], 6)

    def test_untagged_cases_appear_under_no_tag(self):
        report = repeat_suite(
            self._tagged_suite(), _alternating_agent(), repetitions=4
        )
        self.assertEqual(
            sorted(report.tag_breakdown), ["fast", "smoke"]
        )

    def test_no_tags_means_empty_breakdown(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertEqual(report.tag_breakdown, {})
        self.assertEqual(report.to_dict()["tag_breakdown"], {})

    def test_tags_carried_per_case_in_dict(self):
        report = repeat_suite(
            self._tagged_suite(), _alternating_agent(), repetitions=2
        )
        by_id = {c["case_id"]: c for c in report.to_dict()["cases"]}
        self.assertEqual(by_id["smoke-flaky"]["tags"], ["fast", "smoke"])
        self.assertEqual(by_id["untagged"]["tags"], [])

    def test_breakdown_in_dict(self):
        report = repeat_suite(
            self._tagged_suite(), _alternating_agent(), repetitions=4
        )
        self.assertEqual(
            report.to_dict()["tag_breakdown"]["smoke"],
            {"cases": 2, "passes": 6, "runs": 8,
             "pass_rate": 0.75, "mean_score": 0.75, "flaky": 1},
        )

    def test_format_stability_shows_per_tag_section(self):
        report = repeat_suite(
            self._tagged_suite(), _alternating_agent(), repetitions=4
        )
        text = format_stability(report)
        self.assertIn("Per-tag pass rates:", text)
        self.assertIn("smoke", text)
        self.assertIn("6/8 (75.0%)", text)
        self.assertIn("fast", text)
        self.assertIn("2/8 (25.0%)", text)
        self.assertIn("1 flaky", text)

    def test_format_stability_omits_section_without_tags(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertNotIn("Per-tag", format_stability(report))

    def test_cli_repeat_writes_tag_breakdown(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            report_path = os.path.join(tmp, "stability.json")
            code = main([
                "--suite", "examples/calculator.json",
                "--agent", "examples.calculator_agent:agent",
                "--repeat", "2",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                stability = json.load(f)
            breakdown = stability["tag_breakdown"]
            # Calculator cases are deterministic, so every tag is 100%.
            for tag, entry in breakdown.items():
                self.assertEqual(entry["pass_rate"], 1.0)
                self.assertEqual(entry["flaky"], 0)
                self.assertEqual(entry["passes"], entry["runs"])
            self.assertIn("smoke", breakdown)
            self.assertIn("arithmetic", breakdown)


def threshold_demo_agent(prompt):
    """Demo agent for the threshold stability tests, loadable by the
    CLI as ``tests.test_stability:threshold_demo_agent``: finds 3 of
    4 substrings on 'partial', near-misses 'hello world' on 'fuzzy',
    and answers '19' otherwise."""
    if prompt == "partial":
        return "hello world date"
    if prompt == "fuzzy":
        return "hello worle"
    return "19"


class TestStabilityThresholds(unittest.TestCase):
    """Threshold-aware stability reports (v0.44.0).

    A case's pass threshold decides the verdict in every run, so
    the stability report carries the threshold per case: a case
    that passes on raw score but fails its threshold reads as
    failing, and a partial-credit case that only passes because of
    its threshold says so.
    """

    def _threshold_suite(self):
        def case(case_id, prompt, scorer, threshold=None):
            data = {"id": case_id, "prompt": prompt, "scorer": scorer}
            if threshold is not None:
                data["threshold"] = threshold
            return Case.from_dict(data)

        return Suite(
            name="thresholds",
            cases=[
                # contains_all over 4 substrings; the agent finds 3,
                # so the score is 0.75 every run.
                case("meets-threshold", "partial",
                     {"type": "contains_all",
                      "substrings": ["hello", "world", "date", "time"]},
                     threshold=0.75),
                # Same score, higher bar: the scorer fails it and the
                # threshold does not rescue it.
                case("below-threshold", "partial",
                     {"type": "contains_all",
                      "substrings": ["hello", "world", "date", "time"]},
                     threshold=0.9),
                # fuzzy ratio 0.9091 >= 0.8, so the scorer passes it —
                # but the case threshold is 0.95, so it fails anyway.
                case("scorer-passes-threshold-fails", "fuzzy",
                     {"type": "fuzzy", "expected": "hello world"},
                     threshold=0.95),
                case("plain", "p",
                     {"type": "exact", "expected": "19"}),
            ],
        )

    def _agent(self, prompt):
        return threshold_demo_agent(prompt)

    def test_threshold_carried_per_case_in_dict(self):
        report = repeat_suite(
            self._threshold_suite(), self._agent, repetitions=2
        )
        by_id = {c["case_id"]: c for c in report.to_dict()["cases"]}
        self.assertEqual(by_id["meets-threshold"]["threshold"], 0.75)
        self.assertEqual(by_id["below-threshold"]["threshold"], 0.9)
        self.assertEqual(
            by_id["scorer-passes-threshold-fails"]["threshold"], 0.95
        )
        # A case with no threshold carries null, not 1.0.
        self.assertIsNone(by_id["plain"]["threshold"])
        self.assertFalse(by_id["meets-threshold"]["below_threshold"])
        self.assertTrue(by_id["below-threshold"]["below_threshold"])
        self.assertTrue(
            by_id["scorer-passes-threshold-fails"]["below_threshold"]
        )
        self.assertFalse(by_id["plain"]["below_threshold"])

    def test_threshold_promotes_partial_credit_to_pass(self):
        report = repeat_suite(
            self._threshold_suite(), self._agent, repetitions=2
        )
        by_id = {c.case_id: c for c in report.cases}
        # Score 0.75 meets threshold 0.75, so it passes every run.
        self.assertEqual(by_id["meets-threshold"].passes, 2)
        self.assertEqual(by_id["meets-threshold"].pass_rate, 1.0)
        self.assertFalse(by_id["meets-threshold"].flaky)

    def test_case_passing_on_raw_score_but_failing_threshold(self):
        report = repeat_suite(
            self._threshold_suite(), self._agent, repetitions=2
        )
        by_id = {c.case_id: c for c in report.cases}
        # The fuzzy scorer passes the output (ratio 0.9091 >= 0.8),
        # but the case threshold is 0.95, so the case fails.
        demoted = by_id["scorer-passes-threshold-fails"]
        self.assertEqual(demoted.passes, 0)
        self.assertAlmostEqual(demoted.mean_score, 0.9091, places=4)
        self.assertTrue(demoted.below_threshold)
        # Same story for the partial-credit case under a 0.9 bar.
        self.assertEqual(by_id["below-threshold"].passes, 0)
        self.assertAlmostEqual(by_id["below-threshold"].mean_score, 0.75)
        self.assertTrue(by_id["below-threshold"].below_threshold)

    def test_format_stability_shows_threshold_column(self):
        report = repeat_suite(
            self._threshold_suite(), self._agent, repetitions=2
        )
        text = format_stability(report)
        self.assertIn("threshold", text)
        self.assertIn(">=0.75", text)
        self.assertIn(">=0.9", text)
        self.assertIn(">=0.95", text)
        self.assertIn("Below their pass threshold (2):", text)
        self.assertIn("below-threshold  mean score 0.75 < threshold 0.9",
                      text)
        self.assertIn(
            "scorer-passes-threshold-fails  mean score 0.91 "
            "< threshold 0.95",
            text,
        )
        # The passing case is not in the below-threshold section.
        # (Bound the section: the case-outputs section that follows
        # lists every case, including the passing ones.)
        section = text.split("Below their pass threshold")[1]
        section = section.split("Case outputs (last run):")[0]
        self.assertNotIn("meets-threshold", section)
        self.assertNotIn("plain", section)

    def test_format_stability_omits_below_threshold_section(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertNotIn("Below their pass threshold",
                         format_stability(report))

    def test_cli_repeat_writes_thresholds_in_report(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            report_path = os.path.join(tmp, "stability.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._threshold_suite().to_dict(), f)
            code = main([
                "--suite", path,
                "--agent", "tests.test_stability:threshold_demo_agent",
                "--repeat", "2",
                "--report", report_path,
            ])
            self.assertEqual(code, 1)  # threshold cases fail the run
            with open(report_path, encoding="utf-8") as f:
                stability = json.load(f)
            by_id = {c["case_id"]: c for c in stability["cases"]}
            self.assertEqual(by_id["meets-threshold"]["threshold"], 0.75)
            self.assertEqual(by_id["meets-threshold"]["passes"], 2)
            self.assertEqual(by_id["below-threshold"]["threshold"], 0.9)
            self.assertEqual(by_id["below-threshold"]["passes"], 0)
            self.assertIsNone(by_id["plain"]["threshold"])

class TestOutputCapture(unittest.TestCase):
    def test_output_captured_per_case(self):
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=4)
        by_id = {c.case_id: c for c in report.cases}
        # Stable cases: one distinct output, the last run's answer.
        self.assertEqual(by_id["stable-pass"].output, "19")
        self.assertEqual(by_id["stable-pass"].distinct_outputs, ["19"])
        self.assertEqual(by_id["stable-pass"].distinct_output_count, 1)
        # The flaky case flipped between two answers; both are kept,
        # first-seen order, and the last run's answer is the output.
        flaky = by_id["flaky"]
        self.assertEqual(flaky.distinct_outputs, ["19", "20"])
        self.assertEqual(flaky.distinct_output_count, 2)
        self.assertEqual(flaky.output, "20")
        # The JSON report carries the same fields.
        by_json = {c["case_id"]: c for c in report.to_dict()["cases"]}
        self.assertEqual(by_json["flaky"]["output"], "20")
        self.assertEqual(by_json["flaky"]["distinct_outputs"], ["19", "20"])
        self.assertEqual(by_json["flaky"]["distinct_output_count"], 2)
        self.assertEqual(by_json["stable-pass"]["output"], "19")

    def test_long_output_truncated_with_marker(self):
        def agent(prompt):
            return "x" * 1000

        report = repeat_suite(_suite(), agent, repetitions=2)
        by_id = {c.case_id: c for c in report.cases}
        excerpt = by_id["stable-pass"].output
        self.assertTrue(excerpt.startswith("x" * 500))
        self.assertTrue(excerpt.endswith("[+500 chars]"))
        self.assertLessEqual(len(excerpt), 520)

    def test_secret_shaped_output_redacted(self):
        def agent(prompt):
            return (
                "key sk-abcdefgh12345678 and "
                "Bearer abcdefghij.klmnop.qrstuv plus "
                "api_key=topsecretvalue"
            )

        report = repeat_suite(_suite(), agent, repetitions=1)
        output = report.cases[0].output
        self.assertIn("sk-[REDACTED]", output)
        self.assertIn("Bearer [REDACTED]", output)
        self.assertIn("api_key=[REDACTED]", output)
        self.assertNotIn("abcdefgh12345678", output)
        self.assertNotIn("topsecretvalue", output)

    def test_no_output_when_agent_errors(self):
        def agent(prompt):
            raise RuntimeError("boom")

        report = repeat_suite(_suite(), agent, repetitions=2)
        for case in report.cases:
            self.assertIsNone(case.output)
            self.assertEqual(case.distinct_outputs, [])
            self.assertEqual(case.distinct_output_count, 0)

    def test_format_stability_shows_outputs(self):
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=4)
        text = format_stability(report)
        self.assertIn("Case outputs (last run):", text)
        self.assertIn("stable-pass: '19'", text)
        # The flaky case's last answer plus the other answer it gave.
        self.assertIn("flaky: '20'", text)
        self.assertIn("also saw: '19'", text)

    def test_format_stability_marks_missing_output(self):
        def agent(prompt):
            raise RuntimeError("boom")

        report = repeat_suite(_suite(), agent, repetitions=1)
        text = format_stability(report)
        self.assertIn("stable-pass: (no output)", text)


if __name__ == "__main__":
    unittest.main()
