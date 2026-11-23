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


if __name__ == "__main__":
    unittest.main()
