"""Tests for agent-evals stability: repeated runs and flaky detection."""

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


if __name__ == "__main__":
    unittest.main()
