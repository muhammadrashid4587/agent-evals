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

    def test_score_spread_from_alternating_agent(self):
        # The "flaky" case scores 1.0 and 0.0 on alternate runs, so
        # its score spread is the full range; the stable cases
        # score the same every run, so their spread is 0.
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=4)
        by_id = {c.case_id: c for c in report.cases}
        flaky = by_id["flaky"]
        self.assertEqual(flaky.min_score, 0.0)
        self.assertEqual(flaky.max_score, 1.0)
        self.assertEqual(flaky.score_spread, 1.0)
        self.assertEqual(flaky.mean_score, 0.5)
        for case_id in ("stable-pass", "stable-fail"):
            case = by_id[case_id]
            self.assertEqual(case.min_score, case.max_score)
            self.assertEqual(case.score_spread, 0.0)

    def test_latency_extremes_bracket_the_mean(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=3)
        for case in report.cases:
            self.assertIsNotNone(case.min_latency_ms)
            self.assertIsNotNone(case.max_latency_ms)
            self.assertLessEqual(case.min_latency_ms, case.mean_latency_ms)
            self.assertLessEqual(case.mean_latency_ms, case.max_latency_ms)
            self.assertEqual(
                case.latency_spread_ms,
                case.max_latency_ms - case.min_latency_ms,
            )
            self.assertGreaterEqual(case.latency_spread_ms, 0.0)

    def test_latency_spread_grows_with_a_slow_run(self):
        # The "flip" prompt answers after a 50 ms sleep every other
        # call, so the flaky case's max latency is far above its min
        # while the other cases stay fast.
        import time

        calls = {"n": 0}

        def agent(prompt):
            if prompt == "flip":
                calls["n"] += 1
                if calls["n"] % 2 == 0:
                    time.sleep(0.05)
            return "19"

        report = repeat_suite(_suite(), agent, repetitions=4)
        by_id = {c.case_id: c for c in report.cases}
        flaky = by_id["flaky"]
        self.assertGreater(flaky.latency_spread_ms, 20.0)
        self.assertLess(by_id["stable-pass"].latency_spread_ms, 20.0)

    def test_spread_fields_in_dict_and_summary(self):
        report = repeat_suite(_suite(), _alternating_agent(), repetitions=2)
        data = report.to_dict()
        flaky_case = next(c for c in data["cases"] if c["case_id"] == "flaky")
        self.assertEqual(flaky_case["min_score"], 0.0)
        self.assertEqual(flaky_case["max_score"], 1.0)
        self.assertEqual(flaky_case["score_spread"], 1.0)
        self.assertIn("min_latency_ms", flaky_case)
        self.assertIn("max_latency_ms", flaky_case)
        self.assertIn("latency_spread_ms", flaky_case)
        text = format_stability(report)
        self.assertIn("latency ms", text)
        self.assertIn("0.00-1.00", text)
        # A case with no recorded extremes renders a dash, not a crash.
        from agent_evals.stability import CaseStability

        bare = CaseStability(
            case_id="bare", runs=0, passes=0,
            mean_latency_ms=0.0, mean_score=0.0,
        )
        self.assertEqual(bare.latency_spread_ms, 0.0)
        self.assertEqual(bare.score_spread, 0.0)
        self.assertIsNone(bare.to_dict()["min_latency_ms"])
        self.assertIsNone(bare.to_dict()["min_score"])


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


def output_change_demo_agent(prompt):
    """Demo agent for the output-change gate tests, loadable by the
    CLI as ``tests.test_stability:output_change_demo_agent``:
    alternates between '19' and 'answer: 19' on the 'drift' prompt
    (both pass a contains-'19' scorer, so the case passes every run
    while its output changes), and answers '19' otherwise. The
    drift counter only counts 'drift' calls, so the two answers
    alternate however the calls interleave with other cases."""
    if prompt == "drift":
        output_change_demo_agent.drift_calls["n"] += 1
        return (
            "19"
            if output_change_demo_agent.drift_calls["n"] % 2
            else "answer: 19"
        )
    return "19"


output_change_demo_agent.drift_calls = {"n": 0}


class TestOutputChangeGate(unittest.TestCase):
    """--fail-on-output-change gate (v0.52.0).

    A case trips the gate when it produced more than one distinct
    output across the repetitions (``distinct_output_count > 1``),
    whatever its verdict: a case that passes every run on a lenient
    scorer can still answer differently each time, and that drift
    is what the gate catches. Distinctness is on the redacted,
    truncated excerpts the report keeps, not the raw outputs.
    """

    def _drifting_suite(self):
        return Suite(
            name="drift",
            cases=[
                # contains '19': both '19' and 'answer: 19' pass,
                # so this case passes every run while its output
                # changes.
                Case(id="drifting-pass", prompt="drift",
                     scorer={"type": "contains", "substring": "19"}),
                Case(id="stable-pass", prompt="p",
                     scorer={"type": "exact", "expected": "19"}),
            ],
        )

    def _drifting_agent(self):
        calls = {"n": 0}

        def agent(prompt):
            if prompt == "drift":
                calls["n"] += 1
                return "19" if calls["n"] % 2 else "answer: 19"
            return "19"

        return agent

    def test_passing_case_with_changing_output(self):
        report = repeat_suite(
            self._drifting_suite(), self._drifting_agent(), repetitions=4
        )
        by_id = {c.case_id: c for c in report.cases}
        drifting = by_id["drifting-pass"]
        # Passes every run, so it is not flaky — but its output
        # changed, which is exactly the drift the gate catches.
        self.assertEqual(drifting.passes, 4)
        self.assertFalse(drifting.flaky)
        self.assertTrue(drifting.output_changed)
        self.assertEqual(drifting.distinct_output_count, 2)
        self.assertEqual(
            drifting.distinct_outputs, ["19", "answer: 19"]
        )
        self.assertEqual(
            [c.case_id for c in report.output_changed_cases],
            ["drifting-pass"],
        )
        # The stable case answered the same way every run.
        self.assertFalse(by_id["stable-pass"].output_changed)
        self.assertEqual(by_id["stable-pass"].distinct_output_count, 1)

    def test_stable_failure_has_not_changed(self):
        # A case that fails every run with the same wrong answer is
        # a stable failure, not output drift.
        report = repeat_suite(
            _suite(), lambda prompt: "20", repetitions=3
        )
        by_id = {c.case_id: c for c in report.cases}
        self.assertFalse(by_id["stable-fail"].output_changed)
        self.assertEqual(by_id["stable-fail"].distinct_output_count, 1)
        self.assertEqual(report.output_changed_cases, [])

    def test_failing_case_with_changing_output(self):
        # A case that fails every run with a different wrong answer
        # each time changed its output too — the property is about
        # the output, not the verdict — even though it is not
        # flaky (it never passed).
        calls = {"n": 0}

        def flipping(prompt):
            calls["n"] += 1
            return "20" if calls["n"] % 2 else "21"

        report = repeat_suite(_suite(), flipping, repetitions=4)
        by_id = {c.case_id: c for c in report.cases}
        self.assertEqual(by_id["flaky"].passes, 0)
        self.assertFalse(by_id["flaky"].flaky)
        self.assertTrue(by_id["flaky"].output_changed)
        self.assertEqual(by_id["flaky"].distinct_output_count, 2)

    def test_single_repetition_never_changed(self):
        report = repeat_suite(
            self._drifting_suite(), self._drifting_agent(), repetitions=1
        )
        self.assertEqual(report.output_changed_cases, [])

    def test_to_dict_carries_output_changed(self):
        report = repeat_suite(
            self._drifting_suite(), self._drifting_agent(), repetitions=2
        )
        data = report.to_dict()
        self.assertEqual(data["output_changed"], ["drifting-pass"])
        by_id = {c["case_id"]: c for c in data["cases"]}
        self.assertTrue(by_id["drifting-pass"]["output_changed"])
        self.assertFalse(by_id["stable-pass"]["output_changed"])

    def test_format_stability_shows_output_changed_section(self):
        report = repeat_suite(
            self._drifting_suite(), self._drifting_agent(), repetitions=4
        )
        text = format_stability(report)
        self.assertIn("Output changed across repetitions (1):", text)
        self.assertIn(
            "drifting-pass  2 distinct outputs across 4 runs", text
        )
        # The drifting case's last answer plus the other answer it
        # gave: with 4 repetitions the last drift call is even, so
        # the last output is 'answer: 19' and '19' is the also-saw.
        self.assertIn("drifting-pass: 'answer: 19'", text)
        self.assertIn("also saw: '19'", text)
        # And the verdict is still a clean pass: nothing was flaky.
        self.assertIn("Flaky: 0", text)

    def test_format_stability_omits_section_without_change(self):
        report = repeat_suite(_suite(), lambda prompt: "19", repetitions=2)
        self.assertNotIn("Output changed", format_stability(report))

    def test_cli_gate_fails_on_drifting_pass(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            report_path = os.path.join(tmp, "stability.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._drifting_suite().to_dict(), f)
            code = main([
                "--suite", path,
                "--agent",
                "tests.test_stability:output_change_demo_agent",
                "--repeat", "2",
                "--fail-on-output-change",
                "--report", report_path,
            ])
            # The run would pass without the gate: every case
            # passed every time, nothing was flaky.
            self.assertEqual(code, 1)
            with open(report_path, encoding="utf-8") as f:
                stability = json.load(f)
            self.assertEqual(stability["flaky"], [])
            self.assertEqual(
                stability["output_changed"], ["drifting-pass"]
            )

    def test_cli_gate_is_opt_in(self):
        # The same drifting run passes without the flag: the gate
        # is opt-in, and the base rule only fails flaky or
        # always-failing cases.
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._drifting_suite().to_dict(), f)
            code = main([
                "--suite", path,
                "--agent",
                "tests.test_stability:output_change_demo_agent",
                "--repeat", "2",
            ])
            self.assertEqual(code, 0)

    def test_cli_gate_passes_deterministic_agent(self):
        from agent_evals.cli import main

        code = main([
            "--suite", "examples/calculator.json",
            "--agent", "examples.calculator_agent:agent",
            "--repeat", "2",
            "--fail-on-output-change",
        ])
        self.assertEqual(code, 0)

    def test_cli_gate_requires_repeat(self):
        from agent_evals.cli import main

        # With one repetition a case has only one output, so the
        # flag is a clean CLI error instead of a silent no-op.
        with self.assertRaises(SystemExit) as cm:
            main([
                "--suite", "examples/calculator.json",
                "--agent", "examples.calculator_agent:agent",
                "--fail-on-output-change",
            ])
        self.assertEqual(cm.exception.code, 2)


class TestOutputDiffs(unittest.TestCase):
    """Per-case output diffs across repetitions (v0.53.0).

    When a case answered differently across the repetitions, the
    report carries a unified diff of each later distinct output
    against the first one seen, so what changed is visible at a
    glance instead of two blobs to compare by eye. The diffs are on
    the redacted, truncated excerpts the report keeps.
    """

    def _multi_line_suite(self):
        return Suite(
            name="drift",
            cases=[
                # contains 'alpha': all three answers pass, so the
                # case passes every run while its output changes.
                Case(id="drifting-pass", prompt="drift",
                     scorer={"type": "contains", "substring": "alpha"}),
                Case(id="stable-pass", prompt="p",
                     scorer={"type": "exact", "expected": "alpha"}),
            ],
        )

    def _three_way_agent(self):
        answers = [
            "alpha\nbeta\ngamma",
            "alpha\nbeta\ndelta",
            "alpha\nepsilon\ngamma",
        ]
        calls = {"n": 0}

        def agent(prompt):
            if prompt == "drift":
                calls["n"] += 1
                return answers[(calls["n"] - 1) % len(answers)]
            return "alpha"

        return agent

    def test_diffs_compare_each_answer_to_first(self):
        report = repeat_suite(
            self._multi_line_suite(), self._three_way_agent(),
            repetitions=3,
        )
        by_id = {c.case_id: c for c in report.cases}
        drifting = by_id["drifting-pass"]
        self.assertEqual(
            drifting.distinct_outputs,
            ["alpha\nbeta\ngamma", "alpha\nbeta\ndelta",
             "alpha\nepsilon\ngamma"],
        )
        # One diff per later answer, each against the first.
        self.assertEqual(len(drifting.output_diffs), 2)
        first, second = drifting.output_diffs
        self.assertEqual(
            (first["from_index"], first["to_index"]), (0, 1)
        )
        self.assertEqual(
            (second["from_index"], second["to_index"]), (0, 2)
        )
        # The first diff shows only the changed last line.
        self.assertIn("--- first output", first["diff"])
        self.assertIn("+++ output 2", first["diff"])
        self.assertIn("-gamma", first["diff"])
        self.assertIn("+delta", first["diff"])
        self.assertNotIn("-beta", first["diff"])
        # The second diff shows the changed middle line.
        self.assertIn("+++ output 3", second["diff"])
        self.assertIn("-beta", second["diff"])
        self.assertIn("+epsilon", second["diff"])

    def test_stable_case_has_no_diffs(self):
        report = repeat_suite(
            self._multi_line_suite(), self._three_way_agent(),
            repetitions=3,
        )
        by_id = {c.case_id: c for c in report.cases}
        self.assertEqual(by_id["stable-pass"].output_diffs, [])

    def test_no_output_case_has_no_diffs(self):
        def agent(prompt):
            raise RuntimeError("boom")

        report = repeat_suite(_suite(), agent, repetitions=2)
        for case in report.cases:
            self.assertEqual(case.output_diffs, [])

    def test_to_dict_carries_output_diffs(self):
        report = repeat_suite(
            self._multi_line_suite(), self._three_way_agent(),
            repetitions=3,
        )
        by_json = {c["case_id"]: c for c in report.to_dict()["cases"]}
        diffs = by_json["drifting-pass"]["output_diffs"]
        self.assertEqual(len(diffs), 2)
        self.assertIn("-gamma", diffs[0]["diff"])
        self.assertIn("+delta", diffs[0]["diff"])
        self.assertEqual(by_json["stable-pass"]["output_diffs"], [])

    def test_format_stability_shows_output_diffs(self):
        report = repeat_suite(
            self._multi_line_suite(), self._three_way_agent(),
            repetitions=3,
        )
        text = format_stability(report)
        self.assertIn("Output diffs (vs first answer):", text)
        self.assertIn("drifting-pass  output 2 vs output 1:", text)
        self.assertIn("drifting-pass  output 3 vs output 1:", text)
        self.assertIn("-gamma", text)
        self.assertIn("+delta", text)

    def test_format_stability_omits_diffs_section_without_change(self):
        report = repeat_suite(
            _suite(), lambda prompt: "19", repetitions=2
        )
        text = format_stability(report)
        self.assertNotIn("Output diffs", text)

    def test_format_stability_caps_long_diffs(self):
        # A 40-line answer that changes on every line: the JSON
        # keeps the whole diff, the summary caps it with a marker.
        def agent(prompt):
            calls = agent.calls
            calls["n"] += 1
            tag = "A" if calls["n"] % 2 else "B"
            return "\n".join(f"line {i} {tag}" for i in range(40))

        agent.calls = {"n": 0}
        suite = Suite(
            name="long",
            cases=[
                Case(id="long-drift", prompt="drift",
                     scorer={"type": "contains", "substring": "line"}),
            ],
        )
        report = repeat_suite(suite, agent, repetitions=2)
        case = report.cases[0]
        self.assertEqual(len(case.output_diffs), 1)
        self.assertIn("line 39 B", case.output_diffs[0]["diff"])
        text = format_stability(report)
        self.assertIn("more diff lines)", text)
        # The diff section itself is capped: the last changed line
        # is past the cap there (it still shows in full in the
        # "Case outputs" section below, which prints the whole
        # last-run excerpt).
        diff_section = text.split("Output diffs (vs first answer):")[1]
        diff_section = diff_section.split("Case outputs (last run):")[0]
        self.assertNotIn("line 39 B", diff_section)


if __name__ == "__main__":
    unittest.main()
