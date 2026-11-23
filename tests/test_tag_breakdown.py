"""Tests for --list-tags and per-tag pass rates (v0.14.0)."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from agent_evals import Case, Suite, format_summary, run_suite
from agent_evals.cli import main


def _case(case_id, tags=None, prompt="q", expected="yes", weight=1.0):
    data = {
        "id": case_id,
        "prompt": prompt,
        "scorer": {"type": "exact", "expected": expected},
    }
    if tags is not None:
        data["tags"] = tags
    if weight != 1.0:
        data["weight"] = weight
    return Case.from_dict(data)


def _suite():
    return Suite(
        name="tagged",
        cases=[
            _case("smoke-fast", ["fast", "smoke"]),
            _case("smoke-slow", ["slow", "smoke"]),
            _case("fast-only", ["fast"]),
            _case("untagged"),
        ],
    )


def _agent(prompt):
    # Passes every case except the one whose prompt is "no".
    return "no" if prompt == "no" else "yes"


class TestTagCounts(unittest.TestCase):
    def test_counts_per_tag_sorted(self):
        self.assertEqual(
            _suite().tag_counts(),
            [("fast", 2), ("slow", 1), ("smoke", 2)],
        )

    def test_multi_tag_case_counts_under_each_tag(self):
        suite = Suite(name="s", cases=[_case("a", ["x", "y"])])
        self.assertEqual(suite.tag_counts(), [("x", 1), ("y", 1)])

    def test_untagged_cases_count_under_no_tag(self):
        suite = Suite(name="s", cases=[_case("a"), _case("b")])
        self.assertEqual(suite.tag_counts(), [])

    def test_empty_suite_has_no_tags(self):
        self.assertEqual(Suite(name="s").tag_counts(), [])


class TestTagBreakdown(unittest.TestCase):
    def test_pass_rate_per_tag(self):
        # "smoke-slow" fails (prompt "no"); everything else passes.
        suite = Suite(
            name="tagged",
            cases=[
                _case("smoke-fast", ["fast", "smoke"]),
                _case("smoke-slow", ["slow", "smoke"], prompt="no"),
                _case("fast-only", ["fast"]),
                _case("untagged"),
            ],
        )
        breakdown = run_suite(suite, _agent).tag_breakdown
        self.assertEqual(
            breakdown["smoke"],
            {"cases": 2, "passed": 1, "pass_rate": 0.5, "weighted_score": 0.5},
        )
        self.assertEqual(
            breakdown["fast"],
            {"cases": 2, "passed": 2, "pass_rate": 1.0, "weighted_score": 1.0},
        )
        self.assertEqual(
            breakdown["slow"],
            {"cases": 1, "passed": 0, "pass_rate": 0.0, "weighted_score": 0.0},
        )
        # Keys sorted; untagged cases appear under no tag.
        self.assertEqual(list(breakdown), ["fast", "slow", "smoke"])

    def test_weighted_score_per_tag_uses_case_weights(self):
        suite = Suite(
            name="w",
            cases=[
                _case("heavy-pass", ["t"], weight=3.0),
                _case("light-fail", ["t"], prompt="no", weight=1.0),
            ],
        )
        entry = run_suite(suite, _agent).tag_breakdown["t"]
        self.assertEqual(entry["cases"], 2)
        self.assertEqual(entry["passed"], 1)
        self.assertEqual(entry["pass_rate"], 0.5)
        self.assertEqual(entry["weighted_score"], 0.75)

    def test_no_tags_means_empty_breakdown(self):
        suite = Suite(name="plain", cases=[_case("a"), _case("b")])
        self.assertEqual(run_suite(suite, _agent).tag_breakdown, {})

    def test_breakdown_in_json_report(self):
        report = run_suite(_suite(), _agent)
        data = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(
            data["tag_breakdown"]["smoke"],
            {"cases": 2, "passed": 2, "pass_rate": 1.0, "weighted_score": 1.0},
        )
        self.assertEqual(data["tag_breakdown"]["fast"]["cases"], 2)


class TestSummaryTagBreakdown(unittest.TestCase):
    def test_summary_shows_per_tag_section(self):
        suite = Suite(
            name="tagged",
            cases=[
                _case("smoke-fast", ["fast", "smoke"]),
                _case("smoke-slow", ["slow", "smoke"], prompt="no"),
            ],
        )
        summary = format_summary(run_suite(suite, _agent))
        self.assertIn("Per-tag pass rates:", summary)
        self.assertIn("fast   1/1 (100.0%)", summary)
        self.assertIn("smoke  1/2 (50.0%)", summary)
        self.assertIn("slow   0/1 (0.0%)", summary)

    def test_summary_has_no_tag_section_without_tags(self):
        suite = Suite(name="plain", cases=[_case("a")])
        self.assertNotIn("Per-tag", format_summary(run_suite(suite, _agent)))


class TestCliListTags(unittest.TestCase):
    def _write_suite(self, tmp, suite=None):
        path = os.path.join(tmp, "suite.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump((suite or _suite()).to_dict(), f)
        return path

    def _run(self, args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(args)
        return code, out.getvalue()

    def test_lists_tags_with_counts_and_exits_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(["--suite", self._write_suite(tmp), "--list-tags"])
        self.assertEqual(code, 0)
        self.assertIn("Tags in ", out)
        self.assertIn("fast   2 cases", out)
        self.assertIn("slow   1 case", out)
        self.assertIn("smoke  2 cases", out)

    def test_list_tags_needs_no_agent_and_never_imports_one(self):
        # A bogus agent spec must not matter: listing never loads it.
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run([
                "--suite", self._write_suite(tmp),
                "--list-tags",
                "--agent", "does.not.exist:agent",
            ])
        self.assertEqual(code, 0)
        self.assertIn("smoke", out)

    def test_list_tags_applies_tag_filter_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run([
                "--suite", self._write_suite(tmp),
                "--list-tags", "--tag", "smoke",
            ])
        self.assertEqual(code, 0)
        # Only the smoke cases were selected: fast and slow each once.
        self.assertIn("fast   1 case", out)
        self.assertIn("slow   1 case", out)
        self.assertIn("smoke  2 cases", out)

    def test_list_tags_on_untagged_suite(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = Suite(name="plain", cases=[_case("a"), _case("b")])
            code, out = self._run([
                "--suite", self._write_suite(tmp, suite), "--list-tags",
            ])
        self.assertEqual(code, 0)
        self.assertIn("no tags in ", out)

    def test_agent_required_without_list_tags(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as caught:
                main(["--suite", self._write_suite(tmp)])
        self.assertEqual(caught.exception.code, 2)


class TestCliTagBreakdownInReport(unittest.TestCase):
    def test_run_report_json_carries_tag_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            suite = Suite(
                name="tagged",
                cases=[
                    _case("smoke-fast", ["fast", "smoke"],
                          prompt="2 + 3", expected="5"),
                    _case("smoke-slow", ["slow", "smoke"],
                          prompt="2 + 3", expected="5"),
                ],
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(suite.to_dict(), f)
            report_path = os.path.join(tmp, "report.json")
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(
                report["tag_breakdown"]["smoke"],
                {"cases": 2, "passed": 2, "pass_rate": 1.0,
                 "weighted_score": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
