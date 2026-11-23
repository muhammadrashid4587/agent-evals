"""Tests for case-level tags and --tag filtering (v0.12.0)."""

import json
import os
import tempfile
import unittest

from agent_evals import Case, Suite, format_summary, run_suite
from agent_evals.evals import Result


def _case(case_id, tags=None, prompt="q", expected="yes"):
    data = {"id": case_id, "prompt": prompt, "scorer": {"type": "exact", "expected": expected}}
    if tags is not None:
        data["tags"] = tags
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


def _cli_suite():
    """Like _suite, but with prompts the example calculator agent
    answers correctly, so the CLI tests exit 0."""
    return Suite(
        name="tagged",
        cases=[
            _case("smoke-fast", ["fast", "smoke"], prompt="2 + 3", expected="5"),
            _case("smoke-slow", ["slow", "smoke"], prompt="2 + 3", expected="5"),
            _case("fast-only", ["fast"], prompt="2 + 3", expected="5"),
            _case("untagged", prompt="2 + 3", expected="5"),
        ],
    )


class TestCaseTags(unittest.TestCase):
    def test_default_is_no_tags(self):
        self.assertEqual(_case("a").tags, [])

    def test_tags_round_trip_through_dict(self):
        case = _case("a", ["fast", "smoke"])
        self.assertEqual(Case.from_dict(case.to_dict()).tags, ["fast", "smoke"])

    def test_untagged_case_omits_tags_from_dict(self):
        self.assertNotIn("tags", _case("a").to_dict())

    def test_duplicate_tags_deduplicated(self):
        self.assertEqual(_case("a", ["b", "a", "b"]).tags, ["a", "b"])

    def test_invalid_tags_rejected(self):
        for bad in ("smoke", ["smoke", 3], ["smoke", ""], ["  "], [None]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                _case("a", bad)

    def test_tags_change_fingerprint(self):
        self.assertNotEqual(_case("a").fingerprint(), _case("a", ["smoke"]).fingerprint())
        # Tag order does not matter: same set, same fingerprint.
        self.assertEqual(
            _case("a", ["x", "y"]).fingerprint(),
            _case("a", ["y", "x"]).fingerprint(),
        )


class TestSuiteSelect(unittest.TestCase):
    def test_any_keeps_cases_with_at_least_one_tag(self):
        selected = _suite().select(["smoke", "fast"])
        self.assertEqual(
            [c.id for c in selected.cases],
            ["smoke-fast", "smoke-slow", "fast-only"],
        )

    def test_all_keeps_only_cases_with_every_tag(self):
        selected = _suite().select(["smoke", "fast"], match="all")
        self.assertEqual([c.id for c in selected.cases], ["smoke-fast"])

    def test_unknown_tag_selects_nothing(self):
        self.assertEqual(_suite().select(["nope"]).cases, [])

    def test_empty_tag_list_returns_whole_suite(self):
        selected = _suite().select([])
        self.assertEqual([c.id for c in selected.cases],
                         [c.id for c in _suite().cases])

    def test_select_does_not_mutate_original(self):
        suite = _suite()
        suite.select(["smoke"])
        self.assertEqual(len(suite.cases), 4)

    def test_select_keeps_version(self):
        suite = Suite(name="v", cases=[_case("a", ["smoke"])], version="1.1")
        self.assertEqual(suite.select(["smoke"]).version, "1.1")

    def test_invalid_match_rejected(self):
        with self.assertRaises(ValueError):
            _suite().select(["smoke"], match="xor")


class TestSuiteExclude(unittest.TestCase):
    def test_exclude_drops_cases_carrying_the_tag(self):
        excluded = _suite().exclude(["slow"])
        self.assertEqual(
            [c.id for c in excluded.cases],
            ["smoke-fast", "fast-only", "untagged"],
        )

    def test_exclude_multiple_tags(self):
        excluded = _suite().exclude(["slow", "fast"])
        self.assertEqual([c.id for c in excluded.cases], ["untagged"])

    def test_exclude_unknown_tag_drops_nothing(self):
        excluded = _suite().exclude(["nope"])
        self.assertEqual(len(excluded.cases), 4)

    def test_empty_exclude_list_returns_whole_suite(self):
        excluded = _suite().exclude([])
        self.assertEqual(
            [c.id for c in excluded.cases], [c.id for c in _suite().cases]
        )

    def test_exclude_does_not_mutate_original(self):
        suite = _suite()
        suite.exclude(["smoke"])
        self.assertEqual(len(suite.cases), 4)

    def test_exclude_keeps_version(self):
        suite = Suite(name="v", cases=[_case("a", ["smoke"])], version="1.1")
        self.assertEqual(suite.exclude(["smoke"]).version, "1.1")

    def test_select_then_exclude(self):
        # --tag smoke --exclude-tag slow: the smoke cases that are not slow.
        selected = _suite().select(["smoke"]).exclude(["slow"])
        self.assertEqual([c.id for c in selected.cases], ["smoke-fast"])


class TestTagsInReports(unittest.TestCase):
    def test_results_carry_case_tags(self):
        report = run_suite(_suite(), lambda prompt: "yes")
        by_id = {r.case_id: r for r in report.results}
        self.assertEqual(by_id["smoke-fast"].tags, ["fast", "smoke"])
        self.assertEqual(by_id["untagged"].tags, [])
        self.assertEqual(by_id["smoke-fast"].to_dict()["tags"], ["fast", "smoke"])

    def test_summary_shows_tags(self):
        report = run_suite(_suite(), lambda prompt: "yes")
        summary = format_summary(report)
        self.assertIn("[fast, smoke]", summary)
        # An untagged case's row carries no tag bracket.
        untagged_row = [line for line in summary.splitlines()
                        if line.startswith("untagged")][0]
        self.assertNotIn("[", untagged_row)

    def test_filtered_run_reports_only_selected_cases(self):
        report = run_suite(
            _suite().select(["smoke"]), lambda prompt: "yes"
        )
        self.assertEqual([r.case_id for r in report.results],
                         ["smoke-fast", "smoke-slow"])
        self.assertEqual(report.total, 2)


class TestCliTagFlag(unittest.TestCase):
    def _write_suite(self, tmp):
        path = os.path.join(tmp, "suite.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_cli_suite().to_dict(), f)
        return path

    def test_cli_tag_runs_only_matching_cases(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            report_path = os.path.join(tmp, "report.json")
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "smoke",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            # Only the smoke cases ran.
            self.assertEqual(
                [r["case_id"] for r in report["results"]],
                ["smoke-fast", "smoke-slow"],
            )

    def test_cli_tag_mode_all(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            report_path = os.path.join(tmp, "report.json")
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "smoke",
                "--tag", "fast",
                "--tag-mode", "all",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(
                [r["case_id"] for r in report["results"]], ["smoke-fast"]
            )

    def test_cli_tag_with_no_match_exits_2(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "nope",
            ])
        self.assertEqual(code, 2)

    def test_cli_tag_with_repeat(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            report_path = os.path.join(tmp, "stability.json")
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "smoke",
                "--repeat", "2",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                stability = json.load(f)
            self.assertEqual(
                sorted(c["case_id"] for c in stability["cases"]),
                ["smoke-fast", "smoke-slow"],
            )

    def test_cli_exclude_tag_skips_matching_cases(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            report_path = os.path.join(tmp, "report.json")
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--exclude-tag", "slow",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(
                [r["case_id"] for r in report["results"]],
                ["smoke-fast", "fast-only", "untagged"],
            )

    def test_cli_exclude_tag_combines_with_tag(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            report_path = os.path.join(tmp, "report.json")
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "smoke",
                "--exclude-tag", "slow",
                "--report", report_path,
            ])
            self.assertEqual(code, 0)
            with open(report_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertEqual(
                [r["case_id"] for r in report["results"]], ["smoke-fast"]
            )

    def test_cli_exclude_unknown_tag_exits_2(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--exclude-tag", "nope",
            ])
        self.assertEqual(code, 2)

    def test_cli_exclude_everything_exits_2(self):
        # --tag smoke selects two cases; excluding smoke as well
        # leaves none, which is an error, not a silent empty pass.
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_suite(tmp)
            code = main([
                "--suite", path,
                "--agent", "examples.calculator_agent:agent",
                "--tag", "smoke",
                "--exclude-tag", "smoke",
            ])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
