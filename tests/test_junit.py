"""Tests for the JUnit XML export (agent_evals.junit)."""

import json
import os
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

from agent_evals import Case, Suite, run_suite, to_junit_xml


def _parse(xml_text):
    return ET.fromstring(xml_text)


class TestJUnitXml(unittest.TestCase):
    def _report(self, cases, agent):
        return run_suite(Suite("junit-suite", cases), agent)

    def test_all_passing_suite_has_no_failures_or_errors(self):
        report = self._report(
            [
                Case("add", "2 + 3", {"type": "exact", "expected": "5"}),
                Case("hi", "say hi", {"type": "contains", "substring": "hi"}),
            ],
            lambda prompt: "5" if "2" in prompt else "hi",
        )
        root = _parse(to_junit_xml(report))
        self.assertEqual(root.tag, "testsuite")
        self.assertEqual(root.get("name"), "junit-suite")
        self.assertEqual(root.get("tests"), "2")
        self.assertEqual(root.get("failures"), "0")
        self.assertEqual(root.get("errors"), "0")
        cases = root.findall("testcase")
        self.assertEqual(
            [c.get("name") for c in cases], ["add", "hi"]
        )
        for case in cases:
            self.assertEqual(case.get("classname"), "junit-suite")
            self.assertEqual(len(list(case)), 0)
            float(case.get("time"))  # latency in seconds, numeric

    def test_failing_case_becomes_failure_with_detail(self):
        report = self._report(
            [Case("add", "2 + 3", {"type": "exact", "expected": "5"})],
            lambda prompt: "6",
        )
        root = _parse(to_junit_xml(report))
        self.assertEqual(root.get("failures"), "1")
        self.assertEqual(root.get("errors"), "0")
        failure = root.find("testcase/failure")
        self.assertIsNotNone(failure)
        self.assertIn("expected '5', got '6'", failure.get("message"))
        self.assertIn("expected '5', got '6'", failure.text)

    def test_timeout_and_agent_error_become_errors(self):
        def agent(prompt):
            if prompt == "hang":
                time.sleep(5)
            raise RuntimeError("boom")

        report = self._report(
            [
                Case("hang", "hang", {"type": "exact", "expected": "x"},
                     timeout=0.05),
                Case("crash", "crash", {"type": "exact", "expected": "x"}),
            ],
            agent,
        )
        root = _parse(to_junit_xml(report))
        self.assertEqual(root.get("failures"), "0")
        self.assertEqual(root.get("errors"), "2")
        errors = root.findall("testcase/error")
        self.assertEqual(len(errors), 2)
        self.assertIn("timed out", errors[0].get("message"))
        self.assertIn("agent error", errors[1].get("message"))

    def test_scorer_error_becomes_error(self):
        report = self._report(
            [Case("bad", "q", {"type": "nope"})],
            lambda prompt: "x",
        )
        root = _parse(to_junit_xml(report))
        self.assertEqual(root.get("errors"), "1")
        self.assertEqual(root.get("failures"), "0")
        self.assertIn("scorer error", root.find("testcase/error").get("message"))

    def test_budget_violation_becomes_failure(self):
        report = self._report(
            [
                Case("slow", "q", {"type": "exact", "expected": "x"},
                     max_latency_ms=0.001)
            ],
            lambda prompt: "x",
        )
        self.assertEqual(report.results[0].budget_violated, True)
        root = _parse(to_junit_xml(report))
        self.assertEqual(root.get("failures"), "1")
        self.assertEqual(root.get("errors"), "0")
        failure = root.find("testcase/failure")
        self.assertIn("exceeded latency budget", failure.get("message"))

    def test_properties_carry_version_fingerprint_and_score(self):
        suite = Suite(
            "junit-suite",
            [Case("add", "2 + 3", {"type": "exact", "expected": "5"})],
            version="1.2",
        )
        report = run_suite(suite, lambda prompt: "5")
        root = _parse(to_junit_xml(report))
        props = {
            p.get("name"): p.get("value")
            for p in root.findall("properties/property")
        }
        self.assertEqual(props["suite_version"], "1.2")
        self.assertEqual(props["suite_fingerprint"], suite.fingerprint())
        self.assertEqual(props["weighted_score"], "1.0000")

    def test_special_characters_and_control_chars_survive(self):
        # A detail with XML markup, an ampersand, and a raw control
        # character must still parse back. (Details built from repr(),
        # e.g. exact-match failures, already escape control chars;
        # a judge rationale is interpolated raw, so a hand-built
        # report exercises the sanitiser directly.)
        from agent_evals import Result, RunReport

        report = RunReport(
            suite="junit-suite",
            started_at="2026-11-23T00:00:00+00:00",
            ended_at="2026-11-23T00:00:00+00:00",
            results=[
                Result(
                    case_id="weird",
                    passed=False,
                    detail="judge 'stub': score 0.00 — a<b> & \"c\"\x01",
                    latency_ms=1.0,
                )
            ],
        )
        root = _parse(to_junit_xml(report))
        failure = root.find("testcase/failure")
        self.assertIn("a<b> & \"c\"", failure.text)
        self.assertNotIn("\x01", failure.text)
        self.assertIn("\ufffd", failure.text)

    def test_save_junit_writes_parseable_file(self):
        report = self._report(
            [Case("add", "2 + 3", {"type": "exact", "expected": "5"})],
            lambda prompt: "5",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.xml")
            report.save_junit(path)
            with open(path, encoding="utf-8") as f:
                root = _parse(f.read())
        self.assertEqual(root.get("tests"), "1")

    def test_cli_junit_flag_writes_xml(self):
        from agent_evals.cli import main

        data = {
            "name": "cli-junit",
            "cases": [
                {
                    "id": "add",
                    "prompt": "2 + 3",
                    "scorer": {"type": "exact", "expected": "5"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            suite_path = os.path.join(tmp, "suite.json")
            junit_path = os.path.join(tmp, "results.xml")
            with open(suite_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            code = main(
                [
                    "--suite",
                    suite_path,
                    "--agent",
                    "examples.calculator_agent:agent",
                    "--junit",
                    junit_path,
                ]
            )
            self.assertEqual(code, 0)
            with open(junit_path, encoding="utf-8") as f:
                root = _parse(f.read())
            self.assertEqual(root.get("name"), "cli-junit")
            self.assertEqual(root.get("tests"), "1")

    def test_cli_junit_rejected_with_repeat(self):
        from agent_evals.cli import main

        with self.assertRaises(SystemExit):
            main(
                [
                    "--suite",
                    "examples/calculator.json",
                    "--agent",
                    "examples.calculator_agent:agent",
                    "--repeat",
                    "2",
                    "--junit",
                    "out.xml",
                ]
            )


if __name__ == "__main__":
    unittest.main()
