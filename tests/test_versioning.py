"""Tests for eval-set versioning and run diffing (v0.6.0).

Suites carry a human version label plus a content fingerprint; run
reports record both, plus a fingerprint per case. The diff can then
tell whether a score delta came from the agent or from the eval set
changing underneath it.
"""

import json
import unittest

from agent_evals import Suite, run_suite
from agent_evals.compare import compare_reports, format_comparison
from agent_evals.evals import Case


def _suite_dict(name="s", version=None, cases=None):
    data = {
        "name": name,
        "cases": cases
        if cases is not None
        else [
            {
                "id": "a",
                "prompt": "2+2",
                "scorer": {"type": "exact", "expected": "4"},
            }
        ],
    }
    if version is not None:
        data["version"] = version
    return data


def _report(suite="s", version=None, fingerprint=None, results=None):
    results = results or []
    return {
        "suite": suite,
        "suite_version": version,
        "suite_fingerprint": fingerprint,
        "started_at": "2026-11-22T00:00:00+00:00",
        "ended_at": "2026-11-22T00:00:01+00:00",
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "pass_rate": 1.0,
        "total_weight": len(results),
        "earned_weight": float(sum(1 for r in results if r["passed"])),
        "weighted_score": 1.0,
        "total_latency_ms": 10.0,
        "total_tokens": None,
        "results": results,
    }


def _result(case_id, passed, score=1.0, fingerprint=None):
    return {
        "case_id": case_id,
        "passed": passed,
        "detail": "ok" if passed else "fail",
        "latency_ms": 1.0,
        "tokens": None,
        "output": "",
        "timed_out": False,
        "score": score,
        "weight": 1.0,
        "case_fingerprint": fingerprint,
    }


class TestSuiteVersion(unittest.TestCase):
    def test_version_label_round_trips(self):
        suite = Suite.from_dict(_suite_dict(version="1.3"))
        self.assertEqual(suite.version, "1.3")
        self.assertEqual(suite.to_dict()["version"], "1.3")

    def test_version_absent_when_not_given(self):
        suite = Suite.from_dict(_suite_dict())
        self.assertIsNone(suite.version)
        self.assertNotIn("version", suite.to_dict())

    def test_non_string_version_rejected(self):
        with self.assertRaises(ValueError):
            Suite.from_dict(_suite_dict(version=1.3))


class TestFingerprints(unittest.TestCase):
    def test_same_content_same_fingerprint(self):
        a = Suite.from_dict(_suite_dict(version="1.0"))
        b = Suite.from_dict(_suite_dict(version="2.0"))
        # A label-only change does not change the set's identity.
        self.assertEqual(a.fingerprint(), b.fingerprint())

    def test_changed_case_changes_suite_fingerprint(self):
        a = Suite.from_dict(_suite_dict())
        changed = _suite_dict(
            cases=[
                {
                    "id": "a",
                    "prompt": "2+2",
                    "scorer": {"type": "exact", "expected": "5"},
                }
            ]
        )
        b = Suite.from_dict(changed)
        self.assertNotEqual(a.fingerprint(), b.fingerprint())

    def test_case_fingerprint_tracks_content(self):
        a = Case.from_dict(
            {"id": "a", "prompt": "2+2", "scorer": {"type": "exact", "expected": "4"}}
        )
        b = Case.from_dict(
            {"id": "a", "prompt": "2+2", "scorer": {"type": "exact", "expected": "4"}}
        )
        c = Case.from_dict(
            {"id": "a", "prompt": "3+3", "scorer": {"type": "exact", "expected": "6"}}
        )
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertNotEqual(a.fingerprint(), c.fingerprint())

    def test_fingerprint_is_stable_and_short(self):
        suite = Suite.from_dict(_suite_dict())
        fp = suite.fingerprint()
        self.assertEqual(fp, Suite.from_dict(_suite_dict()).fingerprint())
        self.assertEqual(len(fp), 12)
        self.assertTrue(all(ch in "0123456789abcdef" for ch in fp))


class TestReportCarriesVersioning(unittest.TestCase):
    def test_run_report_records_set_and_case_fingerprints(self):
        suite = Suite.from_dict(_suite_dict(version="1.3"))
        report = run_suite(suite, lambda prompt: "4")
        data = report.to_dict()
        self.assertEqual(data["suite_version"], "1.3")
        self.assertEqual(data["suite_fingerprint"], suite.fingerprint())
        self.assertEqual(
            data["results"][0]["case_fingerprint"],
            suite.cases[0].fingerprint(),
        )

    def test_run_report_without_version_has_null_version(self):
        suite = Suite.from_dict(_suite_dict())
        report = run_suite(suite, lambda prompt: "4")
        data = report.to_dict()
        self.assertIsNone(data["suite_version"])
        self.assertEqual(data["suite_fingerprint"], suite.fingerprint())

    def test_report_json_round_trip_keeps_fingerprints(self):
        suite = Suite.from_dict(_suite_dict(version="1.3"))
        report = run_suite(suite, lambda prompt: "4")
        data = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(data["suite_fingerprint"], suite.fingerprint())
        self.assertEqual(
            data["results"][0]["case_fingerprint"],
            suite.cases[0].fingerprint(),
        )


class TestDiffAttribution(unittest.TestCase):
    def test_same_set_not_changed(self):
        comp = compare_reports(
            _report(fingerprint="abc123", results=[_result("a", True, fingerprint="f1")]),
            _report(fingerprint="abc123", results=[_result("a", True, fingerprint="f1")]),
        )
        self.assertFalse(comp.eval_set_changed)
        self.assertEqual(comp.changed_cases, [])
        self.assertFalse(comp.changes[0].case_changed)

    def test_different_suite_fingerprint_marks_set_changed(self):
        comp = compare_reports(
            _report(version="1.0", fingerprint="aaa111",
                    results=[_result("a", True, fingerprint="f1")]),
            _report(version="1.1", fingerprint="bbb222",
                    results=[_result("a", False, fingerprint="f1")]),
        )
        self.assertTrue(comp.eval_set_changed)
        self.assertEqual(comp.baseline_suite_version, "1.0")
        self.assertEqual(comp.current_suite_version, "1.1")
        # The case itself did not change, so this is an agent regression.
        self.assertFalse(comp.changes[0].case_changed)
        self.assertEqual([c.case_id for c in comp.regressions], ["a"])

    def test_changed_case_flagged_on_regression(self):
        comp = compare_reports(
            _report(fingerprint="aaa111",
                    results=[_result("a", True, fingerprint="f1")]),
            _report(fingerprint="bbb222",
                    results=[_result("a", False, fingerprint="f2")]),
        )
        self.assertTrue(comp.eval_set_changed)
        self.assertTrue(comp.changes[0].case_changed)
        self.assertEqual([c.case_id for c in comp.changed_cases], ["a"])
        self.assertEqual(comp.changes[0].kind, "regression")
        text = format_comparison(comp)
        self.assertIn("eval set changed", text)
        self.assertIn("Changed cases (1):", text)
        self.assertIn("case changed", text)

    def test_changed_case_with_same_verdict_still_listed(self):
        comp = compare_reports(
            _report(fingerprint="aaa111",
                    results=[_result("a", True, fingerprint="f1")]),
            _report(fingerprint="bbb222",
                    results=[_result("a", True, fingerprint="f2")]),
        )
        self.assertEqual(comp.verdict, "UNCHANGED")
        self.assertEqual([c.case_id for c in comp.changed_cases], ["a"])
        self.assertIn("Changed cases (1):", format_comparison(comp))

    def test_version_fallback_when_fingerprints_missing(self):
        # Old reports predate fingerprints: differing labels still
        # mark the set as changed.
        comp = compare_reports(
            _report(version="1.0", results=[_result("a", True)]),
            _report(version="1.1", results=[_result("a", True)]),
        )
        self.assertTrue(comp.eval_set_changed)
        self.assertIsNone(comp.changes[0].case_changed)

    def test_no_evidence_means_set_unchanged(self):
        comp = compare_reports(
            _report(results=[_result("a", True)]),
            _report(results=[_result("a", True)]),
        )
        self.assertFalse(comp.eval_set_changed)
        self.assertIsNone(comp.changes[0].case_changed)
        self.assertNotIn("Eval set:", format_comparison(comp))

    def test_new_and_removed_cases_have_no_case_changed(self):
        comp = compare_reports(
            _report(fingerprint="aaa111",
                    results=[_result("old", True, fingerprint="f1")]),
            _report(fingerprint="bbb222",
                    results=[_result("fresh", True, fingerprint="f2")]),
        )
        self.assertTrue(comp.eval_set_changed)
        for change in comp.changes:
            self.assertIsNone(change.case_changed)

    def test_to_dict_carries_attribution(self):
        comp = compare_reports(
            _report(version="1.0", fingerprint="aaa111",
                    results=[_result("a", True, fingerprint="f1")]),
            _report(version="1.1", fingerprint="bbb222",
                    results=[_result("a", False, fingerprint="f2")]),
        )
        data = json.loads(json.dumps(comp.to_dict()))
        self.assertTrue(data["eval_set_changed"])
        self.assertEqual(data["changed"], ["a"])
        self.assertTrue(data["changes"][0]["case_changed"])
        self.assertEqual(data["baseline_suite_fingerprint"], "aaa111")
        self.assertEqual(data["current_suite_fingerprint"], "bbb222")


if __name__ == "__main__":
    unittest.main()
