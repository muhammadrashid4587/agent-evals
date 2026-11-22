"""Tests for agent-evals: scorers, runner aggregation, report round-trip."""

import json
import os
import tempfile
import unittest

from agent_evals.evals import (
    Case,
    Suite,
    format_summary,
    load_report,
    load_suite,
    run_suite,
    score_case,
)


def _case(scorer, prompt="p"):
    return Case(id="c1", prompt=prompt, scorer=scorer)


class TestExactScorer(unittest.TestCase):
    def test_match(self):
        ok, _ = score_case(_case({"type": "exact", "expected": "19"}), "19")
        self.assertTrue(ok)

    def test_mismatch(self):
        ok, detail = score_case(_case({"type": "exact", "expected": "19"}), "20")
        self.assertFalse(ok)
        self.assertIn("19", detail)

    def test_surrounding_whitespace_ignored(self):
        ok, _ = score_case(_case({"type": "exact", "expected": "19"}), "  19\n")
        self.assertTrue(ok)


class TestContainsScorer(unittest.TestCase):
    def test_found(self):
        ok, _ = score_case(_case({"type": "contains", "substring": "hello"}), "say hello there")
        self.assertTrue(ok)

    def test_missing(self):
        ok, _ = score_case(_case({"type": "contains", "substring": "hello"}), "goodbye")
        self.assertFalse(ok)


class TestContainsAllScorer(unittest.TestCase):
    def test_all_present(self):
        ok, _ = score_case(
            _case({"type": "contains_all", "substrings": ["hello", "world"]}),
            "hello world",
        )
        self.assertTrue(ok)

    def test_one_missing_fails_and_names_it(self):
        ok, detail = score_case(
            _case({"type": "contains_all", "substrings": ["hello", "world"]}),
            "hello there",
        )
        self.assertFalse(ok)
        self.assertIn("world", detail)

    def test_empty_list_passes(self):
        ok, _ = score_case(
            _case({"type": "contains_all", "substrings": []}), "anything"
        )
        self.assertTrue(ok)

    def test_non_list_spec_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "contains_all", "substrings": "hello"}), "hello")


class TestContainsAnyScorer(unittest.TestCase):
    def test_one_present_passes(self):
        ok, _ = score_case(
            _case({"type": "contains_any", "substrings": ["hello", "hi"]}),
            "say hi",
        )
        self.assertTrue(ok)

    def test_none_present_fails(self):
        ok, detail = score_case(
            _case({"type": "contains_any", "substrings": ["hello", "hi"]}),
            "goodbye",
        )
        self.assertFalse(ok)
        self.assertIn("hello", detail)

    def test_empty_list_fails(self):
        ok, _ = score_case(
            _case({"type": "contains_any", "substrings": []}), "anything"
        )
        self.assertFalse(ok)

    def test_non_list_spec_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "contains_any", "substrings": "hello"}), "hello")


class TestRegexScorer(unittest.TestCase):
    def test_match(self):
        ok, _ = score_case(_case({"type": "regex", "pattern": r"^\d+$"}), "42")
        self.assertTrue(ok)

    def test_no_match(self):
        ok, _ = score_case(_case({"type": "regex", "pattern": r"^\d+$"}), "4x2")
        self.assertFalse(ok)

    def test_invalid_pattern_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "regex", "pattern": "("}), "x")


class TestJsonSchemaScorer(unittest.TestCase):
    def test_matching_shape(self):
        ok, _ = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "age": "int"}}),
            '{"name": "ada", "age": 36}',
        )
        self.assertTrue(ok)

    def test_wrong_field_type(self):
        ok, detail = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "age": "int"}}),
            '{"name": "ada", "age": "thirty-six"}',
        )
        self.assertFalse(ok)
        self.assertIn("age", detail)

    def test_missing_required_key(self):
        ok, detail = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "age": "int"}}),
            '{"name": "ada"}',
        )
        self.assertFalse(ok)
        self.assertIn("age", detail)

    def test_optional_field_may_be_absent(self):
        ok, _ = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "nick?": "str"}}),
            '{"name": "ada"}',
        )
        self.assertTrue(ok)

    def test_nested_shape_and_array(self):
        ok, _ = score_case(
            _case(
                {
                    "type": "json_schema",
                    "schema": {"user": {"id": "int"}, "tags": ["str"]},
                }
            ),
            '{"user": {"id": 7}, "tags": ["a", "b"]}',
        )
        self.assertTrue(ok)

    def test_invalid_json_fails(self):
        ok, detail = score_case(
            _case({"type": "json_schema", "schema": {"a": "int"}}), "{not json"
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)

    def test_bool_is_not_int(self):
        ok, _ = score_case(
            _case({"type": "json_schema", "schema": {"flag": "int"}}),
            '{"flag": true}',
        )
        self.assertFalse(ok)


class TestJsonPathScorer(unittest.TestCase):
    def test_nested_key_match(self):
        ok, detail = score_case(
            _case({"type": "json_path", "path": "$.result.total", "expected": 42}),
            '{"result": {"total": 42}}',
        )
        self.assertTrue(ok)
        self.assertIn("$.result.total", detail)

    def test_bare_path_without_dollar(self):
        ok, _ = score_case(
            _case({"type": "json_path", "path": "result.total", "expected": 42}),
            '{"result": {"total": 42}}',
        )
        self.assertTrue(ok)

    def test_array_index(self):
        ok, _ = score_case(
            _case({"type": "json_path", "path": "$.items[1].name", "expected": "b"}),
            '{"items": [{"name": "a"}, {"name": "b"}]}',
        )
        self.assertTrue(ok)

    def test_mismatch_names_expected_and_got(self):
        ok, detail = score_case(
            _case({"type": "json_path", "path": "$.total", "expected": 42}),
            '{"total": 43}',
        )
        self.assertFalse(ok)
        self.assertIn("42", detail)
        self.assertIn("43", detail)

    def test_missing_key_fails(self):
        ok, detail = score_case(
            _case({"type": "json_path", "path": "$.missing", "expected": 1}),
            '{"total": 1}',
        )
        self.assertFalse(ok)
        self.assertIn("missing", detail)

    def test_missing_index_fails(self):
        ok, detail = score_case(
            _case({"type": "json_path", "path": "$.items[5]", "expected": 1}),
            '{"items": [1]}',
        )
        self.assertFalse(ok)
        self.assertIn("5", detail)

    def test_invalid_json_fails(self):
        ok, detail = score_case(
            _case({"type": "json_path", "path": "$.a", "expected": 1}), "{not json"
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)

    def test_numeric_tolerance(self):
        ok, _ = score_case(
            _case(
                {
                    "type": "json_path",
                    "path": "$.pi",
                    "expected": 3.14,
                    "tolerance": 0.01,
                }
            ),
            '{"pi": 3.14159}',
        )
        self.assertTrue(ok)

    def test_bool_is_not_int(self):
        ok, _ = score_case(
            _case({"type": "json_path", "path": "$.flag", "expected": 1}),
            '{"flag": true}',
        )
        self.assertFalse(ok)

    def test_string_and_null_values(self):
        ok, _ = score_case(
            _case({"type": "json_path", "path": "$.name", "expected": "ada"}),
            '{"name": "ada"}',
        )
        self.assertTrue(ok)
        ok, _ = score_case(
            _case({"type": "json_path", "path": "$.note", "expected": None}),
            '{"note": null}',
        )
        self.assertTrue(ok)

    def test_malformed_path_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_path", "path": "$.a..b", "expected": 1}),
                '{"a": {}}',
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_path", "path": "$.a[x]", "expected": 1}),
                '{"a": [1]}',
            )

    def test_missing_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "json_path", "path": "$.a"}), '{"a": 1}')


class TestNumericScorer(unittest.TestCase):
    def test_within_tolerance(self):
        ok, _ = score_case(
            _case({"type": "numeric", "expected": 3.333, "tolerance": 0.001}),
            "3.3333",
        )
        self.assertTrue(ok)

    def test_outside_tolerance(self):
        ok, _ = score_case(
            _case({"type": "numeric", "expected": 3.333, "tolerance": 0.001}),
            "3.5",
        )
        self.assertFalse(ok)

    def test_number_embedded_in_text(self):
        ok, _ = score_case(
            _case({"type": "numeric", "expected": 42, "tolerance": 0}),
            "The answer is 42.",
        )
        self.assertTrue(ok)

    def test_no_number_fails(self):
        ok, _ = score_case(
            _case({"type": "numeric", "expected": 42, "tolerance": 0}), "nope"
        )
        self.assertFalse(ok)


class TestUnknownScorer(unittest.TestCase):
    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "vibes"}), "x")


class TestRunner(unittest.TestCase):
    def _suite(self):
        return Suite(
            name="demo",
            cases=[
                Case("ok-1", "q1", {"type": "exact", "expected": "yes"}),
                Case("ok-2", "q2", {"type": "exact", "expected": "yes"}),
                Case("bad-1", "q3", {"type": "exact", "expected": "yes"}),
            ],
        )

    def test_aggregates_pass_rate(self):
        def agent(prompt):
            return "yes" if prompt in ("q1", "q2") else "no"

        report = run_suite(self._suite(), agent)
        self.assertEqual(report.total, 3)
        self.assertEqual(report.passed, 2)
        self.assertAlmostEqual(report.pass_rate, 2 / 3)
        self.assertEqual([r.case_id for r in report.results if r.passed],
                         ["ok-1", "ok-2"])

    def test_latency_recorded(self):
        report = run_suite(self._suite(), lambda prompt: "yes")
        for r in report.results:
            self.assertGreaterEqual(r.latency_ms, 0.0)

    def test_tokens_recorded_when_agent_reports_them(self):
        def agent(prompt):
            return ("yes", 17)

        report = run_suite(self._suite(), agent)
        self.assertEqual(report.total_tokens, 51)
        self.assertTrue(all(r.tokens == 17 for r in report.results))

    def test_tokens_none_when_agent_returns_plain_string(self):
        report = run_suite(self._suite(), lambda prompt: "yes")
        self.assertIsNone(report.total_tokens)

    def test_failing_agent_reported_as_failing(self):
        report = run_suite(self._suite(), lambda prompt: "wrong answer")
        self.assertEqual(report.passed, 0)
        self.assertTrue(all(not r.passed for r in report.results))

    def test_crashing_agent_fails_the_case_and_run_continues(self):
        def agent(prompt):
            if prompt == "q2":
                raise RuntimeError("boom")
            return "yes"

        report = run_suite(self._suite(), agent)
        self.assertEqual(report.passed, 2)
        failed = [r for r in report.results if not r.passed]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].case_id, "ok-2")
        self.assertIn("RuntimeError", failed[0].detail)

    def test_dict_return_with_tokens(self):
        def agent(prompt):
            return {"output": "yes", "tokens": 5}

        report = run_suite(self._suite(), agent)
        self.assertEqual(report.total_tokens, 15)

    def test_summary_table_mentions_pass_and_fail(self):
        report = run_suite(self._suite(), lambda prompt: "yes")
        text = format_summary(report)
        self.assertIn("3/3", text)
        self.assertIn("PASS", text)


class TestReportRoundTrip(unittest.TestCase):
    def test_json_report_round_trips(self):
        suite = Suite("rt", [Case("a", "q", {"type": "exact", "expected": "x"})])
        report = run_suite(suite, lambda prompt: "x")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            report.save(path)
            loaded = load_report(path)
        self.assertEqual(loaded["suite"], "rt")
        self.assertEqual(loaded["total"], 1)
        self.assertEqual(loaded["passed"], 1)
        self.assertEqual(loaded["results"][0]["case_id"], "a")
        self.assertTrue(loaded["results"][0]["passed"])


class TestLoadSuite(unittest.TestCase):
    def test_loads_json_suite(self):
        data = {
            "name": "s",
            "cases": [
                {"id": "c1", "prompt": "p", "scorer": {"type": "exact", "expected": "e"}}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            suite = load_suite(path)
        self.assertEqual(suite.name, "s")
        self.assertEqual(len(suite.cases), 1)
        self.assertEqual(suite.cases[0].id, "c1")


if __name__ == "__main__":
    unittest.main()
