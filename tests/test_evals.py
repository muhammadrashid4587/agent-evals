"""Tests for agent-evals: scorers, runner aggregation, report round-trip."""

import json
import os
import sys
import tempfile
import time
import unittest

from agent_evals.evals import (
    Case,
    Suite,
    format_summary,
    load_report,
    load_suite,
    make_output_diff,
    run_suite,
    score_case,
)


def _case(scorer, prompt="p"):
    return Case(id="c1", prompt=prompt, scorer=scorer)


class TestExactScorer(unittest.TestCase):
    def test_match(self):
        ok, _, _ = score_case(_case({"type": "exact", "expected": "19"}), "19")
        self.assertTrue(ok)

    def test_mismatch(self):
        ok, detail, _ = score_case(_case({"type": "exact", "expected": "19"}), "20")
        self.assertFalse(ok)
        self.assertIn("19", detail)

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(_case({"type": "exact", "expected": "19"}), "  19\n")
        self.assertTrue(ok)


class TestExactIgnoreCaseScorer(unittest.TestCase):
    def test_matches_other_case(self):
        ok, _, score = score_case(
            _case({"type": "exact", "expected": "Hello", "ignore_case": True}),
            "hello",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_case_sensitive_by_default(self):
        ok, _, _ = score_case(
            _case({"type": "exact", "expected": "Hello"}), "hello"
        )
        self.assertFalse(ok)

    def test_ignore_case_false_is_case_sensitive(self):
        ok, _, _ = score_case(
            _case({"type": "exact", "expected": "Hello", "ignore_case": False}),
            "hello",
        )
        self.assertFalse(ok)

    def test_still_all_or_nothing(self):
        ok, _, score = score_case(
            _case({"type": "exact", "expected": "Hello", "ignore_case": True}),
            "goodbye",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_casefold_folds_german_sharp_s(self):
        # str.casefold folds ß to ss, so "straße" matches "STRASSE".
        ok, _, _ = score_case(
            _case({"type": "exact", "expected": "straße", "ignore_case": True}),
            "STRASSE",
        )
        self.assertTrue(ok)

    def test_sharp_s_matches_ss_spelling(self):
        ok, _, _ = score_case(
            _case({"type": "exact", "expected": "straße", "ignore_case": True}),
            "strasse",
        )
        self.assertTrue(ok)

    def test_whitespace_still_ignored_with_ignore_case(self):
        ok, _, _ = score_case(
            _case({"type": "exact", "expected": "Hello", "ignore_case": True}),
            "  HELLO\n",
        )
        self.assertTrue(ok)

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "exact",
                        "expected": "hello",
                        "ignore_case": "true",
                    }
                ),
                "hello",
            )

    def test_flows_through_suite_json(self):
        suite_json = {
            "name": "case",
            "cases": [
                {
                    "id": "c1",
                    "prompt": "say hello",
                    "scorer": {
                        "type": "exact",
                        "expected": "Hello",
                        "ignore_case": True,
                    },
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as f:
            json.dump(suite_json, f)
            path = f.name
        try:
            suite = load_suite(path)
            report = run_suite(suite, lambda prompt: "HELLO")
        finally:
            os.unlink(path)
        self.assertEqual(report.passed, 1)


class TestContainsScorer(unittest.TestCase):
    def test_found(self):
        ok, _, _ = score_case(_case({"type": "contains", "substring": "hello"}), "say hello there")
        self.assertTrue(ok)

    def test_missing(self):
        ok, _, _ = score_case(_case({"type": "contains", "substring": "hello"}), "goodbye")
        self.assertFalse(ok)


class TestContainsAllScorer(unittest.TestCase):
    def test_all_present(self):
        ok, _, _ = score_case(
            _case({"type": "contains_all", "substrings": ["hello", "world"]}),
            "hello world",
        )
        self.assertTrue(ok)

    def test_one_missing_fails_and_names_it(self):
        ok, detail, _ = score_case(
            _case({"type": "contains_all", "substrings": ["hello", "world"]}),
            "hello there",
        )
        self.assertFalse(ok)
        self.assertIn("world", detail)

    def test_empty_list_passes(self):
        ok, _, _ = score_case(
            _case({"type": "contains_all", "substrings": []}), "anything"
        )
        self.assertTrue(ok)

    def test_non_list_spec_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "contains_all", "substrings": "hello"}), "hello")


class TestContainsAnyScorer(unittest.TestCase):
    def test_one_present_passes(self):
        ok, _, _ = score_case(
            _case({"type": "contains_any", "substrings": ["hello", "hi"]}),
            "say hi",
        )
        self.assertTrue(ok)

    def test_none_present_fails(self):
        ok, detail, _ = score_case(
            _case({"type": "contains_any", "substrings": ["hello", "hi"]}),
            "goodbye",
        )
        self.assertFalse(ok)
        self.assertIn("hello", detail)

    def test_empty_list_fails(self):
        ok, _, _ = score_case(
            _case({"type": "contains_any", "substrings": []}), "anything"
        )
        self.assertFalse(ok)

    def test_non_list_spec_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "contains_any", "substrings": "hello"}), "hello")


class TestNotContainsScorer(unittest.TestCase):
    def test_none_present_passes(self):
        ok, detail, score = score_case(
            _case({"type": "not_contains", "substrings": ["as an AI", "sorry"]}),
            "Here is the refund policy.",
        )
        self.assertTrue(ok)
        self.assertIn("no forbidden substrings", detail)
        self.assertEqual(score, 1.0)

    def test_one_present_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case(
                {"type": "not_contains", "substrings": ["hello", "world"]}
            ),
            "hello there",
        )
        self.assertFalse(ok)
        self.assertIn("hello", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_all_present_scores_zero(self):
        ok, _, score = score_case(
            _case(
                {"type": "not_contains", "substrings": ["hello", "world"]}
            ),
            "hello world",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_list_passes_vacuously(self):
        ok, _, score = score_case(
            _case({"type": "not_contains", "substrings": []}), "anything"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_missing_substrings_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "not_contains"}), "hello")

    def test_non_list_spec_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "not_contains", "substrings": "hello"}), "hello")


class TestRegexScorer(unittest.TestCase):
    def test_match(self):
        ok, _, _ = score_case(_case({"type": "regex", "pattern": r"^\d+$"}), "42")
        self.assertTrue(ok)

    def test_no_match(self):
        ok, _, _ = score_case(_case({"type": "regex", "pattern": r"^\d+$"}), "4x2")
        self.assertFalse(ok)

    def test_invalid_pattern_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "regex", "pattern": "("}), "x")


class TestJsonSchemaScorer(unittest.TestCase):
    def test_matching_shape(self):
        ok, _, _ = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "age": "int"}}),
            '{"name": "ada", "age": 36}',
        )
        self.assertTrue(ok)

    def test_wrong_field_type(self):
        ok, detail, _ = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "age": "int"}}),
            '{"name": "ada", "age": "thirty-six"}',
        )
        self.assertFalse(ok)
        self.assertIn("age", detail)

    def test_missing_required_key(self):
        ok, detail, _ = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "age": "int"}}),
            '{"name": "ada"}',
        )
        self.assertFalse(ok)
        self.assertIn("age", detail)

    def test_optional_field_may_be_absent(self):
        ok, _, _ = score_case(
            _case({"type": "json_schema", "schema": {"name": "str", "nick?": "str"}}),
            '{"name": "ada"}',
        )
        self.assertTrue(ok)

    def test_nested_shape_and_array(self):
        ok, _, _ = score_case(
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
        ok, detail, _ = score_case(
            _case({"type": "json_schema", "schema": {"a": "int"}}), "{not json"
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)

    def test_bool_is_not_int(self):
        ok, _, _ = score_case(
            _case({"type": "json_schema", "schema": {"flag": "int"}}),
            '{"flag": true}',
        )
        self.assertFalse(ok)


class TestJsonPathScorer(unittest.TestCase):
    def test_nested_key_match(self):
        ok, detail, _ = score_case(
            _case({"type": "json_path", "path": "$.result.total", "expected": 42}),
            '{"result": {"total": 42}}',
        )
        self.assertTrue(ok)
        self.assertIn("$.result.total", detail)

    def test_bare_path_without_dollar(self):
        ok, _, _ = score_case(
            _case({"type": "json_path", "path": "result.total", "expected": 42}),
            '{"result": {"total": 42}}',
        )
        self.assertTrue(ok)

    def test_array_index(self):
        ok, _, _ = score_case(
            _case({"type": "json_path", "path": "$.items[1].name", "expected": "b"}),
            '{"items": [{"name": "a"}, {"name": "b"}]}',
        )
        self.assertTrue(ok)

    def test_mismatch_names_expected_and_got(self):
        ok, detail, _ = score_case(
            _case({"type": "json_path", "path": "$.total", "expected": 42}),
            '{"total": 43}',
        )
        self.assertFalse(ok)
        self.assertIn("42", detail)
        self.assertIn("43", detail)

    def test_missing_key_fails(self):
        ok, detail, _ = score_case(
            _case({"type": "json_path", "path": "$.missing", "expected": 1}),
            '{"total": 1}',
        )
        self.assertFalse(ok)
        self.assertIn("missing", detail)

    def test_missing_index_fails(self):
        ok, detail, _ = score_case(
            _case({"type": "json_path", "path": "$.items[5]", "expected": 1}),
            '{"items": [1]}',
        )
        self.assertFalse(ok)
        self.assertIn("5", detail)

    def test_invalid_json_fails(self):
        ok, detail, _ = score_case(
            _case({"type": "json_path", "path": "$.a", "expected": 1}), "{not json"
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)

    def test_numeric_tolerance(self):
        ok, _, _ = score_case(
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
        ok, _, _ = score_case(
            _case({"type": "json_path", "path": "$.flag", "expected": 1}),
            '{"flag": true}',
        )
        self.assertFalse(ok)

    def test_string_and_null_values(self):
        ok, _, _ = score_case(
            _case({"type": "json_path", "path": "$.name", "expected": "ada"}),
            '{"name": "ada"}',
        )
        self.assertTrue(ok)
        ok, _, _ = score_case(
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


class TestJsonSubsetScorer(unittest.TestCase):
    def test_extra_keys_ignored(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"status": "ok", "total": 42},
                }
            ),
            '{"status": "ok", "total": 42, "request_id": "abc", "took_ms": 3}',
        )
        self.assertTrue(ok)

    def test_nested_subset(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"result": {"total": 42, "currency": "USD"}},
                }
            ),
            '{"result": {"total": 42, "currency": "USD", "items": []},'
            ' "debug": true}',
        )
        self.assertTrue(ok)

    def test_list_items_are_subsets(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"items": [{"id": 1}, {"id": 2}]},
                }
            ),
            '{"items": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}',
        )
        self.assertTrue(ok)

    def test_missing_key_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"status": "ok", "total": 42},
                }
            ),
            '{"status": "ok"}',
        )
        self.assertFalse(ok)
        self.assertIn("total", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_wrong_value_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"a": 1, "b": 2, "c": 3},
                }
            ),
            '{"a": 1, "b": 99, "c": 3}',
        )
        self.assertFalse(ok)
        self.assertIn("$.b", detail)
        self.assertAlmostEqual(score, 2 / 3)

    def test_wrong_type_fails(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"user": {"id": 7}},
                }
            ),
            '{"user": [7]}',
        )
        self.assertFalse(ok)
        self.assertIn("object", detail)
        self.assertEqual(score, 0.0)

    def test_list_length_mismatch_fails(self):
        ok, detail, _ = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"items": [1, 2]},
                }
            ),
            '{"items": [1]}',
        )
        self.assertFalse(ok)
        self.assertIn("length 2", detail)

    def test_invalid_json_fails(self):
        ok, detail, _ = score_case(
            _case({"type": "json_subset", "expected": {"a": 1}}), "{not json"
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)

    def test_numeric_tolerance(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_subset",
                    "expected": {"pi": 3.14},
                    "tolerance": 0.01,
                }
            ),
            '{"pi": 3.14159}',
        )
        self.assertTrue(ok)

    def test_bool_is_not_int(self):
        ok, _, _ = score_case(
            _case({"type": "json_subset", "expected": {"flag": 1}}),
            '{"flag": true}',
        )
        self.assertFalse(ok)

    def test_null_value_matches(self):
        ok, _, _ = score_case(
            _case({"type": "json_subset", "expected": {"note": None}}),
            '{"note": null, "other": 1}',
        )
        self.assertTrue(ok)

    def test_empty_expected_passes(self):
        ok, _, score = score_case(
            _case({"type": "json_subset", "expected": {}}),
            '{"anything": true}',
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_non_object_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_subset", "expected": [1, 2]}),
                '{"a": 1}',
            )

    def test_missing_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "json_subset"}), '{"a": 1}')


class TestFuzzyScorer(unittest.TestCase):
    def test_exact_match(self):
        ok, _, score = score_case(
            _case({"type": "fuzzy", "expected": "hello world"}), "hello world"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_near_miss_passes(self):
        ok, detail, score = score_case(
            _case({"type": "fuzzy", "expected": "hello world"}), "hello wrld"
        )
        self.assertTrue(ok)
        self.assertIn("0.95", detail)
        self.assertAlmostEqual(score, 0.9523809523809523)

    def test_far_miss_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case({"type": "fuzzy", "expected": "hello world"}),
            "goodbye moon",
        )
        self.assertFalse(ok)
        self.assertIn("similarity", detail)
        self.assertLess(score, 0.8)
        self.assertGreater(score, 0.0)

    def test_threshold_is_respected(self):
        ok, _, _ = score_case(
            _case(
                {"type": "fuzzy", "expected": "hello world", "threshold": 0.99}
            ),
            "hello wrld",
        )
        self.assertFalse(ok)

    def test_threshold_zero_passes_anything(self):
        ok, _, _ = score_case(
            _case(
                {"type": "fuzzy", "expected": "hello world", "threshold": 0}
            ),
            "something else entirely",
        )
        self.assertTrue(ok)

    def test_threshold_above_one_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "fuzzy", "expected": "hello", "threshold": 1.5}
                ),
                "hello",
            )

    def test_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "fuzzy", "expected": "hello", "threshold": -0.1}
                ),
                "hello",
            )

    def test_missing_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "fuzzy"}), "hello")

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(
            _case({"type": "fuzzy", "expected": "hello"}), "  hello\n"
        )
        self.assertTrue(ok)

    def test_empty_expected_only_matches_empty(self):
        ok, _, _ = score_case(_case({"type": "fuzzy", "expected": ""}), "")
        self.assertTrue(ok)
        ok, _, _ = score_case(
            _case({"type": "fuzzy", "expected": ""}), "hello"
        )
        self.assertFalse(ok)


class TestStartsWithScorer(unittest.TestCase):
    def test_prefix_match(self):
        ok, detail, score = score_case(
            _case({"type": "starts_with", "prefix": "The answer is"}),
            "The answer is 42",
        )
        self.assertTrue(ok)
        self.assertIn("starts with", detail)
        self.assertEqual(score, 1.0)

    def test_full_equality_passes(self):
        ok, _, score = score_case(
            _case({"type": "starts_with", "prefix": "hello"}), "hello"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_wrong_start_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case({"type": "starts_with", "prefix": "hello world"}),
            "hello there",
        )
        self.assertFalse(ok)
        self.assertIn("6 of 11", detail)
        self.assertAlmostEqual(score, 6 / 11)

    def test_no_common_prefix_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "starts_with", "prefix": "hello"}), "goodbye"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_missing_prefix_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "starts_with"}), "hello")

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(
            _case({"type": "starts_with", "prefix": "hello"}), "  hello world\n"
        )
        self.assertTrue(ok)

    def test_empty_prefix_passes_vacuously(self):
        ok, _, score = score_case(
            _case({"type": "starts_with", "prefix": ""}), "anything at all"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)


class TestEndsWithScorer(unittest.TestCase):
    def test_suffix_match(self):
        ok, detail, score = score_case(
            _case({"type": "ends_with", "suffix": "is 42"}),
            "The answer is 42",
        )
        self.assertTrue(ok)
        self.assertIn("ends with", detail)
        self.assertEqual(score, 1.0)

    def test_full_equality_passes(self):
        ok, _, score = score_case(
            _case({"type": "ends_with", "suffix": "hello"}), "hello"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_wrong_end_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case({"type": "ends_with", "suffix": "hello world"}),
            "goodbye world",
        )
        self.assertFalse(ok)
        self.assertIn("6 of 11", detail)
        self.assertAlmostEqual(score, 6 / 11)

    def test_no_common_suffix_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "ends_with", "suffix": "hello"}), "goodbye"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_missing_suffix_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "ends_with"}), "hello")

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(
            _case({"type": "ends_with", "suffix": "world"}), "  hello world\n"
        )
        self.assertTrue(ok)

    def test_empty_suffix_passes_vacuously(self):
        ok, _, score = score_case(
            _case({"type": "ends_with", "suffix": ""}), "anything at all"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)


class TestIgnoreCaseScorers(unittest.TestCase):
    def test_contains_matches_other_case(self):
        ok, _, _ = score_case(
            _case(
                {"type": "contains", "substring": "HELLO", "ignore_case": True}
            ),
            "say hello there",
        )
        self.assertTrue(ok)

    def test_contains_is_case_sensitive_by_default(self):
        ok, _, _ = score_case(
            _case({"type": "contains", "substring": "HELLO"}),
            "say hello there",
        )
        self.assertFalse(ok)

    def test_contains_all_matches_mixed_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "contains_all",
                    "substrings": ["Hello", "WORLD"],
                    "ignore_case": True,
                }
            ),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_contains_all_partial_credit_ignores_case(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "contains_all",
                    "substrings": ["hello", "WORLD"],
                    "ignore_case": True,
                }
            ),
            "hello there",
        )
        self.assertFalse(ok)
        self.assertIn("WORLD", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_contains_any_matches_other_case(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "contains_any",
                    "substrings": ["HELLO", "hi"],
                    "ignore_case": True,
                }
            ),
            "say hello",
        )
        self.assertTrue(ok)

    def test_not_contains_catches_other_case(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "not_contains",
                    "substrings": ["As an AI", "sorry"],
                    "ignore_case": True,
                }
            ),
            "AS AN AI language model, I cannot help.",
        )
        self.assertFalse(ok)
        self.assertIn("As an AI", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_not_contains_passes_when_absent_in_any_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "not_contains",
                    "substrings": ["as an AI", "sorry"],
                    "ignore_case": True,
                }
            ),
            "Here is the refund policy.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_starts_with_matches_other_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "starts_with",
                    "prefix": "the answer is",
                    "ignore_case": True,
                }
            ),
            "The Answer Is 42",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_starts_with_partial_credit_ignores_case(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "starts_with",
                    "prefix": "HELLO WORLD",
                    "ignore_case": True,
                }
            ),
            "hello there",
        )
        self.assertFalse(ok)
        self.assertIn("6 of 11", detail)
        self.assertAlmostEqual(score, 6 / 11)

    def test_ends_with_matches_other_case(self):
        ok, _, score = score_case(
            _case(
                {"type": "ends_with", "suffix": "IS 42", "ignore_case": True}
            ),
            "The answer is 42",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ends_with_partial_credit_ignores_case(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "ends_with",
                    "suffix": "HELLO WORLD",
                    "ignore_case": True,
                }
            ),
            "goodbye world",
        )
        self.assertFalse(ok)
        self.assertIn("6 of 11", detail)
        self.assertAlmostEqual(score, 6 / 11)

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "contains",
                        "substring": "hello",
                        "ignore_case": "true",
                    }
                ),
                "hello",
            )

    def test_ignore_case_false_is_case_sensitive(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "contains",
                    "substring": "HELLO",
                    "ignore_case": False,
                }
            ),
            "say hello there",
        )
        self.assertFalse(ok)


class TestNumericScorer(unittest.TestCase):
    def test_within_tolerance(self):
        ok, _, _ = score_case(
            _case({"type": "numeric", "expected": 3.333, "tolerance": 0.001}),
            "3.3333",
        )
        self.assertTrue(ok)

    def test_outside_tolerance(self):
        ok, _, _ = score_case(
            _case({"type": "numeric", "expected": 3.333, "tolerance": 0.001}),
            "3.5",
        )
        self.assertFalse(ok)

    def test_number_embedded_in_text(self):
        ok, _, _ = score_case(
            _case({"type": "numeric", "expected": 42, "tolerance": 0}),
            "The answer is 42.",
        )
        self.assertTrue(ok)

    def test_no_number_fails(self):
        ok, _, _ = score_case(
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


class TestTimeouts(unittest.TestCase):
    def _timeout_case(self, timeout, prompt="q"):
        return Case(
            "slow", prompt, {"type": "exact", "expected": "x"}, timeout=timeout
        )

    def test_timeout_parsed_from_dict_and_round_trips(self):
        case = Case.from_dict(
            {
                "id": "c",
                "prompt": "p",
                "scorer": {"type": "exact", "expected": "x"},
                "timeout": 2,
            }
        )
        self.assertEqual(case.timeout, 2.0)
        self.assertEqual(case.to_dict()["timeout"], 2.0)

    def test_timeout_absent_by_default(self):
        case = Case.from_dict(
            {"id": "c", "prompt": "p", "scorer": {"type": "exact", "expected": "x"}}
        )
        self.assertIsNone(case.timeout)
        self.assertNotIn("timeout", case.to_dict())

    def test_invalid_timeout_rejected(self):
        for bad in (0, -1, "soon"):
            with self.assertRaises(ValueError):
                Case.from_dict(
                    {
                        "id": "c",
                        "prompt": "p",
                        "scorer": {"type": "exact", "expected": "x"},
                        "timeout": bad,
                    }
                )

    def test_slow_agent_times_out_and_run_continues(self):
        suite = Suite(
            "t",
            [
                self._timeout_case(0.05, prompt="q1"),
                Case("fast", "q2", {"type": "exact", "expected": "y"}),
            ],
        )

        def agent(prompt):
            if prompt == "q1":
                time.sleep(0.5)
                return "x"
            return "y"

        report = run_suite(suite, agent)
        self.assertEqual(report.total, 2)
        self.assertEqual(report.passed, 1)
        slow = report.results[0]
        self.assertFalse(slow.passed)
        self.assertTrue(slow.timed_out)
        self.assertIn("timed out", slow.detail)
        self.assertIn("0.05", slow.detail)
        # Failed at the timeout, not after the agent's 0.5 s sleep.
        self.assertGreaterEqual(slow.latency_ms, 40)
        self.assertLess(slow.latency_ms, 400)
        self.assertFalse(report.results[1].timed_out)
        self.assertTrue(report.results[1].passed)

    def test_fast_agent_within_timeout_passes(self):
        suite = Suite("t", [self._timeout_case(5)])
        report = run_suite(suite, lambda prompt: "x")
        self.assertEqual(report.passed, 1)
        self.assertFalse(report.results[0].timed_out)

    def test_timeout_reported_in_json_report(self):
        suite = Suite("t", [self._timeout_case(0.05)])

        def agent(prompt):
            time.sleep(0.5)
            return "x"

        report = run_suite(suite, agent)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            report.save(path)
            loaded = load_report(path)
        result = loaded["results"][0]
        self.assertFalse(result["passed"])
        self.assertTrue(result["timed_out"])
        self.assertIn("timed out", result["detail"])

    def test_summary_marks_timeout(self):
        suite = Suite("t", [self._timeout_case(0.05)])

        def agent(prompt):
            time.sleep(0.5)
            return "x"

        report = run_suite(suite, agent)
        self.assertIn("TIMEOUT", format_summary(report))


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


class TestWeightsAndPartialCredit(unittest.TestCase):
    def test_weight_defaults_to_one_and_round_trips(self):
        case = Case.from_dict(
            {"id": "c", "prompt": "p", "scorer": {"type": "exact", "expected": "x"}}
        )
        self.assertEqual(case.weight, 1.0)
        self.assertNotIn("weight", case.to_dict())

    def test_weight_parsed_from_dict_and_round_trips(self):
        case = Case.from_dict(
            {
                "id": "c",
                "prompt": "p",
                "scorer": {"type": "exact", "expected": "x"},
                "weight": 2.5,
            }
        )
        self.assertEqual(case.weight, 2.5)
        self.assertEqual(case.to_dict()["weight"], 2.5)

    def test_invalid_weight_rejected(self):
        for bad in (0, -1, "heavy"):
            with self.assertRaises(ValueError):
                Case.from_dict(
                    {
                        "id": "c",
                        "prompt": "p",
                        "scorer": {"type": "exact", "expected": "x"},
                        "weight": bad,
                    }
                )

    def test_pass_scores_one_and_fail_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "exact", "expected": "19"}), "19"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "exact", "expected": "19"}), "20"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_contains_all_partial_credit_is_fraction_found(self):
        ok, _, score = score_case(
            _case(
                {"type": "contains_all", "substrings": ["a", "b", "c", "d"]}
            ),
            "a c here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.5)

    def test_contains_all_none_found_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "contains_all", "substrings": ["a", "b"]}),
            "nothing here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_contains_all_all_found_scores_one(self):
        ok, _, score = score_case(
            _case({"type": "contains_all", "substrings": ["a", "b"]}),
            "a and b",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_contains_all_empty_list_scores_one(self):
        ok, _, score = score_case(
            _case({"type": "contains_all", "substrings": []}), "anything"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_weighted_score_aggregates_by_weight(self):
        suite = Suite(
            "w",
            [
                Case("heavy", "q1", {"type": "exact", "expected": "yes"}, weight=2),
                Case("light", "q2", {"type": "exact", "expected": "yes"}, weight=1),
            ],
        )
        report = run_suite(suite, lambda prompt: "yes" if prompt == "q1" else "no")
        self.assertEqual(report.total_weight, 3)
        self.assertEqual(report.earned_weight, 2.0)
        self.assertAlmostEqual(report.weighted_score, 2 / 3)
        self.assertEqual(report.results[0].weight, 2)
        self.assertEqual(report.results[0].score, 1.0)
        self.assertEqual(report.results[1].score, 0.0)

    def test_partial_credit_contributes_to_weighted_score(self):
        suite = Suite(
            "p",
            [
                Case(
                    "multi",
                    "q",
                    {"type": "contains_all", "substrings": ["a", "b"]},
                    weight=2,
                ),
            ],
        )
        report = run_suite(suite, lambda prompt: "a only")
        self.assertFalse(report.results[0].passed)
        self.assertEqual(report.results[0].score, 0.5)
        self.assertEqual(report.earned_weight, 1.0)
        self.assertEqual(report.weighted_score, 0.5)

    def test_weighted_fields_in_json_report(self):
        suite = Suite(
            "w",
            [Case("a", "q", {"type": "exact", "expected": "x"}, weight=3)],
        )
        report = run_suite(suite, lambda prompt: "x")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            report.save(path)
            loaded = load_report(path)
        self.assertEqual(loaded["total_weight"], 3)
        self.assertEqual(loaded["earned_weight"], 3.0)
        self.assertEqual(loaded["weighted_score"], 1.0)
        self.assertEqual(loaded["results"][0]["score"], 1.0)
        self.assertEqual(loaded["results"][0]["weight"], 3)

    def test_summary_reports_weighted_score(self):
        suite = Suite(
            "w",
            [
                Case("a", "q1", {"type": "exact", "expected": "x"}, weight=2),
                Case("b", "q2", {"type": "exact", "expected": "x"}, weight=1),
            ],
        )
        report = run_suite(suite, lambda prompt: "x" if prompt == "q1" else "y")
        text = format_summary(report)
        self.assertIn("Weighted score: 2.00/3 (66.7%)", text)
        self.assertIn("1.00", text)
        self.assertIn("0.00", text)


class TestParallelExecution(unittest.TestCase):
    def _suite(self, n=4, delay=0.3):
        return Suite(
            "p",
            [
                Case(f"c{i}", f"q{i}", {"type": "exact", "expected": "yes"})
                for i in range(n)
            ],
        )

    def _sleepy_agent(self, delay=0.3):
        def agent(prompt):
            time.sleep(delay)
            return "yes"

        return agent

    def test_parallel_matches_sequential_verdicts_and_order(self):
        suite = self._suite()
        agent = self._sleepy_agent(0.05)
        seq = run_suite(suite, agent)
        par = run_suite(suite, agent, workers=4)
        self.assertEqual(
            [r.case_id for r in par.results],
            [r.case_id for r in seq.results],
        )
        self.assertEqual(
            [r.passed for r in par.results],
            [r.passed for r in seq.results],
        )
        self.assertEqual(par.passed, 4)

    def test_parallel_is_actually_faster(self):
        suite = self._suite(n=4, delay=0.3)
        t0 = time.perf_counter()
        report = run_suite(suite, self._sleepy_agent(0.3), workers=4)
        wall = time.perf_counter() - t0
        self.assertEqual(report.passed, 4)
        # Sequential would take ~1.2 s; parallel should take ~0.3 s.
        self.assertLess(wall, 0.9)

    def test_workers_one_matches_default(self):
        suite = self._suite()
        agent = self._sleepy_agent(0.01)
        default = run_suite(suite, agent)
        one = run_suite(suite, agent, workers=1)
        self.assertEqual(
            [r.case_id for r in one.results],
            [r.case_id for r in default.results],
        )
        self.assertEqual(one.passed, default.passed)

    def test_hung_case_times_out_while_others_complete_in_parallel(self):
        suite = Suite(
            "p",
            [
                Case("slow", "q1", {"type": "exact", "expected": "x"}, timeout=0.05),
                Case("fast-1", "q2", {"type": "exact", "expected": "y"}),
                Case("fast-2", "q3", {"type": "exact", "expected": "y"}),
            ],
        )

        def agent(prompt):
            if prompt == "q1":
                time.sleep(0.5)
                return "x"
            return "y"

        t0 = time.perf_counter()
        report = run_suite(suite, agent, workers=3)
        wall = time.perf_counter() - t0
        self.assertEqual(report.total, 3)
        self.assertEqual(report.passed, 2)
        slow = report.results[0]
        self.assertFalse(slow.passed)
        self.assertTrue(slow.timed_out)
        self.assertTrue(report.results[1].passed)
        self.assertTrue(report.results[2].passed)
        # The hung case must not stall the run past its own timeout.
        self.assertLess(wall, 0.4)

    def test_results_stay_in_suite_order_when_later_case_finishes_first(self):
        suite = Suite(
            "p",
            [
                Case("slow", "q1", {"type": "exact", "expected": "x"}),
                Case("fast", "q2", {"type": "exact", "expected": "y"}),
            ],
        )

        def agent(prompt):
            if prompt == "q1":
                time.sleep(0.2)
                return "x"
            return "y"

        report = run_suite(suite, agent, workers=2)
        self.assertEqual(
            [r.case_id for r in report.results], ["slow", "fast"]
        )
        self.assertEqual(report.passed, 2)

    def test_invalid_workers_rejected(self):
        suite = self._suite()
        for bad in (0, -1, "two", 2.5, True):
            with self.assertRaises(ValueError):
                run_suite(suite, lambda prompt: "yes", workers=bad)

    def test_cli_workers_flag_runs_suite(self):
        from agent_evals.cli import main

        data = {
            "name": "cli-p",
            "cases": [
                {
                    "id": "add",
                    "prompt": "2 + 3",
                    "scorer": {"type": "exact", "expected": "5"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            code = main(
                [
                    "--suite",
                    path,
                    "--agent",
                    "examples.calculator_agent:agent",
                    "--workers",
                    "2",
                ]
            )
        self.assertEqual(code, 0)

    def test_cli_rejects_non_positive_workers(self):
        from agent_evals.cli import main

        with self.assertRaises(SystemExit):
            main(
                [
                    "--suite",
                    "examples/calculator.json",
                    "--agent",
                    "examples.calculator_agent:agent",
                    "--workers",
                    "0",
                ]
            )


class TestLatencyBudgets(unittest.TestCase):
    def _budget_case(self, budget_ms, prompt="q"):
        return Case(
            "budget",
            prompt,
            {"type": "exact", "expected": "x"},
            max_latency_ms=budget_ms,
        )

    def test_budget_parsed_from_dict_and_round_trips(self):
        case = Case.from_dict(
            {
                "id": "c",
                "prompt": "p",
                "scorer": {"type": "exact", "expected": "x"},
                "max_latency_ms": 250,
            }
        )
        self.assertEqual(case.max_latency_ms, 250.0)
        self.assertEqual(case.to_dict()["max_latency_ms"], 250.0)

    def test_budget_absent_by_default(self):
        case = Case.from_dict(
            {"id": "c", "prompt": "p", "scorer": {"type": "exact", "expected": "x"}}
        )
        self.assertIsNone(case.max_latency_ms)
        self.assertNotIn("max_latency_ms", case.to_dict())

    def test_invalid_budget_rejected(self):
        for bad in (0, -1, "fast"):
            with self.assertRaises(ValueError):
                Case.from_dict(
                    {
                        "id": "c",
                        "prompt": "p",
                        "scorer": {"type": "exact", "expected": "x"},
                        "max_latency_ms": bad,
                    }
                )

    def test_within_budget_passes(self):
        suite = Suite("b", [self._budget_case(5000)])
        report = run_suite(suite, lambda prompt: "x")
        result = report.results[0]
        self.assertTrue(result.passed)
        self.assertFalse(result.budget_violated)
        self.assertEqual(result.max_latency_ms, 5000.0)
        self.assertEqual(report.budget_violations, 0)

    def test_over_budget_is_flagged_not_passed(self):
        suite = Suite("b", [self._budget_case(10)])

        def agent(prompt):
            time.sleep(0.1)
            return "x"

        report = run_suite(suite, agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.budget_violated)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.max_latency_ms, 10.0)
        self.assertGreater(result.latency_ms, 10.0)
        self.assertIn("latency budget", result.detail)
        self.assertIn("10 ms", result.detail)
        self.assertEqual(report.budget_violations, 1)
        self.assertEqual(report.passed, 0)

    def test_no_budget_leaves_slow_pass_unchanged(self):
        suite = Suite(
            "b", [Case("slow", "q", {"type": "exact", "expected": "x"})]
        )

        def agent(prompt):
            time.sleep(0.1)
            return "x"

        report = run_suite(suite, agent)
        result = report.results[0]
        self.assertTrue(result.passed)
        self.assertFalse(result.budget_violated)
        self.assertIsNone(result.max_latency_ms)
        self.assertEqual(report.budget_violations, 0)

    def test_failing_case_is_not_a_budget_violation(self):
        suite = Suite("b", [self._budget_case(10)])

        def agent(prompt):
            time.sleep(0.1)
            return "wrong"

        report = run_suite(suite, agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertFalse(result.budget_violated)
        self.assertNotIn("latency budget", result.detail)
        self.assertEqual(report.budget_violations, 0)

    def test_timed_out_case_is_not_a_budget_violation(self):
        suite = Suite(
            "b",
            [
                Case(
                    "slow",
                    "q",
                    {"type": "exact", "expected": "x"},
                    timeout=0.05,
                    max_latency_ms=10,
                )
            ],
        )

        def agent(prompt):
            time.sleep(0.5)
            return "x"

        report = run_suite(suite, agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.budget_violated)
        self.assertEqual(report.budget_violations, 0)

    def test_budget_fields_in_json_report(self):
        suite = Suite("b", [self._budget_case(10)])

        def agent(prompt):
            time.sleep(0.1)
            return "x"

        report = run_suite(suite, agent)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            report.save(path)
            loaded = load_report(path)
        self.assertEqual(loaded["budget_violations"], 1)
        result = loaded["results"][0]
        self.assertFalse(result["passed"])
        self.assertTrue(result["budget_violated"])
        self.assertEqual(result["max_latency_ms"], 10.0)
        self.assertIn("latency budget", result["detail"])

    def test_summary_marks_budget_violation(self):
        suite = Suite("b", [self._budget_case(10)])

        def agent(prompt):
            time.sleep(0.1)
            return "x"

        report = run_suite(suite, agent)
        text = format_summary(report)
        self.assertIn("BUDGET", text)
        self.assertIn("Budget violations: 1", text)

    def test_cli_fail_on_budget_flag(self):
        from agent_evals.cli import main

        data = {
            "name": "cli-b",
            "cases": [
                {
                    "id": "slow",
                    "prompt": "2 + 3",
                    "scorer": {"type": "exact", "expected": "5"},
                    "max_latency_ms": 0.01,
                }
            ],
        }
        # The calculator agent answers correctly but takes far longer
        # than 0.01 ms, so the case is a budget violation: reported as a
        # failure, but the run only fails the CLI when --fail-on-budget
        # is given.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            code = main(
                [
                    "--suite",
                    path,
                    "--agent",
                    "examples.calculator_agent:agent",
                ]
            )
            self.assertEqual(code, 0)
            code = main(
                [
                    "--suite",
                    path,
                    "--agent",
                    "examples.calculator_agent:agent",
                    "--fail-on-budget",
                ]
            )
            self.assertEqual(code, 1)


class TestTokenBudgets(unittest.TestCase):
    def _cost_case(self, budget, prompt="q"):
        return Case(
            "cost",
            prompt,
            {"type": "exact", "expected": "x"},
            max_tokens=budget,
        )

    def _token_agent(self, tokens):
        def agent(prompt):
            return "x", tokens

        return agent

    def test_budget_parsed_from_dict_and_round_trips(self):
        case = Case.from_dict(
            {
                "id": "c",
                "prompt": "p",
                "scorer": {"type": "exact", "expected": "x"},
                "max_tokens": 100,
            }
        )
        self.assertEqual(case.max_tokens, 100)
        self.assertEqual(case.to_dict()["max_tokens"], 100)

    def test_budget_absent_by_default(self):
        case = Case.from_dict(
            {"id": "c", "prompt": "p", "scorer": {"type": "exact", "expected": "x"}}
        )
        self.assertIsNone(case.max_tokens)
        self.assertNotIn("max_tokens", case.to_dict())

    def test_invalid_budget_rejected(self):
        for bad in (0, -1, "many", 1.5, True):
            with self.assertRaises(ValueError):
                Case.from_dict(
                    {
                        "id": "c",
                        "prompt": "p",
                        "scorer": {"type": "exact", "expected": "x"},
                        "max_tokens": bad,
                    }
                )

    def test_adding_a_budget_changes_the_fingerprint(self):
        plain = Case("c", "p", {"type": "exact", "expected": "x"})
        budgeted = Case(
            "c", "p", {"type": "exact", "expected": "x"}, max_tokens=100
        )
        self.assertNotEqual(plain.fingerprint(), budgeted.fingerprint())

    def test_within_budget_passes(self):
        suite = Suite("b", [self._cost_case(100)])
        report = run_suite(suite, self._token_agent(42))
        result = report.results[0]
        self.assertTrue(result.passed)
        self.assertFalse(result.cost_violated)
        self.assertEqual(result.max_tokens, 100)
        self.assertEqual(result.tokens, 42)
        self.assertEqual(report.cost_violations, 0)

    def test_at_budget_passes(self):
        suite = Suite("b", [self._cost_case(100)])
        report = run_suite(suite, self._token_agent(100))
        self.assertTrue(report.results[0].passed)
        self.assertFalse(report.results[0].cost_violated)

    def test_over_budget_is_flagged_not_passed(self):
        suite = Suite("b", [self._cost_case(100)])
        report = run_suite(suite, self._token_agent(250))
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.cost_violated)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.max_tokens, 100)
        self.assertEqual(result.tokens, 250)
        self.assertIn("token budget", result.detail)
        self.assertIn("250 tokens > 100 tokens", result.detail)
        self.assertEqual(report.cost_violations, 1)
        self.assertEqual(report.passed, 0)

    def test_no_tokens_reported_leaves_pass_unchanged(self):
        # An agent that returns a plain string reports no token count,
        # so the budget has nothing to compare against and cannot trip.
        suite = Suite("b", [self._cost_case(100)])
        report = run_suite(suite, lambda prompt: "x")
        result = report.results[0]
        self.assertTrue(result.passed)
        self.assertFalse(result.cost_violated)
        self.assertIsNone(result.tokens)
        self.assertEqual(report.cost_violations, 0)

    def test_failing_case_is_not_a_cost_violation(self):
        suite = Suite(
            "b",
            [Case("c", "q", {"type": "exact", "expected": "x"}, max_tokens=100)],
        )
        report = run_suite(suite, lambda prompt: ("wrong", 250))
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertFalse(result.cost_violated)
        self.assertNotIn("token budget", result.detail)
        self.assertEqual(report.cost_violations, 0)

    def test_timed_out_case_is_not_a_cost_violation(self):
        suite = Suite(
            "b",
            [
                Case(
                    "slow",
                    "q",
                    {"type": "exact", "expected": "x"},
                    timeout=0.05,
                    max_tokens=1,
                )
            ],
        )

        def agent(prompt):
            time.sleep(0.5)
            return "x", 100

        report = run_suite(suite, agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)
        self.assertFalse(result.cost_violated)
        self.assertEqual(report.cost_violations, 0)

    def test_latency_budget_is_checked_before_token_budget(self):
        # A case over both budgets is a latency violation, not both:
        # one verdict, one reason.
        suite = Suite(
            "b",
            [
                Case(
                    "both",
                    "q",
                    {"type": "exact", "expected": "x"},
                    max_latency_ms=10,
                    max_tokens=1,
                )
            ],
        )

        def agent(prompt):
            time.sleep(0.1)
            return "x", 100

        report = run_suite(suite, agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.budget_violated)
        self.assertFalse(result.cost_violated)
        self.assertEqual(report.budget_violations, 1)
        self.assertEqual(report.cost_violations, 0)

    def test_cost_fields_in_json_report(self):
        suite = Suite("b", [self._cost_case(100)])
        report = run_suite(suite, self._token_agent(250))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            report.save(path)
            loaded = load_report(path)
        self.assertEqual(loaded["cost_violations"], 1)
        result = loaded["results"][0]
        self.assertFalse(result["passed"])
        self.assertTrue(result["cost_violated"])
        self.assertEqual(result["max_tokens"], 100)
        self.assertEqual(result["tokens"], 250)
        self.assertIn("token budget", result["detail"])

    def test_summary_marks_cost_violation(self):
        suite = Suite("b", [self._cost_case(100)])
        report = run_suite(suite, self._token_agent(250))
        text = format_summary(report)
        self.assertIn("COST", text)
        self.assertIn("Cost violations: 1", text)

    def test_cli_fail_on_cost_flag(self):
        from agent_evals.cli import main

        data = {
            "name": "cli-c",
            "cases": [
                {
                    "id": "chatty",
                    "prompt": "2 + 3",
                    "scorer": {"type": "exact", "expected": "5"},
                    "max_tokens": 10,
                }
            ],
        }
        agent_src = (
            "def agent(prompt):\n"
            "    return \"5\", 120\n"
        )
        # The agent answers correctly but reports 120 tokens against
        # a 10-token budget: a cost violation, reported as a failure,
        # but the run only fails the CLI when --fail-on-cost is given.
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "token_agent.py"), "w") as f:
                f.write(agent_src)
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            sys.path.insert(0, tmp)
            try:
                code = main(
                    ["--suite", path, "--agent", "token_agent:agent"]
                )
                self.assertEqual(code, 0)
                code = main(
                    [
                        "--suite",
                        path,
                        "--agent",
                        "token_agent:agent",
                        "--fail-on-cost",
                    ]
                )
                self.assertEqual(code, 1)
            finally:
                sys.path.remove(tmp)
                sys.modules.pop("token_agent", None)


class TestOutputDiff(unittest.TestCase):
    def test_identical_outputs_have_no_diff(self):
        self.assertIsNone(make_output_diff("same\nlines", "same\nlines"))

    def test_diff_marks_removed_and_added_lines(self):
        diff = make_output_diff("line1\nline2\nline3", "line1\nCHANGED\nline3")
        self.assertIn("-line2", diff)
        self.assertIn("+CHANGED", diff)
        self.assertIn("line1", diff)  # context lines are kept
        self.assertIn("--- expected", diff)
        self.assertIn("+++ actual", diff)

    def test_max_lines_caps_the_diff(self):
        expected = "\n".join(f"line{i}" for i in range(20))
        actual = "\n".join(f"CHANGED{i}" for i in range(20))
        diff = make_output_diff(expected, actual, max_lines=5)
        self.assertEqual(len(diff.splitlines()), 6)  # 5 lines + truncation note
        self.assertIn("more diff lines", diff)

    def test_failing_exact_case_carries_a_diff(self):
        suite = Suite(
            "d",
            [Case("bad", "q", {"type": "exact", "expected": "alpha\nbeta"})],
        )
        report = run_suite(suite, lambda prompt: "alpha\ngamma")
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertIn("-beta", result.diff)
        self.assertIn("+gamma", result.diff)

    def test_passing_exact_case_has_no_diff(self):
        suite = Suite(
            "d", [Case("ok", "q", {"type": "exact", "expected": "yes"})]
        )
        report = run_suite(suite, lambda prompt: "yes")
        self.assertIsNone(report.results[0].diff)

    def test_non_exact_failing_case_has_no_diff(self):
        suite = Suite(
            "d",
            [Case("bad", "q", {"type": "contains", "substring": "hello"})],
        )
        report = run_suite(suite, lambda prompt: "goodbye")
        self.assertFalse(report.results[0].passed)
        self.assertIsNone(report.results[0].diff)

    def test_diff_round_trips_in_json_report(self):
        suite = Suite(
            "d", [Case("bad", "q", {"type": "exact", "expected": "a\nb"})]
        )
        report = run_suite(suite, lambda prompt: "a\nc")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            report.save(path)
            loaded = load_report(path)
        self.assertIn("-b", loaded["results"][0]["diff"])
        self.assertIn("+c", loaded["results"][0]["diff"])

    def test_summary_prints_the_diff_under_the_failing_row(self):
        suite = Suite(
            "d", [Case("bad", "q", {"type": "exact", "expected": "a\nb"})]
        )
        report = run_suite(suite, lambda prompt: "a\nc")
        text = format_summary(report)
        self.assertIn("FAIL", text)
        self.assertIn("-b", text)
        self.assertIn("+c", text)


if __name__ == "__main__":
    unittest.main()
