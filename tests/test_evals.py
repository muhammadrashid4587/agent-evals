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


class TestCountScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        ok, detail, score = score_case(
            _case({"type": "count", "substring": "step", "min": 2, "max": 3}),
            "step one, step two",
        )
        self.assertTrue(ok)
        self.assertIn("found 'step' 2 time(s)", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "count", "substring": "x", "min": 2, "max": 2}),
            "x x",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "count", "substring": "item", "min": 3}),
            "item item item item item",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "count", "substring": "sorry", "max": 1}),
            "sorry about that",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 1 of a 2 minimum: score falls linearly to 0 at 0 occurrences.
        ok, detail, score = score_case(
            _case({"type": "count", "substring": "step", "min": 2}),
            "one step only",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_zero_occurrences_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "count", "substring": "step", "min": 2}),
            "nothing here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 3 of a 2 maximum: halfway to 2x the max, so score 0.5.
        ok, detail, score = score_case(
            _case({"type": "count", "substring": "x", "max": 2}),
            "x x x",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "count", "substring": "x", "max": 2}),
            "x x x x",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_occurrence(self):
        ok, _, score = score_case(
            _case({"type": "count", "substring": "x", "max": 0}),
            "clean output",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "count", "substring": "x", "max": 0}),
            "one x here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_non_overlapping_count(self):
        # "aaa" contains "aa" once non-overlapping, not twice.
        ok, _, score = score_case(
            _case({"type": "count", "substring": "aa", "min": 2}),
            "aaa",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)

    def test_ignore_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "count",
                    "substring": "Step",
                    "min": 2,
                    "ignore_case": True,
                }
            ),
            "STEP one, step two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_missing_substring_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "count", "min": 1}), "hello")

    def test_empty_substring_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "count", "substring": "", "min": 1}), "hello"
            )

    def test_non_string_substring_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "count", "substring": 42, "min": 1}), "hello"
            )

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "count", "substring": "x"}), "x x")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "count", "substring": "x", "min": 3, "max": 1}),
                "x x",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "count", "substring": "x", "min": -1}), "x"
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "count", "substring": "x", "min": 1.5}), "x"
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "count",
                        "substring": "x",
                        "min": 1,
                        "ignore_case": "yes",
                    }
                ),
                "x",
            )


class TestRegexCountScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "regex_count",
                    "pattern": r"\d+",
                    "min": 2,
                    "max": 3,
                }
            ),
            "item 1, item 2",
        )
        self.assertTrue(ok)
        self.assertIn(r"pattern '\\d+' matched 2 time(s)", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"x", "min": 2, "max": 2}),
            "x x",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"\d+", "min": 3}),
            "1 2 3 4 5",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"\bTODO\b", "max": 1}),
            "TODO: fix this",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 1 of a 2 minimum: score falls linearly to 0 at 0 matches.
        ok, detail, score = score_case(
            _case({"type": "regex_count", "pattern": r"\d+", "min": 2}),
            "only 1 here",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_zero_matches_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"\d+", "min": 2}),
            "nothing here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 3 of a 2 maximum: halfway to 2x the max, so score 0.5.
        ok, detail, score = score_case(
            _case({"type": "regex_count", "pattern": r"x", "max": 2}),
            "x x x",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"x", "max": 2}),
            "x x x x",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_match(self):
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"x", "max": 0}),
            "clean output",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"x", "max": 0}),
            "one x here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_non_overlapping_matches(self):
        # "aaa" matches "aa" once non-overlapping, not twice.
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": "aa", "min": 2}),
            "aaa",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)

    def test_capturing_groups_count_matches_not_groups(self):
        # "(ab)+" matches "abab" once and "ab" once: 2 matches, not 4 groups.
        ok, _, score = score_case(
            _case({"type": "regex_count", "pattern": r"(ab)+", "min": 2, "max": 2}),
            "abab ab",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "regex_count",
                    "pattern": r"step",
                    "min": 2,
                    "ignore_case": True,
                }
            ),
            "STEP one, step two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_missing_pattern_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "regex_count", "min": 1}), "hello")

    def test_non_string_pattern_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "regex_count", "pattern": 42, "min": 1}), "hello"
            )

    def test_invalid_regex_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "regex_count", "pattern": r"([", "min": 1}), "hello"
            )

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "regex_count", "pattern": r"x"}), "x x")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "regex_count", "pattern": r"x", "min": 3, "max": 1}
                ),
                "x x",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "regex_count", "pattern": r"x", "min": -1}), "x"
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "regex_count", "pattern": r"x", "min": 1.5}), "x"
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "regex_count",
                        "pattern": r"x",
                        "min": 1,
                        "ignore_case": "yes",
                    }
                ),
                "x",
            )


class TestWordCountScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        ok, detail, score = score_case(
            _case({"type": "word_count", "min": 2, "max": 3}),
            "one two",
        )
        self.assertTrue(ok)
        self.assertIn("word count 2", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "min": 2, "max": 2}),
            "one two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "min": 3}),
            "one two three four five",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "max": 1}),
            "one",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 1 of a 2 minimum: score falls linearly to 0 at 0 words.
        ok, detail, score = score_case(
            _case({"type": "word_count", "min": 2}),
            "only",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_zero_words_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "min": 2}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 3 of a 2 maximum: halfway to 2x the max, so score 0.5.
        ok, detail, score = score_case(
            _case({"type": "word_count", "max": 2}),
            "one two three",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "max": 2}),
            "one two three four",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_word(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "word_count", "max": 0}),
            "one word here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_whitespace_runs_count_as_one_separator(self):
        ok, _, score = score_case(
            _case({"type": "word_count", "min": 2, "max": 2}),
            "  one   two\n",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_punctuation_stays_attached(self):
        # "hello," and "hello" are different tokens, like length's
        # word unit: this is two words, not one.
        ok, _, score = score_case(
            _case({"type": "word_count", "min": 2, "max": 2}),
            "hello, hello",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "word_count"}), "one two")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_count", "min": 3, "max": 1}),
                "one two",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_count", "min": -1}),
                "one",
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_count", "min": 1.5}),
                "one",
            )


class TestUniqueWordsScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        ok, detail, score = score_case(
            _case({"type": "unique_words", "min": 2, "max": 3}),
            "one two one",
        )
        self.assertTrue(ok)
        self.assertIn("unique word count 2", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "unique_words", "min": 2, "max": 2}),
            "one two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "unique_words", "min": 3}),
            "one two three one",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "unique_words", "max": 1}),
            "same same same",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 1 of a 2 minimum: score falls linearly to 0 at 0 words.
        ok, detail, score = score_case(
            _case({"type": "unique_words", "min": 2}),
            "same same same",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_zero_words_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "unique_words", "min": 2}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 3 of a 2 maximum: halfway to 2x the max, so score 0.5.
        ok, detail, score = score_case(
            _case({"type": "unique_words", "max": 2}),
            "one two three",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "unique_words", "max": 2}),
            "one two three four",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_word(self):
        ok, _, score = score_case(
            _case({"type": "unique_words", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "unique_words", "max": 0}),
            "one word here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_repeated_words_count_once(self):
        # Five words, two distinct: a min of 3 fails at 2/3.
        ok, _, score = score_case(
            _case({"type": "unique_words", "min": 3}),
            "yes yes yes no no",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 2 / 3)

    def test_ignore_case_folds_before_deduping(self):
        # Without ignore_case, "The" and "the" are two words.
        ok, _, score = score_case(
            _case({"type": "unique_words", "min": 2, "max": 2}),
            "The the the",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # With ignore_case they fold to one word: below a 2 minimum.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "unique_words",
                    "min": 2,
                    "max": 2,
                    "ignore_case": True,
                }
            ),
            "The the the",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 2", detail)
        self.assertAlmostEqual(score, 0.5)
        # And a folded single word passes a max of 1.
        ok, _, score = score_case(
            _case(
                {
                    "type": "unique_words",
                    "min": 1,
                    "max": 1,
                    "ignore_case": True,
                }
            ),
            "The the THE",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_case_sensitive_by_default(self):
        # Without ignore_case, "The" and "the" are different words.
        ok, _, score = score_case(
            _case({"type": "unique_words", "min": 2, "max": 2}),
            "The the",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "unique_words"}), "one two")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "unique_words", "min": 3, "max": 1}),
                "one two",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "unique_words", "min": -1}),
                "one",
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "unique_words", "min": 1.5}),
                "one",
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "unique_words",
                        "min": 1,
                        "ignore_case": "yes",
                    }
                ),
                "one",
            )


class TestSentenceCountScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        ok, detail, score = score_case(
            _case({"type": "sentence_count", "min": 2, "max": 3}),
            "One sentence. Two sentences.",
        )
        self.assertTrue(ok)
        self.assertIn("sentence count 2", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 2, "max": 2}),
            "One. Two.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 3}),
            "One. Two. Three. Four.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "max": 1}),
            "Just one.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 1 of a 2 minimum: score falls linearly to 0 at 0 sentences.
        ok, detail, score = score_case(
            _case({"type": "sentence_count", "min": 2}),
            "Only one.",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_zero_sentences_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 2}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        # Punctuation alone is no sentence either.
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 1}),
            "...",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 3 of a 2 maximum: halfway to 2x the max, so score 0.5.
        ok, detail, score = score_case(
            _case({"type": "sentence_count", "max": 2}),
            "One. Two. Three.",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "max": 2}),
            "One. Two. Three. Four.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_sentence(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "sentence_count", "max": 0}),
            "One sentence here.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_unterminated_sentence_counts(self):
        # No terminal punctuation: still one sentence.
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 1, "max": 1}),
            "Hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_mixed_terminators_count(self):
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 3, "max": 3}),
            "Really? Yes! Absolutely.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_repeated_punctuation_is_one_terminator(self):
        # "Wait... Really?" is two sentences, not four.
        ok, _, score = score_case(
            _case({"type": "sentence_count", "min": 2, "max": 2}),
            "Wait... Really?",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "sentence_count"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_count", "min": 3, "max": 1}),
                "One. Two.",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_count", "min": -1}),
                "One.",
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_count", "min": 1.5}),
                "One.",
            )


class TestKeywordDensityScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # 4 keyword hits in 8 words: density 0.5.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step", "verify"],
                    "min": 0.4,
                    "max": 0.6,
                }
            ),
            "step one step two verify three step four",
        )
        self.assertTrue(ok)
        self.assertIn("keyword density 0.500 (4 of 8 words)", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # 1 hit in 10 words: density exactly 0.1.
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.1,
                    "max": 0.1,
                }
            ),
            "step one two three four five six seven eight nine",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.2,
                }
            ),
            "step one step two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "max": 0.1,
                }
            ),
            "step one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # Density 0.05 of a 0.10 minimum: halfway to the floor.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.1,
                }
            ),
            "step one two three four five six seven eight nine ten "
            "eleven twelve thirteen fourteen fifteen sixteen "
            "seventeen eighteen nineteen",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 0.1", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_zero_hits_scores_zero(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.05,
                }
            ),
            "one two three four",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # Density 0.30 of a 0.20 maximum: halfway to 2x the max.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "max": 0.2,
                }
            ),
            "step step step one two three four five six seven",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 0.2", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        # Density 0.40 of a 0.20 maximum: at 2x the max.
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "max": 0.2,
                }
            ),
            "step step step step one two three four five six",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_keyword(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "max": 0,
                }
            ),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "max": 0,
                }
            ),
            "step one",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero_unless_min_is_zero(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.05,
                }
            ),
            "   ",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0,
                }
            ),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case_matches(self):
        # "Step" and "STEP" are 2 of 4 words with ignore_case.
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.5,
                    "max": 0.5,
                    "ignore_case": True,
                }
            ),
            "Step one STEP two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # Without it, neither capitalised token matches.
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.5,
                }
            ),
            "Step one STEP two",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_punctuation_and_substrings_do_not_count(self):
        # "step," and "stepping" are not the keyword "step".
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_density",
                    "keywords": ["step"],
                    "min": 0.1,
                }
            ),
            "step, stepping along",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_missing_keywords_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "keyword_density", "min": 0.1}),
                "step one",
            )

    def test_empty_keywords_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": [],
                        "min": 0.1,
                    }
                ),
                "step one",
            )

    def test_non_string_keyword_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step", 7],
                        "min": 0.1,
                    }
                ),
                "step one",
            )

    def test_empty_string_keyword_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": [""],
                        "min": 0.1,
                    }
                ),
                "step one",
            )

    def test_multiword_keyword_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step by step"],
                        "min": 0.1,
                    }
                ),
                "step by step",
            )

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "keyword_density", "keywords": ["step"]}),
                "step one",
            )

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step"],
                        "min": 0.5,
                        "max": 0.1,
                    }
                ),
                "step one",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step"],
                        "min": -0.1,
                    }
                ),
                "step one",
            )

    def test_bound_above_one_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step"],
                        "max": 1.5,
                    }
                ),
                "step one",
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step"],
                        "min": "0.1",
                    }
                ),
                "step one",
            )

    def test_nan_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step"],
                        "min": float("nan"),
                    }
                ),
                "step one",
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_density",
                        "keywords": ["step"],
                        "min": 0.1,
                        "ignore_case": "yes",
                    }
                ),
                "step one",
            )


class TestAvgSentenceLengthScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # 7 words in 2 sentences: average 3.5.
        ok, detail, score = score_case(
            _case({"type": "avg_sentence_length", "min": 3, "max": 4}),
            "One two three. Four five six seven.",
        )
        self.assertTrue(ok)
        self.assertIn("average sentence length 3.50", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # 10 words in 2 sentences: average exactly 5.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 5, "max": 5}),
            "One two three four five. Six seven eight nine ten.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        # 9 words in 2 sentences: average 4.5, above a 3 minimum.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 3}),
            "One two three four. Five six seven eight nine.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # 4 words in 2 sentences: average 2, below a 5 maximum.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "max": 5}),
            "Short one. Short two.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # Average 5 of a 10 minimum: halfway to the floor, score 0.5.
        ok, detail, score = score_case(
            _case({"type": "avg_sentence_length", "min": 10}),
            "One two three four five. Six seven eight nine ten.",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 10", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_fractional_average_partial_credit(self):
        # 10 words in one sentence, 15 in the other: average 12.5
        # of a 10 maximum, so (2*10 - 12.5) / 10 = 0.75.
        first = " ".join(["word"] * 10) + "."
        second = " ".join(["word"] * 15) + "."
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "max": 10}),
            f"{first} {second}",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.75)

    def test_zero_average_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 2}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        # Punctuation alone is no sentence either.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 1}),
            "...",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No sentences: an average of 0 is in range when the
        # minimum is 0, and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "max": 5}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_above_maximum_partial_credit(self):
        # Average 15 of a 10 maximum: halfway to 2x the max, score 0.5.
        sentence = " ".join(["word"] * 15) + "."
        ok, detail, score = score_case(
            _case({"type": "avg_sentence_length", "max": 10}),
            " ".join([sentence] * 3),
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 10", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        sentence = " ".join(["word"] * 20) + "."
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "max": 10}),
            " ".join([sentence] * 2),
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_sentence(self):
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "max": 0}),
            "One sentence here.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_unterminated_sentence_counts(self):
        # No terminal punctuation: still one sentence of four words.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 4, "max": 4}),
            "Hello world foo bar",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_mixed_terminators_count(self):
        # Three two-word sentences: average exactly 2.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 2, "max": 2}),
            "Really now? Yes indeed! Absolutely right.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_repeated_punctuation_is_one_terminator(self):
        # "Wait... Really?" is two one-word sentences, average 1.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 1, "max": 1}),
            "Wait... Really?",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_abbreviation_is_naive(self):
        # "Mr. Smith went home." is two sentences ("Mr", "Smith went
        # home"): 4 words in 2 sentences, average 2 — the same naive
        # split sentence_count uses, abbreviations and all.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 2, "max": 2}),
            "Mr. Smith went home.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_words_counted_within_each_sentence(self):
        # Punctuation with no space between the words still splits
        # the sentences, and each sentence's words are counted
        # inside it: "one...two" is two one-word sentences.
        ok, _, score = score_case(
            _case({"type": "avg_sentence_length", "min": 1, "max": 1}),
            "one...two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "avg_sentence_length"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_sentence_length", "min": 10, "max": 5}),
                "One. Two.",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_sentence_length", "min": -1}),
                "One.",
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_sentence_length", "min": 2.5}),
                "One.",
            )


class TestSentenceLengthVarianceScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # Sentences of 2 and 4 words: mean 3, population variance 1.0.
        ok, detail, score = score_case(
            _case({"type": "sentence_length_variance", "min": 1, "max": 1}),
            "One two. Three four five six.",
        )
        self.assertTrue(ok)
        self.assertIn(
            "sentence length variance 1.00 (sentence lengths 2, 4)",
            detail,
        )
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # Sentences of 1 and 3 words: mean 2, variance 1.0.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 1, "max": 1}),
            "One. Two three four.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_uniform_sentences_have_zero_variance(self):
        # Two two-word sentences: every length equals the mean.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0, "max": 0}),
            "One two. Three four.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # A single sentence always has a variance of 0.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0, "max": 0}),
            "One two three.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_fractional_bounds_accepted(self):
        # Sentences of 1 and 2 words: mean 1.5, variance 0.25 —
        # a variance is a squared word count, so bounds need not
        # be integers.
        ok, _, score = score_case(
            _case(
                {"type": "sentence_length_variance", "min": 0.25, "max": 0.25}
            ),
            "One. Two three.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        # Variance 1.0, above a 0.5 minimum.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0.5}),
            "One two. Three four five six.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # Variance 0, below a 0.5 maximum.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "max": 0.5}),
            "One two. Three four.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # Variance 0.25 of a 1 minimum: score 0.25.
        ok, detail, score = score_case(
            _case({"type": "sentence_length_variance", "min": 1}),
            "One. Two three.",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 1", detail)
        self.assertAlmostEqual(score, 0.25)

    def test_above_maximum_partial_credit(self):
        # Sentences of 1 and 7 words: mean 4, variance 9, of a 5
        # maximum: (2*5 - 9) / 5 = 0.2.
        ok, detail, score = score_case(
            _case({"type": "sentence_length_variance", "max": 5}),
            "One. Two three four five six seven eight.",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 5", detail)
        self.assertAlmostEqual(score, 0.2)

    def test_twice_maximum_scores_zero(self):
        # Sentences of 1, 1, and 7 words: mean 3, variance 8 —
        # twice a 4 maximum, so nothing.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "max": 4}),
            "One. Two. Three four five six seven eight nine.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_variation(self):
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "max": 0}),
            "One two. Three four five six.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no sentences in output", detail)
        self.assertEqual(score, 0.0)
        # Punctuation alone is no sentence either.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 1}),
            "...",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No sentences: a variance of 0 is in range when the
        # minimum is 0, and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "max": 5}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_unterminated_sentence_has_zero_variance(self):
        # No terminal punctuation: still one sentence, variance 0.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0, "max": 0}),
            "Hello world foo bar",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_repeated_punctuation_is_one_terminator(self):
        # "Wait... Really?" is two one-word sentences, variance 0.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 0, "max": 0}),
            "Wait... Really?",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_abbreviation_is_naive(self):
        # "Mr. Smith went home." counts as two sentences of 1 and
        # 3 words: variance 1.0, the same limit avg_sentence_length
        # has, inherited unchanged.
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "min": 1, "max": 1}),
            "Mr. Smith went home.",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_agrees_with_avg_sentence_length(self):
        # The same output's mean and variance are measured by two
        # scorers: sentences of 2 and 4 words average 3.0 and vary
        # by 1.0, so both bands pin exactly.
        output = "One two. Three four five six."
        ok, _, _ = score_case(
            _case({"type": "avg_sentence_length", "min": 3, "max": 3}),
            output,
        )
        self.assertTrue(ok)
        ok, _, _ = score_case(
            _case({"type": "sentence_length_variance", "min": 1, "max": 1}),
            output,
        )
        self.assertTrue(ok)
        # And the two scorers can disagree: sentences of 1 and 6
        # words average 3.5 (in [3, 4]) but vary by 6.25.
        output = "One. Two three four five six seven."
        ok, _, _ = score_case(
            _case({"type": "avg_sentence_length", "min": 3, "max": 4}),
            output,
        )
        self.assertTrue(ok)
        ok, _, score = score_case(
            _case({"type": "sentence_length_variance", "max": 1}),
            output,
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "sentence_length_variance"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_length_variance", "min": 2, "max": 1}),
                "One. Two.",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_length_variance", "min": -1}),
                "One.",
            )

    def test_string_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_length_variance", "min": "1"}),
                "One.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_length_variance", "min": True}),
                "One.",
            )

    def test_non_finite_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_length_variance", "min": float("nan")}),
                "One.",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "sentence_length_variance", "max": float("inf")}),
                "One.",
            )


class TestAvgWordLengthScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "hello world": 10 characters in 2 words, average 5.0.
        ok, detail, score = score_case(
            _case({"type": "avg_word_length", "min": 5, "max": 5}),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertIn(
            "average word length 5.00 characters (10 characters in 2 word(s))",
            detail,
        )
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # Three two-letter words: average 2.0 exactly.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 2, "max": 2}),
            "aa bb cc",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_single_word(self):
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 5, "max": 5}),
            "hello",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_fractional_bounds_accepted(self):
        # "a bb": 3 characters in 2 words, average 1.5 — an average
        # word length is a mean of character counts, so bounds need
        # not be integers.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 1.5, "max": 1.5}),
            "a bb",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_punctuation_counts_toward_word_length(self):
        # Punctuation stays attached to its word: "hello," is 6
        # characters, so "hello, world!" averages 6.0, not 5.0 —
        # the same naive tokenization unique_words has.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 6, "max": 6}),
            "hello, world!",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 5, "max": 5}),
            "hello, world!",
        )
        self.assertFalse(ok)

    def test_min_only_passes(self):
        # Average 5.0, above a 4 minimum.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 4}),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # Average 5.0, below a 6 maximum.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "max": 6}),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # Average 2.0 of a 4 minimum: score 0.5.
        ok, detail, score = score_case(
            _case({"type": "avg_word_length", "min": 4}),
            "aa bb",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 4", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_above_maximum_partial_credit(self):
        # Average 6.0 of a 5 maximum: (2*5 - 6) / 5 = 0.8.
        ok, detail, score = score_case(
            _case({"type": "avg_word_length", "max": 5}),
            "aaaaaa bbbbbb",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 5", detail)
        self.assertAlmostEqual(score, 0.8)

    def test_twice_maximum_scores_zero(self):
        # Average 10.0, twice a 5 maximum, so nothing.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "max": 5}),
            "aaaaaaaaaa bbbbbbbbbb",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_word(self):
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "max": 0}),
            "One",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "avg_word_length", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no words in output", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: an average of 0 is in range when the minimum
        # is 0, and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "avg_word_length", "max": 5}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_agrees_with_word_count(self):
        # The same output's word count and average word length are
        # measured by two scorers: "hello world" is 2 words
        # averaging 5.0 characters, so both bands pin exactly, and
        # the average times the count is the total character count.
        output = "hello world"
        ok, _, _ = score_case(
            _case({"type": "word_count", "min": 2, "max": 2}),
            output,
        )
        self.assertTrue(ok)
        ok, detail, _ = score_case(
            _case({"type": "avg_word_length", "min": 5, "max": 5}),
            output,
        )
        self.assertTrue(ok)
        self.assertIn("10 characters in 2 word(s)", detail)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "avg_word_length"}), "hello world")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_word_length", "min": 6, "max": 5}),
                "hello world",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_word_length", "min": -1}),
                "hello",
            )

    def test_string_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_word_length", "min": "5"}),
                "hello",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_word_length", "min": True}),
                "hello",
            )

    def test_non_finite_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_word_length", "min": float("nan")}),
                "hello",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "avg_word_length", "max": float("inf")}),
                "hello",
            )


class TestWordLengthVarianceScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "a bbb": word lengths 1 and 3, mean 2.0, variance 1.0.
        ok, detail, score = score_case(
            _case({"type": "word_length_variance", "min": 1, "max": 1}),
            "a bbb",
        )
        self.assertTrue(ok)
        self.assertIn(
            "word length variance 1.00 (word lengths 1, 3)",
            detail,
        )
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # A single word always has a variance of 0.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "min": 0, "max": 0}),
            "hello",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_fractional_bounds_accepted(self):
        # "a bb": word lengths 1 and 2, mean 1.5, variance 0.25 —
        # a variance is a squared character count, so bounds need
        # not be integers.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "min": 0.25, "max": 0.25}),
            "a bb",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_punctuation_counts_toward_word_length(self):
        # Punctuation stays attached to its word: "hello," is 6
        # characters, so "hello, world" has lengths 6 and 5 and a
        # variance of 0.25, while "hello world" has lengths 5 and 5
        # and a variance of 0 — the same naive tokenization
        # unique_words has.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "min": 0.25, "max": 0.25}),
            "hello, world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "min": 0.25, "max": 0.25}),
            "hello world",
        )
        self.assertFalse(ok)

    def test_min_only_passes(self):
        # Variance 1.0, above a 0.5 minimum.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "min": 0.5}),
            "a bbb",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # Variance 1.0, below a 2 maximum.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "max": 2}),
            "a bbb",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # Variance 1.0 of a 4 minimum: score 0.25.
        ok, detail, score = score_case(
            _case({"type": "word_length_variance", "min": 4}),
            "a bbb",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 4", detail)
        self.assertAlmostEqual(score, 0.25)

    def test_above_maximum_partial_credit(self):
        # "a bbbbbbb": lengths 1 and 7, mean 4.0, variance 9.0;
        # (2*5 - 9) / 5 = 0.2 of a 5 maximum.
        ok, detail, score = score_case(
            _case({"type": "word_length_variance", "max": 5}),
            "a bbbbbbb",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 5", detail)
        self.assertAlmostEqual(score, 0.2)

    def test_twice_maximum_scores_zero(self):
        # "a bbbbbbbbbbb": lengths 1 and 11, mean 6.0, variance 25.0,
        # well past twice a 5 maximum, so nothing.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "max": 5}),
            "a bbbbbbbbbbb",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_variation(self):
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "max": 0}),
            "a bb",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "word_length_variance", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no words in output", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: a variance of 0 is in range when the minimum
        # is 0, and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "word_length_variance", "max": 5}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_agrees_with_avg_word_length_on_mean(self):
        # The same output's mean and variance are measured by two
        # scorers: "a bbbbbb" has lengths 1 and 6, averaging 3.5
        # with a variance of 6.25, so an average band of [3, 4]
        # passes while a variance band capped at 1 fails — the two
        # scorers can disagree, and both consistency checks are
        # pinned here.
        output = "a bbbbbb"
        ok, detail, _ = score_case(
            _case({"type": "avg_word_length", "min": 3, "max": 4}),
            output,
        )
        self.assertTrue(ok)
        self.assertIn("average word length 3.50 characters", detail)
        ok, detail, _ = score_case(
            _case({"type": "word_length_variance", "min": 6.25, "max": 6.25}),
            output,
        )
        self.assertTrue(ok)
        self.assertIn("word length variance 6.25", detail)
        ok, _, _ = score_case(
            _case({"type": "word_length_variance", "max": 1}),
            output,
        )
        self.assertFalse(ok)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "word_length_variance"}), "hello world")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_length_variance", "min": 6, "max": 5}),
                "hello world",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_length_variance", "min": -1}),
                "hello",
            )

    def test_string_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_length_variance", "min": "5"}),
                "hello",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_length_variance", "min": True}),
                "hello",
            )

    def test_non_finite_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_length_variance", "min": float("nan")}),
                "hello",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "word_length_variance", "max": float("inf")}),
                "hello",
            )


class TestReadabilityScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "The cat sat. The dog ran." — 6 words, 2 sentences, 6
        # syllables: 206.835 - 1.015*3 - 84.6*1 = 119.19.
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 60, "max": 120}),
            "The cat sat. The dog ran.",
        )
        self.assertTrue(ok)
        self.assertIn("readability 119.19", detail)
        self.assertIn("6 word(s), 2 sentence(s), 6 syllable(s)", detail)
        self.assertEqual(score, 1.0)

    def test_multi_syllable_words(self):
        # "Reading is beautiful and easy." — 5 words, 1 sentence, 9
        # syllables (reading 2, is 1, beautiful 3, and 1, easy 2):
        # 206.835 - 1.015*5 - 84.6*1.8 = 49.48.
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 40, "max": 60}),
            "Reading is beautiful and easy.",
        )
        self.assertTrue(ok)
        self.assertIn("readability 49.48", detail)
        self.assertEqual(score, 1.0)

    def test_silent_e_and_le(self):
        # "The table is stable." — 4 words, 1 sentence, 6 syllables:
        # table and stable are 2 each (consonant + "le"), the and is
        # 1 each: 206.835 - 1.015*4 - 84.6*1.5 = 75.88.
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 70, "max": 80}),
            "The table is stable.",
        )
        self.assertTrue(ok)
        self.assertIn("readability 75.88", detail)
        self.assertEqual(score, 1.0)

    def test_silent_e_words_count_one_syllable(self):
        # "The large white horse ate cake." — 6 words, 1 sentence, 6
        # syllables: every word's trailing e is silent, so each word
        # counts once: 206.835 - 1.015*6 - 84.6*1 = 116.15.
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 110, "max": 120}),
            "The large white horse ate cake.",
        )
        self.assertTrue(ok)
        self.assertIn("readability 116.15", detail)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 119.19 of a 150 minimum: 119.19 / 150 = 0.7946.
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 150}),
            "The cat sat. The dog ran.",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 150", detail)
        self.assertAlmostEqual(score, 119.19 / 150)

    def test_above_maximum_partial_credit(self):
        # 119.19 of a 100 maximum: (2*100 - 119.19) / 100 = 0.8081.
        ok, detail, score = score_case(
            _case({"type": "readability", "max": 100}),
            "The cat sat. The dog ran.",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 100", detail)
        self.assertAlmostEqual(score, 0.8081)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "readability", "max": 50}),
            "The cat sat. The dog ran.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_text(self):
        ok, _, score = score_case(
            _case({"type": "readability", "max": 0}),
            "The cat sat.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        ok, _, score = score_case(
            _case({"type": "readability", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 60}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("readability 0.00", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: a readability of 0 is in range when the minimum
        # is 0, and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "readability", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "readability", "max": 60}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_negative_minimum_gives_no_partial_credit_below_it(self):
        # A very hard passage scores below a negative minimum, and
        # there is no linear scale down to a non-positive floor.
        text = " ".join(["antidisestablishmentarianism"] * 20) + "."
        ok, _, score = score_case(
            _case({"type": "readability", "min": -10}),
            text,
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_abbreviation_is_naive(self):
        # "Mr. Smith went home." is two sentences ("Mr", "Smith went
        # home"): 4 words in 2 sentences, 4 syllables — the same
        # naive split sentence_count uses, abbreviations and all:
        # 206.835 - 1.015*2 - 84.6*1 = 120.21.
        ok, detail, score = score_case(
            _case({"type": "readability", "min": 120, "max": 121}),
            "Mr. Smith went home.",
        )
        self.assertTrue(ok)
        self.assertIn("readability 120.21", detail)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "readability"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "readability", "min": 90, "max": 60}),
                "One. Two.",
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "readability", "min": "high"}),
                "One.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "readability", "min": True}),
                "One.",
            )

    def test_non_finite_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "readability", "min": float("nan")}),
                "One.",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "readability", "max": float("inf")}),
                "One.",
            )


class TestSyllableCountScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "The cat sat. The dog ran." — 6 words, 6 syllables: every
        # word is one syllable.
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 5, "max": 10}),
            "The cat sat. The dog ran.",
        )
        self.assertTrue(ok)
        self.assertIn("syllable count 6", detail)
        self.assertEqual(score, 1.0)

    def test_multi_syllable_words(self):
        # "Reading is beautiful and easy." — 9 syllables: reading 2,
        # is 1, beautiful 3, and 1, easy 2.
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 9, "max": 9}),
            "Reading is beautiful and easy.",
        )
        self.assertTrue(ok)
        self.assertIn("syllable count 9", detail)
        self.assertEqual(score, 1.0)

    def test_silent_e_and_le(self):
        # "The table is stable." — 6 syllables: table and stable are
        # 2 each (consonant + "le"), the and is 1 each.
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 6, "max": 6}),
            "The table is stable.",
        )
        self.assertTrue(ok)
        self.assertIn("syllable count 6", detail)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # "hello world" is 3 syllables of a 6 minimum: 3 / 6 = 0.5.
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 6}),
            "hello world",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 6", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_above_maximum_partial_credit(self):
        # 9 syllables of a 6 maximum: (2*6 - 9) / 6 = 0.5.
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "max": 6}),
            "Reading is beautiful and easy.",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 6", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "syllable_count", "max": 4}),
            "Reading is beautiful and easy.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_syllable(self):
        ok, _, score = score_case(
            _case({"type": "syllable_count", "max": 0}),
            "hello",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        ok, _, score = score_case(
            _case({"type": "syllable_count", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("syllable count 0", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No syllables: a count of 0 is in range when the minimum
        # is 0, and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "syllable_count", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "syllable_count", "max": 5}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_number_tokens_have_no_syllables(self):
        # "answer 42" is 2 syllables: "answer" is 2, "42" has no
        # letters and none.
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 2, "max": 2}),
            "answer 42",
        )
        self.assertTrue(ok)
        self.assertIn("syllable count 2", detail)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "syllable_count", "min": 1}),
            "42",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_agrees_with_readability_on_the_syllable_total(self):
        # The same text scored by readability carries the same
        # syllable total in its detail: the two scorers share the
        # heuristic, so they cannot drift apart.
        text = "The cat sat. The dog ran."
        _, readable_detail, _ = score_case(
            _case({"type": "readability", "min": 60, "max": 120}),
            text,
        )
        self.assertIn("6 syllable(s)", readable_detail)
        ok, detail, score = score_case(
            _case({"type": "syllable_count", "min": 6, "max": 6}),
            text,
        )
        self.assertTrue(ok)
        self.assertIn("syllable count 6", detail)
        self.assertEqual(score, 1.0)

    def test_can_disagree_with_word_count(self):
        # "hello world" is 2 words but 3 syllables: a case that
        # pins the syllable count catches long words where a
        # word-count case would pass it.
        words = score_case(
            _case({"type": "word_count", "min": 3}),
            "hello world",
        )
        syllables = score_case(
            _case({"type": "syllable_count", "min": 3, "max": 3}),
            "hello world",
        )
        self.assertFalse(words[0])
        self.assertTrue(syllables[0])

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "syllable_count"}), "One two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "syllable_count", "min": 10, "max": 5}),
                "One two.",
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "syllable_count", "min": 2.5}),
                "One two.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "syllable_count", "min": True}),
                "One.",
            )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "syllable_count", "min": -1}),
                "One.",
            )


class TestTypeTokenRatioScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "one two three one" — 3 distinct of 4 words: 0.75.
        ok, detail, score = score_case(
            _case({"type": "type_token_ratio", "min": 0.5, "max": 0.9}),
            "one two three one",
        )
        self.assertTrue(ok)
        self.assertIn("type-token ratio 0.750 (3 of 4 words)", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # Every word different: a ratio of exactly 1.0.
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "min": 1.0, "max": 1.0}),
            "one two three",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "min": 0.5}),
            "one two one",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # "same same same" — 1 distinct of 3 words: 0.333.
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "max": 0.5}),
            "same same same",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 0.5 of a 0.8 minimum: 0.5 / 0.8 = 0.625.
        ok, detail, score = score_case(
            _case({"type": "type_token_ratio", "min": 0.8}),
            "one two one one",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 0.8", detail)
        self.assertAlmostEqual(score, 0.625)

    def test_repeated_words_score_low(self):
        # 1 of 3 distinct: 0.333 of a 0.5 minimum is 2/3.
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "min": 0.5}),
            "same same same",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, (1 / 3) / 0.5)

    def test_above_maximum_partial_credit(self):
        # 1.0 of a 0.8 maximum: (2*0.8 - 1.0) / 0.8 = 0.75.
        ok, detail, score = score_case(
            _case({"type": "type_token_ratio", "max": 0.8}),
            "one two three",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 0.8", detail)
        self.assertAlmostEqual(score, 0.75)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "max": 0.4}),
            "one two three",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_word(self):
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "max": 0}),
            "one word here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "type_token_ratio", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no words in output", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: a ratio of 0 is in range when the minimum is 0,
        # and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "type_token_ratio", "max": 0.9}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case_folds_before_deduping(self):
        # Without ignore_case, "The" and "the" are two words: 2/3.
        ok, _, score = score_case(
            _case(
                {"type": "type_token_ratio", "min": 0.6, "max": 0.7}
            ),
            "The the the",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # With ignore_case they fold to one word: 1/3, below a 0.6
        # minimum, so the score is (1/3) / 0.6.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "type_token_ratio",
                    "min": 0.6,
                    "max": 0.7,
                    "ignore_case": True,
                }
            ),
            "The the the",
        )
        self.assertFalse(ok)
        self.assertIn("type-token ratio 0.333 (1 of 3 words)", detail)
        self.assertAlmostEqual(score, (1 / 3) / 0.6)

    def test_tokenization_is_naive_about_punctuation(self):
        # "hello," and "hello" are different tokens, so "hello,
        # hello" is two distinct words out of two — a ratio of 1.0
        # even though both tokens are the same word once punctuation
        # is stripped. Same naive tokenization as word_count and
        # unique_words.
        ok, detail, score = score_case(
            _case({"type": "type_token_ratio", "min": 1.0, "max": 1.0}),
            "hello, hello",
        )
        self.assertTrue(ok)
        self.assertIn("type-token ratio 1.000 (2 of 2 words)", detail)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "type_token_ratio"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "type_token_ratio", "min": 0.9, "max": 0.5}),
                "One. Two.",
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "type_token_ratio", "min": "high"}),
                "One.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "type_token_ratio", "min": True}),
                "One.",
            )

    def test_bound_outside_unit_interval_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "type_token_ratio", "min": 1.5}),
                "One.",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "type_token_ratio", "max": -0.1}),
                "One.",
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "type_token_ratio",
                        "min": 0.5,
                        "ignore_case": "true",
                    }
                ),
                "One.",
            )


class TestHapaxRatioScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "one two three one" — only "two" and "three" are used
        # once: a hapax ratio of 2/4 = 0.5. The type-token ratio
        # of the same output is 3/4, so the two measures differ.
        ok, detail, score = score_case(
            _case({"type": "hapax_ratio", "min": 0.4, "max": 0.6}),
            "one two three one",
        )
        self.assertTrue(ok)
        self.assertIn("hapax ratio 0.500 (2 of 4 words used once)", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # Every word used once: a ratio of exactly 1.0.
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "min": 1.0, "max": 1.0}),
            "one two three",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "min": 0.5}),
            "one two three",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # "same same same" — no word used once: 0.0.
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "max": 0.5}),
            "same same same",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 0.25 of a 0.5 minimum: 0.25 / 0.5 = 0.5.
        ok, detail, score = score_case(
            _case({"type": "hapax_ratio", "min": 0.5}),
            "one two one one",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 0.5", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_repeated_words_score_low(self):
        # 0 of 3 words used once: a ratio of 0, so no partial
        # credit below a 0.5 minimum.
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "min": 0.5}),
            "same same same",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 1.0 of a 0.8 maximum: (2*0.8 - 1.0) / 0.8 = 0.75.
        ok, detail, score = score_case(
            _case({"type": "hapax_ratio", "max": 0.8}),
            "one two three",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 0.8", detail)
        self.assertAlmostEqual(score, 0.75)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "max": 0.4}),
            "one two three",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_word(self):
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "max": 0}),
            "one word here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "hapax_ratio", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no words in output", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: a ratio of 0 is in range when the minimum is 0,
        # and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "hapax_ratio", "max": 0.9}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case_folds_before_counting(self):
        # Without ignore_case, "The", "the", and "cat" are three
        # different words used once each: a ratio of 1.0.
        ok, _, score = score_case(
            _case(
                {"type": "hapax_ratio", "min": 0.9, "max": 1.0}
            ),
            "The the cat",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # With ignore_case "The" and "the" fold to one word used
        # twice: only "cat" is used once, a ratio of 1/3, below a
        # 0.9 minimum, so the score is (1/3) / 0.9.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "hapax_ratio",
                    "min": 0.9,
                    "max": 1.0,
                    "ignore_case": True,
                }
            ),
            "The the cat",
        )
        self.assertFalse(ok)
        self.assertIn("hapax ratio 0.333 (1 of 3 words used once)", detail)
        self.assertAlmostEqual(score, (1 / 3) / 0.9)

    def test_tokenization_is_naive_about_punctuation(self):
        # "hello," and "hello" are different tokens, each used
        # once, so the ratio is 1.0 — same naive tokenization as
        # word_count and unique_words.
        ok, detail, score = score_case(
            _case({"type": "hapax_ratio", "min": 1.0, "max": 1.0}),
            "hello, hello",
        )
        self.assertTrue(ok)
        self.assertIn("hapax ratio 1.000 (2 of 2 words used once)", detail)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "hapax_ratio"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "hapax_ratio", "min": 0.9, "max": 0.5}),
                "One. Two.",
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "hapax_ratio", "min": "high"}),
                "One.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "hapax_ratio", "min": True}),
                "One.",
            )

    def test_bound_outside_unit_interval_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "hapax_ratio", "min": 1.5}),
                "One.",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "hapax_ratio", "max": -0.1}),
                "One.",
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "hapax_ratio",
                        "min": 0.5,
                        "ignore_case": "true",
                    }
                ),
                "One.",
            )


class TestStopwordRatioScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "the cat sat on the mat" — "the" twice, "on" once:
        # 3 of 6 words are stopwords, a ratio of 0.5.
        ok, detail, score = score_case(
            _case({"type": "stopword_ratio", "min": 0.4, "max": 0.6}),
            "the cat sat on the mat",
        )
        self.assertTrue(ok)
        self.assertIn("stopword ratio 0.500 (3 of 6 words are stopwords)", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # Every word a stopword: a ratio of exactly 1.0.
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "min": 1.0, "max": 1.0}),
            "the cat and the dog",
        )
        self.assertFalse(ok)  # "cat" and "dog" are content words
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "min": 1.0, "max": 1.0}),
            "the and of the",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "min": 0.5}),
            "the and of the",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # "cats dogs birds" — no stopwords at all: 0.0.
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "max": 0.5}),
            "cats dogs birds",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 0.25 of a 0.5 minimum: 0.25 / 0.5 = 0.5.
        ok, detail, score = score_case(
            _case({"type": "stopword_ratio", "min": 0.5}),
            "the cats sat quietly",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 0.5", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_no_stopwords_score_zero_below_minimum(self):
        # 0 of 3 words are stopwords: a ratio of 0, so no partial
        # credit below a 0.5 minimum.
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "min": 0.5}),
            "cats dogs birds",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 1.0 of a 0.8 maximum: (2*0.8 - 1.0) / 0.8 = 0.75.
        ok, detail, score = score_case(
            _case({"type": "stopword_ratio", "max": 0.8}),
            "the and of the",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 0.8", detail)
        self.assertAlmostEqual(score, 0.75)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "max": 0.4}),
            "the and of the",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_stopword(self):
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "max": 0}),
            "the cat",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "stopword_ratio", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no words in output", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: a ratio of 0 is in range when the minimum is 0,
        # and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "stopword_ratio", "max": 0.9}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case_folds_before_counting(self):
        # Without ignore_case, "The" is not in the lowercase list,
        # so only "and" counts: a ratio of 1/3.
        ok, _, score = score_case(
            _case(
                {"type": "stopword_ratio", "min": 0.9, "max": 1.0}
            ),
            "The and cat",
        )
        self.assertFalse(ok)
        # With ignore_case "The" folds to "the" and counts: 2/3,
        # still below a 0.9 minimum, so the score is (2/3) / 0.9.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "stopword_ratio",
                    "min": 0.9,
                    "max": 1.0,
                    "ignore_case": True,
                }
            ),
            "The and cat",
        )
        self.assertFalse(ok)
        self.assertIn("stopword ratio 0.667 (2 of 3 words are stopwords)", detail)
        self.assertAlmostEqual(score, (2 / 3) / 0.9)

    def test_tokenization_is_naive_about_punctuation(self):
        # "the," is not the stopword "the": a ratio of 0.0.
        ok, detail, score = score_case(
            _case({"type": "stopword_ratio", "min": 0.0, "max": 0.0}),
            "the, cat",
        )
        self.assertTrue(ok)
        self.assertIn("stopword ratio 0.000 (0 of 2 words are stopwords)", detail)
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "stopword_ratio"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "stopword_ratio", "min": 0.9, "max": 0.5}),
                "One. Two.",
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "stopword_ratio", "min": "high"}),
                "One.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "stopword_ratio", "min": True}),
                "One.",
            )

    def test_bound_outside_unit_interval_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "stopword_ratio", "min": 1.5}),
                "One.",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "stopword_ratio", "max": -0.1}),
                "One.",
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "stopword_ratio",
                        "min": 0.5,
                        "ignore_case": "true",
                    }
                ),
                "One.",
            )


class TestContentWordRatioScorer(unittest.TestCase):
    def test_within_bounds_passes(self):
        # "the cat sat on the mat" — "cat", "sat", "mat" are content
        # words: 3 of 6, a ratio of 0.5.
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.4, "max": 0.6}),
            "the cat sat on the mat",
        )
        self.assertTrue(ok)
        self.assertIn(
            "content word ratio 0.500 (3 of 6 words are content words)",
            detail,
        )
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        # Every word a content word: a ratio of exactly 1.0.
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "min": 1.0, "max": 1.0}),
            "cats dogs birds",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "min": 1.0, "max": 1.0}),
            "the and of the",
        )
        self.assertFalse(ok)  # no content words at all

    def test_min_only_passes(self):
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.5}),
            "cats dogs birds",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_max_only_passes(self):
        # "the and of the" — no content words at all: 0.0.
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "max": 0.5}),
            "the and of the",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # "the cats sat quietly" — 3 of 4 words are content words:
        # 0.75 of a 0.9 minimum is 0.75 / 0.9.
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.9}),
            "the cats sat quietly",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 0.9", detail)
        self.assertAlmostEqual(score, 0.75 / 0.9)

    def test_no_content_words_score_zero_below_minimum(self):
        # 0 of 4 words are content words: a ratio of 0, so no partial
        # credit below a 0.5 minimum.
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.5}),
            "the and of the",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 1.0 of a 0.8 maximum: (2*0.8 - 1.0) / 0.8 = 0.75.
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "max": 0.8}),
            "cats dogs birds",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 0.8", detail)
        self.assertAlmostEqual(score, 0.75)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "max": 0.5}),
            "cats dogs birds",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_max_zero_forbids_any_content_word(self):
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "max": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "max": 0}),
            "the cat",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.5}),
            "   ",
        )
        self.assertFalse(ok)
        self.assertIn("no words in output", detail)
        self.assertEqual(score, 0.0)

    def test_empty_output_passes_when_min_zero(self):
        # No words: a ratio of 0 is in range when the minimum is 0,
        # and when there is no minimum at all.
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "min": 0}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "content_word_ratio", "max": 0.9}),
            "   ",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_complements_stopword_ratio(self):
        # The two ratios sum to 1.0 on the same output: 1/3 stopwords
        # means 2/3 content words.
        _, _, stop_score = score_case(
            _case({"type": "stopword_ratio", "min": 0.0, "max": 1.0}),
            "the cats sat",
        )
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.0, "max": 1.0}),
            "the cats sat",
        )
        self.assertTrue(ok)
        self.assertIn(
            "content word ratio 0.667 (2 of 3 words are content words)",
            detail,
        )
        self.assertAlmostEqual(score, 1.0)
        self.assertEqual(stop_score, 1.0)

    def test_ignore_case_folds_before_counting(self):
        # Without ignore_case, "The" is not in the lowercase list,
        # so it counts as a content word: 2 of 3.
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "min": 0.6, "max": 0.7}),
            "The and cat",
        )
        self.assertTrue(ok)
        self.assertIn(
            "content word ratio 0.667 (2 of 3 words are content words)",
            detail,
        )
        # With ignore_case "The" folds to "the" and is a stopword:
        # only "cat" is a content word, 1/3, below a 0.6 minimum,
        # so the score is (1/3) / 0.6.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "content_word_ratio",
                    "min": 0.6,
                    "max": 0.7,
                    "ignore_case": True,
                }
            ),
            "The and cat",
        )
        self.assertFalse(ok)
        self.assertIn(
            "content word ratio 0.333 (1 of 3 words are content words)",
            detail,
        )
        self.assertAlmostEqual(score, (1 / 3) / 0.6)

    def test_tokenization_is_naive_about_punctuation(self):
        # "the," is not the stopword "the", so it counts as a
        # content word: a ratio of 1.0.
        ok, detail, score = score_case(
            _case({"type": "content_word_ratio", "min": 1.0, "max": 1.0}),
            "the, cat",
        )
        self.assertTrue(ok)
        self.assertIn(
            "content word ratio 1.000 (2 of 2 words are content words)",
            detail,
        )
        self.assertEqual(score, 1.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "content_word_ratio"}), "One. Two.")

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "content_word_ratio", "min": 0.9, "max": 0.5}),
                "One. Two.",
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "content_word_ratio", "min": "high"}),
                "One.",
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "content_word_ratio", "min": True}),
                "One.",
            )

    def test_bound_outside_unit_interval_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "content_word_ratio", "min": 1.5}),
                "One.",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "content_word_ratio", "max": -0.1}),
                "One.",
            )

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "content_word_ratio",
                        "min": 0.5,
                        "ignore_case": "true",
                    }
                ),
                "One.",
            )


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

    def test_ignore_case_matches(self):
        ok, _, _ = score_case(
            _case({"type": "regex", "pattern": r"^hello$", "ignore_case": True}),
            "HELLO",
        )
        self.assertTrue(ok)

    def test_ignore_case_false_is_case_sensitive(self):
        ok, _, _ = score_case(
            _case({"type": "regex", "pattern": r"^hello$", "ignore_case": False}),
            "HELLO",
        )
        self.assertFalse(ok)

    def test_ignore_case_defaults_off(self):
        ok, _, _ = score_case(
            _case({"type": "regex", "pattern": r"^hello$"}), "HELLO"
        )
        self.assertFalse(ok)

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "regex",
                        "pattern": r"^hello$",
                        "ignore_case": "true",
                    }
                ),
                "HELLO",
            )


class TestContainsRegexScorer(unittest.TestCase):
    def test_match(self):
        ok, detail, score = score_case(
            _case({"type": "contains_regex", "pattern": r"\d+"}), "order 42"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        self.assertEqual(detail, "contains_regex matched")

    def test_match_anywhere(self):
        ok, _, _ = score_case(
            _case({"type": "contains_regex", "pattern": r"error \d+"}),
            "log: error 500 on retry",
        )
        self.assertTrue(ok)

    def test_no_match(self):
        ok, detail, score = score_case(
            _case({"type": "contains_regex", "pattern": r"^\d+$"}), "4x2"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        self.assertIn("did not match", detail)

    def test_missing_pattern_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "contains_regex"}), "x")

    def test_non_string_pattern_raises(self):
        for bad in (42, 3.5, ["^\\d+$"], None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    score_case(
                        _case({"type": "contains_regex", "pattern": bad}), "42"
                    )

    def test_invalid_pattern_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "contains_regex", "pattern": "("}), "x")

    def test_ignore_case_matches(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "contains_regex",
                    "pattern": r"hello",
                    "ignore_case": True,
                }
            ),
            "say HELLO there",
        )
        self.assertTrue(ok)

    def test_ignore_case_false_is_case_sensitive(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "contains_regex",
                    "pattern": r"hello",
                    "ignore_case": False,
                }
            ),
            "say HELLO there",
        )
        self.assertFalse(ok)

    def test_ignore_case_defaults_off(self):
        ok, _, _ = score_case(
            _case({"type": "contains_regex", "pattern": r"hello"}),
            "say HELLO there",
        )
        self.assertFalse(ok)

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "contains_regex",
                        "pattern": r"hello",
                        "ignore_case": "true",
                    }
                ),
                "HELLO",
            )

    def test_flows_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "regex-suite",
                "cases": [
                    {
                        "id": "has-code",
                        "prompt": "give me the code",
                        "scorer": {
                            "type": "contains_regex",
                            "pattern": r"CODE-\d{4}",
                        },
                    },
                    {
                        "id": "no-code",
                        "prompt": "give me the code",
                        "scorer": {
                            "type": "contains_regex",
                            "pattern": r"CODE-\d{4}",
                        },
                    },
                ],
            }
        )
        # run_suite calls the agent once per case in order; use a
        # stateful agent so each case gets its own output.
        calls = iter(["your code is CODE-1234", "no code here"])
        report = run_suite(suite, lambda prompt: next(calls))
        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.results[0].score, 1.0)
        self.assertFalse(report.results[1].passed)
        self.assertEqual(report.results[1].score, 0.0)


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


class TestJsonPathNumericScorer(unittest.TestCase):
    def test_in_range_passes(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.result.score",
                    "min": 80,
                    "max": 100,
                }
            ),
            '{"result": {"score": 87}}',
        )
        self.assertTrue(ok)
        self.assertIn("$.result.score", detail)
        self.assertIn("87", detail)
        self.assertEqual(score, 1.0)

    def test_bare_path_and_array_index(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "items[0].price",
                    "min": 0,
                    "max": 10,
                }
            ),
            '{"items": [{"price": 9.5}]}',
        )
        self.assertTrue(ok)

    def test_float_value(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.pi",
                    "min": 3,
                    "max": 4,
                }
            ),
            '{"pi": 3.14159}',
        )
        self.assertTrue(ok)

    def test_below_min_partial_credit(self):
        # One scale away is the full range width (20), so 70 is
        # halfway below the floor of 80: score 0.5.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.score",
                    "min": 80,
                    "max": 100,
                }
            ),
            '{"score": 70}',
        )
        self.assertFalse(ok)
        self.assertIn("below minimum", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_above_max_partial_credit(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.score",
                    "min": 0,
                    "max": 100,
                }
            ),
            '{"score": 110}',
        )
        self.assertFalse(ok)
        self.assertIn("above maximum", detail)
        self.assertAlmostEqual(score, 0.9)

    def test_far_outside_scores_zero(self):
        _, _, score = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.score",
                    "min": 0,
                    "max": 100,
                }
            ),
            '{"score": 300}',
        )
        self.assertEqual(score, 0.0)

    def test_single_bound_floor(self):
        ok, _, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "min": 10}),
            '{"n": 12}',
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # Below a lone floor the scale is the floor's magnitude.
        _, _, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "min": 10}),
            '{"n": 5}',
        )
        self.assertAlmostEqual(score, 0.5)

    def test_single_bound_ceiling(self):
        ok, _, _ = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "max": 10}),
            '{"n": 5}',
        )
        self.assertTrue(ok)
        _, _, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "max": 10}),
            '{"n": 15}',
        )
        self.assertAlmostEqual(score, 0.5)

    def test_max_zero_forbids(self):
        ok, _, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "max": 0}),
            '{"n": 5}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_min_equals_max_gives_no_partial_credit(self):
        _, _, score = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.n",
                    "min": 42,
                    "max": 42,
                }
            ),
            '{"n": 43}',
        )
        self.assertEqual(score, 0.0)

    def test_tolerance_widens_band(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.n",
                    "min": 0,
                    "max": 100,
                    "tolerance": 2,
                }
            ),
            '{"n": 101}',
        )
        self.assertTrue(ok)
        ok, _, score = score_case(
            _case(
                {
                    "type": "json_path_numeric",
                    "path": "$.n",
                    "min": 0,
                    "max": 100,
                    "tolerance": 2,
                }
            ),
            '{"n": 103}',
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 1 - 1 / 104)

    def test_invalid_json_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "min": 0}),
            "{not json",
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)
        self.assertEqual(score, 0.0)

    def test_missing_key_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.missing", "min": 0}),
            '{"total": 1}',
        )
        self.assertFalse(ok)
        self.assertIn("missing", detail)
        self.assertEqual(score, 0.0)

    def test_non_numeric_value_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "min": 0}),
            '{"n": "42"}',
        )
        self.assertFalse(ok)
        self.assertIn("number", detail)
        self.assertEqual(score, 0.0)

    def test_bool_is_not_a_number(self):
        ok, _, score = score_case(
            _case({"type": "json_path_numeric", "path": "$.n", "min": 0}),
            '{"n": true}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_malformed_path_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_path_numeric", "path": "$.a..b", "min": 0}
                ),
                '{"a": {}}',
            )

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_path_numeric", "path": "$.n"}),
                '{"n": 1}',
            )

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "json_path_numeric",
                        "path": "$.n",
                        "min": 100,
                        "max": 0,
                    }
                ),
                '{"n": 50}',
            )

    def test_non_numeric_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "json_path_numeric",
                        "path": "$.n",
                        "min": "low",
                    }
                ),
                '{"n": 1}',
            )

    def test_bool_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_path_numeric", "path": "$.n", "min": True}
                ),
                '{"n": 1}',
            )

    def test_bad_tolerance_raises(self):
        for bad in (-1, "0.1", float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                score_case(
                    _case(
                        {
                            "type": "json_path_numeric",
                            "path": "$.n",
                            "min": 0,
                            "max": 100,
                            "tolerance": bad,
                        }
                    ),
                    '{"n": 50}',
                )

    def test_agrees_with_json_path_at_a_point(self):
        # A band collapsed to one value with a tolerance is the
        # same check as json_path with that expected value and
        # tolerance: the two scorers must agree on every value.
        for value in (41, 42, 43):
            numeric = score_case(
                _case(
                    {
                        "type": "json_path_numeric",
                        "path": "$.total",
                        "min": 42,
                        "max": 42,
                        "tolerance": 0.01,
                    }
                ),
                f'{{"total": {value}}}',
            )
            exact = score_case(
                _case(
                    {
                        "type": "json_path",
                        "path": "$.total",
                        "expected": 42,
                        "tolerance": 0.01,
                    }
                ),
                f'{{"total": {value}}}',
            )
            self.assertEqual(numeric[0], exact[0], f"value {value}")

    def test_agrees_with_range_on_the_same_value(self):
        # The same number scored the same way: json_path_numeric
        # on the JSON field and range on the bare number must give
        # the same verdict and partial credit.
        for output, expected_score in (('{"total": 42}', 1.0), ('{"total": 150}', 0.5)):
            numeric = score_case(
                _case(
                    {
                        "type": "json_path_numeric",
                        "path": "$.total",
                        "min": 0,
                        "max": 100,
                    }
                ),
                output,
            )
            ranged = score_case(
                _case({"type": "range", "min": 0, "max": 100}),
                output,
            )
            self.assertEqual(numeric[0], ranged[0])
            self.assertAlmostEqual(numeric[2], expected_score)
            self.assertAlmostEqual(numeric[2], ranged[2])


class TestJsonPathCountScorer(unittest.TestCase):
    def test_array_in_range_passes(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 2,
                    "max": 5,
                }
            ),
            '{"items": [1, 2, 3]}',
        )
        self.assertTrue(ok)
        self.assertIn("$.items", detail)
        self.assertIn("3", detail)
        self.assertEqual(score, 1.0)

    def test_object_key_count_passes(self):
        # A dict counts its keys, the same way a list counts its
        # elements: three keys is a count of 3.
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.config",
                    "min": 3,
                    "max": 3,
                }
            ),
            '{"config": {"a": 1, "b": 2, "c": 3}}',
        )
        self.assertTrue(ok)

    def test_bare_path_and_array_index(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "results[0].tags",
                    "min": 1,
                    "max": 3,
                }
            ),
            '{"results": [{"tags": ["x", "y"]}]}',
        )
        self.assertTrue(ok)

    def test_empty_array_counts_zero(self):
        ok, _, score = score_case(
            _case(
                {"type": "json_path_count", "path": "$.items", "min": 0}
            ),
            '{"items": []}',
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_min_partial_credit(self):
        # Below the minimum the score is n / min, the count
        # convention: 2 items against a floor of 4 scores 0.5.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 4,
                    "max": 6,
                }
            ),
            '{"items": [1, 2]}',
        )
        self.assertFalse(ok)
        self.assertIn("below minimum", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_above_max_partial_credit(self):
        # Above the maximum the score falls linearly to 0 at twice
        # the maximum: 6 items against a ceiling of 5 scores 0.8.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 1,
                    "max": 5,
                }
            ),
            '{"items": [1, 2, 3, 4, 5, 6]}',
        )
        self.assertFalse(ok)
        self.assertIn("above maximum", detail)
        self.assertAlmostEqual(score, 0.8)

    def test_far_above_max_scores_zero(self):
        _, _, score = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 1,
                    "max": 5,
                }
            ),
            '{"items": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}',
        )
        self.assertEqual(score, 0.0)

    def test_single_bound_floor(self):
        ok, _, score = score_case(
            _case({"type": "json_path_count", "path": "$.items", "min": 2}),
            '{"items": [1, 2, 3]}',
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        _, _, score = score_case(
            _case({"type": "json_path_count", "path": "$.items", "min": 4}),
            '{"items": [1, 2]}',
        )
        self.assertAlmostEqual(score, 0.5)

    def test_single_bound_ceiling(self):
        ok, _, _ = score_case(
            _case({"type": "json_path_count", "path": "$.items", "max": 3}),
            '{"items": [1]}',
        )
        self.assertTrue(ok)
        _, _, score = score_case(
            _case({"type": "json_path_count", "path": "$.items", "max": 3}),
            '{"items": [1, 2, 3, 4]}',
        )
        self.assertAlmostEqual(score, 2 / 3)

    def test_max_zero_forbids(self):
        ok, _, score = score_case(
            _case({"type": "json_path_count", "path": "$.items", "max": 0}),
            '{"items": [1]}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_min_equals_max_is_exact_count(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 3,
                    "max": 3,
                }
            ),
            '{"items": [1, 2, 3]}',
        )
        self.assertTrue(ok)
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 3,
                    "max": 3,
                }
            ),
            '{"items": [1, 2]}',
        )
        self.assertFalse(ok)

    def test_invalid_json_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_count", "path": "$.items", "min": 0}),
            "{not json",
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)
        self.assertEqual(score, 0.0)

    def test_missing_key_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_count", "path": "$.missing", "min": 0}),
            '{"items": [1]}',
        )
        self.assertFalse(ok)
        self.assertIn("missing", detail)
        self.assertEqual(score, 0.0)

    def test_scalar_value_fails(self):
        # A string, a number, a bool, and null have no items to
        # count, so they fail the same way a missing key does.
        for output in (
            '{"items": "abc"}',
            '{"items": 42}',
            '{"items": true}',
            '{"items": null}',
        ):
            ok, detail, score = score_case(
                _case(
                    {"type": "json_path_count", "path": "$.items", "min": 0}
                ),
                output,
            )
            self.assertFalse(ok, output)
            self.assertIn("list or object", detail)
            self.assertEqual(score, 0.0)

    def test_malformed_path_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_path_count", "path": "$.a..b", "min": 0}
                ),
                '{"a": {}}',
            )

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_path_count", "path": "$.items"}),
                '{"items": [1]}',
            )

    def test_min_above_max_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "json_path_count",
                        "path": "$.items",
                        "min": 5,
                        "max": 2,
                    }
                ),
                '{"items": [1]}',
            )

    def test_non_integer_bound_raises(self):
        for bad in ("low", 2.5, True):
            with self.assertRaises(ValueError):
                score_case(
                    _case(
                        {
                            "type": "json_path_count",
                            "path": "$.items",
                            "min": bad,
                        }
                    ),
                    '{"items": [1]}',
                )

    def test_negative_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "json_path_count",
                        "path": "$.items",
                        "min": -1,
                    }
                ),
                '{"items": [1]}',
            )

    def test_integral_float_bound_accepted(self):
        # _as_length_bound accepts an integral float, like the
        # other count-style scorers.
        ok, _, _ = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 2.0,
                    "max": 5.0,
                }
            ),
            '{"items": [1, 2, 3]}',
        )
        self.assertTrue(ok)

    def test_agrees_with_count_on_the_same_count(self):
        # The same count scored the same way: json_path_count on
        # the JSON array and count on the bare output must give the
        # same verdict and partial credit — 2 occurrences against a
        # floor of 4 is 0.5 either way.
        counted = score_case(
            _case(
                {
                    "type": "json_path_count",
                    "path": "$.items",
                    "min": 4,
                    "max": 6,
                }
            ),
            '{"items": [1, 2]}',
        )
        plain = score_case(
            _case({"type": "count", "substring": "x", "min": 4, "max": 6}),
            "x x",
        )
        self.assertEqual(counted[0], plain[0])
        self.assertAlmostEqual(counted[2], plain[2])

    def test_list_and_dict_with_same_size_agree(self):
        # A list of 3 and a dict of 3 keys are both a count of 3:
        # the two container shapes must score identically.
        for output in (
            '{"v": [1, 2, 3]}',
            '{"v": {"a": 1, "b": 2, "c": 3}}',
        ):
            ok, _, score = score_case(
                _case(
                    {
                        "type": "json_path_count",
                        "path": "$.v",
                        "min": 3,
                        "max": 3,
                    }
                ),
                output,
            )
            self.assertTrue(ok, output)
            self.assertEqual(score, 1.0)


class TestJsonPathExistsScorer(unittest.TestCase):
    def test_present_key_passes(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_exists", "path": "$.result.id"}),
            '{"result": {"id": 42}}',
        )
        self.assertTrue(ok)
        self.assertIn("$.result.id", detail)
        self.assertIn("exists", detail)
        self.assertEqual(score, 1.0)

    def test_falsy_values_still_exist(self):
        # Presence is not truthiness: 0, false, "", an empty list,
        # and null all count as existing.
        for output in (
            '{"v": 0}',
            '{"v": false}',
            '{"v": ""}',
            '{"v": []}',
            '{"v": null}',
        ):
            ok, _, score = score_case(
                _case({"type": "json_path_exists", "path": "$.v"}),
                output,
            )
            self.assertTrue(ok, output)
            self.assertEqual(score, 1.0, output)

    def test_bare_path_and_array_index(self):
        ok, _, _ = score_case(
            _case({"type": "json_path_exists", "path": "results[0].name"}),
            '{"results": [{"name": "ada"}]}',
        )
        self.assertTrue(ok)

    def test_root_path_exists(self):
        # The empty path is the whole document, which always exists.
        ok, _, _ = score_case(
            _case({"type": "json_path_exists", "path": "$"}),
            '{"a": 1}',
        )
        self.assertTrue(ok)

    def test_missing_key_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_exists", "path": "$.missing"}),
            '{"items": [1]}',
        )
        self.assertFalse(ok)
        self.assertIn("missing", detail)
        self.assertEqual(score, 0.0)

    def test_missing_index_fails(self):
        ok, _, score = score_case(
            _case({"type": "json_path_exists", "path": "$.items[3]"}),
            '{"items": [1, 2]}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_invalid_json_fails(self):
        ok, detail, score = score_case(
            _case({"type": "json_path_exists", "path": "$.v"}),
            "{not json",
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)
        self.assertEqual(score, 0.0)

    def test_non_null_rejects_null(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "json_path_exists",
                    "path": "$.v",
                    "non_null": True,
                }
            ),
            '{"v": null}',
        )
        self.assertFalse(ok)
        self.assertIn("null", detail)
        self.assertEqual(score, 0.0)

    def test_non_null_accepts_present_values(self):
        # With non_null set, every present non-null value passes —
        # including falsy ones like 0 and false.
        for output in ('{"v": 0}', '{"v": false}', '{"v": ""}'):
            ok, _, _ = score_case(
                _case(
                    {
                        "type": "json_path_exists",
                        "path": "$.v",
                        "non_null": True,
                    }
                ),
                output,
            )
            self.assertTrue(ok, output)

    def test_non_null_still_fails_when_missing(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "json_path_exists",
                    "path": "$.missing",
                    "non_null": True,
                }
            ),
            '{"v": 1}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_malformed_path_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_path_exists", "path": "$.a..b"}),
                '{"a": {}}',
            )

    def test_missing_path_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_path_exists"}),
                '{"a": 1}',
            )

    def test_non_boolean_non_null_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "json_path_exists",
                        "path": "$.v",
                        "non_null": "yes",
                    }
                ),
                '{"v": 1}',
            )

    def test_agrees_with_json_path_on_presence(self):
        # Existence passes exactly when json_path, given the value
        # actually at the path as its expected value, passes — the
        # two scorers share the same path language and the same
        # notion of "found".
        for output, path, expected in (
            ('{"result": {"id": 42}}', "$.result.id", 42),
            ('{"result": {"id": null}}', "$.result.id", None),
            ('{"items": ["a", "b"]}', "$.items[1]", "b"),
        ):
            exists = score_case(
                _case({"type": "json_path_exists", "path": path}),
                output,
            )
            equals = score_case(
                _case(
                    {
                        "type": "json_path",
                        "path": path,
                        "expected": expected,
                    }
                ),
                output,
            )
            self.assertEqual(exists[0], equals[0], output)
            self.assertEqual(exists[2], equals[2], output)


class TestKeywordOrderScorer(unittest.TestCase):
    def test_in_order_passes(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "keyword_order",
                    "keywords": ["gather", "verify", "submit"],
                }
            ),
            "first gather the facts, then verify them, then submit",
        )
        self.assertTrue(ok)
        self.assertIn("in order", detail)
        self.assertEqual(score, 1.0)

    def test_out_of_order_fails_with_ordered_prefix_credit(self):
        # "verify" comes before "gather" here, so the ordered prefix
        # stops at one of three keywords.
        ok, detail, score = score_case(
            _case(
                {
                    "type": "keyword_order",
                    "keywords": ["gather", "verify", "submit"],
                }
            ),
            "first verify the facts, then gather them, then submit",
        )
        self.assertFalse(ok)
        self.assertIn("out of order", detail)
        self.assertIn("'verify'", detail)
        self.assertAlmostEqual(score, 1 / 3)

    def test_missing_keyword_fails(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "keyword_order",
                    "keywords": ["gather", "verify", "submit"],
                }
            ),
            "gather the facts, then verify them",
        )
        self.assertFalse(ok)
        self.assertIn("missing", detail)
        self.assertIn("'submit'", detail)
        self.assertAlmostEqual(score, 2 / 3)

    def test_none_found_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "keyword_order", "keywords": ["a", "b"]}),
            "nothing here",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_empty_list_scores_one(self):
        ok, _, score = score_case(
            _case({"type": "keyword_order", "keywords": []}), "anything"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_repeated_keyword_needs_a_second_occurrence(self):
        ok, _, score = score_case(
            _case({"type": "keyword_order", "keywords": ["step", "step"]}),
            "one step only",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)
        ok, _, score = score_case(
            _case({"type": "keyword_order", "keywords": ["step", "step"]}),
            "step one, then step two",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "keyword_order",
                    "keywords": ["Gather", "VERIFY"],
                    "ignore_case": True,
                }
            ),
            "GATHER then verify",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_non_list_keywords_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "keyword_order", "keywords": "gather"}),
                "gather",
            )

    def test_missing_keywords_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "keyword_order"}), "gather")

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "keyword_order",
                        "keywords": ["a"],
                        "ignore_case": "yes",
                    }
                ),
                "a",
            )

    def test_agrees_with_contains_all_when_in_order(self):
        # When the keywords appear in order, keyword_order and
        # contains_all agree exactly — pass and score — because
        # presence in order implies presence.
        for output, keywords in (
            ("gather verify submit", ["gather", "verify", "submit"]),
            ("a and b", ["a", "b"]),
            ("nothing here", ["a", "b"]),
        ):
            ordered = score_case(
                _case({"type": "keyword_order", "keywords": keywords}),
                output,
            )
            unordered = score_case(
                _case({"type": "contains_all", "substrings": keywords}),
                output,
            )
            self.assertEqual(ordered[0], unordered[0], output)
            self.assertEqual(ordered[2], unordered[2], output)

    def test_out_of_order_passes_contains_all_but_not_keyword_order(self):
        # Presence without order is exactly what keyword_order adds
        # over contains_all: every keyword present, wrong sequence.
        output = "verify then gather then submit"
        keywords = ["gather", "verify", "submit"]
        unordered = score_case(
            _case({"type": "contains_all", "substrings": keywords}),
            output,
        )
        ordered = score_case(
            _case({"type": "keyword_order", "keywords": keywords}),
            output,
        )
        self.assertTrue(unordered[0])
        self.assertFalse(ordered[0])
        self.assertLess(ordered[2], 1.0)


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

    def test_ignore_case_matches(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "fuzzy",
                    "expected": "Hello World",
                    "ignore_case": True,
                }
            ),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case_false_is_case_sensitive(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "fuzzy",
                    "expected": "HELLO",
                    "ignore_case": False,
                }
            ),
            "hello",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_ignore_case_defaults_off(self):
        ok, _, score = score_case(
            _case({"type": "fuzzy", "expected": "HELLO"}),
            "hello",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "fuzzy",
                        "expected": "hello",
                        "ignore_case": "true",
                    }
                ),
                "hello",
            )


class TestLevenshteinScorer(unittest.TestCase):
    def test_exact_match(self):
        ok, _, score = score_case(
            _case({"type": "levenshtein", "expected": "hello world"}),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_one_typo_passes_with_edit_distance_score(self):
        ok, detail, score = score_case(
            _case({"type": "levenshtein", "expected": "hello world"}),
            "hello wrld",
        )
        self.assertTrue(ok)
        self.assertIn("distance 1", detail)
        self.assertIn("0.91", detail)
        self.assertAlmostEqual(score, 10 / 11)

    def test_classic_kitten_sitting_distance(self):
        ok, detail, score = score_case(
            _case({"type": "levenshtein", "expected": "kitten"}),
            "sitting",
        )
        self.assertFalse(ok)
        self.assertIn("distance 3", detail)
        self.assertAlmostEqual(score, 4 / 7)

    def test_short_answer_charges_more_per_typo(self):
        # The same single edit costs 1/5 of a short answer but only
        # 1/11 of a long one: edit distance charges per edit, so
        # short answers are punished harder than fuzzy would.
        ok, _, score = score_case(
            _case({"type": "levenshtein", "expected": "hello"}),
            "hallo",
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(score, 0.8)

    def test_far_miss_fails_with_partial_credit(self):
        ok, detail, score = score_case(
            _case({"type": "levenshtein", "expected": "hello world"}),
            "goodbye moon",
        )
        self.assertFalse(ok)
        self.assertIn("levenshtein distance", detail)
        self.assertLess(score, 0.8)
        self.assertGreater(score, 0.0)

    def test_threshold_is_respected(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "levenshtein",
                    "expected": "hello world",
                    "threshold": 0.99,
                }
            ),
            "hello wrld",
        )
        self.assertFalse(ok)

    def test_threshold_zero_passes_anything(self):
        ok, _, _ = score_case(
            _case(
                {
                    "type": "levenshtein",
                    "expected": "hello world",
                    "threshold": 0,
                }
            ),
            "something else entirely",
        )
        self.assertTrue(ok)

    def test_threshold_above_one_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "levenshtein", "expected": "hello", "threshold": 1.5}
                ),
                "hello",
            )

    def test_negative_threshold_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "levenshtein", "expected": "hello", "threshold": -0.1}
                ),
                "hello",
            )

    def test_missing_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "levenshtein"}), "hello")

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(
            _case({"type": "levenshtein", "expected": "hello"}), "  hello\n"
        )
        self.assertTrue(ok)

    def test_both_empty_match(self):
        ok, _, score = score_case(
            _case({"type": "levenshtein", "expected": ""}), ""
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_empty_expected_only_matches_empty(self):
        ok, _, score = score_case(
            _case({"type": "levenshtein", "expected": ""}), "hello"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_ignore_case_matches(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "levenshtein",
                    "expected": "Hello World",
                    "ignore_case": True,
                }
            ),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_case_defaults_off(self):
        ok, _, score = score_case(
            _case({"type": "levenshtein", "expected": "HELLO"}),
            "hello",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "levenshtein",
                        "expected": "hello",
                        "ignore_case": "true",
                    }
                ),
                "hello",
            )


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


class TestOneOfScorer(unittest.TestCase):
    def test_allowed_value_passes(self):
        ok, detail, score = score_case(
            _case({"type": "one_of", "values": ["yes", "no", "maybe"]}),
            "no",
        )
        self.assertTrue(ok)
        self.assertIn("one of", detail)
        self.assertEqual(score, 1.0)

    def test_first_value_passes(self):
        ok, _, score = score_case(
            _case({"type": "one_of", "values": ["yes", "no"]}), "yes"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_value_not_in_list_fails(self):
        ok, detail, score = score_case(
            _case({"type": "one_of", "values": ["yes", "no", "maybe"]}),
            "perhaps",
        )
        self.assertFalse(ok)
        self.assertIn("expected one of", detail)
        self.assertEqual(score, 0.0)

    def test_substring_of_allowed_value_fails(self):
        # one_of is whole-output equality, not containment: "yes
        # please" is not "yes".
        ok, _, score = score_case(
            _case({"type": "one_of", "values": ["yes", "no"]}),
            "yes please",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(
            _case({"type": "one_of", "values": ["yes", "no"]}), "  yes\n"
        )
        self.assertTrue(ok)

    def test_ignore_case_matches_other_case(self):
        ok, _, score = score_case(
            _case(
                {
                    "type": "one_of",
                    "values": ["Yes", "No"],
                    "ignore_case": True,
                }
            ),
            "YES",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_case_sensitive_by_default(self):
        ok, _, score = score_case(
            _case({"type": "one_of", "values": ["Yes", "No"]}), "YES"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_missing_values_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "one_of"}), "yes")

    def test_non_list_values_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "one_of", "values": "yes"}), "yes")

    def test_empty_values_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "one_of", "values": []}), "yes")

    def test_non_boolean_ignore_case_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {
                        "type": "one_of",
                        "values": ["yes"],
                        "ignore_case": "true",
                    }
                ),
                "yes",
            )

    def test_single_value_behaves_like_exact(self):
        ok, _, score = score_case(
            _case({"type": "one_of", "values": ["42"]}), "42"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)


class TestLengthScorer(unittest.TestCase):
    def test_inside_range_passes(self):
        ok, detail, score = score_case(
            _case({"type": "length", "min_length": 10, "max_length": 20}),
            "hello world",
        )
        self.assertTrue(ok)
        self.assertIn("length 11 characters", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "length", "min_length": 10, "max_length": 20}),
            "0123456789",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "length", "min_length": 10, "max_length": 20}),
            "01234567890123456789",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 5 of a 10 minimum: score falls linearly to 0 at length 0.
        ok, detail, score = score_case(
            _case({"type": "length", "min_length": 10, "max_length": 20}),
            "hello",
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 10", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_empty_output_below_minimum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "length", "min_length": 10}), ""
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_above_maximum_partial_credit(self):
        # 15 of a 10 maximum: halfway to 2x the max, so score 0.5.
        ok, detail, score = score_case(
            _case({"type": "length", "max_length": 10}),
            "0123456789abcde",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 10", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_twice_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "length", "max_length": 10}),
            "01234567890123456789",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_wildly_over_maximum_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "length", "max_length": 10}), "x" * 25
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_only_minimum_is_unbounded_above(self):
        ok, _, score = score_case(
            _case({"type": "length", "min_length": 5}), "x" * 100
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_only_maximum_is_unbounded_below(self):
        ok, _, score = score_case(
            _case({"type": "length", "max_length": 5}), ""
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_zero_maximum(self):
        ok, _, score = score_case(
            _case({"type": "length", "max_length": 0}), ""
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "length", "max_length": 0}), "x"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_words_unit(self):
        ok, detail, score = score_case(
            _case(
                {
                    "type": "length",
                    "min_length": 2,
                    "max_length": 4,
                    "unit": "words",
                }
            ),
            "one two three",
        )
        self.assertTrue(ok)
        self.assertIn("length 3 words", detail)
        self.assertEqual(score, 1.0)

    def test_words_unit_partial_credit(self):
        # 5 words against a 4 maximum: (2*4 - 5) / 4 = 0.75.
        ok, _, score = score_case(
            _case(
                {"type": "length", "max_length": 4, "unit": "words"}
            ),
            "one two three four five",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.75)

    def test_words_unit_below_minimum(self):
        # 1 word against a 2 minimum: 1 / 2 = 0.5.
        ok, _, score = score_case(
            _case(
                {"type": "length", "min_length": 2, "unit": "words"}
            ),
            "one",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)

    def test_words_split_on_any_whitespace(self):
        ok, _, _ = score_case(
            _case(
                {"type": "length", "min_length": 3, "max_length": 3,
                 "unit": "words"}
            ),
            "one\ttwo\nthree",
        )
        self.assertTrue(ok)

    def test_surrounding_whitespace_ignored(self):
        ok, _, _ = score_case(
            _case({"type": "length", "min_length": 5, "max_length": 5}),
            "  hello  \n",
        )
        self.assertTrue(ok)

    def test_integral_float_bound_accepted(self):
        ok, _, _ = score_case(
            _case({"type": "length", "min_length": 10.0}), "0123456789"
        )
        self.assertTrue(ok)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "length"}), "hello")

    def test_negative_minimum_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "length", "min_length": -1}), "hello"
            )

    def test_negative_maximum_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "length", "max_length": -5}), "hello"
            )

    def test_non_integer_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "length", "min_length": 10.5}), "hello"
            )

    def test_string_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "length", "min_length": "10"}), "hello"
            )

    def test_boolean_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "length", "min_length": True}), "hello"
            )

    def test_minimum_above_maximum_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "length", "min_length": 20, "max_length": 10}
                ),
                "hello",
            )

    def test_unknown_unit_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "length", "max_length": 10, "unit": "bytes"}
                ),
                "hello",
            )

    def test_non_string_unit_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "length", "max_length": 10, "unit": 5}),
                "hello",
            )

    def test_flows_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "length-demo",
                "cases": [
                    {
                        "id": "terse",
                        "prompt": "Answer briefly.",
                        "scorer": {
                            "type": "length",
                            "max_length": 10,
                        },
                    }
                ],
            }
        )
        report = run_suite(suite, lambda prompt: "short")
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.results[0].score, 1.0)


class TestRangeScorer(unittest.TestCase):
    def test_inside_range_passes(self):
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}),
            "The answer is 42.",
        )
        self.assertTrue(ok)
        self.assertIn("value 42.0 within", detail)
        self.assertEqual(score, 1.0)

    def test_boundaries_pass(self):
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "0"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "100"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_below_minimum_partial_credit(self):
        # 50 below a 0 minimum, range width 100: 1 - 50/100 = 0.5.
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "-50"
        )
        self.assertFalse(ok)
        self.assertIn("below minimum 0.0", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_above_maximum_partial_credit(self):
        # 150 against a 100 maximum, range width 100: 1 - 50/100 = 0.5.
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "150"
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 100.0", detail)
        self.assertAlmostEqual(score, 0.5)

    def test_one_scale_outside_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "200"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "-100"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_wildly_outside_scores_zero(self):
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "9999"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_only_minimum_is_unbounded_above(self):
        ok, _, score = score_case(
            _case({"type": "range", "min": 0}), "1000000"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_only_maximum_is_unbounded_below(self):
        ok, _, score = score_case(
            _case({"type": "range", "max": 100}), "-1000000"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_single_bound_falloff_uses_bound_magnitude(self):
        # 5 against a lone minimum of 10: 1 - (10 - 5) / 10 = 0.5,
        # the same falloff the length scorer gives for 5 of a 10 minimum.
        ok, _, score = score_case(
            _case({"type": "range", "min": 10}), "5"
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)
        # 15 against a lone maximum of 10: 1 - (15 - 10) / 10 = 0.5.
        ok, _, score = score_case(
            _case({"type": "range", "max": 10}), "15"
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)

    def test_negative_range(self):
        ok, _, score = score_case(
            _case({"type": "range", "min": -10, "max": -1}), "-5"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # 1 above a -1 maximum, width 9: 1 - 1/9.
        ok, _, score = score_case(
            _case({"type": "range", "min": -10, "max": -1}), "0"
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 1.0 - 1.0 / 9.0)

    def test_zero_width_range(self):
        ok, _, score = score_case(
            _case({"type": "range", "min": 42, "max": 42}), "42"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "range", "min": 42, "max": 42}), "43"
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_lone_zero_bound(self):
        ok, _, score = score_case(_case({"type": "range", "min": 0}), "0")
        self.assertTrue(ok)
        ok, _, score = score_case(_case({"type": "range", "min": 0}), "-1")
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_number_embedded_in_text(self):
        ok, _, _ = score_case(
            _case({"type": "range", "min": 0, "max": 100}),
            "Score: 87 out of 100",
        )
        self.assertTrue(ok)

    def test_float_and_scientific_notation(self):
        ok, _, _ = score_case(
            _case({"type": "range", "min": 0, "max": 1}), "0.75"
        )
        self.assertTrue(ok)
        ok, _, _ = score_case(
            _case({"type": "range", "min": 0, "max": 1}), "5e-1"
        )
        self.assertTrue(ok)

    def test_no_number_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100}), "no digits here"
        )
        self.assertFalse(ok)
        self.assertIn("no number in output", detail)
        self.assertEqual(score, 0.0)

    def test_no_bounds_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "range"}), "42")

    def test_minimum_above_maximum_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "range", "min": 100, "max": 0}), "42"
            )

    def test_string_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "range", "min": "0"}), "42")

    def test_boolean_bound_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "range", "max": True}), "42")

    def test_tolerance_widens_band_passes(self):
        # The band [0, 100] with tolerance 10 accepts [-10, 110].
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 10}),
            "105",
        )
        self.assertTrue(ok)
        self.assertIn("within [min -10.0, max 110.0]", detail)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 10}),
            "-5",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # The widened boundary itself passes.
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 10}),
            "110",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_tolerance_outside_widened_band_partial_credit(self):
        # 150 against a widened maximum of 110, widened width 120:
        # 1 - 40/120 = 2/3.
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 10}),
            "150",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 110.0", detail)
        self.assertAlmostEqual(score, 2.0 / 3.0)
        # One widened scale outside scores 0.
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 10}),
            "230",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_tolerance_with_single_bound(self):
        # A lone minimum of 10 with tolerance 5 accepts from 5 up.
        ok, _, score = score_case(
            _case({"type": "range", "min": 10, "tolerance": 5}), "5"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # 4 is 1 below the widened minimum of 5, scale 5: 1 - 1/5.
        ok, _, score = score_case(
            _case({"type": "range", "min": 10, "tolerance": 5}), "4"
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.8)
        # A lone maximum of 10 with tolerance 5 accepts up to 15.
        ok, _, score = score_case(
            _case({"type": "range", "max": 10, "tolerance": 5}), "15"
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # 16 is 1 above the widened maximum of 15, scale 15: 1 - 1/15.
        ok, _, score = score_case(
            _case({"type": "range", "max": 10, "tolerance": 5}), "16"
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 1.0 - 1.0 / 15.0)

    def test_tolerance_gives_zero_width_range_partial_credit(self):
        # min == max == 42 with tolerance 5 accepts [37, 47]; the
        # widened width of 10 is the falloff scale, so a zero-width
        # range with tolerance finally has partial credit.
        ok, _, score = score_case(
            _case({"type": "range", "min": 42, "max": 42, "tolerance": 5}),
            "45",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        # 50 is 3 above the widened maximum of 47: 1 - 3/10.
        ok, _, score = score_case(
            _case({"type": "range", "min": 42, "max": 42, "tolerance": 5}),
            "50",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.7)

    def test_tolerance_zero_keeps_old_path(self):
        ok, detail, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 0}),
            "105",
        )
        self.assertFalse(ok)
        self.assertIn("above maximum 100.0", detail)
        self.assertAlmostEqual(score, 0.95)
        ok, _, score = score_case(
            _case({"type": "range", "min": 0, "max": 100, "tolerance": 0}),
            "150",
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(score, 0.5)

    def test_negative_tolerance_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "range", "min": 0, "max": 100,
                     "tolerance": -1}
                ),
                "50",
            )

    def test_non_numeric_tolerance_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "range", "min": 0, "max": 100,
                     "tolerance": "10"}
                ),
                "50",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "range", "min": 0, "max": 100,
                     "tolerance": True}
                ),
                "50",
            )

    def test_nan_and_infinite_tolerance_raise(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "range", "min": 0, "max": 100,
                     "tolerance": float("nan")}
                ),
                "50",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "range", "min": 0, "max": 100,
                     "tolerance": float("inf")}
                ),
                "50",
            )

    def test_tolerance_flows_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "range-tolerance-demo",
                "cases": [
                    {
                        "id": "close-enough",
                        "prompt": "What is 6 times 7, roughly?",
                        "scorer": {
                            "type": "range",
                            "min": 0,
                            "max": 100,
                            "tolerance": 10,
                        },
                    }
                ],
            }
        )
        report = run_suite(suite, lambda prompt: "105")
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.results[0].score, 1.0)

    def test_flows_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "range-demo",
                "cases": [
                    {
                        "id": "in-range",
                        "prompt": "What is 6 times 7?",
                        "scorer": {
                            "type": "range",
                            "min": 0,
                            "max": 100,
                        },
                    }
                ],
            }
        )
        report = run_suite(suite, lambda prompt: "42")
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.results[0].score, 1.0)


class TestJsonEqualsScorer(unittest.TestCase):
    def test_exact_object_passes(self):
        ok, detail, score = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"status": "ok", "total": 42}}
            ),
            '{"status": "ok", "total": 42}',
        )
        self.assertTrue(ok)
        self.assertIn("JSON values are equal", detail)
        self.assertEqual(score, 1.0)

    def test_key_order_does_not_matter(self):
        ok, _, _ = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"a": 1, "b": 2}}
            ),
            '{"b": 2, "a": 1}',
        )
        self.assertTrue(ok)

    def test_extra_key_fails(self):
        # Full equality, unlike json_subset: extra keys fail.
        ok, detail, score = score_case(
            _case(
                {"type": "json_equals", "expected": {"status": "ok"}}
            ),
            '{"status": "ok", "request_id": "abc"}',
        )
        self.assertFalse(ok)
        self.assertIn("expected", detail)
        self.assertEqual(score, 0.0)

    def test_missing_key_fails(self):
        ok, _, score = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"status": "ok", "total": 42}}
            ),
            '{"status": "ok"}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_nested_objects_pass(self):
        ok, _, _ = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"result": {"total": 42, "items": [1, 2]}}}
            ),
            '{"result": {"items": [1, 2], "total": 42}}',
        )
        self.assertTrue(ok)

    def test_nested_mismatch_fails(self):
        ok, detail, score = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"result": {"total": 42}}}
            ),
            '{"result": {"total": 43}}',
        )
        self.assertFalse(ok)
        self.assertIn("43", detail)
        self.assertEqual(score, 0.0)

    def test_int_and_float_are_equal(self):
        ok, _, _ = score_case(
            _case({"type": "json_equals", "expected": 1}),
            "1.0",
        )
        self.assertTrue(ok)

    def test_int_vs_string_is_type_strict(self):
        ok, _, score = score_case(
            _case({"type": "json_equals", "expected": 1}),
            '"1"',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_bool_vs_int_is_type_strict(self):
        ok, _, score = score_case(
            _case({"type": "json_equals", "expected": True}),
            "1",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        ok, _, score = score_case(
            _case({"type": "json_equals", "expected": 1}),
            "true",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_list_order_matters_by_default(self):
        ok, _, score = score_case(
            _case({"type": "json_equals", "expected": [1, 2, 3]}),
            "[3, 2, 1]",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_ignore_order_passes_on_reordered_arrays(self):
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": [1, 2, 3],
                 "ignore_order": True}
            ),
            "[3, 1, 2]",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_ignore_order_is_still_a_multiset(self):
        # [1, 1, 2] is not [1, 2, 2]: duplicates must match too.
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": [1, 1, 2],
                 "ignore_order": True}
            ),
            "[1, 2, 2]",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_ignore_order_covers_nested_arrays(self):
        ok, _, _ = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"tags": ["b", "a"]},
                 "ignore_order": True}
            ),
            '{"tags": ["a", "b"]}',
        )
        self.assertTrue(ok)

    def test_ignore_order_non_bool_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_equals", "expected": [1],
                     "ignore_order": "yes"}
                ),
                "[1]",
            )

    def test_invalid_json_output_scores_zero(self):
        ok, detail, score = score_case(
            _case({"type": "json_equals", "expected": {"a": 1}}),
            "{not json",
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)
        self.assertEqual(score, 0.0)

    def test_scalar_outputs(self):
        ok, _, _ = score_case(
            _case({"type": "json_equals", "expected": "hello"}),
            '"hello"',
        )
        self.assertTrue(ok)
        ok, _, _ = score_case(
            _case({"type": "json_equals", "expected": None}),
            "null",
        )
        self.assertTrue(ok)
        ok, _, _ = score_case(
            _case({"type": "json_equals", "expected": 42}),
            "42",
        )
        self.assertTrue(ok)

    def test_missing_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "json_equals"}), "{}")

    def test_non_serialisable_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_equals", "expected": {1, 2}}),
                "[1, 2]",
            )

    def test_nan_expected_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case({"type": "json_equals", "expected": float("nan")}),
                "null",
            )

    def test_tolerance_within_passes(self):
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": 3.14159,
                 "tolerance": 0.001}
            ),
            "3.141",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)

    def test_tolerance_outside_fails(self):
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": 3.14159,
                 "tolerance": 0.001}
            ),
            "3.14",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_tolerance_applies_to_nested_numbers(self):
        ok, _, _ = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"result": {"total": 42.0, "items": [1.5, 2.5]}},
                 "tolerance": 0.1}
            ),
            '{"result": {"total": 42.05, "items": [1.55, 2.45]}}',
        )
        self.assertTrue(ok)
        ok, _, score = score_case(
            _case(
                {"type": "json_equals",
                 "expected": {"result": {"total": 42.0}},
                 "tolerance": 0.1}
            ),
            '{"result": {"total": 42.2}}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_tolerance_zero_is_exact(self):
        ok, _, _ = score_case(
            _case(
                {"type": "json_equals", "expected": 1.5, "tolerance": 0}
            ),
            "1.5",
        )
        self.assertTrue(ok)
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": 1.5, "tolerance": 0}
            ),
            "1.5000001",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_tolerance_does_not_loosen_non_numbers(self):
        # Strings still compare exactly; bool is still never an int.
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": {"a": "x", "b": True},
                 "tolerance": 10}
            ),
            '{"a": "y", "b": 1}',
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_tolerance_with_ignore_order(self):
        # Reordered arrays still match as multisets, now within
        # tolerance — including the greedy-pairing trap: 1.1 must be
        # free to match 1.2, not consumed by 1.0.
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": [1.0, 1.2],
                 "ignore_order": True, "tolerance": 0.15}
            ),
            "[1.1, 1.0]",
        )
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        ok, _, score = score_case(
            _case(
                {"type": "json_equals", "expected": [1.0, 1.2],
                 "ignore_order": True, "tolerance": 0.15}
            ),
            "[1.1, 1.5]",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)

    def test_negative_tolerance_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_equals", "expected": 1,
                     "tolerance": -0.5}
                ),
                "1",
            )

    def test_non_numeric_tolerance_raises(self):
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_equals", "expected": 1,
                     "tolerance": "0.5"}
                ),
                "1",
            )
        with self.assertRaises(ValueError):
            score_case(
                _case(
                    {"type": "json_equals", "expected": 1, "tolerance": True}
                ),
                "1",
            )

    def test_tolerance_with_invalid_json_still_scores_zero(self):
        ok, detail, score = score_case(
            _case(
                {"type": "json_equals", "expected": {"a": 1},
                 "tolerance": 0.5}
            ),
            "{not json",
        )
        self.assertFalse(ok)
        self.assertIn("not valid JSON", detail)
        self.assertEqual(score, 0.0)

    def test_tolerance_flows_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "json-equals-tolerance-demo",
                "cases": [
                    {
                        "id": "pi-close",
                        "prompt": "Return pi.",
                        "scorer": {
                            "type": "json_equals",
                            "expected": {"pi": 3.14159},
                            "tolerance": 0.001,
                        },
                    }
                ],
            }
        )
        report = run_suite(suite, lambda prompt: '{"pi": 3.141}')
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.results[0].score, 1.0)


    def test_flows_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "json-equals-demo",
                "cases": [
                    {
                        "id": "status-ok",
                        "prompt": "Return the status object.",
                        "scorer": {
                            "type": "json_equals",
                            "expected": {"status": "ok", "total": 42},
                        },
                    }
                ],
            }
        )
        report = run_suite(suite, lambda prompt: '{"status": "ok", "total": 42}')
        self.assertEqual(report.passed, 1)
        self.assertEqual(report.results[0].score, 1.0)


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


class TestFailOnScore(unittest.TestCase):
    def _score_suite(self, threshold=None):
        # A contains_all case over four substrings with a per-case
        # threshold of 0.75: an agent answering three of the four
        # passes with a score of 0.75, so it is exactly the kind of
        # partial-credit pass the --fail-on-score gate is for.
        case = {
            "id": "mostly",
            "prompt": "list a b c d",
            "scorer": {
                "type": "contains_all",
                "substrings": ["a", "b", "c", "d"],
            },
        }
        if threshold is not None:
            case["threshold"] = threshold
        return {"name": "cli-s", "cases": [case]}

    def _partial_agent_src(self):
        return "def agent(prompt):\n    return \"a b c\"\n"

    def _run_cli(self, tmp, suite_data, args, agent_src=None):
        from agent_evals.cli import main

        path = os.path.join(tmp, "suite.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(suite_data, f)
        if agent_src is not None:
            with open(os.path.join(tmp, "partial_agent.py"), "w") as f:
                f.write(agent_src)
            sys.path.insert(0, tmp)
        try:
            return main(
                ["--suite", path, "--agent", "partial_agent:agent"] + args
            )
        finally:
            if agent_src is not None:
                sys.path.remove(tmp)
                sys.modules.pop("partial_agent", None)

    def test_cli_fail_on_score_flag(self):
        # The agent finds 3 of 4 substrings: score 0.75, passed at
        # the case threshold 0.75. Without the flag the run passes;
        # with --fail-on-score 0.9 the same pass fails the run, and
        # at the score itself (0.75) it passes, since the gate only
        # trips below the threshold.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run_cli(
                    tmp,
                    self._score_suite(threshold=0.75),
                    [],
                    agent_src=self._partial_agent_src(),
                ),
                0,
            )
            self.assertEqual(
                self._run_cli(
                    tmp,
                    self._score_suite(threshold=0.75),
                    ["--fail-on-score", "0.9"],
                    agent_src=self._partial_agent_src(),
                ),
                1,
            )
            self.assertEqual(
                self._run_cli(
                    tmp,
                    self._score_suite(threshold=0.75),
                    ["--fail-on-score", "0.75"],
                    agent_src=self._partial_agent_src(),
                ),
                0,
            )
            self.assertEqual(
                self._run_cli(
                    tmp,
                    self._score_suite(threshold=0.75),
                    ["--fail-on-score", "0.5"],
                    agent_src=self._partial_agent_src(),
                ),
                0,
            )

    def test_cli_fail_on_score_rejects_out_of_range(self):
        from agent_evals.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._score_suite(threshold=0.75), f)
            for bad in ("1.5", "-0.1", "nan", "inf"):
                with self.assertRaises(SystemExit) as ctx:
                    main(
                        [
                            "--suite",
                            path,
                            "--agent",
                            "partial_agent:agent",
                            "--fail-on-score",
                            bad,
                        ]
                    )
                self.assertEqual(ctx.exception.code, 2)

    def test_cli_fail_on_score_ignores_budget_violation(self):
        # A budget violation records a score of 0.0, but that is not
        # a scored result — the answer was right, just slow — so the
        # score gate must not trip on it. Without --fail-on-budget
        # the run still passes even with --fail-on-score 0.9.
        from agent_evals.cli import main

        data = {
            "name": "cli-sb",
            "cases": [
                {
                    "id": "slow",
                    "prompt": "2 + 3",
                    "scorer": {"type": "exact", "expected": "5"},
                    "max_latency_ms": 0.01,
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
                    "--fail-on-score",
                    "0.9",
                ]
            )
            self.assertEqual(code, 0)

    def test_cli_fail_on_score_composes_with_fail_on_budget(self):
        # Either gate can fail the run: a low-score pass trips the
        # score gate even when --fail-on-budget is also given and
        # there is no budget violation at all.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run_cli(
                    tmp,
                    self._score_suite(threshold=0.75),
                    ["--fail-on-score", "0.9", "--fail-on-budget"],
                    agent_src=self._partial_agent_src(),
                ),
                1,
            )

    def test_cli_fail_on_score_does_not_mask_genuine_failure(self):
        # A case that fails its checks fails the run with or without
        # the flag: the gate adds failures, it never hides one.
        data = {
            "name": "cli-sf",
            "cases": [
                {
                    "id": "wrong",
                    "prompt": "2 + 3",
                    "scorer": {"type": "exact", "expected": "6"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "suite.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            from agent_evals.cli import main

            code = main(
                [
                    "--suite",
                    path,
                    "--agent",
                    "examples.calculator_agent:agent",
                    "--fail-on-score",
                    "0.9",
                ]
            )
            self.assertEqual(code, 1)


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


class TestPerCaseThreshold(unittest.TestCase):
    """Per-case pass thresholds (v0.43.0).

    A case may carry a ``threshold`` in [0, 1]: it passes when its
    scorer's score reaches the threshold, so a partial-credit case
    can pass without a full score. When unset the case passes only
    as its scorer says, exactly as before.
    """

    def _threshold_case(self, threshold, scorer=None):
        return Case(
            "c1",
            "list the fruits",
            scorer
            or {
                "type": "contains_all",
                "substrings": ["apple", "banana", "cherry", "date"],
            },
            threshold=threshold,
        )

    def test_threshold_defaults_to_none_and_is_omitted(self):
        case = Case.from_dict(
            {"id": "c1", "prompt": "p", "scorer": {"type": "exact", "expected": "x"}}
        )
        self.assertIsNone(case.threshold)
        self.assertNotIn("threshold", case.to_dict())
        # An unset threshold leaves the case identical to before.
        twin = Case("c1", "p", {"type": "exact", "expected": "x"})
        self.assertEqual(case.fingerprint(), twin.fingerprint())

    def test_threshold_round_trips_through_suite_json(self):
        suite = Suite.from_dict(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "c1",
                        "prompt": "list the fruits",
                        "scorer": {
                            "type": "contains_all",
                            "substrings": ["apple", "banana", "cherry", "date"],
                        },
                        "threshold": 0.75,
                    }
                ],
            }
        )
        self.assertEqual(suite.cases[0].threshold, 0.75)
        self.assertEqual(suite.to_dict()["cases"][0]["threshold"], 0.75)
        report = run_suite(suite, lambda prompt: "apple banana cherry")
        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.results[0].score, 0.75)

    def test_threshold_changes_the_case_fingerprint(self):
        plain = self._threshold_case(None)
        with_threshold = self._threshold_case(0.5)
        self.assertNotEqual(plain.fingerprint(), with_threshold.fingerprint())
        other = self._threshold_case(0.75)
        self.assertNotEqual(with_threshold.fingerprint(), other.fingerprint())

    def test_partial_credit_passes_when_score_meets_threshold(self):
        # 3 of 4 substrings: score 0.75, which the scorer alone fails.
        suite = Suite("s", [self._threshold_case(0.75)])
        report = run_suite(suite, lambda prompt: "apple banana cherry")
        result = report.results[0]
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(report.earned_weight, 0.75)
        self.assertIn("meets threshold 0.75", result.detail)

    def test_partial_credit_fails_below_threshold(self):
        suite = Suite("s", [self._threshold_case(0.8)])
        report = run_suite(suite, lambda prompt: "apple banana cherry")
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.75)
        self.assertEqual(report.earned_weight, 0.75)

    def test_threshold_can_fail_a_scorer_pass_on_a_low_score(self):
        # fuzzy ratio 0.9 passes the scorer (its own threshold is
        # 0.8) with a score of 0.9; a case threshold of 0.95 fails it.
        case = Case(
            "c1",
            "p",
            {"type": "fuzzy", "expected": "abcdefghij"},
            threshold=0.95,
        )
        report = run_suite(Suite("s", [case]), lambda prompt: "abcdefghix")
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.9)
        self.assertIn("below threshold 0.95", result.detail)

    def test_threshold_zero_passes_any_scored_case(self):
        case = Case(
            "c1", "p", {"type": "exact", "expected": "yes"}, threshold=0.0
        )
        report = run_suite(Suite("s", [case]), lambda prompt: "no")
        self.assertTrue(report.results[0].passed)
        self.assertEqual(report.results[0].score, 0.0)

    def test_scorer_error_still_fails_with_zero_threshold(self):
        case = Case(
            "c1",
            "p",
            {"type": "regex", "pattern": "[invalid"},
            threshold=0.0,
        )
        report = run_suite(Suite("s", [case]), lambda prompt: "anything")
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.detail.startswith("scorer error:"))
        self.assertEqual(result.score, 0.0)

    def test_timeout_still_fails_with_zero_threshold(self):
        def slow_agent(prompt):
            time.sleep(0.5)
            return "done"

        case = Case(
            "c1",
            "p",
            {"type": "exact", "expected": "done"},
            timeout=0.05,
            threshold=0.0,
        )
        report = run_suite(Suite("s", [case]), slow_agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.timed_out)

    def test_agent_error_still_fails_with_zero_threshold(self):
        def broken_agent(prompt):
            raise ValueError("boom")

        case = Case(
            "c1", "p", {"type": "exact", "expected": "x"}, threshold=0.0
        )
        report = run_suite(Suite("s", [case]), broken_agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertIn("agent error", result.detail)

    def test_threshold_passed_case_can_still_trip_a_latency_budget(self):
        def slow_agent(prompt):
            time.sleep(0.01)
            return "apple banana cherry"

        case = Case(
            "c1",
            "list the fruits",
            {
                "type": "contains_all",
                "substrings": ["apple", "banana", "cherry", "date"],
            },
            threshold=0.5,
            max_latency_ms=1,
        )
        report = run_suite(Suite("s", [case]), slow_agent)
        result = report.results[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.budget_violated)
        self.assertEqual(result.score, 0.0)

    def test_threshold_in_report_and_summary(self):
        suite = Suite("s", [self._threshold_case(0.5)])
        report = run_suite(suite, lambda prompt: "apple banana cherry")
        data = report.to_dict()
        self.assertEqual(data["results"][0]["threshold"], 0.5)
        self.assertEqual(data["passed"], 1)
        text = format_summary(report)
        self.assertIn("PASS", text)
        self.assertIn("meets threshold 0.5", text)

    def test_threshold_flows_through_suite_json_file(self):
        suite_json = {
            "name": "threshold-suite",
            "cases": [
                {
                    "id": "fruits",
                    "prompt": "list the fruits",
                    "scorer": {
                        "type": "contains_all",
                        "substrings": ["apple", "banana", "cherry", "date"],
                    },
                    "threshold": 0.5,
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
            report = run_suite(suite, lambda prompt: "apple banana cherry")
        finally:
            os.unlink(path)
        self.assertEqual(report.passed, 1)

    def test_non_numeric_threshold_rejected(self):
        for bad in ("high", True, "0.5"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Case.from_dict(
                        {
                            "id": "c1",
                            "prompt": "p",
                            "scorer": {"type": "exact", "expected": "x"},
                            "threshold": bad,
                        }
                    )

    def test_out_of_range_threshold_rejected(self):
        for bad in (-0.1, 1.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Case.from_dict(
                        {
                            "id": "c1",
                            "prompt": "p",
                            "scorer": {"type": "exact", "expected": "x"},
                            "threshold": bad,
                        }
                    )

    def test_integer_threshold_accepted(self):
        case = Case.from_dict(
            {
                "id": "c1",
                "prompt": "p",
                "scorer": {"type": "exact", "expected": "x"},
                "threshold": 1,
            }
        )
        self.assertEqual(case.threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
