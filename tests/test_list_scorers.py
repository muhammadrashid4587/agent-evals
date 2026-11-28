"""Tests for --list-scorers and the scorer registry (v0.69.0).

``--list-scorers`` prints every scorer the harness knows, with an
example spec and a one-line description, and exits without a suite or
an agent — for writing suite files without opening the README. The
registry in ``agent_evals.evals`` is the source of truth: its example
specs are real specs, so the listing can never drift from the
dispatcher — every listed type must be accepted by ``score_case``.
"""

import io
import unittest
from contextlib import redirect_stdout

from agent_evals.cli import main
from agent_evals.evals import SCORER_TYPES, Case, format_scorer_list, score_case

# The registry order mirrors the README scorer table, so a new scorer
# is added at the end of both in the same change.
EXPECTED_TYPES = [
    "exact",
    "contains",
    "contains_all",
    "contains_any",
    "keyword_order",
    "not_contains",
    "count",
    "regex",
    "regex_count",
    "word_count",
    "unique_words",
    "sentence_count",
    "keyword_density",
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_word_length",
    "word_length_variance",
    "readability",
    "syllable_count",
    "type_token_ratio",
    "hapax_ratio",
    "stopword_ratio",
    "content_word_ratio",
    "contains_regex",
    "json_schema",
    "json_path",
    "json_path_numeric",
    "json_path_count",
    "json_path_exists",
    "json_subset",
    "json_equals",
    "fuzzy",
    "levenshtein",
    "starts_with",
    "ends_with",
    "one_of",
    "length",
    "range",
    "numeric",
    "judge",
]


class TestScorerRegistry(unittest.TestCase):
    def test_registry_lists_every_scorer_once_in_readme_order(self):
        self.assertEqual(
            [kind for kind, _, _ in SCORER_TYPES], EXPECTED_TYPES
        )

    def test_every_example_spec_is_accepted_by_score_case(self):
        # The listing must never name a scorer the dispatcher does
        # not know, and its example specs must be real specs: score
        # a sample output with each one and require a score in
        # [0, 1], not a scorer error. Pass or fail does not matter
        # here — only that the type is known and the spec is valid.
        for kind, spec, _ in SCORER_TYPES:
            case = Case(
                id="x",
                prompt="q",
                scorer={"type": kind, **spec},
            )
            with self.subTest(kind=kind):
                passed, _, score = score_case(case, "hello world 42")
                self.assertIsInstance(passed, bool)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_registry_descriptions_are_nonempty_one_liners(self):
        for kind, _, description in SCORER_TYPES:
            with self.subTest(kind=kind):
                self.assertTrue(description)
                self.assertNotIn("\n", description)

    def test_format_lists_every_scorer_with_its_example_spec(self):
        rendered = format_scorer_list()
        self.assertIn(f"Scorers ({len(SCORER_TYPES)}):", rendered)
        for kind, spec, _ in SCORER_TYPES:
            with self.subTest(kind=kind):
                self.assertIn(kind, rendered)
                # The example spec renders as JSON with the type first.
                self.assertIn(f'"type": "{kind}"', rendered)
                for key in spec:
                    self.assertIn(f'"{key}"', rendered)


class TestListScorersCli(unittest.TestCase):
    def _run(self, args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(args)
        return code, out.getvalue()

    def test_lists_scorers_with_no_suite_and_no_agent(self):
        code, out = self._run(["--list-scorers"])
        self.assertEqual(code, 0)
        self.assertIn("Scorers (40):", out)
        for kind in EXPECTED_TYPES:
            self.assertIn(kind, out)
        self.assertIn('"type": "exact"', out)
        self.assertIn('"type": "avg_word_length"', out)
        self.assertIn("average word length in characters", out)
        self.assertIn('"type": "word_length_variance"', out)
        self.assertIn("word-length variance falls in a bounded range", out)
        self.assertIn('"type": "json_path_count"', out)
        self.assertIn(
            "number of items at the JSON path falls in a bounded range",
            out,
        )
        self.assertIn('"type": "json_path_exists"', out)
        self.assertIn("the value at the JSON path exists", out)
        self.assertIn('"type": "syllable_count"', out)
        self.assertIn(
            "the output has a bounded number of syllables", out
        )

    def test_list_scorers_never_imports_an_agent(self):
        # A bogus agent spec must not matter: listing never loads it.
        code, out = self._run(
            ["--list-scorers", "--agent", "does.not.exist:agent"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Scorers (40):", out)

    def test_list_scorers_never_reads_a_suite(self):
        # A nonexistent suite path must not matter either: listing
        # needs no suite file.
        code, out = self._run(
            ["--list-scorers", "--suite", "does/not/exist.json"]
        )
        self.assertEqual(code, 0)
        self.assertIn("Scorers (40):", out)

    def test_suite_still_required_without_list_scorers(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--agent", "examples.calculator_agent:agent"])
        self.assertEqual(caught.exception.code, 2)

    def test_agent_still_required_without_a_list_flag(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--suite", "examples/calculator.json"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
