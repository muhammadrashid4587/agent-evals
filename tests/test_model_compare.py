"""Tests for per-model comparison across agents (model_compare.py)."""

import json
import os
import tempfile
import unittest

from agent_evals import Suite, run_suite
from agent_evals.evals import Case
from agent_evals.model_compare import (
    compare_models,
    format_model_comparison,
    main,
)


def _suite(cases=None):
    return Suite.from_dict(
        {
            "name": "calc",
            "version": "1.0",
            "cases": cases
            if cases is not None
            else [
                {
                    "id": "add",
                    "prompt": "2+2",
                    "scorer": {"type": "exact", "expected": "4"},
                },
                {
                    "id": "sub",
                    "prompt": "5-3",
                    "scorer": {"type": "exact", "expected": "2"},
                },
            ],
        }
    )


def _good(prompt):
    """An agent that answers both cases correctly, reporting tokens."""
    answers = {"2+2": ("4", 10), "5-3": ("2", 12)}
    return answers[prompt]


def _bad(prompt):
    """An agent that answers both cases wrong, reporting tokens."""
    return ("0", 50)


def _mixed(prompt):
    """An agent that gets the first case right and the second wrong.

    Slightly slower than ``_good`` on purpose, so a tied case is won
    by ``_good`` on the latency tie-break, deterministically.
    """
    import time

    time.sleep(0.01)
    return ("4", 20) if prompt == "2+2" else ("9", 20)


class TestCompareModels(unittest.TestCase):
    def test_runs_same_suite_against_each_agent(self):
        comp = compare_models(_suite(), {"good": _good, "bad": _bad})
        self.assertEqual(comp.suite, "calc")
        self.assertEqual(comp.suite_version, "1.0")
        self.assertEqual(comp.reports["good"].passed, 2)
        self.assertEqual(comp.reports["bad"].passed, 0)
        self.assertEqual(comp.reports["good"].total_tokens, 22)
        self.assertEqual(comp.reports["bad"].total_tokens, 100)

    def test_winner_is_best_agent_overall(self):
        comp = compare_models(_suite(), {"good": _good, "bad": _bad})
        self.assertEqual(comp.winner, "good")
        self.assertEqual(comp.ranking, ["good", "bad"])

    def test_ranking_breaks_ties_by_latency(self):
        # Both pass everything; the genuinely slower agent ranks last.
        import time

        fast = lambda prompt: ("4" if prompt == "2+2" else "2", 1)

        def slow(prompt):
            time.sleep(0.05)
            return ("4" if prompt == "2+2" else "2", 1)

        comp = compare_models(_suite(), {"slow": slow, "fast": fast})
        self.assertEqual(comp.ranking, ["fast", "slow"])

    def test_ranking_breaks_exact_ties_by_name(self):
        # With identical scores, pass rates, and latencies, the name
        # decides, so the ranking is deterministic. Built directly
        # from reports because measured latencies are never exactly
        # equal in a live run.
        from agent_evals.evals import Result, RunReport

        def report():
            return RunReport(
                suite="calc",
                started_at="2026-11-23T00:00:00+00:00",
                ended_at="2026-11-23T00:00:01+00:00",
                results=[
                    Result(
                        case_id="add",
                        passed=True,
                        detail="ok",
                        latency_ms=1.0,
                        score=1.0,
                    )
                ],
            )

        from agent_evals.model_compare import ModelComparison

        comp = ModelComparison(
            suite="calc", reports={"b": report(), "a": report()}
        )
        self.assertEqual(comp.ranking, ["a", "b"])
        self.assertEqual(comp.winner, "a")

    def test_per_case_winner_is_fastest_of_the_best(self):
        comp = compare_models(
            _suite(), {"good": _good, "mixed": _mixed, "bad": _bad}
        )
        by_id = {c.case_id: c for c in comp.cases}
        self.assertEqual(by_id["add"].winner, "good")
        self.assertEqual(by_id["sub"].winner, "good")
        self.assertEqual(comp.reports["mixed"].passed, 1)

    def test_case_every_agent_fails_has_no_winner(self):
        suite = _suite(
            [
                {
                    "id": "add",
                    "prompt": "2+2",
                    "scorer": {"type": "exact", "expected": "4"},
                },
                {
                    "id": "hard",
                    "prompt": "sqrt(-1)",
                    "scorer": {"type": "exact", "expected": "i"},
                },
            ]
        )
        comp = compare_models(suite, {"good": _good, "bad": _bad})
        by_id = {c.case_id: c for c in comp.cases}
        self.assertEqual(by_id["add"].winner, "good")
        self.assertIsNone(by_id["hard"].winner)

    def test_partial_credit_case_winner_is_highest_score(self):
        suite = Suite.from_dict(
            {
                "name": "s",
                "cases": [
                    {
                        "id": "all",
                        "prompt": "hi",
                        "scorer": {
                            "type": "contains_all",
                            "substrings": ["hello", "world"],
                        },
                    }
                ],
            }
        )
        comp = compare_models(
            suite,
            {
                "some": lambda prompt: "hello there",
                "none": lambda prompt: "goodbye",
            },
        )
        self.assertEqual(comp.cases[0].winner, "some")
        # The partial-credit agent still loses the case overall.
        self.assertEqual(comp.reports["some"].passed, 0)

    def test_single_agent_is_an_error(self):
        with self.assertRaises(ValueError):
            compare_models(_suite(), {"only": _good})

    def test_no_agents_is_an_error(self):
        with self.assertRaises(ValueError):
            compare_models(_suite(), {})

    def test_empty_suite_compares_cleanly(self):
        comp = compare_models(
            Suite.from_dict({"name": "empty", "cases": []}),
            {"a": _good, "b": _bad},
        )
        self.assertEqual(comp.cases, [])
        self.assertEqual(comp.winner, "a")
        self.assertIn("No cases in the suite.", format_model_comparison(comp))

    def test_to_dict_round_trips_through_json(self):
        comp = compare_models(
            _suite(), {"good": _good, "mixed": _mixed, "bad": _bad}
        )
        data = json.loads(json.dumps(comp.to_dict()))
        self.assertEqual(data["winner"], "good")
        self.assertEqual(data["ranking"], ["good", "mixed", "bad"])
        self.assertEqual(data["agents"]["good"]["passed"], 2)
        self.assertEqual(data["agents"]["good"]["total_tokens"], 22)
        self.assertEqual(data["agents"]["bad"]["total_tokens"], 100)
        by_id = {c["case_id"]: c for c in data["cases"]}
        self.assertEqual(by_id["add"]["winner"], "good")
        self.assertEqual(by_id["sub"]["winner"], "good")
        self.assertEqual(by_id["add"]["agents"]["mixed"]["score"], 1.0)
        self.assertEqual(by_id["sub"]["agents"]["mixed"]["score"], 0.0)

    def test_format_shows_ranking_and_per_case_table(self):
        comp = compare_models(
            _suite(), {"good": _good, "mixed": _mixed, "bad": _bad}
        )
        text = format_model_comparison(comp)
        self.assertIn("Suite: calc", text)
        self.assertIn("1. good", text)
        self.assertIn("<-- winner", text)
        self.assertIn("add", text)
        self.assertIn("PASS 1.00", text)
        self.assertIn("FAIL 0.00", text)
        self.assertIn("Case wins: good 2, mixed 0, bad 0", text)

    def test_format_marks_no_winner_case_with_dash(self):
        suite = _suite(
            [
                {
                    "id": "hard",
                    "prompt": "sqrt(-1)",
                    "scorer": {"type": "exact", "expected": "i"},
                }
            ]
        )
        comp = compare_models(suite, {"good": _good, "bad": _bad})
        text = format_model_comparison(comp)
        self.assertIn("Case wins: good 0, bad 0", text)
        self.assertIn("hard", text)


class TestLosersToBaseline(unittest.TestCase):
    def test_loser_is_agent_the_baseline_beats(self):
        comp = compare_models(_suite(), {"good": _good, "bad": _bad})
        self.assertEqual(comp.losers_to("good"), ["bad"])
        self.assertEqual(comp.losers_to("bad"), [])

    def test_only_agents_strictly_beaten_are_losers(self):
        comp = compare_models(
            _suite(), {"good": _good, "mixed": _mixed, "bad": _bad}
        )
        # mixed beats bad but loses to good; good beats both.
        self.assertEqual(comp.losers_to("mixed"), ["bad"])
        self.assertEqual(comp.losers_to("good"), ["mixed", "bad"])

    def test_exact_tie_is_not_a_loss(self):
        # Identical reports: the ranking still orders them by name,
        # but neither agent has lost to the other.
        from agent_evals.evals import Result, RunReport

        def report():
            return RunReport(
                suite="calc",
                started_at="2026-11-23T00:00:00+00:00",
                ended_at="2026-11-23T00:00:01+00:00",
                results=[
                    Result(
                        case_id="add",
                        passed=True,
                        detail="ok",
                        latency_ms=1.0,
                        score=1.0,
                    )
                ],
            )

        from agent_evals.model_compare import ModelComparison

        comp = ModelComparison(
            suite="calc", reports={"b": report(), "a": report()}
        )
        self.assertEqual(comp.ranking, ["a", "b"])
        self.assertEqual(comp.losers_to("a"), [])
        self.assertEqual(comp.losers_to("b"), [])

    def test_unknown_baseline_is_an_error(self):
        comp = compare_models(_suite(), {"good": _good, "bad": _bad})
        with self.assertRaises(ValueError):
            comp.losers_to("nope")


class TestModelCompareCli(unittest.TestCase):
    def _write_suite(self, tmp):
        path = os.path.join(tmp, "suite.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_suite().to_dict(), f)
        return path

    def test_cli_compares_example_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            out = os.path.join(tmp, "compare.json")
            code = main(
                [
                    "--suite",
                    suite,
                    "--agent",
                    "calc=examples.calculator_agent:agent",
                    "--agent",
                    "kv=examples.kv_agent:agent",
                    "--json",
                    out,
                ]
            )
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["winner"], "calc")
            self.assertEqual(data["ranking"], ["calc", "kv"])
            self.assertEqual(data["agents"]["calc"]["passed"], 2)
            self.assertEqual(data["agents"]["kv"]["passed"], 0)

    def test_cli_rejects_one_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--suite",
                        suite,
                        "--agent",
                        "calc=examples.calculator_agent:agent",
                    ]
                )

    def test_cli_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--suite",
                        suite,
                        "--agent",
                        "a=examples.calculator_agent:agent",
                        "--agent",
                        "a=examples.kv_agent:agent",
                    ]
                )

    def test_cli_rejects_bad_agent_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--suite",
                        suite,
                        "--agent",
                        "a=examples.no_such_module:agent",
                        "--agent",
                        "b=examples.kv_agent:agent",
                    ]
                )

    def test_cli_fail_on_lose_exits_1_when_baseline_wins(self):
        # calc passes both cases, kv passes none: kv loses to calc.
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            code = main(
                [
                    "--suite",
                    suite,
                    "--agent",
                    "calc=examples.calculator_agent:agent",
                    "--agent",
                    "kv=examples.kv_agent:agent",
                    "--fail-on-lose",
                    "calc",
                ]
            )
            self.assertEqual(code, 1)

    def test_cli_fail_on_lose_exits_0_when_nobody_loses(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            code = main(
                [
                    "--suite",
                    suite,
                    "--agent",
                    "calc=examples.calculator_agent:agent",
                    "--agent",
                    "kv=examples.kv_agent:agent",
                    "--fail-on-lose",
                    "kv",
                ]
            )
            self.assertEqual(code, 0)

    def test_cli_fail_on_lose_rejects_unknown_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            suite = self._write_suite(tmp)
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--suite",
                        suite,
                        "--agent",
                        "calc=examples.calculator_agent:agent",
                        "--agent",
                        "kv=examples.kv_agent:agent",
                        "--fail-on-lose",
                        "nope",
                    ]
                )


class TestPackageExports(unittest.TestCase):
    def test_compare_models_exported_from_package(self):
        import agent_evals

        self.assertIs(agent_evals.compare_models, compare_models)
        self.assertIs(
            agent_evals.format_model_comparison, format_model_comparison
        )

    def test_case_comparison_matches_run_suite_results(self):
        # The comparison's per-case rows must agree with a plain
        # run_suite of the same agent on the same suite.
        suite = _suite()
        comp = compare_models(suite, {"good": _good, "bad": _bad})
        report = run_suite(suite, _good)
        row = comp.cases[0]
        self.assertEqual(
            row.results["good"].score, report.results[0].score
        )
        self.assertEqual(
            row.results["good"].passed, report.results[0].passed
        )
        self.assertEqual(
            row.results["good"].case_id, report.results[0].case_id
        )


if __name__ == "__main__":
    unittest.main()
