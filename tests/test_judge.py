"""Tests for agent-evals: the judge / LLM-as-judge scorer."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_evals.evals import Case, Suite, run_suite, score_case
from agent_evals.judge import (
    JudgeVerdict,
    coerce_verdict,
    make_http_judge,
    stub_judge,
)


def freeform_agent(prompt):
    """Canned support agent for judge-case tests."""
    return (
        "Our refund policy gives you 30 days from purchase to request "
        "a refund. Contact support with your order number and we will "
        "process it within 5 business days."
    )


RUBRIC = {
    "criteria": [
        {"name": "names the refund window", "must_mention": ["30 days", "refund"]},
        {"name": "offers the next step", "must_mention": ["Contact support"]},
    ],
}


def _case(scorer, prompt="What is your refund policy?"):
    return Case(id="c1", prompt=prompt, scorer=scorer)


def _judge_case(rubric=None, judge=None, prompt="What is your refund policy?"):
    spec = {"type": "judge", "rubric": rubric if rubric is not None else RUBRIC}
    if judge is not None:
        spec["judge"] = judge
    return _case(spec, prompt)


class TestJudgeVerdict(unittest.TestCase):
    def test_valid_verdict(self):
        v = JudgeVerdict(passed=True, score=0.75, rationale="mostly right")
        self.assertEqual(v.to_dict()["score"], 0.75)

    def test_score_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            JudgeVerdict(passed=True, score=1.5)

    def test_non_bool_passed_raises(self):
        with self.assertRaises(ValueError):
            JudgeVerdict(passed=1, score=0.5)


class TestCoerceVerdict(unittest.TestCase):
    def test_verdict_passthrough(self):
        v = JudgeVerdict(passed=False, score=0.25, rationale="weak")
        self.assertIs(coerce_verdict(v), v)

    def test_dict_verdict(self):
        v = coerce_verdict({"passed": True, "score": 1.0, "rationale": "nailed it"})
        self.assertTrue(v.passed)
        self.assertEqual(v.score, 1.0)
        self.assertEqual(v.rationale, "nailed it")

    def test_dict_without_rationale(self):
        v = coerce_verdict({"passed": False, "score": 0.0})
        self.assertEqual(v.rationale, "")

    def test_dict_missing_score_raises(self):
        with self.assertRaises(ValueError):
            coerce_verdict({"passed": True})

    def test_plain_string_raises(self):
        with self.assertRaises(ValueError):
            coerce_verdict("looks good")


class TestStubJudge(unittest.TestCase):
    def test_all_criteria_met_passes(self):
        v = stub_judge("p", freeform_agent("p"), RUBRIC)
        self.assertTrue(v.passed)
        self.assertEqual(v.score, 1.0)
        self.assertIn("met:", v.rationale)

    def test_one_criterion_missed_partial_score(self):
        v = stub_judge(
            "p",
            "Our refund policy gives you 30 days from purchase.",
            RUBRIC,
        )
        self.assertFalse(v.passed)
        self.assertEqual(v.score, 0.5)
        self.assertIn("missed:", v.rationale)
        self.assertIn("offers the next step", v.rationale)

    def test_lower_threshold_passes_on_partial(self):
        rubric = dict(RUBRIC, pass_threshold=0.5)
        v = stub_judge(
            "p",
            "Our refund policy gives you 30 days from purchase.",
            rubric,
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.score, 0.5)

    def test_weights_change_the_score(self):
        rubric = {
            "criteria": [
                {"name": "big", "must_mention": ["30 days"], "weight": 3},
                {"name": "small", "must_mention": ["Contact support"], "weight": 1},
            ]
        }
        v = stub_judge("p", "30 days only", rubric)
        self.assertEqual(v.score, 0.75)

    def test_forbidden_substring_always_fails(self):
        rubric = dict(RUBRIC, must_not_mention=["as an AI"])
        v = stub_judge("p", "As an AI, our refund policy gives you 30 days. Contact support.", rubric)
        # Case-sensitive: "As an AI" is not "as an AI", so this passes.
        self.assertTrue(v.passed)
        v = stub_judge(
            "p",
            "as an AI, our refund policy gives you 30 days. Contact support.",
            rubric,
        )
        self.assertFalse(v.passed)
        self.assertIn("forbidden:", v.rationale)

    def test_empty_rubric_raises(self):
        with self.assertRaises(ValueError):
            stub_judge("p", "anything", {})

    def test_criterion_without_must_mention_raises(self):
        with self.assertRaises(ValueError):
            stub_judge("p", "anything", {"criteria": [{"name": "x"}]})

    def test_threshold_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            stub_judge("p", "anything", dict(RUBRIC, pass_threshold=1.5))


class TestJudgeScorer(unittest.TestCase):
    def test_stub_judge_is_the_default(self):
        ok, detail, score = score_case(_judge_case(), freeform_agent("p"))
        self.assertTrue(ok)
        self.assertEqual(score, 1.0)
        self.assertIn("stub", detail)
        self.assertIn("met:", detail)

    def test_partial_credit_flows_through(self):
        ok, _, score = score_case(
            _judge_case(),
            "Our refund policy gives you 30 days from purchase.",
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.5)

    def test_missing_rubric_raises(self):
        with self.assertRaises(ValueError):
            score_case(_case({"type": "judge"}), "anything")

    def test_unknown_judge_raises_and_names_known(self):
        with self.assertRaises(ValueError) as ctx:
            score_case(_judge_case(judge="gpt-5"), "anything")
        self.assertIn("stub", str(ctx.exception))

    def test_custom_judge_from_registry(self):
        def strict_judge(prompt, output, rubric):
            return {"passed": False, "score": 0.0, "rationale": "never good enough"}

        ok, detail, score = score_case(
            _judge_case(judge="strict"),
            freeform_agent("p"),
            judges={"strict": strict_judge},
        )
        self.assertFalse(ok)
        self.assertEqual(score, 0.0)
        self.assertIn("never good enough", detail)

    def test_judge_returning_verdict_object(self):
        def verdict_judge(prompt, output, rubric):
            return JudgeVerdict(passed=True, score=0.9, rationale="solid")

        ok, _, score = score_case(
            _judge_case(judge="v"),
            "anything",
            judges={"v": verdict_judge},
        )
        self.assertTrue(ok)
        self.assertEqual(score, 0.9)

    def test_judge_raising_is_a_scorer_error_in_a_run(self):
        def broken_judge(prompt, output, rubric):
            raise RuntimeError("model exploded")

        suite = Suite(name="s", cases=[_judge_case(judge="broken")])
        report = run_suite(
            suite, freeform_agent, judges={"broken": broken_judge}
        )
        self.assertFalse(report.results[0].passed)
        self.assertIn("scorer error", report.results[0].detail)
        self.assertIn("model exploded", report.results[0].detail)


class _VerdictHandler(BaseHTTPRequestHandler):
    """Tiny judge endpoint for tests: echoes a verdict from the request."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/boom":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"internal error")
            return
        if self.path == "/garbage":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"not json{")
            return
        if self.path == "/nonsense":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"verdict": "fine"}).encode())
            return
        output = body.get("output", "")
        passed = "30 days" in output
        payload = json.dumps(
            {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "rationale": "mentions the window" if passed else "no window",
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TestHttpJudge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _VerdictHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path=""):
        return f"http://127.0.0.1:{self.port}{path}"

    def test_verdict_from_endpoint(self):
        judge = make_http_judge(self._url())
        v = judge("p", freeform_agent("p"), RUBRIC)
        self.assertTrue(v.passed)
        self.assertEqual(v.score, 1.0)
        self.assertEqual(v.rationale, "mentions the window")

    def test_failing_verdict_from_endpoint(self):
        judge = make_http_judge(self._url())
        v = judge("p", "no idea", RUBRIC)
        self.assertFalse(v.passed)
        self.assertEqual(v.score, 0.0)

    def test_http_error_raises(self):
        judge = make_http_judge(self._url("/boom"))
        with self.assertRaises(ValueError) as ctx:
            judge("p", "x", RUBRIC)
        self.assertIn("500", str(ctx.exception))

    def test_invalid_json_raises(self):
        judge = make_http_judge(self._url("/garbage"))
        with self.assertRaises(ValueError):
            judge("p", "x", RUBRIC)

    def test_non_verdict_response_raises(self):
        judge = make_http_judge(self._url("/nonsense"))
        with self.assertRaises(ValueError):
            judge("p", "x", RUBRIC)

    def test_unreachable_endpoint_raises(self):
        judge = make_http_judge("http://127.0.0.1:1/nope", timeout=1)
        with self.assertRaises(ValueError):
            judge("p", "x", RUBRIC)

    def test_http_judge_wires_into_run_suite(self):
        suite = Suite(name="s", cases=[_judge_case(judge="http")])
        report = run_suite(
            suite, freeform_agent, judges={"http": make_http_judge(self._url())}
        )
        self.assertTrue(report.results[0].passed)
        self.assertIn("mentions the window", report.results[0].detail)


class TestJudgeInRun(unittest.TestCase):
    def test_weighted_score_uses_judge_partial_credit(self):
        cases = [
            _judge_case(),
            _case({"type": "exact", "expected": "OK"}, prompt="say OK"),
        ]
        cases[0].id = "judged"
        cases[1].id = "plain"
        suite = Suite(name="s", cases=cases)
        agent = lambda prompt: (  # noqa: E731
            freeform_agent(prompt)
            if "refund" in prompt
            else "OK"
        )
        report = run_suite(suite, agent)
        self.assertEqual(report.passed, 2)
        self.assertEqual(report.weighted_score, 1.0)


if __name__ == "__main__":
    unittest.main()
