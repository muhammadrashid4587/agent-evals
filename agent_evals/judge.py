"""LLM-as-judge scoring for agent-evals.

The eight deterministic scorers in :mod:`agent_evals.evals` can only check
what is mechanically checkable: exact strings, substrings, regexes, JSON
shape. Free-form agent output — an explanation, a summary, a plan — needs
a judge that reads the answer and scores it against a rubric.

A judge is a plain callable::

    judge(prompt, output, rubric) -> JudgeVerdict

where ``rubric`` is the dict from the case's scorer spec and the return
value is a :class:`JudgeVerdict` (or a dict with the same keys). The
harness ships two judges and lets you bring your own:

* :func:`stub_judge` — the built-in deterministic judge, used when a
  case's scorer spec does not name a judge. It scores the output
  against rubric criteria mechanically (each criterion lists
  substrings the output must mention), so judge cases run offline with
  no model and no network, and are reproducible. Use it in tests and
  demos, and as the reference implementation for a real judge.
* :func:`make_http_judge` — a factory that returns a judge callable
  posting ``{"prompt", "output", "rubric"}`` as JSON to an HTTP
  endpoint and expecting ``{"passed", "score", "rationale"}`` back.
  Point it at any LLM-judge service you run — a small wrapper around
  an LLM API, a local model server, a teammate's endpoint — and the
  harness treats its verdict like any other scorer result. The harness
  itself still makes no model calls and holds no API keys.

A case uses the judge scorer like this::

    {"type": "judge", "judge": "stub", "rubric": {
        "criteria": [
            {"name": "names the refund window",
             "must_mention": ["30 days", "refund"]},
            {"name": "offers the next step",
             "must_mention": ["contact support"]},
        ],
        "must_not_mention": ["as an AI"],
        "pass_threshold": 1.0,
    }}

The ``"judge"`` key names a judge in the registry passed to
:func:`agent_evals.run_suite` (``judges={"my-judge": my_judge}``) or to
:func:`agent_evals.score_case`; it defaults to ``"stub"``. The CLI
registers a judge from ``--judge module:attribute`` under the name
``"custom"`` and one from ``--judge-url URL`` under the name ``"http"``.

A judge's verdict is a score in [0, 1] plus a pass/fail and a rationale
string, so judge cases earn partial credit like ``contains_all`` does:
a case that meets half the rubric's weighted criteria earns half its
weight, and the rationale lands in the run report's detail so a human
can see *why* the judge scored it that way. A judge that raises, or
returns something that is not a verdict, is a scorer error: the case
fails and the run continues, exactly like a malformed regex.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

__all__ = [
    "DEFAULT_JUDGES",
    "JudgeVerdict",
    "coerce_verdict",
    "make_http_judge",
    "stub_judge",
]

# A judge is a callable (prompt, output, rubric) -> JudgeVerdict | dict.
Judge = Callable[[str, str, dict], Any]


@dataclass
class JudgeVerdict:
    """What a judge thought of one agent output.

    ``passed`` is the judge's verdict, ``score`` is the fraction of the
    rubric earned (in [0, 1]), and ``rationale`` is a human-readable
    reason — it lands in the run report's detail.
    """

    passed: bool
    score: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError(f"judge verdict 'passed' must be a bool, got {self.passed!r}")
        try:
            self.score = float(self.score)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"judge verdict 'score' must be a number, got {self.score!r}"
            ) from exc
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"judge verdict 'score' must be in [0, 1], got {self.score!r}"
            )

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "rationale": self.rationale,
        }


def coerce_verdict(result: Any) -> JudgeVerdict:
    """Normalise whatever a judge returned into a :class:`JudgeVerdict`.

    Accepts a ``JudgeVerdict`` or a dict with ``passed`` and ``score``
    keys (``rationale`` optional). Anything else raises ``ValueError``.
    """
    if isinstance(result, JudgeVerdict):
        return result
    if isinstance(result, dict):
        if "passed" not in result or "score" not in result:
            raise ValueError(
                "judge must return a verdict with 'passed' and 'score', "
                f"got {result!r}"
            )
        return JudgeVerdict(
            passed=result["passed"],
            score=result["score"],
            rationale=str(result.get("rationale", "")),
        )
    raise ValueError(
        "judge must return a JudgeVerdict or a dict with 'passed' and "
        f"'score', got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# The built-in deterministic stub judge
# ---------------------------------------------------------------------------


def _parse_criterion(criterion: Any, index: int) -> tuple:
    """Normalise one rubric criterion to (name, must_mention, weight)."""
    if not isinstance(criterion, dict):
        raise ValueError(f"rubric criterion {index} must be an object, got {criterion!r}")
    name = str(criterion.get("name", f"criterion {index + 1}"))
    must_mention = criterion.get("must_mention")
    if not isinstance(must_mention, list) or not must_mention:
        raise ValueError(
            f"rubric criterion {name!r} requires a non-empty "
            "'must_mention' list of substrings"
        )
    weight = criterion.get("weight", 1.0)
    try:
        weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"rubric criterion {name!r}: weight must be a number, got {weight!r}"
        ) from exc
    if weight <= 0:
        raise ValueError(
            f"rubric criterion {name!r}: weight must be positive, got {weight!r}"
        )
    return name, [str(s) for s in must_mention], weight


def stub_judge(prompt: str, output: str, rubric: dict) -> JudgeVerdict:
    """Deterministic built-in judge: score the output against a rubric.

    The rubric is::

        {"criteria": [{"name": ..., "must_mention": [...], "weight": 1.0}],
         "must_not_mention": [...],          # optional, forbidden substrings
         "pass_threshold": 1.0}               # optional, default 1.0

    A criterion is met when the output contains every listed substring
    (same case-sensitive matching as the other scorers). The score is
    the weight-weighted fraction of criteria met. The case passes when
    the score reaches ``pass_threshold`` and no forbidden substring
    appears — a forbidden substring always fails the case, whatever the
    score. The rationale names the criteria met and missed, so the
    verdict is explainable without re-running anything.
    """
    if not isinstance(rubric, dict):
        raise ValueError(f"judge rubric must be an object, got {rubric!r}")
    criteria = [_parse_criterion(c, i) for i, c in enumerate(rubric.get("criteria", []))]
    must_not_mention = rubric.get("must_not_mention", [])
    if not isinstance(must_not_mention, list):
        raise ValueError("'must_not_mention' must be a list of substrings")
    forbidden = [s for s in (str(s) for s in must_not_mention) if s in output]
    if not criteria and not must_not_mention:
        raise ValueError(
            "judge rubric needs at least one criterion or a "
            "'must_not_mention' list"
        )
    pass_threshold = rubric.get("pass_threshold", 1.0)
    try:
        pass_threshold = float(pass_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'pass_threshold' must be a number, got {pass_threshold!r}"
        ) from exc
    if not 0.0 <= pass_threshold <= 1.0:
        raise ValueError(
            f"'pass_threshold' must be in [0, 1], got {pass_threshold!r}"
        )
    total_weight = sum(w for _, _, w in criteria)
    met, missed = [], []
    earned = 0.0
    for name, substrings, weight in criteria:
        missing = [s for s in substrings if s not in output]
        if not missing:
            met.append(name)
            earned += weight
        else:
            missed.append(f"{name} (missing {', '.join(repr(s) for s in missing)})")
    score = (earned / total_weight) if total_weight else 1.0
    parts = []
    if met:
        parts.append("met: " + "; ".join(met))
    if missed:
        parts.append("missed: " + "; ".join(missed))
    if forbidden:
        parts.append(
            "forbidden: " + ", ".join(repr(s) for s in forbidden)
        )
    rationale = "; ".join(parts) if parts else "no criteria to check"
    passed = score >= pass_threshold and not forbidden
    return JudgeVerdict(passed=passed, score=score, rationale=rationale)


# ---------------------------------------------------------------------------
# HTTP judge: call any LLM-judge endpoint over HTTP
# ---------------------------------------------------------------------------


def make_http_judge(
    url: str, timeout: float = 30.0, headers: Optional[dict] = None
) -> Judge:
    """Build a judge callable that POSTs to an HTTP LLM-judge endpoint.

    Each call POSTs ``{"prompt": ..., "output": ..., "rubric": ...}`` as
    JSON to ``url`` and expects a JSON response with ``passed`` (bool),
    ``score`` (number in [0, 1]) and ``rationale`` (string). A non-2xx
    status, an unreadable body, or a response that is not a verdict
    raises ``ValueError``, which the runner records as a scorer error
    for that case. ``timeout`` is in seconds; ``headers`` are merged
    over the default ``Content-Type: application/json``.
    """
    if not isinstance(url, str) or not url:
        raise ValueError(f"HTTP judge needs a URL, got {url!r}")
    if timeout <= 0:
        raise ValueError(f"HTTP judge timeout must be positive, got {timeout!r}")

    def http_judge(prompt: str, output: str, rubric: dict) -> JudgeVerdict:
        payload = json.dumps(
            {"prompt": prompt, "output": output, "rubric": rubric}
        ).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            url, data=payload, headers=request_headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                if not 200 <= response.status < 300:
                    raise ValueError(
                        f"judge endpoint {url} returned HTTP {response.status}"
                    )
        except urllib.error.HTTPError as exc:
            raise ValueError(
                f"judge endpoint {url} returned HTTP {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ValueError(
                f"could not reach judge endpoint {url}: {exc.reason}"
            ) from exc
        try:
            result = json.loads(body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"judge endpoint {url} returned invalid JSON: {exc}"
            ) from exc
        return coerce_verdict(result)

    return http_judge


#: Judges available when a run does not supply its own registry.
DEFAULT_JUDGES: dict = {"stub": stub_judge}
