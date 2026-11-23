"""Core of agent-evals: suites, deterministic scorers, the runner, reports.

Design
------
* A ``Suite`` is a name plus a list of ``Case`` objects. A case is one
  prompt plus one scorer spec. A case may also carry an optional
  ``timeout`` in seconds: if the agent takes longer, the case fails as
  timed out and the run moves on to the next case.
* A scorer spec is a plain dict, so suites can live in JSON files::

      {"type": "exact", "expected": "19"}
      {"type": "contains", "substring": "hello"}
      {"type": "contains_all", "substrings": ["hello", "world"]}
      {"type": "contains_any", "substrings": ["hello", "hi"]}
      {"type": "regex", "pattern": "^\\d+$"}
      {"type": "json_schema", "schema": {"name": "str", "age": "int"}}
      {"type": "json_path", "path": "$.result.total", "expected": 42}
      {"type": "numeric", "expected": 3.14159, "tolerance": 0.001}
      {"type": "judge", "judge": "stub", "rubric": {"criteria": [...]}}

* A case may carry a ``weight`` (default 1.0): how much of the suite's
  total score it is worth. Scorers also return a score in [0, 1] — 1.0
  on a pass, 0.0 on a fail — and ``contains_all`` earns partial credit
  equal to the fraction of required substrings found. The report totals
  ``earned_weight`` over ``total_weight`` as ``weighted_score``.

* A case may carry a ``max_latency_ms`` budget: if the agent's answer
  passes its checks but took longer than the budget, the case is
  reported as a budget violation instead of a pass (``passed`` is
  false, ``budget_violated`` is true, and the latency and the budget
  are in the detail). A case that fails its checks is never also a
  budget violation, and a timed-out case is not one either.

* The agent under test is a plain callable ``agent(prompt) -> ...``. It may
  return:
    - a ``str`` (the agent's output; token count unknown),
    - a ``(output, tokens)`` tuple (output str, token count int),
    - a ``dict`` with an ``"output"`` key and an optional ``"tokens"`` key.
* Every scorer is deterministic: same output, same verdict, every run —
  except the ``judge`` scorer, which calls a judge you supply. The
  built-in ``stub`` judge is deterministic (it scores the output
  against rubric criteria mechanically, with no model and no network),
  and any other judge is a plain callable you register with
  ``run_suite`` — see :mod:`agent_evals.judge`.
"""

from __future__ import annotations

import concurrent.futures
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .judge import DEFAULT_JUDGES, coerce_verdict

__all__ = [
    "Case",
    "Result",
    "RunReport",
    "Suite",
    "format_summary",
    "load_report",
    "load_suite",
    "run_suite",
    "score_case",
]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

_TYPE_NAMES = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "number": (int, float),
}


# ---------------------------------------------------------------------------
# Suites and cases
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One eval case: a prompt and the scorer that judges the answer.

    ``timeout`` is an optional per-case limit in seconds: if the agent
    takes longer, the case fails as timed out and the run continues.
    ``weight`` is how much of the suite's total score the case is worth
    (default 1.0); a case earns ``score * weight`` points, where the
    score is 1.0 on a pass and may be a fraction on a fail when the
    scorer gives partial credit.
    ``max_latency_ms`` is an optional per-case latency budget in
    milliseconds: a case that passes its checks but takes longer than
    the budget is reported as a budget violation instead of a pass.
    """

    id: str
    prompt: str
    scorer: dict
    timeout: Optional[float] = None
    weight: float = 1.0
    max_latency_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        timeout = data.get("timeout")
        if timeout is not None:
            try:
                timeout = float(timeout)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"case {data.get('id')!r}: timeout must be a number of "
                    f"seconds, got {timeout!r}"
                ) from exc
            if timeout <= 0:
                raise ValueError(
                    f"case {data.get('id')!r}: timeout must be positive, "
                    f"got {timeout!r}"
                )
        weight = data.get("weight", 1.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"case {data.get('id')!r}: weight must be a number, "
                f"got {weight!r}"
            ) from exc
        if weight <= 0:
            raise ValueError(
                f"case {data.get('id')!r}: weight must be positive, "
                f"got {weight!r}"
            )
        max_latency_ms = data.get("max_latency_ms")
        if max_latency_ms is not None:
            try:
                max_latency_ms = float(max_latency_ms)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"case {data.get('id')!r}: max_latency_ms must be a "
                    f"number of milliseconds, got {max_latency_ms!r}"
                ) from exc
            if max_latency_ms <= 0:
                raise ValueError(
                    f"case {data.get('id')!r}: max_latency_ms must be "
                    f"positive, got {max_latency_ms!r}"
                )
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            scorer=dict(data["scorer"]),
            timeout=timeout,
            weight=weight,
            max_latency_ms=max_latency_ms,
        )

    def to_dict(self) -> dict:
        data = {"id": self.id, "prompt": self.prompt, "scorer": self.scorer}
        if self.timeout is not None:
            data["timeout"] = self.timeout
        if self.weight != 1.0:
            data["weight"] = self.weight
        if self.max_latency_ms is not None:
            data["max_latency_ms"] = self.max_latency_ms
        return data


@dataclass
class Suite:
    """A named list of cases."""

    name: str
    cases: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Suite":
        return cls(
            name=str(data.get("name", "suite")),
            cases=[Case.from_dict(c) for c in data.get("cases", [])],
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "cases": [c.to_dict() for c in self.cases]}


def load_suite(path: str) -> Suite:
    """Load a suite from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return Suite.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def score_exact(output: str, expected: str) -> tuple:
    """Pass when the output equals the expected string (ignoring
    surrounding whitespace on both sides)."""
    ok = output.strip() == expected.strip()
    detail = "exact match" if ok else f"expected {expected!r}, got {output!r}"
    return ok, detail, 1.0 if ok else 0.0


def score_contains(output: str, substring: str) -> tuple:
    """Pass when the output contains the substring."""
    ok = substring in output
    detail = "substring found" if ok else f"{substring!r} not in output"
    return ok, detail, 1.0 if ok else 0.0


def _as_substring_list(spec: dict, kind: str) -> list:
    """Coerce a scorer spec's ``substrings`` value to a list of strings."""
    substrings = spec.get("substrings")
    if not isinstance(substrings, list):
        raise ValueError(f"{kind} scorer requires a 'substrings' list")
    return [str(s) for s in substrings]


def score_contains_all(output: str, substrings: list) -> tuple:
    """Pass when the output contains every listed substring.

    An empty list passes: there is nothing required to be present.
    Partial credit: the score is the fraction of listed substrings
    found, so a case that gets most of the way there still earns most
    of its weight. A case only counts as passed when every substring
    is found (score 1.0).
    """
    missing = [s for s in substrings if s not in output]
    if not substrings:
        return True, "all substrings found", 1.0
    score = (len(substrings) - len(missing)) / len(substrings)
    if not missing:
        return True, "all substrings found", 1.0
    return (
        False,
        f"missing substrings: {', '.join(repr(s) for s in missing)}",
        score,
    )


def score_contains_any(output: str, substrings: list) -> tuple:
    """Pass when the output contains at least one listed substring.

    An empty list fails: no listed substring can be present.
    """
    found = [s for s in substrings if s in output]
    if found:
        return True, f"found {found[0]!r}", 1.0
    return (
        False,
        "none of the substrings found: "
        + ", ".join(repr(s) for s in substrings),
        0.0,
    )


def score_regex(output: str, pattern: str) -> tuple:
    """Pass when the regex matches anywhere in the output."""
    try:
        match = re.search(pattern, output)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
    ok = match is not None
    detail = "regex matched" if ok else f"pattern {pattern!r} did not match"
    return ok, detail, 1.0 if ok else 0.0


def _check_shape(value: Any, schema: Any, path: str = "$") -> Optional[str]:
    """Return the first shape mismatch as a string, or None if it matches.

    A schema is either:
      * a type name (``"str"``, ``"int"``, ``"float"``, ``"bool"``,
        ``"list"``, ``"dict"``, ``"number"``),
      * a dict of ``field -> subschema`` for objects (a field name ending
        in ``"?"`` is optional),
      * a one-element list ``[subschema]`` for arrays of that shape.
    """
    if isinstance(schema, str):
        want = _TYPE_NAMES.get(schema)
        if want is None:
            raise ValueError(f"unknown type name in schema: {schema!r}")
        # bool is a subclass of int; treat them as distinct types.
        if isinstance(value, bool) and want is not bool:
            return f"{path}: expected {schema}, got bool"
        if not isinstance(value, want):
            return f"{path}: expected {schema}, got {type(value).__name__}"
        return None
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            return f"{path}: expected object, got {type(value).__name__}"
        for key, subschema in schema.items():
            optional = key.endswith("?")
            field = key[:-1] if optional else key
            if field not in value:
                if optional:
                    continue
                return f"{path}: missing key {field!r}"
            err = _check_shape(value[field], subschema, f"{path}.{field}")
            if err is not None:
                return err
        return None
    if isinstance(schema, list):
        if len(schema) != 1:
            raise ValueError("array schema must have exactly one element")
        if not isinstance(value, list):
            return f"{path}: expected array, got {type(value).__name__}"
        for i, item in enumerate(value):
            err = _check_shape(item, schema[0], f"{path}[{i}]")
            if err is not None:
                return err
        return None
    raise ValueError(f"invalid schema: {schema!r}")


def score_json_schema(output: str, schema: Any) -> tuple:
    """Pass when the output is valid JSON with the given shape."""
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"output is not valid JSON: {exc}", 0.0
    err = _check_shape(value, schema)
    if err is None:
        return True, "JSON shape matches", 1.0
    return False, err, 0.0


def _parse_json_path(path: str) -> list:
    """Parse a JSON path into steps: dict keys (str) and indices (int).

    Accepts an optional leading ``$`` (``$.a.b`` or ``$a.b``) or a bare
    dotted path (``a.b``). Array indices use bracket syntax: ``a[0].b``.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("json_path scorer requires a non-empty 'path' string")
    rest = path[1:] if path.startswith("$") else path
    if rest.startswith("."):
        rest = rest[1:]
    if not rest:
        return []
    steps: list = []
    for segment in rest.split("."):
        if not segment:
            raise ValueError(f"invalid JSON path {path!r}: empty segment")
        key, _, brackets = segment.partition("[")
        if key:
            steps.append(key)
        while brackets:
            index, sep, brackets = brackets.partition("]")
            if not sep or not index.isdigit():
                raise ValueError(f"invalid JSON path {path!r}: bad index")
            steps.append(int(index))
            if brackets and not brackets.startswith("["):
                raise ValueError(f"invalid JSON path {path!r}: bad index")
            brackets = brackets[1:] if brackets.startswith("[") else brackets
    return steps


def _extract_json_path(value: Any, steps: list, path: str) -> tuple:
    """Walk ``steps`` through ``value``; return (found, extracted)."""
    current = value
    walked = "$"
    for step in steps:
        if isinstance(step, str):
            walked += f".{step}"
            if not isinstance(current, dict) or step not in current:
                return False, f"path {path!r}: no key {step!r} at {walked}"
            current = current[step]
        else:
            walked += f"[{step}]"
            if not isinstance(current, list) or step >= len(current):
                return False, f"path {path!r}: no index {step} at {walked}"
            current = current[step]
    return True, current


def _values_equal(got: Any, expected: Any, tolerance: float) -> bool:
    """Equality for a json_path comparison, with bool/int kept distinct.

    Numbers compare within ``tolerance`` when it is nonzero; everything
    else compares with plain equality.
    """
    if isinstance(got, bool) or isinstance(expected, bool):
        return got is expected
    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(got) - float(expected)) <= tolerance
    return got == expected


def score_json_path(
    output: str, path: str, expected: Any, tolerance: float = 0.0
) -> tuple:
    """Pass when the value at ``path`` in the output JSON equals ``expected``.

    ``path`` is a dotted JSON path with optional ``$`` prefix and ``[n]``
    array indices, e.g. ``$.result.total`` or ``items[0].name``. A missing
    key or index fails the case; a malformed path raises ``ValueError``.
    """
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"output is not valid JSON: {exc}", 0.0
    steps = _parse_json_path(path)
    found, extracted = _extract_json_path(value, steps, path)
    if not found:
        return False, extracted, 0.0
    if _values_equal(extracted, expected, float(tolerance)):
        return True, f"{path} == {expected!r}", 1.0
    return False, f"expected {path} == {expected!r}, got {extracted!r}", 0.0


def score_numeric(output: str, expected: float, tolerance: float = 0.0) -> tuple:
    """Pass when the first number in the output is within ``tolerance`` of
    ``expected``. The number may be embedded in surrounding text."""
    match = _NUMBER_RE.search(output)
    if match is None:
        return False, "no number in output", 0.0
    got = float(match.group(0))
    diff = abs(got - float(expected))
    ok = diff <= float(tolerance)
    detail = (
        f"got {got} (diff {diff})"
        if ok
        else f"expected {expected} ± {tolerance}, got {got} (diff {diff})"
    )
    return ok, detail, 1.0 if ok else 0.0


def score_judge(output: str, case: Case, judges: Optional[dict] = None) -> tuple:
    """Score one case's output with a judge.

    The scorer spec names the judge (default ``"stub"``) and carries a
    ``rubric`` dict::

        {"type": "judge", "judge": "stub", "rubric": {"criteria": [...]}}

    ``judges`` maps judge names to callables
    ``judge(prompt, output, rubric) -> JudgeVerdict | dict``; when it
    is None the built-in registry (:data:`agent_evals.judge.DEFAULT_JUDGES`,
    just the deterministic stub judge) is used. A judge that raises, or
    returns something that is not a verdict, is a scorer error. The
    detail carries the judge's rationale, so the report says *why* the
    output scored the way it did.
    """
    spec = case.scorer
    rubric = spec.get("rubric")
    if not isinstance(rubric, dict):
        raise ValueError("judge scorer requires a 'rubric' object")
    name = str(spec.get("judge", "stub"))
    registry = DEFAULT_JUDGES if judges is None else judges
    judge = registry.get(name)
    if judge is None:
        known = ", ".join(sorted(registry)) or "(none)"
        raise ValueError(
            f"unknown judge {name!r}; known judges: {known}"
        )
    if not callable(judge):
        raise ValueError(f"judge {name!r} is not callable")
    try:
        verdict = coerce_verdict(judge(case.prompt, output, rubric))
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - a failing judge is a scorer error
        raise ValueError(f"judge {name!r} failed: {exc}") from exc
    detail = f"judge {name!r}: score {verdict.score:.2f}"
    if verdict.rationale:
        detail += f" — {verdict.rationale}"
    return verdict.passed, detail, verdict.score


def score_case(case: Case, output: str, judges: Optional[dict] = None) -> tuple:
    """Score one case's output. Returns (passed, detail, score) where
    ``score`` is in [0, 1]: 1.0 on a pass, 0.0 on a fail, or a fraction
    when the scorer gives partial credit (see ``score_contains_all`` and
    ``score_judge``). ``judges`` is the registry of judge callables
    used by a ``judge`` scorer; None means the built-in registry."""
    spec = case.scorer
    kind = spec.get("type")
    if kind == "exact":
        return score_exact(output, str(spec["expected"]))
    if kind == "contains":
        return score_contains(output, str(spec["substring"]))
    if kind == "contains_all":
        return score_contains_all(output, _as_substring_list(spec, kind))
    if kind == "contains_any":
        return score_contains_any(output, _as_substring_list(spec, kind))
    if kind == "regex":
        return score_regex(output, str(spec["pattern"]))
    if kind == "json_schema":
        return score_json_schema(output, spec["schema"])
    if kind == "json_path":
        if "expected" not in spec:
            raise ValueError("json_path scorer requires an 'expected' value")
        return score_json_path(
            output,
            str(spec["path"]),
            spec["expected"],
            float(spec.get("tolerance", 0.0)),
        )
    if kind == "numeric":
        return score_numeric(
            output, float(spec["expected"]), float(spec.get("tolerance", 0.0))
        )
    if kind == "judge":
        return score_judge(output, case, judges)
    raise ValueError(f"unknown scorer type: {kind!r}")


# ---------------------------------------------------------------------------
# Runner and report
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """The outcome of running one case.

    ``score`` is the fraction of the case's weight earned: 1.0 on a
    pass, 0.0 on a fail, or a fraction when the scorer gives partial
    credit. ``weight`` is the case's weight (default 1.0).
    ``budget_violated`` is true when the case passed its checks but
    took longer than its ``max_latency_ms`` budget; such a case is not
    a pass (``passed`` is false, ``score`` is 0.0) and the latency and
    the budget are in the detail.
    """

    case_id: str
    passed: bool
    detail: str
    latency_ms: float
    tokens: Optional[int] = None
    output: str = ""
    timed_out: bool = False
    score: float = 0.0
    weight: float = 1.0
    max_latency_ms: Optional[float] = None
    budget_violated: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 3),
            "tokens": self.tokens,
            "output": self.output,
            "timed_out": self.timed_out,
            "score": round(self.score, 4),
            "weight": self.weight,
            "max_latency_ms": self.max_latency_ms,
            "budget_violated": self.budget_violated,
        }


def _split_agent_result(raw: Any) -> tuple:
    """Normalise an agent's return value to (output, tokens)."""
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, tuple) and len(raw) == 2:
        output, tokens = raw
        return str(output), int(tokens)
    if isinstance(raw, dict) and "output" in raw:
        tokens = raw.get("tokens")
        return str(raw["output"]), (int(tokens) if tokens is not None else None)
    return str(raw), None


@dataclass
class RunReport:
    """Everything a run produced, serialisable to JSON."""

    suite: str
    started_at: str
    ended_at: str
    results: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total) if self.total else 0.0

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.results)

    @property
    def total_tokens(self) -> Optional[int]:
        tokens = [r.tokens for r in self.results if r.tokens is not None]
        return sum(tokens) if tokens else None

    @property
    def total_weight(self) -> float:
        return sum(r.weight for r in self.results)

    @property
    def earned_weight(self) -> float:
        return sum(r.score * r.weight for r in self.results)

    @property
    def weighted_score(self) -> float:
        return (self.earned_weight / self.total_weight) if self.total_weight else 0.0

    @property
    def budget_violations(self) -> int:
        return sum(1 for r in self.results if r.budget_violated)

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "total_weight": self.total_weight,
            "earned_weight": round(self.earned_weight, 4),
            "weighted_score": round(self.weighted_score, 4),
            "budget_violations": self.budget_violations,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_tokens": self.total_tokens,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def load_report(path: str) -> dict:
    """Load a saved JSON report back as a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_agent(
    agent: Callable[[str], Any], prompt: str, timeout: Optional[float]
) -> tuple:
    """Call the agent, enforcing ``timeout`` seconds when set.

    Returns ``(status, output, tokens, error)`` where status is ``"ok"``,
    ``"error"`` (the agent raised), or ``"timeout"``. With a timeout the
    agent runs on a daemon thread: on expiry the case is failed and the
    run moves on, and the abandoned thread cannot hang the process. A
    Python thread cannot be killed, so a hung agent keeps running in the
    background until it returns on its own.
    """
    if timeout is None:
        try:
            output, tokens = _split_agent_result(agent(prompt))
        except Exception as exc:  # noqa: BLE001 - a crashing agent fails the case
            return "error", "", None, exc
        return "ok", output, tokens, None
    results: queue.Queue = queue.Queue()

    def _target() -> None:
        try:
            output, tokens = _split_agent_result(agent(prompt))
        except Exception as exc:  # noqa: BLE001 - a crashing agent fails the case
            results.put(("error", "", None, exc))
        else:
            results.put(("ok", output, tokens, None))

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    try:
        return results.get(timeout=timeout)
    except queue.Empty:
        return "timeout", "", None, None


def _run_case(
    case: Case, agent: Callable[[str], Any], judges: Optional[dict] = None
) -> Result:
    """Run one case against the agent and return its Result.

    An agent that raises is recorded as a failed case with the exception
    in the detail. A case with a ``timeout`` that the agent exceeds is
    recorded as failed and timed out.
    """
    t0 = time.perf_counter()
    status, output, tokens, exc = _run_agent(agent, case.prompt, case.timeout)
    latency = (time.perf_counter() - t0) * 1000
    if status == "timeout":
        return Result(
            case_id=case.id,
            passed=False,
            detail=f"timed out after {case.timeout:g} s",
            latency_ms=latency,
            timed_out=True,
            score=0.0,
            weight=case.weight,
        )
    if status == "error":
        return Result(
            case_id=case.id,
            passed=False,
            detail=f"agent error: {type(exc).__name__}: {exc}",
            latency_ms=latency,
            output="",
            score=0.0,
            weight=case.weight,
        )
    try:
        passed, detail, score = score_case(case, output, judges)
    except ValueError as exc:
        passed, detail, score = False, f"scorer error: {exc}", 0.0
    budget_violated = False
    if (
        passed
        and case.max_latency_ms is not None
        and latency > case.max_latency_ms
    ):
        # The answer was right but too slow: a budget violation, not a pass.
        budget_violated = True
        passed = False
        score = 0.0
        detail = (
            f"{detail}; exceeded latency budget "
            f"{latency:.1f} ms > {case.max_latency_ms:g} ms"
        )
    return Result(
        case_id=case.id,
        passed=passed,
        detail=detail,
        latency_ms=latency,
        tokens=tokens,
        output=output,
        score=score,
        weight=case.weight,
        max_latency_ms=case.max_latency_ms,
        budget_violated=budget_violated,
    )


def run_suite(
    suite: Suite,
    agent: Callable[[str], Any],
    workers: int = 1,
    judges: Optional[dict] = None,
) -> RunReport:
    """Run every case in ``suite`` against ``agent`` and return a report.

    An agent that raises is recorded as a failed case with the exception
    in the detail; the run continues with the remaining cases. A case
    with a ``timeout`` that the agent exceeds is recorded as failed and
    timed out, and the run continues as well.

    ``workers`` is the size of the thread pool running cases: 1 runs
    them one at a time, in suite order. With more workers, cases run
    concurrently, so a slow agent finishes a suite in roughly the time
    of its slowest case instead of the sum of all of them. Results are
    always reported in suite order, whatever order the cases finished
    in, and per-case timeouts are still enforced per case. Parallel
    mode needs a thread-safe agent: a stateful agent whose cases
    depend on each other's order (for example a SET in one case and a
    GET in a later one) can give different verdicts when cases run at
    the same time, so leave ``workers`` at 1 for those suites.

    ``judges`` maps judge names to callables for ``judge`` scorers;
    None means the built-in registry (just the deterministic stub
    judge). A judge that calls a remote model must be thread-safe for
    ``workers`` greater than 1.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    started = datetime.now(timezone.utc).isoformat()
    if workers == 1:
        results = [_run_case(case, agent, judges) for case in suite.cases]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            # map() yields results in suite order, not finish order.
            results = list(
                executor.map(
                    lambda case: _run_case(case, agent, judges), suite.cases
                )
            )
    ended = datetime.now(timezone.utc).isoformat()
    return RunReport(suite=suite.name, started_at=started, ended_at=ended, results=results)


def format_summary(report: RunReport) -> str:
    """Render a run report as a plain-text summary table."""
    lines = [
        f"Suite: {report.suite}",
        f"Passed: {report.passed}/{report.total} "
        f"({report.pass_rate * 100:.1f}%)  "
        f"total {report.total_latency_ms:.1f} ms"
        + (
            f"  tokens {report.total_tokens}"
            if report.total_tokens is not None
            else ""
        ),
        f"Weighted score: {report.earned_weight:.2f}/{report.total_weight:g} "
        f"({report.weighted_score * 100:.1f}%)",
    ]
    if report.budget_violations:
        lines.append(f"Budget violations: {report.budget_violations}")
    lines += [
        "",
        f"{'case':<22} {'pass':<7} {'score':>6} {'ms':>8} {'tokens':>7}  detail",
        "-" * 86,
    ]
    for r in report.results:
        if r.budget_violated:
            mark = "BUDGET"
        elif r.timed_out:
            mark = "TIMEOUT"
        else:
            mark = "PASS" if r.passed else "FAIL"
        tokens = str(r.tokens) if r.tokens is not None else "-"
        lines.append(
            f"{r.case_id:<22} {mark:<7} {r.score:>6.2f} {r.latency_ms:>8.1f} "
            f"{tokens:>7}  {r.detail}"
        )
    return "\n".join(lines)
