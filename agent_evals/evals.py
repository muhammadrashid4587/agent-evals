"""Core of agent-evals: suites, deterministic scorers, the runner, reports.

Design
------
* A ``Suite`` is a name plus a list of ``Case`` objects. A case is one
  prompt plus one scorer spec.
* A scorer spec is a plain dict, so suites can live in JSON files::

      {"type": "exact", "expected": "19"}
      {"type": "contains", "substring": "hello"}
      {"type": "regex", "pattern": "^\\d+$"}
      {"type": "json_schema", "schema": {"name": "str", "age": "int"}}
      {"type": "numeric", "expected": 3.14159, "tolerance": 0.001}

* The agent under test is a plain callable ``agent(prompt) -> ...``. It may
  return:
    - a ``str`` (the agent's output; token count unknown),
    - a ``(output, tokens)`` tuple (output str, token count int),
    - a ``dict`` with an ``"output"`` key and an optional ``"tokens"`` key.
* Every scorer is deterministic: same output, same verdict, every run.
  There is no model-as-judge here on purpose.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

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
    """One eval case: a prompt and the scorer that judges the answer."""

    id: str
    prompt: str
    scorer: dict

    @classmethod
    def from_dict(cls, data: dict) -> "Case":
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            scorer=dict(data["scorer"]),
        )

    def to_dict(self) -> dict:
        return {"id": self.id, "prompt": self.prompt, "scorer": self.scorer}


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
    return ok, detail


def score_contains(output: str, substring: str) -> tuple:
    """Pass when the output contains the substring."""
    ok = substring in output
    detail = "substring found" if ok else f"{substring!r} not in output"
    return ok, detail


def score_regex(output: str, pattern: str) -> tuple:
    """Pass when the regex matches anywhere in the output."""
    try:
        match = re.search(pattern, output)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
    ok = match is not None
    detail = "regex matched" if ok else f"pattern {pattern!r} did not match"
    return ok, detail


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
        return False, f"output is not valid JSON: {exc}"
    err = _check_shape(value, schema)
    if err is None:
        return True, "JSON shape matches"
    return False, err


def score_numeric(output: str, expected: float, tolerance: float = 0.0) -> tuple:
    """Pass when the first number in the output is within ``tolerance`` of
    ``expected``. The number may be embedded in surrounding text."""
    match = _NUMBER_RE.search(output)
    if match is None:
        return False, "no number in output"
    got = float(match.group(0))
    diff = abs(got - float(expected))
    ok = diff <= float(tolerance)
    detail = (
        f"got {got} (diff {diff})"
        if ok
        else f"expected {expected} ± {tolerance}, got {got} (diff {diff})"
    )
    return ok, detail


def score_case(case: Case, output: str) -> tuple:
    """Score one case's output. Returns (passed, detail)."""
    spec = case.scorer
    kind = spec.get("type")
    if kind == "exact":
        return score_exact(output, str(spec["expected"]))
    if kind == "contains":
        return score_contains(output, str(spec["substring"]))
    if kind == "regex":
        return score_regex(output, str(spec["pattern"]))
    if kind == "json_schema":
        return score_json_schema(output, spec["schema"])
    if kind == "numeric":
        return score_numeric(
            output, float(spec["expected"]), float(spec.get("tolerance", 0.0))
        )
    raise ValueError(f"unknown scorer type: {kind!r}")


# ---------------------------------------------------------------------------
# Runner and report
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """The outcome of running one case."""

    case_id: str
    passed: bool
    detail: str
    latency_ms: float
    tokens: Optional[int] = None
    output: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 3),
            "tokens": self.tokens,
            "output": self.output,
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

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
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


def run_suite(
    suite: Suite, agent: Callable[[str], Any]
) -> RunReport:
    """Run every case in ``suite`` against ``agent`` and return a report.

    An agent that raises is recorded as a failed case with the exception in
    the detail; the run continues with the remaining cases.
    """
    started = datetime.now(timezone.utc).isoformat()
    results = []
    for case in suite.cases:
        t0 = time.perf_counter()
        try:
            output, tokens = _split_agent_result(agent(case.prompt))
        except Exception as exc:  # noqa: BLE001 - a crashing agent fails the case
            latency = (time.perf_counter() - t0) * 1000
            results.append(
                Result(
                    case_id=case.id,
                    passed=False,
                    detail=f"agent error: {type(exc).__name__}: {exc}",
                    latency_ms=latency,
                    output="",
                )
            )
            continue
        latency = (time.perf_counter() - t0) * 1000
        try:
            passed, detail = score_case(case, output)
        except ValueError as exc:
            passed, detail = False, f"scorer error: {exc}"
        results.append(
            Result(
                case_id=case.id,
                passed=passed,
                detail=detail,
                latency_ms=latency,
                tokens=tokens,
                output=output,
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
        "",
        f"{'case':<22} {'pass':<6} {'ms':>8} {'tokens':>7}  detail",
        "-" * 78,
    ]
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        tokens = str(r.tokens) if r.tokens is not None else "-"
        lines.append(
            f"{r.case_id:<22} {mark:<6} {r.latency_ms:>8.1f} {tokens:>7}  {r.detail}"
        )
    return "\n".join(lines)
