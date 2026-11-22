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
import queue
import re
import threading
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
    """One eval case: a prompt and the scorer that judges the answer.

    ``timeout`` is an optional per-case limit in seconds: if the agent
    takes longer, the case fails as timed out and the run continues.
    """

    id: str
    prompt: str
    scorer: dict
    timeout: Optional[float] = None

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
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            scorer=dict(data["scorer"]),
            timeout=timeout,
        )

    def to_dict(self) -> dict:
        data = {"id": self.id, "prompt": self.prompt, "scorer": self.scorer}
        if self.timeout is not None:
            data["timeout"] = self.timeout
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
    return ok, detail


def score_contains(output: str, substring: str) -> tuple:
    """Pass when the output contains the substring."""
    ok = substring in output
    detail = "substring found" if ok else f"{substring!r} not in output"
    return ok, detail


def _as_substring_list(spec: dict, kind: str) -> list:
    """Coerce a scorer spec's ``substrings`` value to a list of strings."""
    substrings = spec.get("substrings")
    if not isinstance(substrings, list):
        raise ValueError(f"{kind} scorer requires a 'substrings' list")
    return [str(s) for s in substrings]


def score_contains_all(output: str, substrings: list) -> tuple:
    """Pass when the output contains every listed substring.

    An empty list passes: there is nothing required to be present.
    """
    missing = [s for s in substrings if s not in output]
    if not missing:
        return True, "all substrings found"
    return False, f"missing substrings: {', '.join(repr(s) for s in missing)}"


def score_contains_any(output: str, substrings: list) -> tuple:
    """Pass when the output contains at least one listed substring.

    An empty list fails: no listed substring can be present.
    """
    found = [s for s in substrings if s in output]
    if found:
        return True, f"found {found[0]!r}"
    return (
        False,
        "none of the substrings found: "
        + ", ".join(repr(s) for s in substrings),
    )


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
        return False, f"output is not valid JSON: {exc}"
    steps = _parse_json_path(path)
    found, extracted = _extract_json_path(value, steps, path)
    if not found:
        return False, extracted
    if _values_equal(extracted, expected, float(tolerance)):
        return True, f"{path} == {expected!r}"
    return False, f"expected {path} == {expected!r}, got {extracted!r}"


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
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "detail": self.detail,
            "latency_ms": round(self.latency_ms, 3),
            "tokens": self.tokens,
            "output": self.output,
            "timed_out": self.timed_out,
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


def run_suite(
    suite: Suite, agent: Callable[[str], Any]
) -> RunReport:
    """Run every case in ``suite`` against ``agent`` and return a report.

    An agent that raises is recorded as a failed case with the exception in
    the detail; the run continues with the remaining cases. A case with a
    ``timeout`` that the agent exceeds is recorded as failed and timed out,
    and the run continues as well.
    """
    started = datetime.now(timezone.utc).isoformat()
    results = []
    for case in suite.cases:
        t0 = time.perf_counter()
        status, output, tokens, exc = _run_agent(agent, case.prompt, case.timeout)
        latency = (time.perf_counter() - t0) * 1000
        if status == "timeout":
            results.append(
                Result(
                    case_id=case.id,
                    passed=False,
                    detail=f"timed out after {case.timeout:g} s",
                    latency_ms=latency,
                    timed_out=True,
                )
            )
            continue
        if status == "error":
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
        f"{'case':<22} {'pass':<7} {'ms':>8} {'tokens':>7}  detail",
        "-" * 79,
    ]
    for r in report.results:
        mark = "TIMEOUT" if r.timed_out else ("PASS" if r.passed else "FAIL")
        tokens = str(r.tokens) if r.tokens is not None else "-"
        lines.append(
            f"{r.case_id:<22} {mark:<7} {r.latency_ms:>8.1f} {tokens:>7}  {r.detail}"
        )
    return "\n".join(lines)
