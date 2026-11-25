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
      {"type": "not_contains", "substrings": ["as an AI", "sorry"]}
      {"type": "starts_with", "prefix": "The answer is"}
      {"type": "ends_with", "suffix": "is 42"}
      {"type": "length", "min_length": 10, "max_length": 100}

* The ``exact`` scorer and the string scorers (``contains``,
  ``contains_all``, ``contains_any``, ``not_contains``,
  ``starts_with``, ``ends_with``), and the ``fuzzy``,
  ``levenshtein``, ``regex``, and ``contains_regex`` scorers, also accept an optional
  ``"ignore_case": true``: the expected strings and the output are
  case-folded before comparing (``regex`` and ``contains_regex``
  match with ``re.IGNORECASE``), so ``"Hello"``
  matches ``"hello"``. Scores and partial credit are unchanged;
  the option defaults to off, and a non-boolean value is a scorer
  error.
      {"type": "regex", "pattern": "^\\d+$"}
      {"type": "contains_regex", "pattern": "^\\d+$"}
      {"type": "json_schema", "schema": {"name": "str", "age": "int"}}
      {"type": "json_path", "path": "$.result.total", "expected": 42}
      {"type": "json_subset", "expected": {"status": "ok", "total": 42}}
      {"type": "json_equals", "expected": {"status": "ok", "total": 42}}
      {"type": "fuzzy", "expected": "hello world", "threshold": 0.8}
      {"type": "levenshtein", "expected": "hello world", "threshold": 0.8}
      {"type": "numeric", "expected": 3.14159, "tolerance": 0.001}
      {"type": "judge", "judge": "stub", "rubric": {"criteria": [...]}}

* A case may carry a ``weight`` (default 1.0): how much of the suite's
  total score it is worth. Scorers also return a score in [0, 1] — 1.0
  on a pass, 0.0 on a fail — and ``contains_all`` earns partial credit
  equal to the fraction of required substrings found. The report totals
  ``earned_weight`` over ``total_weight`` as ``weighted_score``.

* A case may carry a ``threshold`` in [0, 1]: the minimum score it
  needs to pass. By default a case passes only on a full score, but a
  case with ``threshold`` 0.75 passes when its scorer gives it 0.75 or
  more — so a partial-credit case can pass without earning a full
  score, and a case whose scorer passes it on a low score can still
  fail. The threshold only applies to a score the scorer actually
  produced: a timeout, an agent error, or a scorer error still fails
  the case whatever the threshold is.

* A case may carry a ``max_latency_ms`` budget: if the agent's answer
  passes its checks but took longer than the budget, the case is
  reported as a budget violation instead of a pass (``passed`` is
  false, ``budget_violated`` is true, and the latency and the budget
  are in the detail). A case that fails its checks is never also a
  budget violation, and a timed-out case is not one either.

* A case may carry a ``max_tokens`` budget: if the agent's answer
  passes its checks but the agent reported more tokens than the
  budget, the case is reported as a cost violation instead of a pass
  (``passed`` is false, ``cost_violated`` is true, and the token
  count and the budget are in the detail). It only applies when the
  agent reports a token count — an agent that returns a plain string
  reports no tokens, so its cases can never trip the budget. A case
  that fails its checks is never also a cost violation, and a
  timed-out case is not one either.

* A case may carry ``tags``, a list of labels such as ``"smoke"`` or
  ``"slow"``. Tags are recorded per result in the report and shown in
  the summary, and a suite can be filtered to the cases carrying a
  tag (see :meth:`Suite.select`) — so a quick smoke run or a
  slow-only run is one flag away. :meth:`Suite.tag_counts` lists
  every tag with its case count, and a run report carries a
  per-tag pass-rate breakdown (``RunReport.tag_breakdown``), so a
  tag that keeps failing is visible without digging through rows.

* When an ``exact`` case fails, the result carries a unified diff of
  the expected output vs the actual output (``Result.diff``, also in
  the JSON report and under the row in the text summary), so a
  regression is visible at a glance. Other scorers have no expected
  text to diff against, so they carry no diff.

* An eval set is versioned: a suite file may carry a ``"version"``
  label, and every suite and case has a content fingerprint (a short
  SHA-256 of its canonical JSON). A run report records the suite's
  version and fingerprint plus each case's fingerprint, so a later
  diff (see :mod:`agent_evals.compare`) can tell whether a score
  delta came from the agent or from the eval set changing underneath
  it. The fingerprint covers the name and the cases only — relabelling
  a version without touching a case keeps the same fingerprint.

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
import difflib
import hashlib
import json
import math
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
    "make_output_diff",
    "run_suite",
    "score_case",
]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

# Fingerprints are the first 12 hex chars of a SHA-256 over a canonical
# JSON encoding: short enough to read, long enough that a collision
# between two distinct eval sets is not a practical concern.


def _canonical_json(data: Any) -> str:
    """Encode ``data`` as JSON with sorted keys and no whitespace, so
    the same logical content always encodes to the same string no
    matter how a dict was built or ordered."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(data: Any) -> str:
    """A short content hash of ``data`` (see ``_canonical_json``)."""
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()[:12]

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
    ``max_tokens`` is an optional per-case token budget: a case that
    passes its checks but whose agent reported more tokens than the
    budget is reported as a cost violation instead of a pass. The
    token budget only applies when the agent reports a token count.
    ``tags`` is an optional list of labels (e.g. ``["smoke"]``) used
    to select cases for a run; it defaults to no tags.
    ``threshold`` is an optional per-case pass threshold in [0, 1]:
    when set, a case passes when its scorer's score reaches the
    threshold, so a partial-credit case can pass without a full
    score. When unset the case passes only as its scorer says. A
    timeout, an agent error, or a scorer error still fails the case
    whatever the threshold is.
    """

    id: str
    prompt: str
    scorer: dict
    timeout: Optional[float] = None
    weight: float = 1.0
    max_latency_ms: Optional[float] = None
    max_tokens: Optional[int] = None
    threshold: Optional[float] = None
    tags: list = field(default_factory=list)

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
        max_tokens = data.get("max_tokens")
        if max_tokens is not None:
            # Tokens are a count, so the budget is a whole number.
            if (
                isinstance(max_tokens, bool)
                or not isinstance(max_tokens, int)
                or max_tokens <= 0
            ):
                raise ValueError(
                    f"case {data.get('id')!r}: max_tokens must be a "
                    f"positive integer, got {max_tokens!r}"
                )
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(
                f"case {data.get('id')!r}: tags must be a list of strings, "
                f"got {tags!r}"
            )
        threshold = data.get("threshold")
        if threshold is not None:
            # A threshold is a score in [0, 1]: bools are rejected
            # (bool is a subclass of int), and NaN/infinity fail the
            # range check below, since no comparison with them is true.
            if isinstance(threshold, bool) or not isinstance(
                threshold, (int, float)
            ):
                raise ValueError(
                    f"case {data.get('id')!r}: threshold must be a number "
                    f"in [0, 1], got {threshold!r}"
                )
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"case {data.get('id')!r}: threshold must be in "
                    f"[0, 1], got {threshold!r}"
                )
        seen: set = set()
        clean_tags = []
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise ValueError(
                    f"case {data.get('id')!r}: tags must be non-empty "
                    f"strings, got {tag!r}"
                )
            if tag not in seen:
                seen.add(tag)
                clean_tags.append(tag)
        # Canonical order: the tag set is what identifies the case, so
        # the order tags are listed in must not change its fingerprint.
        clean_tags.sort()
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            scorer=dict(data["scorer"]),
            timeout=timeout,
            weight=weight,
            max_latency_ms=max_latency_ms,
            max_tokens=max_tokens,
            threshold=threshold,
            tags=clean_tags,
        )

    def to_dict(self) -> dict:
        data = {"id": self.id, "prompt": self.prompt, "scorer": self.scorer}
        if self.timeout is not None:
            data["timeout"] = self.timeout
        if self.weight != 1.0:
            data["weight"] = self.weight
        if self.max_latency_ms is not None:
            data["max_latency_ms"] = self.max_latency_ms
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.threshold is not None:
            data["threshold"] = self.threshold
        if self.tags:
            data["tags"] = list(self.tags)
        return data

    def fingerprint(self) -> str:
        """A short content hash of this case: its id, prompt, scorer,
        timeout, weight, latency budget, token budget, threshold, and tags. Two cases with the
        same fingerprint are the same case; a changed prompt or scorer
        changes the fingerprint, so a run diff can tell a case that
        changed from an agent that changed."""
        return _fingerprint(self.to_dict())


@dataclass
class Suite:
    """A named list of cases.

    ``version`` is an optional human label for this eval set, e.g.
    ``"1.3"`` — read from a suite file's ``"version"`` field and
    recorded in run reports. The label is not part of the suite's
    identity: :meth:`fingerprint` hashes only the name and the cases,
    so renaming a version without touching a case keeps the same
    fingerprint, and changing a case changes it even if the label
    was not bumped.
    """

    name: str
    cases: list = field(default_factory=list)
    version: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Suite":
        version = data.get("version")
        if version is not None and not isinstance(version, str):
            raise ValueError(
                f"suite version must be a string, got {version!r}"
            )
        return cls(
            name=str(data.get("name", "suite")),
            cases=[Case.from_dict(c) for c in data.get("cases", [])],
            version=version,
        )

    def to_dict(self) -> dict:
        data: dict = {"name": self.name, "cases": [c.to_dict() for c in self.cases]}
        if self.version is not None:
            data["version"] = self.version
        return data

    def select(self, tags: list, match: str = "any") -> "Suite":
        """Return a new suite with only the cases carrying ``tags``.

        ``match`` is ``"any"`` (default) or ``"all"``: ``"any"`` keeps
        a case carrying at least one of the tags (OR), ``"all"`` keeps
        only a case carrying every one of them (AND). An empty tag
        list returns the whole suite unchanged. The original suite is
        not modified, and the version label carries over, so a
        filtered run still records which eval set it was measured
        against (its fingerprint differs, because the cases differ).
        """
        if match not in ("any", "all"):
            raise ValueError(f"match must be 'any' or 'all', got {match!r}")
        wanted = [str(t) for t in tags]
        if not wanted:
            return Suite(name=self.name, cases=list(self.cases), version=self.version)
        if match == "any":
            kept = [c for c in self.cases if any(t in c.tags for t in wanted)]
        else:
            kept = [c for c in self.cases if all(t in c.tags for t in wanted)]
        return Suite(name=self.name, cases=kept, version=self.version)

    def exclude(self, tags: list) -> "Suite":
        """Return a new suite without the cases carrying any of ``tags``.

        A case is dropped when it carries at least one of the given
        tags, whatever its other tags are. An empty tag list returns
        the whole suite unchanged. The original suite is not modified,
        and the version label carries over. Pairs with
        :meth:`select`: select first, then exclude, so ``--tag smoke
        --exclude-tag slow`` runs the smoke cases that are not slow.
        """
        dropped = [str(t) for t in tags]
        if not dropped:
            return Suite(name=self.name, cases=list(self.cases), version=self.version)
        kept = [c for c in self.cases if not any(t in c.tags for t in dropped)]
        return Suite(name=self.name, cases=kept, version=self.version)

    def tag_counts(self) -> list:
        """Count cases per tag: a sorted list of ``(tag, count)`` pairs.

        A case carrying several tags counts once under each of them;
        untagged cases appear under no tag. An empty list means the
        suite uses no tags at all. This is what the CLI's
        ``--list-tags`` prints.
        """
        counts: dict = {}
        for case in self.cases:
            for tag in case.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return sorted(counts.items())

    def fingerprint(self) -> str:
        """A short content hash of this eval set: its name and every
        case. Two suites with the same fingerprint contain the same
        cases, whatever version label they carry; a different
        fingerprint means the set itself changed, so a score delta
        between runs on the two sets may come from the set rather
        than from the agent."""
        return _fingerprint(
            {"name": self.name, "cases": [c.to_dict() for c in self.cases]}
        )


def load_suite(path: str) -> Suite:
    """Load a suite from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return Suite.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def score_exact(output: str, expected: str, ignore_case: bool = False) -> tuple:
    """Pass when the output equals the expected string (ignoring
    surrounding whitespace on both sides).

    With ``ignore_case``, both strings are case-folded before
    comparing, so ``"Hello"`` matches ``"hello"``. Exact stays
    all-or-nothing: the score is 1.0 on a pass and 0.0 on a fail.
    """
    ok = _fold(output.strip(), ignore_case) == _fold(expected.strip(), ignore_case)
    detail = "exact match" if ok else f"expected {expected!r}, got {output!r}"
    return ok, detail, 1.0 if ok else 0.0


def _fold(text: str, ignore_case: bool) -> str:
    """Fold ``text`` for comparison when ``ignore_case`` is set.

    ``str.casefold`` is used rather than ``lower`` so non-ASCII case
    pairs (e.g. German ß/ss) compare equal too. With ``ignore_case``
    off the text is returned unchanged.
    """
    return text.casefold() if ignore_case else text


def _as_bool_option(spec: dict, key: str, kind: str) -> bool:
    """Read an optional boolean scorer option, defaulting to False.

    A non-boolean value is a scorer error: a JSON ``"false"`` string
    would otherwise silently enable the option, since ``bool("false")``
    is True.
    """
    value = spec.get(key, False)
    if not isinstance(value, bool):
        raise ValueError(
            f"{kind} scorer requires {key!r} to be a boolean, "
            f"got {value!r}"
        )
    return value


def _validate_tolerance(tolerance: Any, kind: str) -> float:
    """Validate a scorer's numeric ``tolerance`` option.

    It must be a real number (not a bool, not a string, not NaN or
    infinity) and at least 0. Anything else is a scorer error: a
    negative tolerance would make even an exact match fail, and a
    non-numeric one would fail confusingly inside the comparison.
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ValueError(
            f"{kind} scorer requires a numeric 'tolerance', "
            f"got {tolerance!r}"
        )
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            f"{kind} scorer requires a 'tolerance' >= 0, got {tolerance!r}"
        )
    return tolerance


def score_contains(output: str, substring: str, ignore_case: bool = False) -> tuple:
    """Pass when the output contains the substring.

    With ``ignore_case``, both the substring and the output are
    case-folded before comparing, so ``"Hello"`` matches ``"hello"``.
    """
    ok = _fold(substring, ignore_case) in _fold(output, ignore_case)
    detail = "substring found" if ok else f"{substring!r} not in output"
    return ok, detail, 1.0 if ok else 0.0


def _as_substring_list(spec: dict, kind: str) -> list:
    """Coerce a scorer spec's ``substrings`` value to a list of strings."""
    substrings = spec.get("substrings")
    if not isinstance(substrings, list):
        raise ValueError(f"{kind} scorer requires a 'substrings' list")
    return [str(s) for s in substrings]


def score_contains_all(
    output: str, substrings: list, ignore_case: bool = False
) -> tuple:
    """Pass when the output contains every listed substring.

    An empty list passes: there is nothing required to be present.
    Partial credit: the score is the fraction of listed substrings
    found, so a case that gets most of the way there still earns most
    of its weight. A case only counts as passed when every substring
    is found (score 1.0). With ``ignore_case``, the comparison is
    case-insensitive; the detail still names the substrings as
    written in the spec.
    """
    folded = _fold(output, ignore_case)
    missing = [s for s in substrings if _fold(s, ignore_case) not in folded]
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


def score_contains_any(
    output: str, substrings: list, ignore_case: bool = False
) -> tuple:
    """Pass when the output contains at least one listed substring.

    An empty list fails: no listed substring can be present. With
    ``ignore_case``, the comparison is case-insensitive.
    """
    folded = _fold(output, ignore_case)
    found = [s for s in substrings if _fold(s, ignore_case) in folded]
    if found:
        return True, f"found {found[0]!r}", 1.0
    return (
        False,
        "none of the substrings found: "
        + ", ".join(repr(s) for s in substrings),
        0.0,
    )


def score_not_contains(
    output: str, substrings: list, ignore_case: bool = False
) -> tuple:
    """Pass when the output contains none of the listed substrings.

    The mirror of :func:`score_contains_all`: where that scorer
    requires every listed substring to be present, this one requires
    every listed substring to be absent — for output that must not
    leak a phrase ("as an AI"), must not apologise, or must not
    contain a forbidden token. An empty list passes vacuously: there
    is nothing forbidden to find. Partial credit: the score is the
    fraction of listed substrings absent, so an output that slips in
    one of four forbidden phrases still earns most of its weight. A
    case only counts as passed when no listed substring is found
    (score 1.0). With ``ignore_case``, the comparison is
    case-insensitive, so a forbidden phrase in any casing fails the
    case; the detail still names the substrings as written in the
    spec.
    """
    folded = _fold(output, ignore_case)
    found = [s for s in substrings if _fold(s, ignore_case) in folded]
    if not substrings:
        return True, "no forbidden substrings found", 1.0
    score = (len(substrings) - len(found)) / len(substrings)
    if not found:
        return True, "no forbidden substrings found", 1.0
    return (
        False,
        f"found forbidden substrings: {', '.join(repr(s) for s in found)}",
        score,
    )


def score_starts_with(
    output: str, prefix: str, ignore_case: bool = False
) -> tuple:
    """Pass when the output starts with the prefix string.

    Surrounding whitespace is ignored on both sides, like ``exact``
    and ``fuzzy``. Partial credit: the score is the fraction of the
    prefix the output matches from the start (the longest common
    prefix over the prefix length), so an output that gets the first
    few words right still earns part of its weight. An empty prefix
    passes vacuously: every string starts with it. With
    ``ignore_case``, the comparison is case-insensitive; the detail
    still shows the prefix as written in the spec.
    """
    out = output.strip()
    pre = prefix.strip()
    if not pre:
        return True, "starts with empty prefix", 1.0
    out_cmp = _fold(out, ignore_case)
    pre_cmp = _fold(pre, ignore_case)
    common = 0
    for a, b in zip(out_cmp, pre_cmp):
        if a != b:
            break
        common += 1
    score = common / len(pre_cmp)
    if common == len(pre_cmp):
        return True, f"starts with {pre!r}", 1.0
    return (
        False,
        f"expected output to start with {pre!r}, "
        f"matched {common} of {len(pre_cmp)} prefix characters; got {out!r}",
        score,
    )


def score_ends_with(
    output: str, suffix: str, ignore_case: bool = False
) -> tuple:
    """Pass when the output ends with the suffix string.

    Surrounding whitespace is ignored on both sides, like ``exact``
    and ``fuzzy``. Partial credit: the score is the fraction of the
    suffix the output matches from the end (the longest common
    suffix over the suffix length), so an output that gets the last
    few words right still earns part of its weight. An empty suffix
    passes vacuously: every string ends with it. With
    ``ignore_case``, the comparison is case-insensitive; the detail
    still shows the suffix as written in the spec.
    """
    out = output.strip()
    suf = suffix.strip()
    if not suf:
        return True, "ends with empty suffix", 1.0
    out_cmp = _fold(out, ignore_case)
    suf_cmp = _fold(suf, ignore_case)
    common = 0
    for a, b in zip(reversed(out_cmp), reversed(suf_cmp)):
        if a != b:
            break
        common += 1
    score = common / len(suf_cmp)
    if common == len(suf_cmp):
        return True, f"ends with {suf!r}", 1.0
    return (
        False,
        f"expected output to end with {suf!r}, "
        f"matched {common} of {len(suf_cmp)} suffix characters; got {out!r}",
        score,
    )


def _as_value_list(spec: dict, kind: str) -> list:
    """Coerce a scorer spec's ``values`` value to a non-empty list of strings.

    An empty list is a scorer error: a set of allowed values with
    nothing in it can never pass, so the suite is wrong, not the
    agent — the same fail-loudly rule as a non-string
    ``contains_regex`` pattern.
    """
    values = spec.get("values")
    if not isinstance(values, list):
        raise ValueError(f"{kind} scorer requires a 'values' list")
    if not values:
        raise ValueError(f"{kind} scorer requires a non-empty 'values' list")
    return [str(v) for v in values]


def score_one_of(
    output: str, values: list, ignore_case: bool = False
) -> tuple:
    """Pass when the output equals one of the allowed values.

    The scorer for classification-style evals: an agent that must
    answer with a label from a fixed set — "yes", "no", "maybe" — a
    verdict from a fixed vocabulary, a choice from a menu. The
    output is stripped before comparing, like ``exact``, so
    surrounding whitespace does not fail the case. It is
    all-or-nothing like ``exact``: 1.0 on a pass, 0.0 on a fail.
    With ``ignore_case``, the comparison is case-insensitive; the
    detail still names the values as written in the spec.
    """
    out = output.strip()
    folded = _fold(out, ignore_case)
    if any(folded == _fold(v.strip(), ignore_case) for v in values):
        return True, f"output is one of {values!r}", 1.0
    return (
        False,
        f"expected one of {values!r}, got {out!r}",
        0.0,
    )


def _as_length_bound(spec: dict, key: str, kind: str) -> Optional[int]:
    """Read an optional length bound from a scorer spec.

    A bound must be a non-negative integer: ``True``/``False`` are
    rejected (``bool`` is a subclass of ``int``), a float is only
    accepted when it is integral (``10.0``), and a string, a
    non-integral float, or a negative value is a scorer error.
    Absent means no bound on that side.
    """
    if key not in spec or spec[key] is None:
        return None
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{kind} scorer requires {key!r} to be a non-negative integer"
        )
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"{kind} scorer requires {key!r} to be a non-negative integer"
            )
        value = int(value)
    if value < 0:
        raise ValueError(
            f"{kind} scorer requires {key!r} to be a non-negative integer"
        )
    return value


def score_length(
    output: str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    unit: str = "characters",
) -> tuple:
    """Pass when the output's length falls in ``[min_length, max_length]``.

    At least one bound is required; a missing bound is unbounded on
    that side, so ``{"min_length": 10}`` only sets a floor and
    ``{"max_length": 100}`` only sets a ceiling. Length is measured
    in characters by default (``len`` of the stripped output, like
    the other string scorers) or in words with ``unit="words"``
    (whitespace-split word count). Partial credit: the score falls
    linearly from 1.0 at the bound to 0.0 at a length of 0 (too
    short: score is ``n / min_length``) or at twice the maximum
    (too long: score is ``max(0, (2 * max_length - n) / max_length)``),
    so an output just outside the range still earns most of its
    weight and one wildly outside earns nothing. A bound that is
    not a non-negative integer, ``min_length`` above ``max_length``,
    or a ``unit`` other than ``"characters"`` or ``"words"`` is a
    scorer error.
    """
    if unit not in ("characters", "words"):
        raise ValueError(
            "length scorer requires 'unit' to be 'characters' or 'words'"
        )
    if min_length is None and max_length is None:
        raise ValueError(
            "length scorer requires at least one of 'min_length' or "
            "'max_length'"
        )
    if (
        min_length is not None
        and max_length is not None
        and min_length > max_length
    ):
        raise ValueError(
            "length scorer requires 'min_length' <= 'max_length'"
        )
    stripped = output.strip()
    if unit == "words":
        n = len(stripped.split())
    else:
        n = len(stripped)
    lo = f"min {min_length}" if min_length is not None else "no minimum"
    hi = f"max {max_length}" if max_length is not None else "no maximum"
    if (min_length is None or n >= min_length) and (
        max_length is None or n <= max_length
    ):
        return True, f"length {n} {unit} within [{lo}, {hi}]", 1.0
    if min_length is not None and n < min_length:
        score = n / min_length
        return (
            False,
            f"length {n} {unit} below minimum {min_length} "
            f"(score {score:.2f})",
            score,
        )
    # n > max_length here; max_length is 0 only when the output is
    # non-empty, which already scores 0.
    if max_length == 0:
        score = 0.0
    else:
        score = max(0.0, (2 * max_length - n) / max_length)
    return (
        False,
        f"length {n} {unit} above maximum {max_length} "
        f"(score {score:.2f})",
        score,
    )


def _as_range_bound(spec: dict, key: str, kind: str) -> Optional[float]:
    """Read an optional numeric range bound from a scorer spec.

    A bound must be a real number: ``True``/``False`` are rejected
    (``bool`` is a subclass of ``int``), and a string is a scorer
    error. Unlike a length bound, a range bound may be negative.
    Absent means no bound on that side.
    """
    if key not in spec or spec[key] is None:
        return None
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{kind} scorer requires {key!r} to be a number")
    return float(value)


def score_range(
    output: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    tolerance: float = 0.0,
) -> tuple:
    """Pass when the first number in the output falls in ``[min, max]``.

    At least one bound is required; a missing bound is unbounded on
    that side, so ``{"min": 0}`` only sets a floor and
    ``{"max": 100}`` only sets a ceiling. The number may be embedded
    in surrounding text (the same first-number parse as the
    ``numeric`` scorer); an output with no number fails with a
    score of 0.0. An optional ``tolerance`` widens the accepted band
    by that amount on each bounded side — the value passes when it
    falls in ``[min - tolerance, max + tolerance]`` — so a float
    answer that is close but just outside the range still passes.
    Partial credit: the score falls linearly from 1.0 at the widened
    bound to 0.0 one scale away, where the scale is the widened
    range width when both bounds are present (``max - min`` plus
    twice the tolerance) or the magnitude of the single widened
    bound otherwise — so a value just outside the widened range
    still earns most of its weight and one a full scale outside
    earns nothing. A zero scale (``min == max`` with no tolerance,
    or a lone bound of 0 with no tolerance) gives no partial
    credit: any value outside the range scores 0.0. A non-numeric
    bound, ``min`` above ``max``, or a non-numeric or negative
    ``tolerance`` is a scorer error. A tolerance of 0 keeps the
    old behaviour exactly.
    """
    tolerance = _validate_tolerance(tolerance, "range")
    if min_value is None and max_value is None:
        raise ValueError(
            "range scorer requires at least one of 'min' or 'max'"
        )
    if (
        min_value is not None
        and max_value is not None
        and min_value > max_value
    ):
        raise ValueError("range scorer requires 'min' <= 'max'")
    match = _NUMBER_RE.search(output)
    if match is None:
        return False, "no number in output", 0.0
    got = float(match.group(0))
    eff_min = None if min_value is None else min_value - tolerance
    eff_max = None if max_value is None else max_value + tolerance
    lo = f"min {eff_min}" if eff_min is not None else "no minimum"
    hi = f"max {eff_max}" if eff_max is not None else "no maximum"
    if (eff_min is None or got >= eff_min) and (
        eff_max is None or got <= eff_max
    ):
        return True, f"value {got} within [{lo}, {hi}]", 1.0
    if eff_min is not None and eff_max is not None:
        scale = eff_max - eff_min
    elif eff_min is not None:
        scale = abs(eff_min)
    else:
        scale = abs(eff_max)  # type: ignore[arg-type]
    if scale == 0:
        score = 0.0
    elif eff_min is not None and got < eff_min:
        score = max(0.0, 1.0 - (eff_min - got) / scale)
    else:
        score = max(0.0, 1.0 - (got - eff_max) / scale)  # type: ignore[operator]
    side = "below minimum" if eff_min is not None and got < eff_min else "above maximum"
    bound = eff_min if side == "below minimum" else eff_max
    return (
        False,
        f"value {got} {side} {bound} (score {score:.2f})",
        score,
    )


def score_regex(output: str, pattern: str, ignore_case: bool = False) -> tuple:
    """Pass when the regex matches anywhere in the output.

    With ``ignore_case``, the match is case-insensitive
    (``re.IGNORECASE``), so a pattern written in lower case still
    matches upper-case output.
    """
    flags = re.IGNORECASE if ignore_case else 0
    try:
        match = re.search(pattern, output, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
    ok = match is not None
    detail = "regex matched" if ok else f"pattern {pattern!r} did not match"
    return ok, detail, 1.0 if ok else 0.0


def score_contains_regex(output: str, pattern: Any, ignore_case: bool = False) -> tuple:
    """Pass when the regex matches anywhere in the output.

    The strict counterpart of ``score_regex``: the ``pattern`` must
    be a string, so a suite that passes a number (or any other
    non-string) where a pattern belongs is a scorer error instead
    of silently matching the stringified value. An invalid regex
    is a scorer error too. With ``ignore_case``, the match is
    case-insensitive (``re.IGNORECASE``), like ``score_regex``.
    All-or-nothing: 1.0 on a match, 0.0 otherwise.
    """
    if not isinstance(pattern, str):
        raise ValueError(
            f"contains_regex scorer requires a string 'pattern', "
            f"got {type(pattern).__name__}"
        )
    flags = re.IGNORECASE if ignore_case else 0
    try:
        match = re.search(pattern, output, flags)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
    ok = match is not None
    detail = "contains_regex matched" if ok else f"pattern {pattern!r} did not match"
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


def _count_leaves(value: Any) -> int:
    """Count the scalar leaves in an expected JSON value: one per
    scalar, the sum of the elements for a list, the sum of the values
    for a dict. Used to weight partial credit for a subset match."""
    if isinstance(value, dict):
        return sum(_count_leaves(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_leaves(v) for v in value)
    return 1


def _subset_check(
    expected: Any, actual: Any, path: str, tolerance: float
) -> tuple:
    """Compare ``expected`` against ``actual`` as a subset.

    Returns ``(matched, total, mismatches)``: ``matched``/``total``
    count the expected leaf values that matched, and ``mismatches``
    are human-readable strings naming what did not match. Dicts
    match by subset — every expected key must be present, extra
    actual keys are ignored — and lists match pairwise by position
    with the same length; a dict inside a list is itself a subset
    match, so an expected list item only pins the fields it names.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0, _count_leaves(expected), [
                f"{path}: expected object, got {type(actual).__name__}"
            ]
        matched, total, mismatches = 0, 0, []
        for key, sub_expected in expected.items():
            child = f"{path}.{key}"
            if key not in actual:
                total += _count_leaves(sub_expected)
                mismatches.append(f"missing key {key!r} at {child}")
                continue
            m, t, errs = _subset_check(sub_expected, actual[key], child, tolerance)
            matched += m
            total += t
            mismatches.extend(errs)
        return matched, total, mismatches
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0, _count_leaves(expected), [
                f"{path}: expected array, got {type(actual).__name__}"
            ]
        if len(actual) != len(expected):
            return 0, _count_leaves(expected), [
                f"{path}: expected array of length {len(expected)}, "
                f"got {len(actual)}"
            ]
        matched, total, mismatches = 0, 0, []
        for i, (sub_expected, sub_actual) in enumerate(zip(expected, actual)):
            m, t, errs = _subset_check(
                sub_expected, sub_actual, f"{path}[{i}]", tolerance
            )
            matched += m
            total += t
            mismatches.extend(errs)
        return matched, total, mismatches
    if _values_equal(actual, expected, tolerance):
        return 1, 1, []
    return 0, 1, [f"expected {path} == {expected!r}, got {actual!r}"]


def score_json_subset(output: str, expected: Any, tolerance: float = 0.0) -> tuple:
    """Pass when the output JSON contains every expected value.

    ``expected`` is an object: every key it names must be present in
    the output with a matching value, recursively, while extra keys
    in the output are ignored — the check is a subset match, not a
    full equality. Numbers compare with an optional ``tolerance``;
    ``bool`` is never accepted as ``int``. Partial credit: the score
    is the fraction of expected leaf values that matched, so a case
    that gets most of the fields right still earns most of its
    weight. An empty expected object passes vacuously. A non-object
    ``expected`` raises ``ValueError``.
    """
    if not isinstance(expected, dict):
        raise ValueError("json_subset scorer requires an 'expected' object")
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"output is not valid JSON: {exc}", 0.0
    matched, total, mismatches = _subset_check(
        expected, value, "$", float(tolerance)
    )
    if not mismatches:
        return True, "JSON subset matches", 1.0
    score = (matched / total) if total else 1.0
    return False, "; ".join(mismatches), score


def _json_multiset_match(
    got: list, expected: list, ignore_order: bool, tolerance: float
) -> bool:
    """Whether ``got`` matches ``expected`` as a multiset under
    :func:`_json_values_equal` with the given ``tolerance``.

    Greedy pairing can miss a valid match — e.g. expected ``[1.0,
    1.2]`` vs got ``[1.1, 1.0]`` with tolerance 0.15 pairs 1.1 with
    1.0 and then has nothing left for 1.2, even though 1.1 matches
    1.2 and 1.0 matches 1.0 — so this is a maximum bipartite
    matching (Kuhn's algorithm). Lists here are small eval values,
    so the quadratic worst case is fine.
    """
    match_to: dict = {}

    def _try_match(gi: int, seen: set) -> bool:
        for ei, e in enumerate(expected):
            if ei in seen:
                continue
            if not _json_values_equal(got[gi], e, ignore_order, tolerance):
                continue
            seen.add(ei)
            if ei not in match_to or _try_match(match_to[ei], seen):
                match_to[ei] = gi
                return True
        return False

    for gi in range(len(got)):
        if not _try_match(gi, set()):
            return False
    return True


def _json_values_equal(
    got: Any, expected: Any, ignore_order: bool = False, tolerance: float = 0.0
) -> bool:
    """Deep equality for JSON values, with bool/int kept distinct.

    Numbers compare numerically, so ``1`` equals ``1.0``; with a
    nonzero ``tolerance`` they compare within that tolerance
    instead. ``True`` never equals ``1``. Dicts compare key by key —
    extra keys fail, unlike ``json_subset`` — and lists compare
    pairwise by position unless ``ignore_order`` is set, in which
    case they compare as multisets (each element encoded as
    canonical JSON, so nested structure is covered too). With a
    nonzero tolerance the multiset comparison matches elements
    pairwise instead, since canonical encodings cannot express
    "within tolerance".
    """
    if isinstance(got, bool) or isinstance(expected, bool):
        return got is expected
    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(got) - float(expected)) <= tolerance
    if isinstance(got, dict) and isinstance(expected, dict):
        if set(got) != set(expected):
            return False
        return all(
            _json_values_equal(got[k], expected[k], ignore_order, tolerance)
            for k in expected
        )
    if isinstance(got, list) and isinstance(expected, list):
        if len(got) != len(expected):
            return False
        if ignore_order:
            if tolerance == 0.0:
                return sorted(_canonical_json(v) for v in got) == sorted(
                    _canonical_json(v) for v in expected
                )
            return _json_multiset_match(got, expected, ignore_order, tolerance)
        return all(
            _json_values_equal(a, b, ignore_order, tolerance)
            for a, b in zip(got, expected)
        )
    return got == expected


def score_json_equals(
    output: str,
    expected: Any,
    ignore_order: bool = False,
    tolerance: float = 0.0,
) -> tuple:
    """Pass when the output JSON is deeply equal to ``expected``.

    Unlike ``json_subset``, this is full equality: extra keys in the
    output fail the case, and lists must match pairwise by position
    (or as multisets with ``ignore_order``). Numbers compare
    numerically, so ``1`` equals ``1.0``; with a ``tolerance`` they
    compare within that tolerance instead, recursively, so a float
    answer that is close but not exact still passes. ``bool`` is
    never accepted as ``int``, so ``true`` does not equal ``1`` and
    ``1`` does not equal ``"1"``. An output that is not valid JSON
    scores 0.0, and an ``expected`` that is not valid strict JSON
    (not serialisable, or containing NaN/Infinity) is a scorer
    error, as is a non-numeric or negative ``tolerance``.
    """
    tolerance = _validate_tolerance(tolerance, "json_equals")
    try:
        json.dumps(expected, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "json_equals scorer requires an 'expected' value that is "
            f"valid JSON: {exc}"
        ) from exc
    try:
        value = json.loads(output)
    except json.JSONDecodeError as exc:
        return False, f"output is not valid JSON: {exc}", 0.0
    if _json_values_equal(value, expected, ignore_order, tolerance):
        return True, "JSON values are equal", 1.0
    return False, f"expected {expected!r}, got {value!r}", 0.0


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


def score_fuzzy(
    output: str, expected: str, threshold: float = 0.8, ignore_case: bool = False
) -> tuple:
    """Pass when the output is close enough to the expected string.

    Similarity is :class:`difflib.SequenceMatcher`'s ratio on the
    stripped strings — deterministic, standard library only, no
    model. Pass when the ratio reaches ``threshold`` (default 0.8),
    which must be in [0, 1]. Partial credit: the score is the ratio
    itself, so a near miss still earns most of its weight. An exact
    match is a ratio of 1.0; an empty expected string only matches
    an empty output. With ``ignore_case``, both strings are
    case-folded before the ratio is computed, so ``"Hello World"``
    and ``"hello world"`` score 1.0.
    """
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("fuzzy scorer requires a 'threshold' in [0, 1]")
    ratio = difflib.SequenceMatcher(
        None,
        _fold(output.strip(), ignore_case),
        _fold(expected.strip(), ignore_case),
    ).ratio()
    ok = ratio >= threshold
    detail = (
        f"similarity {ratio:.2f} >= threshold {threshold:.2f}"
        if ok
        else f"similarity {ratio:.2f} < threshold {threshold:.2f}"
    )
    return ok, detail, ratio


def _levenshtein_distance(a: str, b: str) -> int:
    """The edit distance between ``a`` and ``b``: the minimum number
    of single-character insertions, deletions, and substitutions
    needed to turn one into the other.

    Classic dynamic programming with two rows, so the memory is
    O(min(len(a), len(b))) instead of the full matrix. Deterministic,
    standard library only, no model.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Keep the shorter string as the row, so the working row is small.
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for j, b_char in enumerate(b, start=1):
        current = [j]
        for i, a_char in enumerate(a, start=1):
            cost = 0 if a_char == b_char else 1
            current.append(
                min(
                    previous[i] + 1,  # deletion
                    current[i - 1] + 1,  # insertion
                    previous[i - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def score_levenshtein(
    output: str, expected: str, threshold: float = 0.8, ignore_case: bool = False
) -> tuple:
    """Pass when the output is close enough to the expected string,
    measured by edit distance.

    Similarity is ``1 - distance / max_len``, where ``distance`` is
    the Levenshtein edit distance and ``max_len`` is the longer of
    the two stripped strings — so a one-character typo in a long
    answer scores near 1.0 and a short answer with the same typo
    scores much lower, which is what an edit-distance scorer is
    for: it charges per edit, where ``fuzzy`` (SequenceMatcher)
    charges per matching block. Pass when the similarity reaches
    ``threshold`` (default 0.8), which must be in [0, 1]. Partial
    credit: the score is the similarity itself, so a near miss
    still earns most of its weight. An exact match is a similarity
    of 1.0; two empty strings match at 1.0, and an empty string
    against a non-empty one scores 0.0. With ``ignore_case``, both
    strings are case-folded before the distance is computed, so
    ``"Hello World"`` and ``"hello world"`` score 1.0.
    """
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("levenshtein scorer requires a 'threshold' in [0, 1]")
    out = _fold(output.strip(), ignore_case)
    exp = _fold(expected.strip(), ignore_case)
    max_len = max(len(out), len(exp))
    if max_len == 0:
        similarity = 1.0
        distance = 0
    else:
        distance = _levenshtein_distance(out, exp)
        similarity = 1.0 - distance / max_len
    ok = similarity >= threshold
    detail = (
        f"levenshtein distance {distance}, similarity {similarity:.2f} "
        f">= threshold {threshold:.2f}"
        if ok
        else f"levenshtein distance {distance}, similarity {similarity:.2f} "
        f"< threshold {threshold:.2f}"
    )
    return ok, detail, similarity


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
    when the scorer gives partial credit (see ``score_contains_all``,
    ``score_not_contains``, ``score_fuzzy``, ``score_levenshtein``, ``score_starts_with``, ``score_ends_with``, ``score_length``, ``score_range``, and ``score_judge``). ``judges`` is the registry of judge callables
    used by a ``judge`` scorer; None means the built-in registry.
    The string scorers (``contains``, ``contains_all``,
    ``contains_any``, ``not_contains``, ``starts_with``,
    ``ends_with``, ``one_of``) and the ``exact``, ``fuzzy``,
    ``levenshtein``, ``regex``, and
    ``contains_regex`` scorers accept an optional
    ``"ignore_case": true`` in the spec:
    both the expected strings and the output are case-folded before
    comparing (``regex`` and ``contains_regex`` match with
    ``re.IGNORECASE``), so
    ``"Hello"`` matches ``"hello"``; scores and partial credit are
    unchanged, and a non-boolean value is a scorer error."""
    spec = case.scorer
    kind = spec.get("type")
    if kind == "exact":
        return score_exact(
            output,
            str(spec["expected"]),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "contains":
        return score_contains(
            output,
            str(spec["substring"]),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "contains_all":
        return score_contains_all(
            output,
            _as_substring_list(spec, kind),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "contains_any":
        return score_contains_any(
            output,
            _as_substring_list(spec, kind),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "not_contains":
        return score_not_contains(
            output,
            _as_substring_list(spec, kind),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "regex":
        return score_regex(
            output,
            str(spec["pattern"]),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "contains_regex":
        if "pattern" not in spec:
            raise ValueError("contains_regex scorer requires a 'pattern' value")
        return score_contains_regex(
            output,
            spec["pattern"],
            _as_bool_option(spec, "ignore_case", kind),
        )
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
    if kind == "json_subset":
        if "expected" not in spec:
            raise ValueError("json_subset scorer requires an 'expected' object")
        return score_json_subset(
            output, spec["expected"], float(spec.get("tolerance", 0.0))
        )
    if kind == "json_equals":
        if "expected" not in spec:
            raise ValueError("json_equals scorer requires an 'expected' value")
        return score_json_equals(
            output,
            spec["expected"],
            _as_bool_option(spec, "ignore_order", kind),
            spec.get("tolerance", 0.0),
        )
    if kind == "numeric":
        return score_numeric(
            output, float(spec["expected"]), float(spec.get("tolerance", 0.0))
        )
    if kind == "fuzzy":
        if "expected" not in spec:
            raise ValueError("fuzzy scorer requires an 'expected' value")
        return score_fuzzy(
            output,
            str(spec["expected"]),
            float(spec.get("threshold", 0.8)),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "levenshtein":
        if "expected" not in spec:
            raise ValueError("levenshtein scorer requires an 'expected' value")
        return score_levenshtein(
            output,
            str(spec["expected"]),
            float(spec.get("threshold", 0.8)),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "starts_with":
        if "prefix" not in spec:
            raise ValueError("starts_with scorer requires a 'prefix' value")
        return score_starts_with(
            output,
            str(spec["prefix"]),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "ends_with":
        if "suffix" not in spec:
            raise ValueError("ends_with scorer requires a 'suffix' value")
        return score_ends_with(
            output,
            str(spec["suffix"]),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "one_of":
        return score_one_of(
            output,
            _as_value_list(spec, kind),
            _as_bool_option(spec, "ignore_case", kind),
        )
    if kind == "length":
        return score_length(
            output,
            _as_length_bound(spec, "min_length", kind),
            _as_length_bound(spec, "max_length", kind),
            spec.get("unit", "characters"),
        )
    if kind == "range":
        return score_range(
            output,
            _as_range_bound(spec, "min", kind),
            _as_range_bound(spec, "max", kind),
            spec.get("tolerance", 0.0),
        )
    if kind == "judge":
        return score_judge(output, case, judges)
    raise ValueError(f"unknown scorer type: {kind!r}")


def make_output_diff(
    expected: str, actual: str, max_lines: Optional[int] = None
) -> Optional[str]:
    """A unified diff of ``expected`` vs ``actual``, or None when they
    are identical.

    Line-based, so a one-word change in a multi-line answer shows just
    the changed lines instead of two full blobs. Only meaningful for
    scorers with a full expected output (today: ``exact``) — a
    ``contains`` case has no expected output to diff against, and a
    ``numeric``/``json_path`` case compares values, not text.
    ``max_lines`` caps the diff (the text summary uses it); the JSON
    report keeps the full diff.
    """
    lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    if not lines:
        return None
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines] + [
            f"... ({len(lines) - max_lines} more diff lines)"
        ]
    return "\n".join(lines)


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
    ``cost_violated`` is true when the case passed its checks but the
    agent reported more tokens than the case's ``max_tokens`` budget;
    such a case is not a pass either (``passed`` is false, ``score``
    is 0.0) and the token count and the budget are in the detail. A
    case whose agent reported no token count can never be a cost
    violation. A case is never both a budget violation and a cost
    violation: the latency budget is checked first.
    ``case_fingerprint`` is the case's content hash (see
    :meth:`Case.fingerprint`): a run diff can compare it with the
    baseline's to tell whether the case itself changed between runs.
    ``diff`` is a unified diff of the expected output vs the actual
    output, present only when an ``exact`` case failed — a failing
    answer is visible at a glance instead of requiring the reader to
    compare two blobs by eye.
    ``tags`` is the case's tag list, carried into the report so a
    filtered run still says which cases it ran.
    ``threshold`` is the case's pass threshold (None when the case
    carries none): the score the case needed to pass, so a report
    says why a case that scored 0.75 counted as passed.
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
    max_tokens: Optional[int] = None
    cost_violated: bool = False
    case_fingerprint: Optional[str] = None
    diff: Optional[str] = None
    tags: list = field(default_factory=list)
    threshold: Optional[float] = None

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
            "max_tokens": self.max_tokens,
            "cost_violated": self.cost_violated,
            "case_fingerprint": self.case_fingerprint,
            "diff": self.diff,
            "tags": list(self.tags),
            "threshold": self.threshold,
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
    """Everything a run produced, serialisable to JSON.

    ``suite_version`` is the eval set's human version label (None when
    the suite carries none) and ``suite_fingerprint`` is the set's
    content hash (see :meth:`Suite.fingerprint`): together they say
    exactly which eval set a run was measured against, so a later
    diff can tell whether a score delta came from the agent or from
    the set changing underneath it.
    """

    suite: str
    started_at: str
    ended_at: str
    results: list = field(default_factory=list)
    suite_version: Optional[str] = None
    suite_fingerprint: Optional[str] = None

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

    @property
    def cost_violations(self) -> int:
        return sum(1 for r in self.results if r.cost_violated)

    @property
    def tag_breakdown(self) -> dict:
        """Per-tag pass rates: tag -> ``{"cases", "passed",
        "pass_rate", "weighted_score"}``.

        A result carrying several tags counts under each of them, so
        a case tagged both ``"smoke"`` and ``"fast"`` counts toward
        both tags' pass rates; untagged results appear under no tag.
        ``weighted_score`` is the tag's earned weight over its total
        weight, so a heavy case counts more, the same as in the
        run-wide weighted score. An empty dict means no case in the
        run carried a tag.
        """
        stats: dict = {}
        for r in self.results:
            for tag in r.tags:
                entry = stats.setdefault(
                    tag,
                    {"cases": 0, "passed": 0, "earned": 0.0, "total": 0.0},
                )
                entry["cases"] += 1
                entry["passed"] += 1 if r.passed else 0
                entry["earned"] += r.score * r.weight
                entry["total"] += r.weight
        return {
            tag: {
                "cases": entry["cases"],
                "passed": entry["passed"],
                "pass_rate": (
                    entry["passed"] / entry["cases"] if entry["cases"] else 0.0
                ),
                "weighted_score": (
                    entry["earned"] / entry["total"] if entry["total"] else 0.0
                ),
            }
            for tag, entry in sorted(stats.items())
        }

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "suite_version": self.suite_version,
            "suite_fingerprint": self.suite_fingerprint,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "total_weight": self.total_weight,
            "earned_weight": round(self.earned_weight, 4),
            "weighted_score": round(self.weighted_score, 4),
            "budget_violations": self.budget_violations,
            "cost_violations": self.cost_violations,
            "tag_breakdown": {
                tag: {
                    "cases": entry["cases"],
                    "passed": entry["passed"],
                    "pass_rate": round(entry["pass_rate"], 4),
                    "weighted_score": round(entry["weighted_score"], 4),
                }
                for tag, entry in self.tag_breakdown.items()
            },
            "total_latency_ms": round(self.total_latency_ms, 3),
            "total_tokens": self.total_tokens,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def save_junit(self, path: str) -> None:
        """Write this report as JUnit XML (see :mod:`agent_evals.junit`),
        so CI systems can ingest eval results."""
        from .junit import to_junit_xml

        with open(path, "w", encoding="utf-8") as f:
            f.write(to_junit_xml(self))
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
    recorded as failed and timed out. A case carrying a ``threshold``
    passes when its scorer's score reaches it, and fails when it does
    not — a timeout, an agent error, or a scorer error still fails the
    case whatever the threshold is.
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
            case_fingerprint=case.fingerprint(),
            tags=list(case.tags),
            threshold=case.threshold,
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
            case_fingerprint=case.fingerprint(),
            tags=list(case.tags),
            threshold=case.threshold,
        )
    try:
        passed, detail, score = score_case(case, output, judges)
        if case.threshold is not None:
            # The case passes when its score reaches its own
            # threshold, either way: a partial-credit fail can pass
            # on a good-enough score, and a scorer pass on a low
            # score can fail. A timeout, an agent error, or a scorer
            # error never reaches this line, so those still fail
            # whatever the threshold is. The case keeps the scorer's
            # score either way, so it earns its partial weight.
            if score >= case.threshold and not passed:
                passed = True
                detail = (
                    f"{detail}; score {score:.2f} meets threshold "
                    f"{case.threshold:g}"
                )
            elif score < case.threshold and passed:
                passed = False
                detail = (
                    f"{detail}; score {score:.2f} below threshold "
                    f"{case.threshold:g}"
                )
    except ValueError as exc:
        passed, detail, score = False, f"scorer error: {exc}", 0.0
    diff = None
    if not passed and case.scorer.get("type") == "exact":
        # The expected output is a full string, so a line diff shows
        # exactly what changed; other scorers have no expected text
        # to diff against.
        expected = case.scorer.get("expected")
        if isinstance(expected, str):
            diff = make_output_diff(expected, output)
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
    cost_violated = False
    if (
        passed
        and case.max_tokens is not None
        and tokens is not None
        and tokens > case.max_tokens
    ):
        # The answer was right but too expensive: a cost violation,
        # not a pass. An agent that reports no token count can never
        # trip this — there is nothing to compare against.
        cost_violated = True
        passed = False
        score = 0.0
        detail = (
            f"{detail}; exceeded token budget "
            f"{tokens} tokens > {case.max_tokens} tokens"
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
        max_tokens=case.max_tokens,
        cost_violated=cost_violated,
        case_fingerprint=case.fingerprint(),
        diff=diff,
        tags=list(case.tags),
        threshold=case.threshold,
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
    return RunReport(
        suite=suite.name,
        started_at=started,
        ended_at=ended,
        results=results,
        suite_version=suite.version,
        suite_fingerprint=suite.fingerprint(),
    )


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
    if report.cost_violations:
        lines.append(f"Cost violations: {report.cost_violations}")
    lines += [
        "",
        f"{'case':<22} {'pass':<7} {'score':>6} {'ms':>8} {'tokens':>7}  detail",
        "-" * 86,
    ]
    for r in report.results:
        if r.budget_violated:
            mark = "BUDGET"
        elif r.cost_violated:
            mark = "COST"
        elif r.timed_out:
            mark = "TIMEOUT"
        else:
            mark = "PASS" if r.passed else "FAIL"
        tokens = str(r.tokens) if r.tokens is not None else "-"
        detail = r.detail
        if r.tags:
            detail += f"  [{', '.join(r.tags)}]"
        lines.append(
            f"{r.case_id:<22} {mark:<7} {r.score:>6.2f} {r.latency_ms:>8.1f} "
            f"{tokens:>7}  {detail}"
        )
        if r.diff is not None:
            # The diff shows what changed; cap it so one huge answer
            # does not drown the table (the JSON report keeps it whole).
            for diff_line in r.diff.splitlines()[:30]:
                lines.append(f"    {diff_line}")
            if len(r.diff.splitlines()) > 30:
                lines.append(
                    f"    ... ({len(r.diff.splitlines()) - 30} more diff lines)"
                )
    breakdown = report.tag_breakdown
    if breakdown:
        # A tag that keeps failing is visible at a glance, without
        # digging through the per-case rows.
        lines.append("")
        lines.append("Per-tag pass rates:")
        width = max(len(tag) for tag in breakdown)
        for tag, stats in breakdown.items():
            lines.append(
                f"  {tag:<{width}}  {stats['passed']}/{stats['cases']} "
                f"({stats['pass_rate'] * 100:.1f}%)"
            )
    return "\n".join(lines)
