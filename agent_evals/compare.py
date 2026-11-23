"""Compare two run reports and surface regressions.

An eval harness is only useful if a new agent version can be checked
against the last known-good one. ``compare_reports`` takes two saved
JSON reports (as loaded by ``load_report``) and classifies every case
by how it moved between the runs:

* ``regressions`` — passed in the baseline, failed in the current run.
  A case that passed and now times out is a regression too.
* ``fixes`` — failed in the baseline, passed in the current run.
* ``new`` — present only in the current run.
* ``removed`` — present only in the baseline.
* ``score_changes`` — failed in both runs but earned a different
  partial score, so progress (or decay) is visible even before a pass.
* ``unchanged`` — same verdict and same score in both runs.

On top of the verdict movement, the comparison attributes movement to
its cause. Run reports record the eval set's version label and
content fingerprint plus a fingerprint per case, so the comparison
can tell whether the set itself changed between the runs:

* ``eval_set_changed`` — the two runs were measured against different
  eval sets (different suite fingerprints, or different version
  labels when a fingerprint is missing from an old report).
* ``case_changed`` per shared case — the case's own definition
  (prompt, scorer, timeout, weight, budget) changed between the runs.
  A regression or fix on a changed case may come from the set, not
  the agent; a regression on an unchanged case is an agent regression.

The comparison also carries the headline deltas: pass rate,
weighted score, total latency, and total tokens, plus a per-case
latency and token change for shared cases, so a run that passes
everything but got much slower or more expensive is visible too.
It also names an overall winner — the run with the better weighted
score, then pass rate, then lowest total latency, then lowest total
tokens — and prints every case side by side, baseline vs current,
with its score, latency, and token deltas.

Usage:
    python -m agent_evals.compare --baseline reports/old.json \
        --current reports/new.json
    python -m agent_evals.compare --baseline reports/old.json \
        --current reports/new.json --json reports/diff.json

Exit code is 0 when there is no regression, 1 when at least one case
regressed. Fixes, new cases, and score improvements never fail the
comparison — only a regression does.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional

from .evals import load_report

__all__ = [
    "CaseChange",
    "ReportComparison",
    "compare_reports",
    "format_comparison",
]


@dataclass
class CaseChange:
    """How one case moved between the baseline and the current run.

    ``case_changed`` is True when the case's own definition (prompt,
    scorer, timeout, weight, budget) differs between the runs, False
    when it is identical, and None when either report predates
    per-case fingerprints and the comparison cannot tell.
    """

    case_id: str
    baseline_passed: Optional[bool]
    current_passed: Optional[bool]
    baseline_score: Optional[float]
    current_score: Optional[float]
    baseline_latency_ms: Optional[float] = None
    current_latency_ms: Optional[float] = None
    baseline_tokens: Optional[int] = None
    current_tokens: Optional[int] = None
    case_changed: Optional[bool] = None

    @property
    def latency_delta_ms(self) -> Optional[float]:
        """Change in latency in ms (positive = slower), or None when
        either run has no latency for the case."""
        if self.baseline_latency_ms is None or self.current_latency_ms is None:
            return None
        return round(self.current_latency_ms - self.baseline_latency_ms, 3)

    @property
    def token_delta(self) -> Optional[int]:
        """Change in tokens (positive = more expensive), or None when
        either run has no token count for the case — an agent that
        reports no tokens has nothing to compare."""
        if self.baseline_tokens is None or self.current_tokens is None:
            return None
        return self.current_tokens - self.baseline_tokens

    @property
    def score_delta(self) -> Optional[float]:
        """Change in score (positive = better), or None when either
        run has no score for the case."""
        if self.baseline_score is None or self.current_score is None:
            return None
        return round(self.current_score - self.baseline_score, 4)

    @property
    def kind(self) -> str:
        """One of: regression, fix, new, removed, score_change, unchanged."""
        if self.baseline_passed is None:
            return "new"
        if self.current_passed is None:
            return "removed"
        if self.baseline_passed and not self.current_passed:
            return "regression"
        if not self.baseline_passed and self.current_passed:
            return "fix"
        if self.baseline_score != self.current_score:
            return "score_change"
        return "unchanged"

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "case_changed": self.case_changed,
            "baseline_passed": self.baseline_passed,
            "current_passed": self.current_passed,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "score_delta": self.score_delta,
            "baseline_latency_ms": self.baseline_latency_ms,
            "current_latency_ms": self.current_latency_ms,
            "latency_delta_ms": self.latency_delta_ms,
            "baseline_tokens": self.baseline_tokens,
            "current_tokens": self.current_tokens,
            "token_delta": self.token_delta,
        }


@dataclass
class ReportComparison:
    """The result of comparing two run reports.

    ``baseline_suite_version`` / ``current_suite_version`` are the
    eval sets' human version labels (None when a report predates
    versioning), and the ``*_suite_fingerprint`` fields are the sets'
    content hashes. ``eval_set_changed`` is True when the two runs
    were measured against different eval sets — different suite
    fingerprints, or different version labels when a fingerprint is
    missing — so a score delta between the runs may come from the set
    rather than the agent.
    """

    baseline_suite: str
    current_suite: str
    baseline_passed: int
    baseline_total: int
    current_passed: int
    current_total: int
    baseline_weighted_score: float
    current_weighted_score: float
    baseline_total_latency_ms: float
    current_total_latency_ms: float
    baseline_total_tokens: Optional[int] = None
    current_total_tokens: Optional[int] = None
    changes: list = field(default_factory=list)
    baseline_suite_version: Optional[str] = None
    current_suite_version: Optional[str] = None
    baseline_suite_fingerprint: Optional[str] = None
    current_suite_fingerprint: Optional[str] = None

    @property
    def eval_set_changed(self) -> bool:
        """True when the two runs were measured against different eval
        sets. Fingerprints decide when both reports carry them; when
        one is missing (a report predating versioning), differing
        version labels decide; when neither is available the sets
        are treated as unchanged, since there is no evidence either
        way."""
        if (
            self.baseline_suite_fingerprint is not None
            and self.current_suite_fingerprint is not None
        ):
            return (
                self.baseline_suite_fingerprint
                != self.current_suite_fingerprint
            )
        if (
            self.baseline_suite_version is not None
            and self.current_suite_version is not None
        ):
            return self.baseline_suite_version != self.current_suite_version
        return False

    @property
    def changed_cases(self) -> list:
        """Shared cases whose own definition changed between the runs
        (a different prompt, scorer, timeout, weight, or budget)."""
        return [c for c in self.changes if c.case_changed]

    @property
    def regressions(self) -> list:
        return [c for c in self.changes if c.kind == "regression"]

    @property
    def fixes(self) -> list:
        return [c for c in self.changes if c.kind == "fix"]

    @property
    def new_cases(self) -> list:
        return [c for c in self.changes if c.kind == "new"]

    @property
    def removed_cases(self) -> list:
        return [c for c in self.changes if c.kind == "removed"]

    @property
    def score_changes(self) -> list:
        return [c for c in self.changes if c.kind == "score_change"]

    @property
    def has_regression(self) -> bool:
        return bool(self.regressions)

    @property
    def baseline_pass_rate(self) -> float:
        return (
            (self.baseline_passed / self.baseline_total)
            if self.baseline_total
            else 0.0
        )

    @property
    def current_pass_rate(self) -> float:
        return (
            (self.current_passed / self.current_total)
            if self.current_total
            else 0.0
        )

    @property
    def pass_rate_delta(self) -> float:
        return round(self.current_pass_rate - self.baseline_pass_rate, 4)

    @property
    def weighted_score_delta(self) -> float:
        return round(
            self.current_weighted_score - self.baseline_weighted_score, 4
        )

    @property
    def latency_delta_ms(self) -> float:
        return round(
            self.current_total_latency_ms - self.baseline_total_latency_ms, 3
        )

    @property
    def token_delta(self) -> Optional[int]:
        """Change in total tokens (positive = more expensive), or
        None when either run reported no token count — an agent that
        reports no tokens has nothing to compare."""
        if (
            self.baseline_total_tokens is None
            or self.current_total_tokens is None
        ):
            return None
        return self.current_total_tokens - self.baseline_total_tokens

    @property
    def winner(self) -> Optional[str]:
        """Which run won overall: ``"baseline"``, ``"current"``, or
        None on a tie. Ranked the same way the per-model comparison
        ranks agents: weighted score, then pass rate, then lowest
        total latency, then lowest total tokens (only when both runs
        reported tokens — otherwise cost does not break the tie).
        A regression can still win on score: the winner is the
        better run overall, the verdict is the CI gate."""
        base_tokens = (
            self.baseline_total_tokens
            if self.token_delta is not None
            else 0
        )
        cur_tokens = (
            self.current_total_tokens if self.token_delta is not None else 0
        )
        base = (
            -self.baseline_weighted_score,
            -self.baseline_pass_rate,
            self.baseline_total_latency_ms,
            base_tokens,
        )
        cur = (
            -self.current_weighted_score,
            -self.current_pass_rate,
            self.current_total_latency_ms,
            cur_tokens,
        )
        if base < cur:
            return "baseline"
        if cur < base:
            return "current"
        return None

    @property
    def verdict(self) -> str:
        """REGRESSION when anything got worse, IMPROVED when anything
        got better and nothing regressed, UNCHANGED otherwise."""
        if self.has_regression:
            return "REGRESSION"
        if self.fixes or self.score_changes or self.new_cases:
            return "IMPROVED"
        return "UNCHANGED"

    def to_dict(self) -> dict:
        return {
            "baseline_suite": self.baseline_suite,
            "current_suite": self.current_suite,
            "baseline_suite_version": self.baseline_suite_version,
            "current_suite_version": self.current_suite_version,
            "baseline_suite_fingerprint": self.baseline_suite_fingerprint,
            "current_suite_fingerprint": self.current_suite_fingerprint,
            "eval_set_changed": self.eval_set_changed,
            "baseline_passed": self.baseline_passed,
            "baseline_total": self.baseline_total,
            "current_passed": self.current_passed,
            "current_total": self.current_total,
            "pass_rate_delta": self.pass_rate_delta,
            "baseline_weighted_score": self.baseline_weighted_score,
            "current_weighted_score": self.current_weighted_score,
            "weighted_score_delta": self.weighted_score_delta,
            "baseline_total_latency_ms": self.baseline_total_latency_ms,
            "current_total_latency_ms": self.current_total_latency_ms,
            "latency_delta_ms": self.latency_delta_ms,
            "baseline_total_tokens": self.baseline_total_tokens,
            "current_total_tokens": self.current_total_tokens,
            "token_delta": self.token_delta,
            "winner": self.winner,
            "verdict": self.verdict,
            "regressions": [c.case_id for c in self.regressions],
            "fixes": [c.case_id for c in self.fixes],
            "new": [c.case_id for c in self.new_cases],
            "removed": [c.case_id for c in self.removed_cases],
            "changed": [c.case_id for c in self.changed_cases],
            "changes": [c.to_dict() for c in self.changes],
        }


def _results_by_id(report: dict) -> dict:
    """Index a report's results by case_id. Duplicate ids are a
    malformed report: the first occurrence wins and the rest are
    ignored."""
    by_id = {}
    for result in report.get("results", []):
        case_id = result.get("case_id")
        if case_id is not None and case_id not in by_id:
            by_id[case_id] = result
    return by_id


def _total_tokens(report: dict) -> Optional[int]:
    """A report's total token count: the report-level total when the
    report carries one, else the sum of the per-result counts, else
    None when no result reported tokens."""
    total = report.get("total_tokens")
    if total is not None:
        return int(total)
    tokens = [
        r.get("tokens")
        for r in report.get("results", [])
        if r.get("tokens") is not None
    ]
    return sum(tokens) if tokens else None


def compare_reports(baseline: dict, current: dict) -> ReportComparison:
    """Compare two saved run reports (dicts as returned by
    ``load_report``) and return the classified differences.

    The two reports need not be from the same suite: a suite-name
    mismatch is recorded on the comparison, and cases are matched by
    ``case_id`` only, so a renamed suite still compares sensibly.
    """
    base_results = _results_by_id(baseline)
    cur_results = _results_by_id(current)
    changes = []
    # Baseline order first, then any cases only in the current run.
    for case_id in list(base_results) + [
        c for c in cur_results if c not in base_results
    ]:
        base = base_results.get(case_id)
        cur = cur_results.get(case_id)
        base_fp = base.get("case_fingerprint") if base else None
        cur_fp = cur.get("case_fingerprint") if cur else None
        # Both fingerprints present: the case definition changed iff
        # they differ. Either missing (a report predating versioning):
        # unknown — a new or removed case has no counterpart to compare.
        if base is not None and cur is not None:
            case_changed = (
                None if base_fp is None or cur_fp is None else base_fp != cur_fp
            )
        else:
            case_changed = None
        changes.append(
            CaseChange(
                case_id=case_id,
                baseline_passed=base["passed"] if base else None,
                current_passed=cur["passed"] if cur else None,
                baseline_score=base.get("score") if base else None,
                current_score=cur.get("score") if cur else None,
                baseline_latency_ms=(
                    base.get("latency_ms") if base else None
                ),
                current_latency_ms=(
                    cur.get("latency_ms") if cur else None
                ),
                baseline_tokens=base.get("tokens") if base else None,
                current_tokens=cur.get("tokens") if cur else None,
                case_changed=case_changed,
            )
        )
    return ReportComparison(
        baseline_suite=str(baseline.get("suite", "?")),
        current_suite=str(current.get("suite", "?")),
        baseline_passed=int(baseline.get("passed", 0)),
        baseline_total=int(baseline.get("total", 0)),
        current_passed=int(current.get("passed", 0)),
        current_total=int(current.get("total", 0)),
        baseline_weighted_score=float(
            baseline.get("weighted_score", 0.0)
        ),
        current_weighted_score=float(
            current.get("weighted_score", 0.0)
        ),
        baseline_total_latency_ms=float(
            baseline.get("total_latency_ms", 0.0)
        ),
        current_total_latency_ms=float(
            current.get("total_latency_ms", 0.0)
        ),
        baseline_total_tokens=_total_tokens(baseline),
        current_total_tokens=_total_tokens(current),
        changes=changes,
        baseline_suite_version=baseline.get("suite_version"),
        current_suite_version=current.get("suite_version"),
        baseline_suite_fingerprint=baseline.get("suite_fingerprint"),
        current_suite_fingerprint=current.get("suite_fingerprint"),
    )


def _set_label(version, fingerprint) -> str:
    """Render one run's eval set as ``v1.2 (sha abc123...)`` or just
    the fingerprint, or "" when the report predates versioning."""
    if version is not None and fingerprint is not None:
        return f"v{version} (sha {fingerprint})"
    if version is not None:
        return f"v{version}"
    if fingerprint is not None:
        return f"sha {fingerprint}"
    return ""


def _format_side_by_side(comp: ReportComparison) -> str:
    """Every case in one table, baseline vs current side by side:
    score, latency, and tokens with their deltas, plus how the case
    moved. A missing side (a new or removed case, or a run that
    reported no tokens) shows "-". Token columns appear only when
    some case reported tokens in either run."""
    lines = ["Per-case (side by side):"]
    if not comp.changes:
        lines.append("  (no cases in either run)")
        return "\n".join(lines)

    def num(value, fmt):
        return fmt.format(value) if value is not None else "-"

    show_tokens = any(
        c.baseline_tokens is not None or c.current_tokens is not None
        for c in comp.changes
    )
    # Each column: (header, cells, right-aligned?).
    columns = [
        ("case", [c.case_id for c in comp.changes], False),
        (
            "base_score",
            [num(c.baseline_score, "{:.2f}") for c in comp.changes],
            True,
        ),
        (
            "cur_score",
            [num(c.current_score, "{:.2f}") for c in comp.changes],
            True,
        ),
        (
            "Δscore",
            [num(c.score_delta, "{:+.2f}") for c in comp.changes],
            True,
        ),
        (
            "base_ms",
            [num(c.baseline_latency_ms, "{:.1f}") for c in comp.changes],
            True,
        ),
        (
            "cur_ms",
            [num(c.current_latency_ms, "{:.1f}") for c in comp.changes],
            True,
        ),
        (
            "Δms",
            [num(c.latency_delta_ms, "{:+.1f}") for c in comp.changes],
            True,
        ),
    ]
    if show_tokens:
        columns += [
            (
                "base_tok",
                [num(c.baseline_tokens, "{:d}") for c in comp.changes],
                True,
            ),
            (
                "cur_tok",
                [num(c.current_tokens, "{:d}") for c in comp.changes],
                True,
            ),
            (
                "Δtok",
                [num(c.token_delta, "{:+d}") for c in comp.changes],
                True,
            ),
        ]
    columns.append(
        (
            "movement",
            [c.kind if c.kind != "unchanged" else "" for c in comp.changes],
            False,
        )
    )
    widths = [
        max(len(header), *(len(cell) for cell in cells))
        for header, cells, _ in columns
    ]

    def render(cells):
        return "  ".join(
            f"{cell:>{w}}" if right else f"{cell:<{w}}"
            for cell, w, (_, _, right) in zip(cells, widths, columns)
        )

    header = render([header for header, _, _ in columns])
    lines.append(header)
    lines.append("-" * len(header))
    for i in range(len(comp.changes)):
        lines.append(
            render([cells[i] for _, cells, _ in columns]).rstrip()
        )
    return "\n".join(lines)


def format_comparison(comp: ReportComparison) -> str:
    """Render a comparison as a plain-text summary."""
    lines = [
        f"Baseline: {comp.baseline_suite} "
        f"{comp.baseline_passed}/{comp.baseline_total} passed",
        f"Current:  {comp.current_suite} "
        f"{comp.current_passed}/{comp.current_total} passed",
    ]
    if comp.baseline_suite != comp.current_suite:
        lines.append(
            f"Note: suite names differ "
            f"({comp.baseline_suite!r} vs {comp.current_suite!r}); "
            f"cases matched by case_id only"
        )
    base_set = _set_label(
        comp.baseline_suite_version, comp.baseline_suite_fingerprint
    )
    cur_set = _set_label(
        comp.current_suite_version, comp.current_suite_fingerprint
    )
    if base_set or cur_set:
        lines.append(f"Eval set: baseline {base_set or '(unknown)'} -> "
                     f"current {cur_set or '(unknown)'}")
        if comp.eval_set_changed:
            lines.append(
                "Note: the eval set changed between these runs — "
                "score deltas may come from the set, not the agent"
            )
    delta = comp.pass_rate_delta
    lines.append(
        f"Pass rate: {delta * 100:+.1f}%  "
        f"Weighted score: {comp.weighted_score_delta:+.4f}  "
        f"Latency: {comp.latency_delta_ms:+.1f} ms"
    )
    if comp.token_delta is not None:
        lines.append(
            f"Tokens: {comp.token_delta:+d} "
            f"({comp.baseline_total_tokens} -> {comp.current_total_tokens})"
        )
    lines.append(f"Winner: {comp.winner or 'tie'}")
    lines.append(f"Verdict: {comp.verdict}")
    lines.append("")
    lines.append(_format_side_by_side(comp))
    lines.append("")
    sections = [
        ("Regressions", comp.regressions),
        ("Fixes", comp.fixes),
        ("Score changes (still failing)", comp.score_changes),
        ("New cases", comp.new_cases),
        ("Removed cases", comp.removed_cases),
    ]
    for title, group in sections:
        if not group:
            continue
        lines.append(f"{title} ({len(group)}):")
        for change in group:
            latency = change.latency_delta_ms
            latency_note = (
                f"  latency {latency:+.1f} ms" if latency is not None else ""
            )
            changed_note = (
                "  (case changed — may be the set, not the agent)"
                if change.case_changed
                else ""
            )
            if change.kind in ("regression", "fix"):
                lines.append(
                    f"  {change.case_id}{latency_note}{changed_note}"
                )
            elif change.kind == "score_change":
                lines.append(
                    f"  {change.case_id}: score "
                    f"{change.baseline_score} -> {change.current_score}"
                    f"{latency_note}{changed_note}"
                )
            else:
                lines.append(f"  {change.case_id}")
        lines.append("")
    if comp.changed_cases:
        lines.append(
            f"Changed cases ({len(comp.changed_cases)}): the case "
            f"definition changed between runs, so its movement may "
            f"come from the set, not the agent:"
        )
        for change in comp.changed_cases:
            lines.append(f"  {change.case_id}: {change.kind}")
        lines.append("")
    if comp.verdict == "UNCHANGED":
        lines.append("No changes: same verdicts, scores, and case set.")
    return "\n".join(lines).rstrip("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals-compare",
        description="Compare two saved run reports and report regressions.",
    )
    parser.add_argument(
        "--baseline", required=True, help="path to the baseline JSON report"
    )
    parser.add_argument(
        "--current", required=True, help="path to the current JSON report"
    )
    parser.add_argument(
        "--json", help="optional path to write the comparison as JSON"
    )
    args = parser.parse_args(argv)

    baseline = load_report(args.baseline)
    current = load_report(args.current)
    comp = compare_reports(baseline, current)
    print(format_comparison(comp))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(comp.to_dict(), f, indent=2)
            f.write("\n")
        print(f"\nComparison written to {args.json}")
    return 1 if comp.has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
