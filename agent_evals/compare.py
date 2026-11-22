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

The comparison also carries the headline deltas: pass rate,
weighted score, and total latency, plus a per-case latency change
for shared cases, so a run that passes everything but got much
slower is visible too.

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
    """How one case moved between the baseline and the current run."""

    case_id: str
    baseline_passed: Optional[bool]
    current_passed: Optional[bool]
    baseline_score: Optional[float]
    current_score: Optional[float]
    baseline_latency_ms: Optional[float] = None
    current_latency_ms: Optional[float] = None

    @property
    def latency_delta_ms(self) -> Optional[float]:
        """Change in latency in ms (positive = slower), or None when
        either run has no latency for the case."""
        if self.baseline_latency_ms is None or self.current_latency_ms is None:
            return None
        return round(self.current_latency_ms - self.baseline_latency_ms, 3)

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
            "baseline_passed": self.baseline_passed,
            "current_passed": self.current_passed,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "baseline_latency_ms": self.baseline_latency_ms,
            "current_latency_ms": self.current_latency_ms,
            "latency_delta_ms": self.latency_delta_ms,
        }


@dataclass
class ReportComparison:
    """The result of comparing two run reports."""

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
    changes: list = field(default_factory=list)

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
    def pass_rate_delta(self) -> float:
        base = (self.baseline_passed / self.baseline_total) if self.baseline_total else 0.0
        cur = (self.current_passed / self.current_total) if self.current_total else 0.0
        return round(cur - base, 4)

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
            "verdict": self.verdict,
            "regressions": [c.case_id for c in self.regressions],
            "fixes": [c.case_id for c in self.fixes],
            "new": [c.case_id for c in self.new_cases],
            "removed": [c.case_id for c in self.removed_cases],
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
        changes=changes,
    )


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
    delta = comp.pass_rate_delta
    lines.append(
        f"Pass rate: {delta * 100:+.1f}%  "
        f"Weighted score: {comp.weighted_score_delta:+.4f}  "
        f"Latency: {comp.latency_delta_ms:+.1f} ms"
    )
    lines.append(f"Verdict: {comp.verdict}")
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
            if change.kind in ("regression", "fix"):
                lines.append(f"  {change.case_id}{latency_note}")
            elif change.kind == "score_change":
                lines.append(
                    f"  {change.case_id}: score "
                    f"{change.baseline_score} -> {change.current_score}"
                    f"{latency_note}"
                )
            else:
                lines.append(f"  {change.case_id}")
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
