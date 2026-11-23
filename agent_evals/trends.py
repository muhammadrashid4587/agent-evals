"""Compare run reports across a whole reports directory.

A two-run diff (see :mod:`agent_evals.compare`) says what changed
between yesterday and today. A directory of reports says how the
agent has been trending: which cases pass reliably, which are
decaying, and which regressions were never fixed.

``compare_history`` takes a list of saved run reports (dicts as
returned by ``load_report``), ordered oldest first, and builds a
:class:`TrendReport`:

* per-case pass rate over the whole history, plus the score in each
  run, so a case that used to pass and now fails sometimes is visible
  as a number, not a feeling;
* newly failing / newly passing cases — what moved in the latest run
  compared with the previous run that contained the case;
* worst regressions across the history — cases that passed at least
  once and are failing in their latest appearance, ranked by how
  reliably they used to pass (a case that passed 9 of 10 runs and
  now fails is a worse regression than one that failed half the
  time anyway);
* flaky cases — cases that passed sometimes and failed other times
  across the history, the same signal as a repeated run, but over
  real runs instead of one burst;
* the headline trend: pass rate and weighted score from the first
  run to the latest, and a per-run table.

Usage:
    python -m agent_evals.trends --reports-dir reports/
    python -m agent_evals.trends --reports-dir reports/ --json trends.json

The CLI loads every ``*.json`` file in the directory (sorted by the
report's ``started_at``, falling back to filename order) and skips
files that are not run reports. Exit code is 0 when nothing newly
failed in the latest run, 1 when at least one case newly failed —
the same gating rule as the two-run comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .evals import load_report

__all__ = [
    "CaseTrend",
    "RunSummary",
    "TrendReport",
    "compare_history",
    "format_trend",
    "load_reports_dir",
]


# ---------------------------------------------------------------------------
# Per-case trend
# ---------------------------------------------------------------------------


@dataclass
class CaseTrend:
    """How one case behaved across the whole report history.

    ``passed_history`` / ``score_history`` / ``latency_history`` are
    aligned with the reports list: one entry per report, ``None``
    when the case was absent from that run (a case added or removed
    mid-history).
    """

    case_id: str
    passed_history: list = field(default_factory=list)
    score_history: list = field(default_factory=list)
    latency_history: list = field(default_factory=list)

    @property
    def runs_present(self) -> int:
        """How many of the reports contain this case."""
        return sum(1 for p in self.passed_history if p is not None)

    @property
    def passes(self) -> int:
        return sum(1 for p in self.passed_history if p)

    @property
    def pass_rate(self) -> float:
        return (self.passes / self.runs_present) if self.runs_present else 0.0

    @property
    def ever_passed(self) -> bool:
        return any(p for p in self.passed_history if p is not None)

    @property
    def ever_failed(self) -> bool:
        return any(p is False for p in self.passed_history)

    @property
    def first_passed(self) -> Optional[bool]:
        """The verdict in the first run that contained the case."""
        for p in self.passed_history:
            if p is not None:
                return p
        return None

    @property
    def last_passed(self) -> Optional[bool]:
        """The verdict in the latest run that contained the case."""
        for p in reversed(self.passed_history):
            if p is not None:
                return p
        return None

    @property
    def first_score(self) -> Optional[float]:
        for s in self.score_history:
            if s is not None:
                return s
        return None

    @property
    def last_score(self) -> Optional[float]:
        for s in reversed(self.score_history):
            if s is not None:
                return s
        return None

    @property
    def score_delta(self) -> Optional[float]:
        """Last score minus first score (positive = improved)."""
        if self.first_score is None or self.last_score is None:
            return None
        return round(self.last_score - self.first_score, 4)

    @property
    def worst_drop(self) -> float:
        """The largest score drop between two consecutive runs that
        contained the case (0.0 when the score never dropped). A
        pass (1.0) that became a fail (0.0) is a drop of 1.0; partial
        credit shrinking shows up as a smaller drop."""
        drop = 0.0
        previous = None
        for score in self.score_history:
            if score is None:
                continue
            if previous is not None:
                drop = max(drop, previous - score)
            previous = score
        return round(drop, 4)

    @property
    def newly_failing(self) -> bool:
        """Failed in its latest run, after passing in the previous
        run that contained it. A case that failed from the start, or
        that only ever appeared once, is not newly failing."""
        present = [p for p in self.passed_history if p is not None]
        return len(present) >= 2 and present[-1] is False and present[-2] is True

    @property
    def newly_passing(self) -> bool:
        """Passed in its latest run, after failing in the previous
        run that contained it."""
        present = [p for p in self.passed_history if p is not None]
        return len(present) >= 2 and present[-1] is True and present[-2] is False

    @property
    def current_failing_streak(self) -> int:
        """Consecutive failures ending at the latest run that
        contained the case (0 when the latest verdict was a pass)."""
        streak = 0
        for p in reversed(self.passed_history):
            if p is None:
                continue
            if p:
                break
            streak += 1
        return streak

    @property
    def flaky(self) -> bool:
        """Passed at least once and failed at least once across the
        history, so the verdict is not reproducible run to run."""
        return self.ever_passed and self.ever_failed

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "runs_present": self.runs_present,
            "passes": self.passes,
            "pass_rate": round(self.pass_rate, 4),
            "passed_history": self.passed_history,
            "score_history": self.score_history,
            "first_score": self.first_score,
            "last_score": self.last_score,
            "score_delta": self.score_delta,
            "worst_drop": self.worst_drop,
            "newly_failing": self.newly_failing,
            "newly_passing": self.newly_passing,
            "current_failing_streak": self.current_failing_streak,
            "flaky": self.flaky,
        }


# ---------------------------------------------------------------------------
# Whole-history report
# ---------------------------------------------------------------------------


@dataclass
class RunSummary:
    """One run in the history, as a row in the trend table."""

    path: str
    started_at: Optional[str]
    suite: str
    passed: int
    total: int
    pass_rate: float
    weighted_score: float
    total_latency_ms: float

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "started_at": self.started_at,
            "suite": self.suite,
            "passed": self.passed,
            "total": self.total,
            "pass_rate": round(self.pass_rate, 4),
            "weighted_score": round(self.weighted_score, 4),
            "total_latency_ms": round(self.total_latency_ms, 3),
        }


@dataclass
class TrendReport:
    """The trend across a directory of run reports, oldest first."""

    runs: list = field(default_factory=list)
    cases: list = field(default_factory=list)

    @property
    def n_reports(self) -> int:
        return len(self.runs)

    @property
    def newly_failing(self) -> list:
        return [c for c in self.cases if c.newly_failing]

    @property
    def newly_passing(self) -> list:
        return [c for c in self.cases if c.newly_passing]

    @property
    def flaky_cases(self) -> list:
        return [c for c in self.cases if c.flaky]

    @property
    def worst_regressions(self) -> list:
        """Cases that passed at least once and are failing in their
        latest appearance, worst first: a case that passed nearly
        every run and now fails outranks one that was already
        failing half the time; ties break on the largest single
        score drop between consecutive runs."""
        regressed = [
            c
            for c in self.cases
            if c.ever_passed and c.last_passed is False
        ]
        return sorted(
            regressed,
            key=lambda c: (-c.pass_rate, -c.worst_drop, c.case_id),
        )

    @property
    def has_new_failure(self) -> bool:
        return bool(self.newly_failing)

    @property
    def pass_rate_delta(self) -> Optional[float]:
        """Latest run's pass rate minus the first run's (None when
        there is only one run)."""
        if len(self.runs) < 2:
            return None
        return round(self.runs[-1].pass_rate - self.runs[0].pass_rate, 4)

    @property
    def weighted_score_delta(self) -> Optional[float]:
        """Latest run's weighted score minus the first run's."""
        if len(self.runs) < 2:
            return None
        return round(
            self.runs[-1].weighted_score - self.runs[0].weighted_score, 4
        )

    def to_dict(self) -> dict:
        return {
            "n_reports": self.n_reports,
            "pass_rate_delta": self.pass_rate_delta,
            "weighted_score_delta": self.weighted_score_delta,
            "newly_failing": [c.case_id for c in self.newly_failing],
            "newly_passing": [c.case_id for c in self.newly_passing],
            "worst_regressions": [c.case_id for c in self.worst_regressions],
            "flaky": [c.case_id for c in self.flaky_cases],
            "runs": [r.to_dict() for r in self.runs],
            "cases": [c.to_dict() for c in self.cases],
        }


def _results_by_id(report: dict) -> dict:
    """Index a report's results by case_id (first occurrence wins on
    duplicates, same as the two-run comparison)."""
    by_id = {}
    for result in report.get("results", []) or []:
        case_id = result.get("case_id")
        if case_id is not None and case_id not in by_id:
            by_id[case_id] = result
    return by_id


def compare_history(reports: list) -> TrendReport:
    """Build the trend across a list of run reports, oldest first.

    Reports are dicts as returned by :func:`agent_evals.evals.load_report`.
    The caller is responsible for the order; the CLI sorts by
    ``started_at``. Cases are matched by ``case_id``; a case absent
    from a run gets ``None`` in that run's slot, so a case added or
    removed mid-history does not skew the others' pass rates.
    """
    runs = []
    by_case: dict[str, CaseTrend] = {}
    for report in reports:
        results = _results_by_id(report)
        total = int(report.get("total", len(results)))
        passed = int(report.get("passed", 0))
        runs.append(
            RunSummary(
                path=str(report.get("_path", "?")),
                started_at=report.get("started_at"),
                suite=str(report.get("suite", "?")),
                passed=passed,
                total=total,
                pass_rate=(passed / total) if total else 0.0,
                weighted_score=float(report.get("weighted_score", 0.0)),
                total_latency_ms=float(report.get("total_latency_ms", 0.0)),
            )
        )
        for case_id, trend in by_case.items():
            result = results.get(case_id)
            trend.passed_history.append(
                bool(result["passed"]) if result else None
            )
            trend.score_history.append(
                result.get("score") if result else None
            )
            trend.latency_history.append(
                result.get("latency_ms") if result else None
            )
        for case_id, result in results.items():
            if case_id not in by_case:
                # Backfill None for the earlier runs this case was
                # not part of, so every history lines up with runs.
                trend = CaseTrend(
                    case_id=case_id,
                    passed_history=[None] * (len(runs) - 1),
                    score_history=[None] * (len(runs) - 1),
                    latency_history=[None] * (len(runs) - 1),
                )
                by_case[case_id] = trend
                trend.passed_history.append(bool(result["passed"]))
                trend.score_history.append(result.get("score"))
                trend.latency_history.append(result.get("latency_ms"))
    # Cases first seen in an early run were appended to on every later
    # run; cases first seen later got backfilled. Order the trend by
    # first appearance.
    def _first_seen(case: CaseTrend) -> int:
        for i, passed in enumerate(case.passed_history):
            if passed is not None:
                return i
        return len(case.passed_history)

    cases = sorted(by_case.values(), key=_first_seen)
    return TrendReport(runs=runs, cases=cases)


def load_reports_dir(directory: str) -> list:
    """Load every ``*.json`` run report in ``directory``, oldest first.

    Reports sort by ``started_at`` when present, falling back to
    filename order for reports without one. A file that is not valid
    JSON, or not a run report (no ``results`` list), is skipped with a
    warning on stderr — one bad file must not hide the history.
    Each loaded report carries its filename under ``"_path"``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"reports directory not found: {directory}")
    loaded = []
    for path in sorted(directory.glob("*.json")):
        try:
            report = load_report(str(path))
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(report, dict) or not isinstance(
            report.get("results"), list
        ):
            print(
                f"warning: skipping {path.name}: not a run report "
                f"(no 'results' list)",
                file=sys.stderr,
            )
            continue
        report["_path"] = path.name
        loaded.append(report)
    loaded.sort(
        key=lambda r: (r.get("started_at") or "", r.get("_path", ""))
    )
    return loaded


def format_trend(trend: TrendReport) -> str:
    """Render a trend report as a plain-text summary."""
    lines = [f"Reports: {trend.n_reports}"]
    if trend.runs:
        first = trend.runs[0]
        last = trend.runs[-1]
        span = (
            f"{first.started_at or '?'} -> {last.started_at or '?'}"
            if trend.n_reports > 1
            else f"{first.started_at or '?'}"
        )
        lines.append(f"Span: {span}")
        suites = {r.suite for r in trend.runs}
        if len(suites) > 1:
            lines.append(
                f"Note: reports span {len(suites)} suites "
                f"({', '.join(sorted(suites))}); cases matched by case_id"
            )
    if trend.pass_rate_delta is not None:
        lines.append(
            f"Pass rate trend: {trend.pass_rate_delta * 100:+.1f}%  "
            f"Weighted score trend: {trend.weighted_score_delta:+.4f}"
        )
    lines.append("")
    lines.append(
        f"{'run':<28} {'passed':>10} {'pass rate':>10} "
        f"{'weighted':>9} {'latency':>9}"
    )
    lines.append("-" * 72)
    for run in trend.runs:
        lines.append(
            f"{run.path:<28} {run.passed}/{run.total:<8} "
            f"{run.pass_rate * 100:>9.1f}% {run.weighted_score:>9.4f} "
            f"{run.total_latency_ms:>8.1f} ms"
        )
    lines.append("")

    def _section(title, group, render):
        if not group:
            return
        lines.append(f"{title} ({len(group)}):")
        for item in group:
            lines.append(f"  {render(item)}")
        lines.append("")

    _section(
        "Newly failing",
        trend.newly_failing,
        lambda c: (
            f"{c.case_id} (pass rate {c.pass_rate * 100:.0f}% "
            f"over {c.runs_present} runs, failing {c.current_failing_streak} "
            f"in a row)"
        ),
    )
    _section(
        "Newly passing",
        trend.newly_passing,
        lambda c: (
            f"{c.case_id} (pass rate {c.pass_rate * 100:.0f}% "
            f"over {c.runs_present} runs)"
        ),
    )
    _section(
        "Worst regressions",
        trend.worst_regressions,
        lambda c: (
            f"{c.case_id} — was passing {c.pass_rate * 100:.0f}% of "
            f"{c.runs_present} runs, now failing "
            f"{c.current_failing_streak} in a row, worst single score "
            f"drop {c.worst_drop:.2f}"
        ),
    )
    _section(
        "Flaky across history",
        trend.flaky_cases,
        lambda c: (
            f"{c.case_id} (pass rate {c.pass_rate * 100:.0f}% over "
            f"{c.runs_present} runs)"
        ),
    )
    lines.append("Per-case trend:")
    lines.append(
        f"{'case':<22} {'runs':>4} {'pass rate':>10} "
        f"{'score':>13} {'worst drop':>10}  latest"
    )
    lines.append("-" * 72)
    for case in trend.cases:
        first = case.first_score
        last = case.last_score
        score = (
            f"{first:.2f} -> {last:.2f}"
            if first is not None and last is not None
            else "-"
        )
        latest = "PASS" if case.last_passed else "FAIL"
        lines.append(
            f"{case.case_id:<22} {case.runs_present:>4} "
            f"{case.pass_rate * 100:>9.1f}% {score:>13} "
            f"{case.worst_drop:>10.2f}  {latest}"
        )
    return "\n".join(lines).rstrip("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals-trends",
        description="Compare run reports across a reports directory "
        "and show per-case trends, newly failing/passing cases, and "
        "the worst regressions in the history.",
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="directory of saved run report JSON files",
    )
    parser.add_argument(
        "--json", help="optional path to write the trend report as JSON"
    )
    args = parser.parse_args(argv)

    try:
        reports = load_reports_dir(args.reports_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not reports:
        print(
            f"no run reports found in {args.reports_dir}",
            file=sys.stderr,
        )
        return 2
    trend = compare_history(reports)
    print(format_trend(trend))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(trend.to_dict(), f, indent=2)
            f.write("\n")
        print(f"\nTrend report written to {args.json}")
    return 1 if trend.has_new_failure else 0


if __name__ == "__main__":
    sys.exit(main())
