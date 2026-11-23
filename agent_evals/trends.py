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
* per-case latency trends — mean/min/max latency, first-to-last
  change, and a least-squares slope per run, so a case that keeps
  passing but gets steadily slower shows up before it blows its
  latency budget;
* per-case token-cost trends — the same machinery applied to the
  token counts agents report, so a case that keeps passing but
  gets steadily more expensive shows up before it blows its token
  budget, plus a cheapest/most-expensive ranking per case;
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
    python -m agent_evals.trends --reports-dir reports/ --fail-on-trend 2.0

The CLI loads every ``*.json`` file in the directory (sorted by the
report's ``started_at``, falling back to filename order) and skips
files that are not run reports. Exit code is 0 when nothing newly
failed in the latest run, 1 when at least one case newly failed —
the same gating rule as the two-run comparison. ``--fail-on-trend
N`` additionally fails the run when any case's latency slope exceeds
N ms per run, and ``--fail-on-slowing`` fails it when any case is
labelled ``"slowing"`` — so the trend report can gate a CI step on
latency creep, not just on failures. ``--fail-on-cost-trend N`` and
``--fail-on-rising-cost`` do the same for token cost: they fail the
run when a case's token slope exceeds N tokens per run, or when any
case is labelled ``"rising"`` on tokens. Token trends only cover
cases whose agent reported a token count; an agent that returns a
plain string reports no tokens, so its cases have no token history
at all.
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

    ``passed_history`` / ``score_history`` / ``latency_history`` /
    ``tokens_history`` are aligned with the reports list: one entry
    per report, ``None`` when the case was absent from that run (a
    case added or removed mid-history). A token entry is also
    ``None`` when the agent reported no token count for that run —
    an agent that returns a plain string has no token history.
    """

    case_id: str
    passed_history: list = field(default_factory=list)
    score_history: list = field(default_factory=list)
    latency_history: list = field(default_factory=list)
    tokens_history: list = field(default_factory=list)

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

    # -- latency trends -------------------------------------------------

    @property
    def latencies(self) -> list:
        """Latencies in ms for the runs that contained the case,
        skipping runs where it was absent."""
        return [lat for lat in self.latency_history if lat is not None]

    @property
    def first_latency_ms(self) -> Optional[float]:
        for lat in self.latency_history:
            if lat is not None:
                return round(float(lat), 3)
        return None

    @property
    def last_latency_ms(self) -> Optional[float]:
        for lat in reversed(self.latency_history):
            if lat is not None:
                return round(float(lat), 3)
        return None

    @property
    def mean_latency_ms(self) -> Optional[float]:
        latencies = self.latencies
        if not latencies:
            return None
        return round(sum(latencies) / len(latencies), 3)

    @property
    def min_latency_ms(self) -> Optional[float]:
        latencies = self.latencies
        if not latencies:
            return None
        return round(min(latencies), 3)

    @property
    def max_latency_ms(self) -> Optional[float]:
        latencies = self.latencies
        if not latencies:
            return None
        return round(max(latencies), 3)

    @property
    def latency_delta_ms(self) -> Optional[float]:
        """Last latency minus first latency (positive = slower)."""
        if self.first_latency_ms is None or self.last_latency_ms is None:
            return None
        return round(self.last_latency_ms - self.first_latency_ms, 3)

    @property
    def latency_delta_pct(self) -> Optional[float]:
        """Last latency minus first, as a percent of the first
        (None when there are no samples, or the first latency
        was 0)."""
        if self.first_latency_ms in (None, 0) or self.last_latency_ms is None:
            return None
        return round(
            (self.last_latency_ms - self.first_latency_ms)
            / self.first_latency_ms
            * 100,
            1,
        )

    @property
    def latency_slope_ms_per_run(self) -> Optional[float]:
        """Least-squares slope of latency against run index, in ms
        per run (positive = getting slower run over run). Uses the
        run index, not the compacted sample position, so a case
        absent from a run keeps its spacing. None with fewer than
        two samples."""
        points = [
            (i, lat)
            for i, lat in enumerate(self.latency_history)
            if lat is not None
        ]
        if len(points) < 2:
            return None
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        denom = sum((x - mean_x) ** 2 for x, _ in points)
        if denom == 0:
            return 0.0
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
        )
        return round(slope, 3)

    @property
    def latency_trend(self) -> Optional[str]:
        """``"slowing"``, ``"improving"``, or ``"flat"`` from the
        latency slope relative to the case's mean latency: a slope
        bigger than 10% of the mean per run counts as slowing (or
        improving, when negative). A single sample, or a mean of
        zero, has no trend (None). Latency is noisy, so this is a
        signal to look at the case, not a verdict."""
        slope = self.latency_slope_ms_per_run
        mean = self.mean_latency_ms
        if slope is None or not mean:
            return None
        relative = slope / mean
        if relative > 0.10:
            return "slowing"
        if relative < -0.10:
            return "improving"
        return "flat"

    # -- token-cost trends ------------------------------------------------

    @property
    def tokens(self) -> list:
        """Token counts for the runs that contained the case and
        where the agent reported a count, skipping the rest."""
        return [t for t in self.tokens_history if t is not None]

    @property
    def first_tokens(self) -> Optional[int]:
        for tokens in self.tokens_history:
            if tokens is not None:
                return int(tokens)
        return None

    @property
    def last_tokens(self) -> Optional[int]:
        for tokens in reversed(self.tokens_history):
            if tokens is not None:
                return int(tokens)
        return None

    @property
    def mean_tokens(self) -> Optional[float]:
        tokens = self.tokens
        if not tokens:
            return None
        return round(sum(tokens) / len(tokens), 1)

    @property
    def min_tokens(self) -> Optional[int]:
        tokens = self.tokens
        if not tokens:
            return None
        return int(min(tokens))

    @property
    def max_tokens(self) -> Optional[int]:
        tokens = self.tokens
        if not tokens:
            return None
        return int(max(tokens))

    @property
    def tokens_delta(self) -> Optional[int]:
        """Last token count minus first (positive = more expensive)."""
        if self.first_tokens is None or self.last_tokens is None:
            return None
        return self.last_tokens - self.first_tokens

    @property
    def tokens_delta_pct(self) -> Optional[float]:
        """Last token count minus first, as a percent of the first
        (None when there are no samples, or the first count was 0)."""
        if self.first_tokens in (None, 0) or self.last_tokens is None:
            return None
        return round(
            (self.last_tokens - self.first_tokens)
            / self.first_tokens
            * 100,
            1,
        )

    @property
    def tokens_slope_per_run(self) -> Optional[float]:
        """Least-squares slope of the token count against run index,
        in tokens per run (positive = getting more expensive run
        over run). Uses the run index, not the compacted sample
        position, so a case absent from a run — or a run where the
        agent reported no tokens — keeps its spacing. None with
        fewer than two samples."""
        points = [
            (i, tokens)
            for i, tokens in enumerate(self.tokens_history)
            if tokens is not None
        ]
        if len(points) < 2:
            return None
        n = len(points)
        mean_x = sum(x for x, _ in points) / n
        mean_y = sum(y for _, y in points) / n
        denom = sum((x - mean_x) ** 2 for x, _ in points)
        if denom == 0:
            return 0.0
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
        )
        return round(slope, 3)

    @property
    def tokens_trend(self) -> Optional[str]:
        """``"rising"``, ``"falling"``, or ``"flat"`` from the token
        slope relative to the case's mean token count: a slope
        bigger than 10% of the mean per run counts as rising (or
        falling, when negative). A case with fewer than two token
        samples, or a mean of zero, has no trend (None). Token
        counts are noisy too — an agent's verbosity drifts — so
        this is a signal to look at the case, not a verdict."""
        slope = self.tokens_slope_per_run
        mean = self.mean_tokens
        if slope is None or not mean:
            return None
        relative = slope / mean
        if relative > 0.10:
            return "rising"
        if relative < -0.10:
            return "falling"
        return "flat"

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
            "latency_history": self.latency_history,
            "first_latency_ms": self.first_latency_ms,
            "last_latency_ms": self.last_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "latency_delta_ms": self.latency_delta_ms,
            "latency_delta_pct": self.latency_delta_pct,
            "latency_slope_ms_per_run": self.latency_slope_ms_per_run,
            "latency_trend": self.latency_trend,
            "tokens_history": self.tokens_history,
            "first_tokens": self.first_tokens,
            "last_tokens": self.last_tokens,
            "mean_tokens": self.mean_tokens,
            "min_tokens": self.min_tokens,
            "max_tokens": self.max_tokens,
            "tokens_delta": self.tokens_delta,
            "tokens_delta_pct": self.tokens_delta_pct,
            "tokens_slope_per_run": self.tokens_slope_per_run,
            "tokens_trend": self.tokens_trend,
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
    total_tokens: Optional[int] = None

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
            "total_tokens": self.total_tokens,
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

    def latency_gate_failures(
        self, threshold_ms_per_run: float
    ) -> list:
        """Cases whose latency slope exceeds ``threshold_ms_per_run``
        (ms per run), worst first: ranked by slope descending, ties
        break on case_id. A case with fewer than two latency samples
        has no slope and never trips the gate; a flat or improving
        case has a slope at or below zero and never trips it either,
        so the gate only fires on real, sustained slowing. The
        comparison is strict — a slope exactly at the threshold
        passes, so the threshold is a budget, not a tripwire."""
        failures = [
            c
            for c in self.cases
            if c.latency_slope_ms_per_run is not None
            and c.latency_slope_ms_per_run > threshold_ms_per_run
        ]
        return sorted(
            failures,
            key=lambda c: (-c.latency_slope_ms_per_run, c.case_id),
        )

    def latency_gate_passed(self, threshold_ms_per_run: float) -> bool:
        """True when no case's latency slope exceeds the threshold."""
        return not self.latency_gate_failures(threshold_ms_per_run)

    @property
    def slowing_cases(self) -> list:
        """Cases labelled ``"slowing"`` — slope bigger than 10% of
        the case's mean latency per run — worst first, same ranking
        as :attr:`slowest_growing`."""
        return self.slowest_growing

    @property
    def most_expensive(self) -> list:
        """Cases ranked by mean token count, most expensive first;
        ties break on the latest token count, then case_id. A case
        whose agent never reported a token count has no mean and
        never appears here."""
        expensive = [c for c in self.cases if c.mean_tokens is not None]
        return sorted(
            expensive,
            key=lambda c: (
                -c.mean_tokens,
                -(c.last_tokens or 0),
                c.case_id,
            ),
        )

    @property
    def costliest_growing(self) -> list:
        """Cases whose token count is trending up, worst first:
        ranked by the token slope relative to the case's mean token
        count (a case growing 20 tokens per run on a 100-token mean
        outranks one growing 20 tokens per run on a 1000-token
        mean); ties break on the first-to-last percent change, then
        case_id. A case with fewer than two token samples has no
        slope and never appears here."""
        growing = [c for c in self.cases if c.tokens_trend == "rising"]

        def _relative_slope(case: CaseTrend) -> float:
            mean = case.mean_tokens
            slope = case.tokens_slope_per_run
            if not mean or slope is None:
                return 0.0
            return slope / mean

        return sorted(
            growing,
            key=lambda c: (
                -_relative_slope(c),
                -(c.tokens_delta_pct or 0.0),
                c.case_id,
            ),
        )

    @property
    def rising_cost_cases(self) -> list:
        """Cases labelled ``"rising"`` on tokens — slope bigger
        than 10% of the case's mean token count per run — worst
        first, same ranking as :attr:`costliest_growing`."""
        return self.costliest_growing

    def cost_gate_failures(self, threshold_tokens_per_run: float) -> list:
        """Cases whose token slope exceeds ``threshold_tokens_per_run``
        (tokens per run), worst first: ranked by slope descending,
        ties break on case_id. A case with fewer than two token
        samples has no slope and never trips the gate; a flat or
        falling case has a slope at or below zero and never trips
        it either, so the gate only fires on real, sustained cost
        growth. The comparison is strict — a slope exactly at the
        threshold passes, so the threshold is a budget, not a
        tripwire."""
        failures = [
            c
            for c in self.cases
            if c.tokens_slope_per_run is not None
            and c.tokens_slope_per_run > threshold_tokens_per_run
        ]
        return sorted(
            failures,
            key=lambda c: (-c.tokens_slope_per_run, c.case_id),
        )

    def cost_gate_passed(self, threshold_tokens_per_run: float) -> bool:
        """True when no case's token slope exceeds the threshold."""
        return not self.cost_gate_failures(threshold_tokens_per_run)

    @property
    def slowest_growing(self) -> list:
        """Cases whose latency is trending up, worst first: ranked
        by the latency slope relative to the case's mean latency
        (a case growing 2 ms per run on a 10 ms mean outranks one
        growing 2 ms per run on a 200 ms mean); ties break on the
        first-to-last percent change, then case_id. A case with
        fewer than two latency samples has no slope and never
        appears here."""
        growing = [
            c for c in self.cases if c.latency_trend == "slowing"
        ]

        def _relative_slope(case: CaseTrend) -> float:
            mean = case.mean_latency_ms
            slope = case.latency_slope_ms_per_run
            if not mean or slope is None:
                return 0.0
            return slope / mean

        return sorted(
            growing,
            key=lambda c: (
                -_relative_slope(c),
                -(c.latency_delta_pct or 0.0),
                c.case_id,
            ),
        )

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

    def to_dict(
        self,
        latency_gate_threshold_ms_per_run: Optional[float] = None,
        fail_on_slowing: bool = False,
        cost_gate_threshold_tokens_per_run: Optional[float] = None,
        fail_on_rising_cost: bool = False,
    ) -> dict:
        data = {
            "n_reports": self.n_reports,
            "pass_rate_delta": self.pass_rate_delta,
            "weighted_score_delta": self.weighted_score_delta,
            "newly_failing": [c.case_id for c in self.newly_failing],
            "newly_passing": [c.case_id for c in self.newly_passing],
            "worst_regressions": [c.case_id for c in self.worst_regressions],
            "flaky": [c.case_id for c in self.flaky_cases],
            "slowest_growing": [c.case_id for c in self.slowest_growing],
            "most_expensive": [c.case_id for c in self.most_expensive],
            "costliest_growing": [
                c.case_id for c in self.costliest_growing
            ],
            "runs": [r.to_dict() for r in self.runs],
            "cases": [c.to_dict() for c in self.cases],
        }
        if latency_gate_threshold_ms_per_run is not None:
            failures = self.latency_gate_failures(
                latency_gate_threshold_ms_per_run
            )
            data["latency_gate_threshold_ms_per_run"] = (
                latency_gate_threshold_ms_per_run
            )
            data["latency_gate_failures"] = [c.case_id for c in failures]
            data["latency_gate_passed"] = not failures
        if fail_on_slowing:
            data["slowing_cases"] = [
                c.case_id for c in self.slowing_cases
            ]
        if cost_gate_threshold_tokens_per_run is not None:
            failures = self.cost_gate_failures(
                cost_gate_threshold_tokens_per_run
            )
            data["cost_gate_threshold_tokens_per_run"] = (
                cost_gate_threshold_tokens_per_run
            )
            data["cost_gate_failures"] = [c.case_id for c in failures]
            data["cost_gate_passed"] = not failures
        if fail_on_rising_cost:
            data["rising_cost_cases"] = [
                c.case_id for c in self.rising_cost_cases
            ]
        return data


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
                total_tokens=(
                    int(report["total_tokens"])
                    if report.get("total_tokens") is not None
                    else None
                ),
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
            trend.tokens_history.append(
                result.get("tokens") if result else None
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
                    tokens_history=[None] * (len(runs) - 1),
                )
                by_case[case_id] = trend
                trend.passed_history.append(bool(result["passed"]))
                trend.score_history.append(result.get("score"))
                trend.latency_history.append(result.get("latency_ms"))
                trend.tokens_history.append(result.get("tokens"))
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


def format_trend(
    trend: TrendReport,
    latency_gate_threshold_ms_per_run: Optional[float] = None,
    fail_on_slowing: bool = False,
    cost_gate_threshold_tokens_per_run: Optional[float] = None,
    fail_on_rising_cost: bool = False,
) -> str:
    """Render a trend report as a plain-text summary.

    When ``latency_gate_threshold_ms_per_run`` is given, a section
    lists the cases whose slope tripped the gate (or says the gate
    passed); when ``fail_on_slowing`` is given, the slowing cases
    are called out the same way. ``cost_gate_threshold_tokens_per_run``
    and ``fail_on_rising_cost`` do the same for token cost."""
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
        f"{'weighted':>9} {'latency':>9} {'tokens':>7}"
    )
    lines.append("-" * 81)
    for run in trend.runs:
        tokens = (
            f"{run.total_tokens}"
            if run.total_tokens is not None
            else "-"
        )
        lines.append(
            f"{run.path:<28} {run.passed}/{run.total:<8} "
            f"{run.pass_rate * 100:>9.1f}% {run.weighted_score:>9.4f} "
            f"{run.total_latency_ms:>8.1f} ms {tokens:>7}"
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
    _section(
        "Slowest growing latency",
        trend.slowest_growing,
        lambda c: (
            f"{c.case_id} ({c.first_latency_ms:.1f} -> "
            f"{c.last_latency_ms:.1f} ms, "
            f"{c.latency_delta_pct:+.1f}%, "
            f"+{c.latency_slope_ms_per_run:.2f} ms/run, "
            f"mean {c.mean_latency_ms:.1f} ms)"
        ),
    )
    _section(
        "Most expensive cases",
        trend.most_expensive,
        lambda c: (
            f"{c.case_id} (mean {c.mean_tokens:.0f} tokens, "
            f"{c.first_tokens} -> {c.last_tokens} tokens, "
            f"{c.tokens_delta_pct:+.1f}%)"
            if c.tokens_delta_pct is not None
            else f"{c.case_id} (mean {c.mean_tokens:.0f} tokens, "
            f"{c.first_tokens} -> {c.last_tokens} tokens)"
        ),
    )
    _section(
        "Costliest growing",
        trend.costliest_growing,
        lambda c: (
            f"{c.case_id} ({c.first_tokens} -> "
            f"{c.last_tokens} tokens, "
            f"{c.tokens_delta_pct:+.1f}%, "
            f"+{c.tokens_slope_per_run:.2f} tokens/run, "
            f"mean {c.mean_tokens:.0f} tokens)"
        ),
    )
    if latency_gate_threshold_ms_per_run is not None:
        failures = trend.latency_gate_failures(
            latency_gate_threshold_ms_per_run
        )
        if failures:
            lines.append(
                f"Latency gate FAILED "
                f"(threshold +{latency_gate_threshold_ms_per_run:.2f} "
                f"ms/run, {len(failures)} case(s) over):"
            )
            for case in failures:
                lines.append(
                    f"  {case.case_id} "
                    f"(+{case.latency_slope_ms_per_run:.2f} ms/run, "
                    f"mean {case.mean_latency_ms:.1f} ms, "
                    f"trend {case.latency_trend})"
                )
            lines.append("")
        else:
            lines.append(
                f"Latency gate passed "
                f"(no case over +{latency_gate_threshold_ms_per_run:.2f} "
                f"ms/run)"
            )
            lines.append("")
    if fail_on_slowing:
        slowing = trend.slowing_cases
        if slowing:
            lines.append(
                f"Slowing cases ({len(slowing)}): "
                f"the gate fails on any case labelled slowing:"
            )
            for case in slowing:
                lines.append(
                    f"  {case.case_id} "
                    f"(+{case.latency_slope_ms_per_run:.2f} ms/run, "
                    f"mean {case.mean_latency_ms:.1f} ms)"
                )
            lines.append("")
        else:
            lines.append("Slowing cases (0): none labelled slowing")
            lines.append("")
    if cost_gate_threshold_tokens_per_run is not None:
        failures = trend.cost_gate_failures(
            cost_gate_threshold_tokens_per_run
        )
        if failures:
            lines.append(
                f"Cost gate FAILED "
                f"(threshold +{cost_gate_threshold_tokens_per_run:.2f} "
                f"tokens/run, {len(failures)} case(s) over):"
            )
            for case in failures:
                lines.append(
                    f"  {case.case_id} "
                    f"(+{case.tokens_slope_per_run:.2f} tokens/run, "
                    f"mean {case.mean_tokens:.0f} tokens, "
                    f"trend {case.tokens_trend})"
                )
            lines.append("")
        else:
            lines.append(
                f"Cost gate passed "
                f"(no case over +{cost_gate_threshold_tokens_per_run:.2f} "
                f"tokens/run)"
            )
            lines.append("")
    if fail_on_rising_cost:
        rising = trend.rising_cost_cases
        if rising:
            lines.append(
                f"Rising cost cases ({len(rising)}): "
                f"the gate fails on any case labelled rising:"
            )
            for case in rising:
                lines.append(
                    f"  {case.case_id} "
                    f"(+{case.tokens_slope_per_run:.2f} tokens/run, "
                    f"mean {case.mean_tokens:.0f} tokens)"
                )
            lines.append("")
        else:
            lines.append("Rising cost cases (0): none labelled rising")
            lines.append("")
    lines.append("Per-case trend:")
    lines.append(
        f"{'case':<22} {'runs':>4} {'pass rate':>10} "
        f"{'score':>13} {'worst drop':>10}  {'latency':>20}  "
        f"{'tokens':>15}  latest"
    )
    lines.append("-" * 109)
    for case in trend.cases:
        first = case.first_score
        last = case.last_score
        score = (
            f"{first:.2f} -> {last:.2f}"
            if first is not None and last is not None
            else "-"
        )
        latest = "PASS" if case.last_passed else "FAIL"
        if (
            case.first_latency_ms is not None
            and case.last_latency_ms is not None
        ):
            pct = case.latency_delta_pct
            pct_text = f" ({pct:+.0f}%)" if pct is not None else ""
            latency = (
                f"{case.first_latency_ms:.1f} -> "
                f"{case.last_latency_ms:.1f} ms{pct_text}"
            )
        else:
            latency = "-"
        if case.first_tokens is not None and case.last_tokens is not None:
            pct = case.tokens_delta_pct
            pct_text = f" ({pct:+.0f}%)" if pct is not None else ""
            tokens = (
                f"{case.first_tokens} -> {case.last_tokens}{pct_text}"
            )
        else:
            tokens = "-"
        lines.append(
            f"{case.case_id:<22} {case.runs_present:>4} "
            f"{case.pass_rate * 100:>9.1f}% {score:>13} "
            f"{case.worst_drop:>10.2f}  {latency:>20}  {tokens:>15}  "
            f"{latest}"
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
    parser.add_argument(
        "--fail-on-trend",
        type=float,
        default=None,
        metavar="MS_PER_RUN",
        help="fail (exit 1) when any case's latency slope exceeds "
        "this many ms per run — a CI gate on latency creep, so a "
        "case that keeps passing but gets steadily slower fails "
        "the build before it trips its latency budget",
    )
    parser.add_argument(
        "--fail-on-slowing",
        action="store_true",
        help="fail (exit 1) when any case is labelled \"slowing\" "
        "(slope bigger than 10% of the case's mean latency per "
        "run); catches slow creep on fast cases whose absolute "
        "slope is too small for --fail-on-trend to notice",
    )
    parser.add_argument(
        "--fail-on-cost-trend",
        type=float,
        default=None,
        metavar="TOKENS_PER_RUN",
        help="fail (exit 1) when any case's token slope exceeds "
        "this many tokens per run — a CI gate on cost creep, so a "
        "case that keeps passing but gets steadily more expensive "
        "fails the build before it trips its token budget",
    )
    parser.add_argument(
        "--fail-on-rising-cost",
        action="store_true",
        help="fail (exit 1) when any case is labelled \"rising\" "
        "on tokens (slope bigger than 10% of the case's mean "
        "token count per run); catches cost creep on cheap cases "
        "whose absolute slope is too small for "
        "--fail-on-cost-trend to notice",
    )
    args = parser.parse_args(argv)
    if args.fail_on_trend is not None and args.fail_on_trend < 0:
        parser.error("--fail-on-trend must not be negative")
    if args.fail_on_cost_trend is not None and args.fail_on_cost_trend < 0:
        parser.error("--fail-on-cost-trend must not be negative")

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
    gate_failures = (
        trend.latency_gate_failures(args.fail_on_trend)
        if args.fail_on_trend is not None
        else []
    )
    slowing = trend.slowing_cases if args.fail_on_slowing else []
    cost_gate_failures = (
        trend.cost_gate_failures(args.fail_on_cost_trend)
        if args.fail_on_cost_trend is not None
        else []
    )
    rising = trend.rising_cost_cases if args.fail_on_rising_cost else []
    print(
        format_trend(
            trend,
            latency_gate_threshold_ms_per_run=args.fail_on_trend,
            fail_on_slowing=args.fail_on_slowing,
            cost_gate_threshold_tokens_per_run=args.fail_on_cost_trend,
            fail_on_rising_cost=args.fail_on_rising_cost,
        )
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                trend.to_dict(
                    latency_gate_threshold_ms_per_run=args.fail_on_trend,
                    fail_on_slowing=args.fail_on_slowing,
                    cost_gate_threshold_tokens_per_run=(
                        args.fail_on_cost_trend
                    ),
                    fail_on_rising_cost=args.fail_on_rising_cost,
                ),
                f,
                indent=2,
            )
            f.write("\n")
        print(f"\nTrend report written to {args.json}")
    if gate_failures:
        print(
            f"\nLatency gate FAILED: {len(gate_failures)} case(s) over "
            f"+{args.fail_on_trend:.2f} ms/run: "
            + ", ".join(c.case_id for c in gate_failures),
            file=sys.stderr,
        )
    if slowing:
        print(
            f"\nSlowing gate FAILED: {len(slowing)} case(s) labelled "
            f"slowing: " + ", ".join(c.case_id for c in slowing),
            file=sys.stderr,
        )
    if cost_gate_failures:
        print(
            f"\nCost gate FAILED: {len(cost_gate_failures)} case(s) over "
            f"+{args.fail_on_cost_trend:.2f} tokens/run: "
            + ", ".join(c.case_id for c in cost_gate_failures),
            file=sys.stderr,
        )
    if rising:
        print(
            f"\nRising cost gate FAILED: {len(rising)} case(s) labelled "
            f"rising: " + ", ".join(c.case_id for c in rising),
            file=sys.stderr,
        )
    return (
        1
        if (
            trend.has_new_failure
            or gate_failures
            or slowing
            or cost_gate_failures
            or rising
        )
        else 0
    )


if __name__ == "__main__":
    sys.exit(main())
