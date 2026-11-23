"""Flaky-case history across stability reports.

A single stability report (see :mod:`agent_evals.stability`) says
which cases were flaky in one burst of repeated runs. A directory of
stability reports says which cases *keep* being flaky: a case that
flips in every report is a different problem from one that flipped
once, months ago, and has been stable since.

``compare_flaky_history`` takes a list of saved stability reports
(dicts as returned by ``StabilityReport.to_dict``), ordered oldest
first, and builds a :class:`FlakyHistoryReport`:

* per-case pass-rate history and flaky history — one entry per
  stability report, ``None`` when the case was absent from that
  report (a case added or removed mid-history);
* persistently flaky cases — flaky in at least two reports, so the
  flakiness is a property of the case or the agent, not one bad day;
* newly flaky cases — stable in every earlier report, flaky in the
  latest one, so something just got unstable;
* stopped-flaky cases — flaky in an earlier report, stable in the
  latest one, so a fix (or a quieter agent) landed;
* flaky in the latest report — the live signal, and the CI gate:
  the CLI exits 1 while any case is flaky in the latest stability
  report, 0 otherwise.

Usage:
    python -m agent_evals.flaky_history --reports-dir stability/
    python -m agent_evals.flaky_history --reports-dir stability/ \
        --json flaky-history.json

Or from Python:

    from agent_evals import compare_flaky_history, format_flaky_history
    from agent_evals import load_stability_reports_dir

    history = compare_flaky_history(
        load_stability_reports_dir("stability/"))
    print(format_flaky_history(history))

The CLI loads every ``*.json`` file in the directory (sorted by the
report's ``started_at``, falling back to filename order) and skips
files that are not stability reports — a single-run report has a
``results`` list, not a ``cases`` list with ``repetitions``, so a
directory that mixes run reports and stability reports still works.
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
    "CaseFlakyHistory",
    "FlakyHistoryReport",
    "StabilityRunSummary",
    "compare_flaky_history",
    "format_flaky_history",
    "load_stability_reports_dir",
]


# ---------------------------------------------------------------------------
# Per-case flaky history
# ---------------------------------------------------------------------------


@dataclass
class CaseFlakyHistory:
    """How one case's flakiness behaved across stability reports.

    ``pass_rate_history`` / ``flaky_history`` / ``runs_history`` /
    ``passes_history`` are aligned with the reports list: one entry
    per stability report, ``None`` when the case was absent from
    that report (a case added or removed mid-history).
    """

    case_id: str
    pass_rate_history: list = field(default_factory=list)
    flaky_history: list = field(default_factory=list)
    runs_history: list = field(default_factory=list)
    passes_history: list = field(default_factory=list)

    @property
    def reports_present(self) -> int:
        """How many of the stability reports contain this case."""
        return sum(1 for r in self.pass_rate_history if r is not None)

    @property
    def total_runs(self) -> int:
        return sum(r for r in self.runs_history if r is not None)

    @property
    def total_passes(self) -> int:
        return sum(p for p in self.passes_history if p is not None)

    @property
    def total_fails(self) -> int:
        return self.total_runs - self.total_passes

    @property
    def flaky_reports(self) -> int:
        """How many stability reports flagged this case flaky."""
        return sum(1 for f in self.flaky_history if f)

    @property
    def flaky(self) -> bool:
        """Flaky in at least one stability report."""
        return self.flaky_reports > 0

    @property
    def persistently_flaky(self) -> bool:
        """Flaky in at least two stability reports: the flakiness
        is a property of the case or the agent, not one bad day. A
        case flaky in only one report so far is flaky, not yet
        persistent — it earns the label on its second flaky
        report."""
        return self.flaky_reports >= 2

    @property
    def flaky_in_latest(self) -> bool:
        """Flaky in the latest stability report that contained the
        case. This is the live signal: a case that stopped flipping
        is not flaky now, whatever its history says."""
        for f in reversed(self.flaky_history):
            if f is not None:
                return bool(f)
        return False

    @property
    def newly_flaky(self) -> bool:
        """Stable in every earlier report that contained the case,
        flaky in the latest one — something just got unstable."""
        present = [f for f in self.flaky_history if f is not None]
        return (
            len(present) >= 2
            and present[-1] is True
            and not any(present[:-1])
        )

    @property
    def stopped_flaky(self) -> bool:
        """Flaky in an earlier report, stable in the latest one —
        a fix (or a quieter agent) landed, or the case changed."""
        present = [f for f in self.flaky_history if f is not None]
        return (
            len(present) >= 2
            and present[-1] is False
            and any(present[:-1])
        )

    @property
    def first_pass_rate(self) -> Optional[float]:
        for rate in self.pass_rate_history:
            if rate is not None:
                return rate
        return None

    @property
    def last_pass_rate(self) -> Optional[float]:
        for rate in reversed(self.pass_rate_history):
            if rate is not None:
                return rate
        return None

    @property
    def pass_rate_delta(self) -> Optional[float]:
        """Last pass rate minus first pass rate (positive =
        more reliable). None with fewer than two reports."""
        if self.first_pass_rate is None or self.last_pass_rate is None:
            return None
        if self.reports_present < 2:
            return None
        return round(self.last_pass_rate - self.first_pass_rate, 4)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "reports_present": self.reports_present,
            "total_runs": self.total_runs,
            "total_passes": self.total_passes,
            "total_fails": self.total_fails,
            "flaky_reports": self.flaky_reports,
            "flaky": self.flaky,
            "persistently_flaky": self.persistently_flaky,
            "flaky_in_latest": self.flaky_in_latest,
            "newly_flaky": self.newly_flaky,
            "stopped_flaky": self.stopped_flaky,
            "pass_rate_history": self.pass_rate_history,
            "flaky_history": self.flaky_history,
            "runs_history": self.runs_history,
            "passes_history": self.passes_history,
            "first_pass_rate": self.first_pass_rate,
            "last_pass_rate": self.last_pass_rate,
            "pass_rate_delta": self.pass_rate_delta,
        }


# ---------------------------------------------------------------------------
# Whole-history report
# ---------------------------------------------------------------------------


@dataclass
class StabilityRunSummary:
    """One stability report in the history, as a row in the table."""

    path: str
    started_at: Optional[str]
    suite: str
    repetitions: int
    total_cases: int
    flaky: int

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "started_at": self.started_at,
            "suite": self.suite,
            "repetitions": self.repetitions,
            "total_cases": self.total_cases,
            "flaky": self.flaky,
        }


@dataclass
class FlakyHistoryReport:
    """The flaky-case history across a directory of stability
    reports, oldest first."""

    runs: list = field(default_factory=list)
    cases: list = field(default_factory=list)

    @property
    def n_reports(self) -> int:
        return len(self.runs)

    @property
    def flaky_cases(self) -> list:
        return [c for c in self.cases if c.flaky]

    @property
    def persistently_flaky_cases(self) -> list:
        """Flaky in at least two reports, most flaky reports first;
        ties break on the lowest latest pass rate, then case_id."""
        return sorted(
            (c for c in self.cases if c.persistently_flaky),
            key=lambda c: (
                -c.flaky_reports,
                c.last_pass_rate or 0.0,
                c.case_id,
            ),
        )

    @property
    def flaky_in_latest(self) -> list:
        return [c for c in self.cases if c.flaky_in_latest]

    @property
    def newly_flaky(self) -> list:
        return [c for c in self.cases if c.newly_flaky]

    @property
    def stopped_flaky(self) -> list:
        return [c for c in self.cases if c.stopped_flaky]

    @property
    def has_flaky_in_latest(self) -> bool:
        return bool(self.flaky_in_latest)

    def to_dict(self) -> dict:
        return {
            "n_reports": self.n_reports,
            "flaky": [c.case_id for c in self.flaky_cases],
            "persistently_flaky": [
                c.case_id for c in self.persistently_flaky_cases
            ],
            "flaky_in_latest": [c.case_id for c in self.flaky_in_latest],
            "newly_flaky": [c.case_id for c in self.newly_flaky],
            "stopped_flaky": [c.case_id for c in self.stopped_flaky],
            "runs": [r.to_dict() for r in self.runs],
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def _cases_by_id(report: dict) -> dict:
    """Index a stability report's per-case rows by case_id (first
    occurrence wins on duplicates)."""
    by_id = {}
    for case in report.get("cases", []) or []:
        case_id = case.get("case_id")
        if case_id is not None and case_id not in by_id:
            by_id[case_id] = case
    return by_id


def _case_flaky(case: dict) -> bool:
    """Whether one per-case stability row was flaky: prefer the
    report's own ``flaky`` flag, and derive it from passes/runs
    when the flag is absent (an old report, or a hand-built one)."""
    if isinstance(case.get("flaky"), bool):
        return case["flaky"]
    runs = case.get("runs")
    passes = case.get("passes")
    if isinstance(runs, int) and isinstance(passes, int) and runs:
        return 0 < passes < runs
    return False


def compare_flaky_history(reports: list) -> FlakyHistoryReport:
    """Build the flaky-case history across a list of stability
    reports, oldest first.

    Reports are dicts as returned by
    :func:`agent_evals.stability.StabilityReport.to_dict`. The caller
    is responsible for the order; the CLI sorts by ``started_at``.
    Cases are matched by ``case_id``; a case absent from a report
    gets ``None`` in that report's slot, so a case added or removed
    mid-history does not skew the others' histories.
    """
    runs = []
    by_case: dict[str, CaseFlakyHistory] = {}
    for report in reports:
        cases = _cases_by_id(report)
        repetitions = int(report.get("repetitions", 0))
        runs.append(
            StabilityRunSummary(
                path=str(report.get("_path", "?")),
                started_at=report.get("started_at"),
                suite=str(report.get("suite", "?")),
                repetitions=repetitions,
                total_cases=len(cases),
                flaky=sum(1 for c in cases.values() if _case_flaky(c)),
            )
        )
        for case_id, history in by_case.items():
            case = cases.get(case_id)
            if case is None:
                history.pass_rate_history.append(None)
                history.flaky_history.append(None)
                history.runs_history.append(None)
                history.passes_history.append(None)
            else:
                history.pass_rate_history.append(case.get("pass_rate"))
                history.flaky_history.append(_case_flaky(case))
                history.runs_history.append(case.get("runs"))
                history.passes_history.append(case.get("passes"))
        for case_id, case in cases.items():
            if case_id not in by_case:
                # Backfill None for the earlier reports this case was
                # not part of, so every history lines up with runs.
                history = CaseFlakyHistory(
                    case_id=case_id,
                    pass_rate_history=[None] * (len(runs) - 1),
                    flaky_history=[None] * (len(runs) - 1),
                    runs_history=[None] * (len(runs) - 1),
                    passes_history=[None] * (len(runs) - 1),
                )
                by_case[case_id] = history
                history.pass_rate_history.append(case.get("pass_rate"))
                history.flaky_history.append(_case_flaky(case))
                history.runs_history.append(case.get("runs"))
                history.passes_history.append(case.get("passes"))
    # Cases first seen in an early report were appended to on every
    # later report; cases first seen later got backfilled. Order the
    # history by first appearance.
    def _first_seen(case: CaseFlakyHistory) -> int:
        for i, rate in enumerate(case.pass_rate_history):
            if rate is not None:
                return i
        return len(case.pass_rate_history)

    cases = sorted(by_case.values(), key=_first_seen)
    return FlakyHistoryReport(runs=runs, cases=cases)


def load_stability_reports_dir(directory: str) -> list:
    """Load every ``*.json`` stability report in ``directory``,
    oldest first.

    Reports sort by ``started_at`` when present, falling back to
    filename order for reports without one. A file that is not valid
    JSON, or not a stability report (no ``cases`` list and no
    ``repetitions`` — a single-run report has a ``results`` list
    instead), is skipped with a warning on stderr — one bad file
    must not hide the history. Each loaded report carries its
    filename under ``"_path"``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"stability reports directory not found: {directory}"
        )
    loaded = []
    for path in sorted(directory.glob("*.json")):
        try:
            report = load_report(str(path))
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(report, dict) or not isinstance(
            report.get("cases"), list
        ) or not isinstance(report.get("repetitions"), int):
            print(
                f"warning: skipping {path.name}: not a stability report "
                f"(no 'cases' list with 'repetitions')",
                file=sys.stderr,
            )
            continue
        report["_path"] = path.name
        loaded.append(report)
    loaded.sort(
        key=lambda r: (r.get("started_at") or "", r.get("_path", ""))
    )
    return loaded


def format_flaky_history(history: FlakyHistoryReport) -> str:
    """Render a flaky-case history as a plain-text summary."""
    lines = [f"Stability reports: {history.n_reports}"]
    if history.runs:
        first = history.runs[0]
        last = history.runs[-1]
        span = (
            f"{first.started_at or '?'} -> {last.started_at or '?'}"
            if history.n_reports > 1
            else f"{first.started_at or '?'}"
        )
        lines.append(f"Span: {span}")
        suites = {r.suite for r in history.runs}
        if len(suites) > 1:
            lines.append(
                f"Note: reports span {len(suites)} suites "
                f"({', '.join(sorted(suites))}); cases matched by case_id"
            )
    lines.append(
        f"Flaky in latest report: {len(history.flaky_in_latest)}  "
        f"Persistently flaky: {len(history.persistently_flaky_cases)}  "
        f"Newly flaky: {len(history.newly_flaky)}  "
        f"Stopped flaky: {len(history.stopped_flaky)}"
    )
    lines.append("")
    lines.append(
        f"{'report':<28} {'reps':>5} {'cases':>6} {'flaky':>6}"
    )
    lines.append("-" * 48)
    for run in history.runs:
        lines.append(
            f"{run.path:<28} {run.repetitions:>5} "
            f"{run.total_cases:>6} {run.flaky:>6}"
        )
    lines.append("")

    def _section(title, group, render):
        if not group:
            return
        lines.append(f"{title} ({len(group)}):")
        for item in group:
            lines.append(f"  {render(item)}")
        lines.append("")

    def _history_text(case: CaseFlakyHistory) -> str:
        parts = []
        for rate, flaky in zip(
            case.pass_rate_history, case.flaky_history
        ):
            if rate is None:
                parts.append("-")
            elif flaky:
                parts.append(f"{rate * 100:.0f}%F")
            else:
                parts.append(f"{rate * 100:.0f}%")
        return " ".join(parts)

    _section(
        "Persistently flaky",
        history.persistently_flaky_cases,
        lambda c: (
            f"{c.case_id} (flaky in {c.flaky_reports} of "
            f"{c.reports_present} reports: {_history_text(c)})"
        ),
    )
    _section(
        "Flaky in latest report",
        history.flaky_in_latest,
        lambda c: (
            f"{c.case_id} (flaky in {c.flaky_reports} of "
            f"{c.reports_present} reports: {_history_text(c)})"
        ),
    )
    _section(
        "Newly flaky",
        history.newly_flaky,
        lambda c: (
            f"{c.case_id} (stable before, flaky in latest report: "
            f"{_history_text(c)})"
        ),
    )
    _section(
        "Stopped flaky",
        history.stopped_flaky,
        lambda c: (
            f"{c.case_id} (flaky before, stable in latest report: "
            f"{_history_text(c)})"
        ),
    )
    lines.append("Per-case flaky history:")
    lines.append(
        f"{'case':<22} {'reports':>7} {'flaky in':>8}  "
        f"{'pass-rate history':<28}  latest"
    )
    lines.append("-" * 84)
    for case in history.cases:
        latest = "FLAKY" if case.flaky_in_latest else "stable"
        lines.append(
            f"{case.case_id:<22} {case.reports_present:>7} "
            f"{case.flaky_reports:>8}  {_history_text(case):<28}  {latest}"
        )
    return "\n".join(lines).rstrip("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals-flaky-history",
        description="Show the flaky-case history across a directory "
        "of stability reports: which cases keep flipping, which "
        "just started, and which stopped.",
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="directory of saved stability report JSON files "
        "(--repeat reports)",
    )
    parser.add_argument(
        "--json", help="optional path to write the flaky history as JSON"
    )
    args = parser.parse_args(argv)

    try:
        reports = load_stability_reports_dir(args.reports_dir)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not reports:
        print(
            f"no stability reports found in {args.reports_dir}",
            file=sys.stderr,
        )
        return 2
    history = compare_flaky_history(reports)
    print(format_flaky_history(history))
    if args.json:
        history.save(args.json)
        print(f"\nFlaky history written to {args.json}")
    if history.has_flaky_in_latest:
        print(
            f"\nFlaky gate FAILED: {len(history.flaky_in_latest)} "
            f"case(s) flaky in the latest stability report: "
            + ", ".join(c.case_id for c in history.flaky_in_latest),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
