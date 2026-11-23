"""Repeated runs and flaky-case detection.

A single eval run cannot tell a genuinely broken case from a flaky
one: an agent that answers nondeterministically (sampling, retries,
timeouts under load) passes a case sometimes and fails it other
times. ``repeat_suite`` runs the whole suite ``repetitions`` times
and aggregates per case, so a case whose pass rate is strictly
between 0 and 1 is flagged ``flaky``.

A flaky case is not the same as a failing case: it passed at least
once, so the agent can do it, but the verdict is not reproducible.
That is the signal to look at the agent (nondeterminism, a race, a
timeout that is too tight) or at the case (a scorer that is too
brittle for the agent's output) instead of treating the run as a
clean pass or a clean fail.

Usage:
    from agent_evals import repeat_suite, format_stability

    stability = repeat_suite(suite, agent, repetitions=5)
    print(format_stability(stability))

Or from the CLI: ``python -m agent_evals.cli --suite ... --agent ...
--repeat 5``. Tag filters (``--tag`` / ``--exclude-tag``) apply to
repeated runs too: the suite is scoped first, then repeated, and the
stability report records which tag filter was applied, so a report
says which subset of the suite it covers. The report also carries a
per-tag pass-rate breakdown (``tag_breakdown`` in JSON, a
``Per-tag pass rates:`` section in the text summary), reusing the
same shape as a single run's breakdown, so a tag that is flaky
across repetitions is visible at a glance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .evals import Suite, run_suite

__all__ = [
    "CaseStability",
    "StabilityReport",
    "format_stability",
    "repeat_suite",
]


@dataclass
class CaseStability:
    """How one case behaved across the repeated runs.

    ``pass_rate`` is ``passes / runs``. A case is ``flaky`` when it
    passed at least once and failed at least once: 0 < pass_rate < 1.
    ``mean_latency_ms`` and ``mean_score`` are the averages across
    the runs; ``timeouts`` counts the runs in which the case timed out.
    ``tags`` is the case's tag list, carried so the stability report
    can break pass rates down per tag.
    """

    case_id: str
    runs: int
    passes: int
    mean_latency_ms: float
    mean_score: float
    timeouts: int = 0
    tags: list = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passes / self.runs) if self.runs else 0.0

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.runs

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "runs": self.runs,
            "passes": self.passes,
            "pass_rate": round(self.pass_rate, 4),
            "flaky": self.flaky,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "mean_score": round(self.mean_score, 4),
            "timeouts": self.timeouts,
            "tags": list(self.tags),
        }


@dataclass
class StabilityReport:
    """The aggregated result of running a suite several times.

    ``tag_filter`` records the tag filter the run was scoped to, so a
    stability report can say which subset of the suite it covers:
    ``{"tags": [...], "mode": "any"|"all", "exclude_tags": [...]}``,
    or ``None`` when no tag filter was applied.
    """

    suite: str
    repetitions: int
    started_at: str
    ended_at: str
    cases: list = field(default_factory=list)
    tag_filter: Optional[dict] = None

    @property
    def flaky_cases(self) -> list:
        return [c for c in self.cases if c.flaky]

    @property
    def always_passing(self) -> list:
        return [c for c in self.cases if c.passes == c.runs and c.runs]

    @property
    def always_failing(self) -> list:
        return [c for c in self.cases if c.passes == 0 and c.runs]

    @property
    def tag_breakdown(self) -> dict:
        """Per-tag pass rates across the repetitions: tag ->
        ``{"cases", "passes", "runs", "pass_rate", "mean_score",
        "flaky"}``.

        The same shape as a single run's ``tag_breakdown``
        (``RunReport.tag_breakdown``), aggregated over repetitions:
        ``passes`` is the total passes across all case-runs under
        the tag and ``runs`` is ``cases * repetitions``, so the pass
        rate is the tag's pass rate over every repetition. A case
        carrying several tags counts under each of them; untagged
        cases appear under no tag. ``mean_score`` is the mean of the
        cases' mean scores, and ``flaky`` counts how many cases
        under the tag were flaky, so a tag that is unstable across
        repetitions stands out. An empty dict means no case in the
        report carried a tag.
        """
        stats: dict = {}
        for c in self.cases:
            for tag in c.tags:
                entry = stats.setdefault(
                    tag,
                    {
                        "cases": 0,
                        "passes": 0,
                        "score": 0.0,
                        "flaky": 0,
                    },
                )
                entry["cases"] += 1
                entry["passes"] += c.passes
                entry["score"] += c.mean_score
                entry["flaky"] += 1 if c.flaky else 0
        return {
            tag: {
                "cases": entry["cases"],
                "passes": entry["passes"],
                "runs": entry["cases"] * self.repetitions,
                "pass_rate": (
                    entry["passes"] / (entry["cases"] * self.repetitions)
                    if entry["cases"] and self.repetitions
                    else 0.0
                ),
                "mean_score": (
                    entry["score"] / entry["cases"] if entry["cases"] else 0.0
                ),
                "flaky": entry["flaky"],
            }
            for tag, entry in sorted(stats.items())
        }

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "repetitions": self.repetitions,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "tag_filter": self.tag_filter,
            "flaky": [c.case_id for c in self.flaky_cases],
            "always_passing": [c.case_id for c in self.always_passing],
            "always_failing": [c.case_id for c in self.always_failing],
            "tag_breakdown": {
                tag: {
                    "cases": entry["cases"],
                    "passes": entry["passes"],
                    "runs": entry["runs"],
                    "pass_rate": round(entry["pass_rate"], 4),
                    "mean_score": round(entry["mean_score"], 4),
                    "flaky": entry["flaky"],
                }
                for tag, entry in self.tag_breakdown.items()
            },
            "cases": [c.to_dict() for c in self.cases],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


def repeat_suite(
    suite: Suite,
    agent: Callable[[str], Any],
    repetitions: int,
    workers: int = 1,
    judges: Optional[dict] = None,
    tags: Optional[list] = None,
    tag_mode: str = "any",
    exclude_tags: Optional[list] = None,
) -> StabilityReport:
    """Run ``suite`` against ``agent`` ``repetitions`` times and
    aggregate per-case pass rates.

    ``repetitions`` must be a positive integer; a single repetition is
    just one run and flags nothing as flaky. ``workers`` is passed
    through to ``run_suite`` for each repetition (see its docstring:
    parallel mode needs a thread-safe agent). ``judges`` is passed
    through to ``run_suite`` as well, so ``judge`` scorers work in
    repeated runs. Cases are reported in suite order.

    ``tags`` / ``tag_mode`` / ``exclude_tags`` describe the tag filter
    that was applied to the suite before this call (the CLI applies
    ``--tag`` / ``--tag-mode`` / ``--exclude-tag`` first and passes
    them here): the filter is recorded on the report, so a stability
    report says which subset of the suite it covers. Pass the filter
    that produced ``suite``; this function does not filter the suite
    itself. ``tag_mode`` must be ``"any"`` or ``"all"``.
    """
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError(
            f"repetitions must be a positive integer, got {repetitions!r}"
        )
    if tag_mode not in ("any", "all"):
        raise ValueError(f"tag_mode must be 'any' or 'all', got {tag_mode!r}")
    tag_filter = None
    if tags or exclude_tags:
        tag_filter = {
            "tags": list(tags) if tags else [],
            "mode": tag_mode,
            "exclude_tags": list(exclude_tags) if exclude_tags else [],
        }
    started = datetime.now(timezone.utc).isoformat()
    # Accumulate per case_id, in suite order.
    aggregate: dict = {}
    order: list = []
    by_id: dict = {}
    for case in suite.cases:
        aggregate[case.id] = {"passes": 0, "latency": 0.0, "score": 0.0, "timeouts": 0}
        order.append(case.id)
        by_id[case.id] = case
    for _ in range(repetitions):
        report = run_suite(suite, agent, workers=workers, judges=judges)
        for result in report.results:
            entry = aggregate[result.case_id]
            entry["passes"] += 1 if result.passed else 0
            entry["latency"] += result.latency_ms
            entry["score"] += result.score
            entry["timeouts"] += 1 if result.timed_out else 0
    ended = datetime.now(timezone.utc).isoformat()
    cases = [
        CaseStability(
            case_id=case_id,
            runs=repetitions,
            passes=aggregate[case_id]["passes"],
            mean_latency_ms=aggregate[case_id]["latency"] / repetitions,
            mean_score=aggregate[case_id]["score"] / repetitions,
            timeouts=aggregate[case_id]["timeouts"],
            tags=list(by_id[case_id].tags),
        )
        for case_id in order
    ]
    return StabilityReport(
        suite=suite.name,
        repetitions=repetitions,
        started_at=started,
        ended_at=ended,
        cases=cases,
        tag_filter=tag_filter,
    )


def _format_tag_filter(tag_filter: Optional[dict]) -> str:
    """Render an applied tag filter as one summary line, e.g.
    ``Tags: smoke, fast (mode any); excluded: slow``."""
    parts = []
    tags = tag_filter.get("tags") or []
    if tags:
        parts.append(
            f"Tags: {', '.join(tags)} (mode {tag_filter.get('mode', 'any')})"
        )
    excluded = tag_filter.get("exclude_tags") or []
    if excluded:
        parts.append(f"Excluded tags: {', '.join(excluded)}")
    return "; ".join(parts)


def format_stability(report: StabilityReport) -> str:
    """Render a stability report as a plain-text summary table."""
    lines = [
        f"Suite: {report.suite}  (repeated {report.repetitions}x)",
    ]
    if report.tag_filter:
        lines.append(_format_tag_filter(report.tag_filter))
    lines += [
        f"Flaky: {len(report.flaky_cases)}  "
        f"Always passing: {len(report.always_passing)}  "
        f"Always failing: {len(report.always_failing)}",
        "",
        f"{'case':<22} {'pass rate':<10} {'passes':<8} {'mean ms':>8}  verdict",
        "-" * 66,
    ]
    for c in report.cases:
        if c.flaky:
            verdict = "FLAKY"
        elif c.passes == c.runs:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        lines.append(
            f"{c.case_id:<22} {c.pass_rate * 100:>8.1f}%  "
            f"{c.passes}/{c.runs:<6} {c.mean_latency_ms:>8.1f}  {verdict}"
        )
    if report.flaky_cases:
        lines += [
            "",
            "Flaky cases passed at least once and failed at least once; "
            "the verdict is not reproducible. Check the agent for "
            "nondeterminism or the case for a brittle scorer.",
        ]
    breakdown = report.tag_breakdown
    if breakdown:
        # A tag that keeps failing — or keeps flipping — is visible at
        # a glance, without digging through the per-case rows.
        lines.append("")
        lines.append("Per-tag pass rates:")
        width = max(len(tag) for tag in breakdown)
        for tag, stats in breakdown.items():
            line = (
                f"  {tag:<{width}}  {stats['passes']}/{stats['runs']} "
                f"({stats['pass_rate'] * 100:.1f}%)"
            )
            if stats["flaky"]:
                line += f"  {stats['flaky']} flaky"
            lines.append(line)
    return "\n".join(lines)
