"""Compare several agents (or models) on the same eval suite.

``compare_reports`` (see :mod:`agent_evals.compare`) compares two runs of
one agent over time. This module answers the other question: given the
same suite, which agent is best? ``compare_models`` runs the suite
against every named agent and produces one comparison view: per-case
and aggregate scores, latency, and token/cost totals side by side,
with a winner per case and an overall ranking. The case-wins count
is also broken down per tag, so a tag an agent keeps losing on is
visible without digging through the per-case table.

The per-case winner is the agent with the highest score on that case;
a tie is broken by the lowest latency, then by name, so the winner is
deterministic. A case where every agent scored 0 has no winner — there
is nothing to win. The overall ranking sorts by weighted score, then
pass rate, then total latency (lowest first), then name.

Usage:
    python -m agent_evals.model_compare --suite examples/calculator.json \\
        --agent fast=examples.calculator_agent:agent \\
        --agent slow=my_agents:slow_calc
    python -m agent_evals.model_compare --suite examples/kv_store.json \\
        --agent a=examples.kv_agent:agent \\
        --agent b=examples.kv_agent:agent --json reports/compare.json

An agent spec is ``name=module:attribute``; the name is what the
comparison view shows. Two specs may not share a name, and at least
two agents are required — comparing one agent against itself is a
single run, not a comparison.

Exit code is 0 by default: a comparison view reports which agent
won, it does not gate CI. Use ``agent-evals-compare`` to gate on
regressions between two runs of the same agent. ``--fail-on-lose
BASELINE`` turns the view into a gate: it exits 1 when the named
baseline agent beats any other agent in the comparison (higher
weighted score, then higher pass rate, then lower total latency —
an exact tie on all three is not a loss, so the name tie-break that
decides the ranking never fails the gate), and 0 when no agent
loses to the baseline. ``--fail-on-lose-per-case BASELINE`` is the
per-case variant of the same gate: it exits 1 when the baseline
won strictly more individual cases than any other agent (the
"Case wins" count the view already prints), and 0 when no agent
loses to the baseline on case wins. A tie in case wins is not a
loss, and a case no agent won counts for nobody. The aggregate
gate can pass while the per-case gate fails (a baseline that wins
narrowly on many cases but loses the aggregate on score), and
vice versa — use both when a candidate must beat the baseline
case by case as well as overall.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional

from .cli import load_agent
from .evals import RunReport, load_suite, run_suite

__all__ = [
    "CaseComparison",
    "ModelComparison",
    "compare_models",
    "format_model_comparison",
]


@dataclass
class CaseComparison:
    """One case's outcome across every agent.

    ``results`` maps agent name -> the case's ``Result`` in that
    agent's run. ``winner`` is the name of the agent that won the
    case — the highest score, breaking ties by lowest latency, then
    by name — or None when every agent scored 0 on the case and
    there is nothing to win. ``tags`` is the case's tag list, so
    the comparison can break case wins down per tag.
    """

    case_id: str
    results: dict
    winner: Optional[str] = None
    tags: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "winner": self.winner,
            "tags": list(self.tags),
            "agents": {
                name: {
                    "passed": result.passed,
                    "score": round(result.score, 4),
                    "latency_ms": round(result.latency_ms, 3),
                    "tokens": result.tokens,
                    "timed_out": result.timed_out,
                }
                for name, result in self.results.items()
            },
        }


@dataclass
class ModelComparison:
    """The result of running one suite against several agents.

    ``reports`` maps agent name -> that agent's :class:`RunReport`.
    ``ranking`` lists the agent names best first (weighted score,
    then pass rate, then lowest total latency, then name). ``winner``
    is the first name in the ranking, or None when there are no
    agents (``compare_models`` always requires at least two, so a
    comparison built by hand with no agents has no winner).
    """

    suite: str
    reports: dict = field(default_factory=dict)
    cases: list = field(default_factory=list)
    suite_version: Optional[str] = None
    suite_fingerprint: Optional[str] = None

    @property
    def agent_names(self) -> list:
        """Agent names in the order they were given."""
        return list(self.reports)

    def _score_key(self, name: str) -> tuple:
        """The comparison key for one agent: weighted score, then
        pass rate, then lowest total latency. Higher is better. The
        name tie-break in ``ranking`` is not part of this key — a
        dead heat on all three is a tie, not a win, and the
        ``--fail-on-lose`` gate must not fail on it."""
        report = self.reports[name]
        return (
            report.weighted_score,
            report.pass_rate,
            -report.total_latency_ms,
        )

    @property
    def ranking(self) -> list:
        """Agent names best first: weighted score, then pass rate,
        then lowest total latency, then name."""
        return sorted(
            self.reports,
            key=lambda name: (
                -self._score_key(name)[0],
                -self._score_key(name)[1],
                -self._score_key(name)[2],
                name,
            ),
        )

    def losers_to(self, baseline: str) -> list:
        """The agents the named baseline agent beats, best first.

        An agent loses to the baseline when the baseline is strictly
        better on the ranking key — higher weighted score, then
        higher pass rate, then lower total latency. An agent tied
        with the baseline on all three has not lost: the name
        tie-break that decides the ranking never counts as a loss.
        Raises ``ValueError`` when ``baseline`` is not one of the
        compared agents.
        """
        if baseline not in self.reports:
            raise ValueError(
                f"baseline {baseline!r} is not one of the compared "
                f"agents: {', '.join(self.agent_names)}"
            )
        baseline_key = self._score_key(baseline)
        return [
            name
            for name in self.ranking
            if name != baseline
            and baseline_key > self._score_key(name)
        ]

    def case_wins(self) -> dict:
        """How many individual cases each agent won outright.

        A case with no winner (every agent scored 0) counts for
        nobody. The keys follow ``agent_names`` order.
        """
        wins = {name: 0 for name in self.agent_names}
        for case in self.cases:
            if case.winner is not None:
                wins[case.winner] += 1
        return wins

    def case_wins_by_tag(self) -> dict:
        """Case wins broken down per tag.

        Maps each tag to ``{"cases": n, "ties": n, "wins": {name: n}}``,
        sorted by tag. A case carrying several tags counts under each
        of them; untagged cases appear under no tag. A case with no
        winner (every agent scored 0 on it) is a tie: it counts
        towards the tag's case count but towards no agent's wins, so
        wins plus ties always equals the tag's case count.
        """
        breakdown = {}
        for case in self.cases:
            for tag in case.tags:
                entry = breakdown.setdefault(
                    tag,
                    {
                        "cases": 0,
                        "ties": 0,
                        "wins": {
                            name: 0 for name in self.agent_names
                        },
                    },
                )
                entry["cases"] += 1
                if case.winner is None:
                    entry["ties"] += 1
                else:
                    entry["wins"][case.winner] += 1
        return dict(sorted(breakdown.items()))

    def per_case_losers_to(self, baseline: str) -> list:
        """The agents the named baseline agent beats on case wins,
        best first (by the overall ranking).

        An agent loses per-case to the baseline when the baseline
        won strictly more individual cases than it did. A tie in
        case wins is not a loss, mirroring ``losers_to``: the
        aggregate gate and the per-case gate can disagree, which
        is the point — a candidate can win the aggregate on score
        while losing case by case, or the other way round. Raises
        ``ValueError`` when ``baseline`` is not one of the
        compared agents.
        """
        if baseline not in self.reports:
            raise ValueError(
                f"baseline {baseline!r} is not one of the compared "
                f"agents: {', '.join(self.agent_names)}"
            )
        wins = self.case_wins()
        return [
            name
            for name in self.ranking
            if name != baseline and wins[baseline] > wins[name]
        ]

    @property
    def winner(self) -> Optional[str]:
        """The best agent overall, or None when there are no agents."""
        ranked = self.ranking
        return ranked[0] if ranked else None

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "suite_version": self.suite_version,
            "suite_fingerprint": self.suite_fingerprint,
            "agents": {
                name: {
                    "passed": report.passed,
                    "total": report.total,
                    "pass_rate": round(report.pass_rate, 4),
                    "weighted_score": round(report.weighted_score, 4),
                    "total_latency_ms": round(report.total_latency_ms, 3),
                    "total_tokens": report.total_tokens,
                }
                for name, report in self.reports.items()
            },
            "ranking": self.ranking,
            "winner": self.winner,
            "case_wins": self.case_wins(),
            "case_wins_by_tag": self.case_wins_by_tag(),
            "cases": [case.to_dict() for case in self.cases],
        }


def _case_winner(case_id: str, results: dict) -> Optional[str]:
    """The agent that won one case, or None when every agent scored 0.

    Highest score wins; a tie is broken by the lowest latency, then
    by name, so the winner is deterministic. A case where the best
    score is 0 has no winner: every agent failed it, so none of them
    beat the others.
    """
    best_score = max((r.score for r in results.values()), default=0.0)
    if best_score <= 0.0:
        return None
    contenders = [n for n, r in results.items() if r.score == best_score]
    return min(contenders, key=lambda n: (results[n].latency_ms, n))


def compare_models(
    suite,
    agents: dict,
    workers: int = 1,
    judges: Optional[dict] = None,
) -> ModelComparison:
    """Run ``suite`` against every agent in ``agents`` (name -> callable)
    and return the comparison view.

    At least two agents are required — one agent is a single run, not
    a comparison. Every agent runs the same suite in the same order,
    so each case is directly comparable across agents. ``workers`` is
    passed through to :func:`run_suite` per agent; a stateful agent
    whose cases depend on each other's order needs ``workers`` at 1,
    the same as a single run.
    """
    if len(agents) < 2:
        raise ValueError(
            "comparing models needs at least two agents, "
            f"got {len(agents)}"
        )
    reports = {
        name: run_suite(suite, agent, workers=workers, judges=judges)
        for name, agent in agents.items()
    }
    cases = []
    for case in suite.cases:
        results = {}
        for name, report in reports.items():
            # Every report ran the same suite, so every case is in
            # every report; index by case_id rather than position.
            match = next(
                (r for r in report.results if r.case_id == case.id), None
            )
            if match is not None:
                results[name] = match
        cases.append(
            CaseComparison(
                case_id=case.id,
                results=results,
                winner=_case_winner(case.id, results),
                tags=list(case.tags),
            )
        )
    return ModelComparison(
        suite=suite.name,
        reports=reports,
        cases=cases,
        suite_version=suite.version,
        suite_fingerprint=suite.fingerprint(),
    )


def _agent_line(rank: int, name: str, report: RunReport, is_winner: bool) -> str:
    """One ranking row: rank, name, pass count, weighted score,
    latency, and tokens when the agent reported any."""
    tokens = (
        f"  tokens {report.total_tokens}"
        if report.total_tokens is not None
        else ""
    )
    mark = "  <-- winner" if is_winner else ""
    return (
        f"  {rank}. {name}  {report.passed}/{report.total} passed  "
        f"weighted {report.weighted_score:.4f}  "
        f"{report.total_latency_ms:.1f} ms{tokens}{mark}"
    )


def format_model_comparison(comp: ModelComparison) -> str:
    """Render a model comparison as a plain-text summary."""
    lines = [f"Suite: {comp.suite}"]
    if comp.suite_version is not None:
        lines.append(f"Eval set: v{comp.suite_version}")
    lines.append("")
    lines.append("Ranking:")
    for i, name in enumerate(comp.ranking, start=1):
        lines.append(
            _agent_line(i, name, comp.reports[name], name == comp.winner)
        )
    lines.append("")
    if not comp.cases:
        lines.append("No cases in the suite.")
        return "\n".join(lines)
    names = comp.agent_names
    # Per-case table: one column per agent, one for the winner.
    # A cell is "PASS 1.00  12.3ms" (or FAIL / TIMEOUT / BUDGET / COST).
    id_width = max(len(c.case_id) for c in comp.cases)
    id_width = max(id_width, len("case"))
    cells = {}
    col_widths = {}
    for name in names:
        col_widths[name] = len(name)
    for case in comp.cases:
        for name in names:
            result = case.results.get(name)
            if result is None:
                cell = "-"
            else:
                if result.budget_violated:
                    mark = "BUDGET"
                elif result.cost_violated:
                    mark = "COST"
                elif result.timed_out:
                    mark = "TIMEOUT"
                else:
                    mark = "PASS" if result.passed else "FAIL"
                cell = (
                    f"{mark} {result.score:.2f}  {result.latency_ms:.1f} ms"
                )
            cells[(case.case_id, name)] = cell
            col_widths[name] = max(col_widths[name], len(cell))
    winner_width = max(
        len(c.winner) if c.winner else 1 for c in comp.cases
    )
    winner_width = max(winner_width, len("winner"))
    header = (
        f"{'case':<{id_width}}  "
        + "  ".join(f"{name:<{col_widths[name]}}" for name in names)
        + f"  {'winner':<{winner_width}}"
    )
    lines.append("Per-case:")
    lines.append(header)
    lines.append("-" * len(header))
    for case in comp.cases:
        row = (
            f"{case.case_id:<{id_width}}  "
            + "  ".join(
                f"{cells[(case.case_id, name)]:<{col_widths[name]}}"
                for name in names
            )
            + f"  {(case.winner or '-'): <{winner_width}}"
        )
        lines.append(row.rstrip())
    # Wins per agent: how many cases each agent won outright.
    wins = comp.case_wins()
    lines.append("")
    lines.append(
        "Case wins: "
        + ", ".join(f"{name} {wins[name]}" for name in names)
    )
    # Wins per tag: which agent won how many cases under each tag.
    # A case carrying several tags counts under each of them; a case
    # with no winner is a tie, counted but won by nobody.
    by_tag = comp.case_wins_by_tag()
    if by_tag:
        lines.append("")
        lines.append("Case wins by tag:")
        for tag, entry in by_tag.items():
            wins_text = ", ".join(
                f"{name} {entry['wins'][name]}" for name in names
            )
            count = entry["cases"]
            noun = "case" if count == 1 else "cases"
            lines.append(
                f"  {tag} ({count} {noun}): {wins_text}, "
                f"ties {entry['ties']}"
            )
    return "\n".join(lines)


def _parse_agent_spec(spec: str) -> tuple:
    """Split a ``name=module:attribute`` spec into (name, spec)."""
    if "=" not in spec:
        raise ValueError(
            f"agent must be given as name=module:attribute, got {spec!r}"
        )
    name, _, agent_spec = spec.partition("=")
    if not name.strip():
        raise ValueError(f"agent name must not be empty, got {spec!r}")
    return name, agent_spec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals-compare-models",
        description="Run a suite against several agents and compare them.",
    )
    parser.add_argument("--suite", required=True, help="path to a suite JSON file")
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        metavar="NAME=MODULE:ATTR",
        help="an agent to compare, as name=module:attribute "
        "(e.g. fast=examples.calculator_agent:agent); repeatable, "
        "at least two required",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="cases to run at once per agent (default 1, sequential); "
        "the agents must be thread-safe for more than 1",
    )
    parser.add_argument(
        "--judge",
        help="judge callable for 'judge' scorers, as "
        "module:attribute; registered under the name \"custom\"",
    )
    parser.add_argument(
        "--judge-url",
        help="HTTP endpoint for 'judge' scorers; registered under "
        "the name \"http\"",
    )
    parser.add_argument(
        "--json", help="optional path to write the comparison as JSON"
    )
    parser.add_argument(
        "--fail-on-lose",
        metavar="BASELINE",
        help="exit 1 when the named baseline agent beats any other "
        "agent in the comparison (higher weighted score, then "
        "higher pass rate, then lower total latency; an exact tie "
        "on all three is not a loss), so a candidate agent that "
        "regresses against the baseline fails CI",
    )
    parser.add_argument(
        "--fail-on-lose-per-case",
        metavar="BASELINE",
        help="exit 1 when the named baseline agent won strictly "
        "more individual cases than any other agent (the \"Case "
        "wins\" count; a tie in case wins is not a loss), so a "
        "candidate agent that loses case by case against the "
        "baseline fails CI even if it wins the aggregate",
    )
    args = parser.parse_args(argv)
    if len(args.agent) < 2:
        parser.error("at least two --agent specs are required")
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    agents = {}
    for spec in args.agent:
        name, agent_spec = _parse_agent_spec(spec)
        if name in agents:
            parser.error(f"duplicate agent name {name!r}")
        try:
            agents[name] = load_agent(agent_spec)
        except (ImportError, AttributeError, ValueError) as exc:
            parser.error(f"cannot load agent {spec!r}: {exc}")
    if args.fail_on_lose and args.fail_on_lose not in agents:
        parser.error(
            f"--fail-on-lose baseline {args.fail_on_lose!r} is not "
            f"one of the compared agents: {', '.join(agents)}"
        )
    if (
        args.fail_on_lose_per_case
        and args.fail_on_lose_per_case not in agents
    ):
        parser.error(
            f"--fail-on-lose-per-case baseline "
            f"{args.fail_on_lose_per_case!r} is not "
            f"one of the compared agents: {', '.join(agents)}"
        )
    from .judge import DEFAULT_JUDGES, make_http_judge

    judges = dict(DEFAULT_JUDGES)
    if args.judge:
        from .cli import load_callable

        try:
            judges["custom"] = load_callable(args.judge, "judge")
        except (ImportError, AttributeError, ValueError) as exc:
            parser.error(f"cannot load judge {args.judge!r}: {exc}")
    if args.judge_url:
        judges["http"] = make_http_judge(args.judge_url)
    suite = load_suite(args.suite)
    comp = compare_models(suite, agents, workers=args.workers, judges=judges)
    print(format_model_comparison(comp))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(comp.to_dict(), f, indent=2)
            f.write("\n")
        print(f"\nComparison written to {args.json}")
    failed = False
    if args.fail_on_lose:
        losers = comp.losers_to(args.fail_on_lose)
        if losers:
            print(
                f"\nFail-on-lose gate FAILED: baseline "
                f"{args.fail_on_lose!r} beats " + ", ".join(losers),
                file=sys.stderr,
            )
            failed = True
    if args.fail_on_lose_per_case:
        wins = comp.case_wins()
        per_case_losers = comp.per_case_losers_to(
            args.fail_on_lose_per_case
        )
        if per_case_losers:
            beaten = ", ".join(
                f"{name} ({wins[name]})" for name in per_case_losers
            )
            print(
                f"\nFail-on-lose-per-case gate FAILED: baseline "
                f"{args.fail_on_lose_per_case!r} won "
                f"{wins[args.fail_on_lose_per_case]} cases, beating "
                f"{beaten}",
                file=sys.stderr,
            )
            failed = True
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
