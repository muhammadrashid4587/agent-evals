"""CLI: run an eval suite (JSON) against an agent (dotted import path).

Usage:
    python -m agent_evals.cli --suite examples/calculator.json \
        --agent examples.calculator_agent:agent
    python -m agent_evals.cli --suite examples/kv_store.json \
        --agent examples.kv_agent:agent --report reports/kv.json

The agent is named as ``module:attribute`` and must be a callable taking a
prompt string. Exit code is 0 when every case passes, 1 otherwise. A case
that passes its checks but exceeds its ``max_latency_ms`` budget is
reported as a budget violation, not a pass, but does not fail the run
unless ``--fail-on-budget`` is given; a case that passes its checks but
whose agent reported more tokens than its ``max_tokens`` budget is
reported as a cost violation, not a pass, but does not fail the run
unless ``--fail-on-cost`` is given. ``--report`` writes the JSON
report; ``--junit`` writes the same run as JUnit XML for CI ingestion.
``--fail-on-score 0.9`` exits 1 when a case passes its checks but
scores below 0.9, so a partial-credit pass can still fail the run.
``--workers N`` runs cases concurrently on a thread pool (default 1); the agent must be
thread-safe for more than 1. ``--repeat N`` runs the whole suite N
times and reports per-case pass rates, flagging flaky cases; the run
only passes when every case passed every time. Cases with a ``judge``
scorer use the built-in deterministic ``stub`` judge unless a judge is
registered: ``--judge module:attribute`` registers a judge callable
under the name ``"custom"``, and ``--judge-url URL`` registers an HTTP
judge under the name ``"http"``. ``--tag TAG`` (repeatable) runs only
the cases carrying that tag, so a smoke run is ``--tag smoke``;
``--tag-mode all`` requires a case to carry every given tag instead
of any one of them. ``--exclude-tag TAG`` (repeatable) skips the
cases carrying that tag, and combines with ``--tag`` (select first,
then exclude). ``--list-tags`` lists the tags used in the suite with
their per-tag case counts and exits without running the agent, and a
run report carries a per-tag pass-rate breakdown (see
``RunReport.tag_breakdown``).
"""

from __future__ import annotations

import argparse
import importlib
import sys

from .evals import format_summary, load_suite, run_suite
from .judge import DEFAULT_JUDGES, make_http_judge
from .stability import format_stability, repeat_suite


def load_callable(spec: str, kind: str):
    """Import a callable from a ``module:attribute`` spec."""
    if ":" not in spec:
        raise ValueError(
            f"{kind} must be given as module:attribute, got {spec!r}"
        )
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    if not callable(value):
        raise ValueError(f"{kind} {spec!r} is not callable")
    return value


def load_agent(spec: str):
    """Import an agent from a ``module:attribute`` spec."""
    return load_callable(spec, "agent")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals",
        description="Run an eval suite against a Python agent callable.",
    )
    parser.add_argument("--suite", required=True, help="path to a suite JSON file")
    parser.add_argument(
        "--agent",
        help="agent to test, as module:attribute (e.g. examples.calculator_agent:agent); "
        "required unless --list-tags is given, which never runs the agent",
    )
    parser.add_argument(
        "--report", help="optional path to write the JSON report to"
    )
    parser.add_argument(
        "--junit",
        help="optional path to write the report as JUnit XML, so CI "
        "systems can ingest eval results (not supported with "
        "--repeat, which reports stability, not a single run)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="cases to run at once (default 1, sequential); the agent "
        "must be thread-safe for more than 1",
    )
    parser.add_argument(
        "--fail-on-budget",
        action="store_true",
        help="exit 1 when a case passes its checks but exceeds its "
        "max_latency_ms budget (by default a budget violation is "
        "reported but does not fail the run)",
    )
    parser.add_argument(
        "--fail-on-cost",
        action="store_true",
        help="exit 1 when a case passes its checks but the agent "
        "reported more tokens than its max_tokens budget (by "
        "default a cost violation is reported but does not fail "
        "the run; a case whose agent reports no token count can "
        "never trip it)",
    )
    parser.add_argument(
        "--fail-on-score",
        type=float,
        default=None,
        metavar="THRESHOLD",
        help="exit 1 when a case passes its checks but scores below "
        "THRESHOLD, in [0, 1] (by default a partial-credit pass is "
        "still a pass); a case that fails its checks, times out, "
        "or errors already fails the run, so this gate only covers "
        "passed cases scoring below the bar",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the suite N times and report per-case pass rates, "
        "flagging flaky cases that pass sometimes and fail other "
        "times (default 1, a single run)",
    )
    parser.add_argument(
        "--judge",
        help="judge callable for 'judge' scorers, as "
        "module:attribute (e.g. my_judges:llm_judge); registered "
        "under the name \"custom\"",
    )
    parser.add_argument(
        "--judge-url",
        help="HTTP endpoint for 'judge' scorers; registered under "
        "the name \"http\" (POSTs {prompt, output, rubric}, expects "
        "{passed, score, rationale})",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="TAG",
        help="run only cases carrying this tag; repeatable, e.g. "
        "--tag smoke --tag fast (see --tag-mode for how multiple "
        "tags combine)",
    )
    parser.add_argument(
        "--tag-mode",
        choices=["any", "all"],
        default="any",
        help="how multiple --tag values combine: 'any' runs a case "
        "carrying at least one of them (default), 'all' runs only a "
        "case carrying every one of them",
    )
    parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        metavar="TAG",
        help="skip cases carrying this tag; repeatable, e.g. "
        "--exclude-tag slow skips the slow cases, and it combines "
        "with --tag (select first, then exclude)",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="list the tags used in the suite with their per-tag case "
        "counts and exit, without running the agent (applies --tag "
        "and --exclude-tag first, so it lists the tags of the "
        "selected set)",
    )
    args = parser.parse_args(argv)
    if not args.list_tags and not args.agent:
        parser.error("--agent is required unless --list-tags is given")
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.repeat < 1:
        parser.error("--repeat must be a positive integer")
    if args.fail_on_score is not None and not (
        0.0 <= args.fail_on_score <= 1.0
    ):
        # The chained comparison also rejects NaN and infinity: no
        # comparison with them is true.
        parser.error("--fail-on-score must be in [0, 1]")

    loaded = load_suite(args.suite)
    suite = loaded
    if args.tag:
        suite = suite.select(args.tag, match=args.tag_mode)
        if not suite.cases:
            print(
                f"no cases match tag(s) {', '.join(args.tag)} "
                f"(mode {args.tag_mode}) in {args.suite}",
                file=sys.stderr,
            )
            return 2
    if args.exclude_tag:
        # A typo in an excluded tag would silently exclude nothing,
        # so an excluded tag that matches no case in the loaded suite
        # is an error, same as a --tag that matches nothing.
        known = {t for c in loaded.cases for t in c.tags}
        unknown = [t for t in args.exclude_tag if t not in known]
        if unknown:
            print(
                f"no cases carry excluded tag(s) {', '.join(unknown)} "
                f"in {args.suite}",
                file=sys.stderr,
            )
            return 2
        suite = suite.exclude(args.exclude_tag)
        if not suite.cases:
            print(
                f"no cases remain after excluding tag(s) "
                f"{', '.join(args.exclude_tag)} in {args.suite}",
                file=sys.stderr,
            )
            return 2
    if args.list_tags:
        # Listing never runs the agent, so --agent is not needed and
        # is never imported here.
        counts = suite.tag_counts()
        if not counts:
            print(f"no tags in {args.suite}")
        else:
            print(f"Tags in {args.suite}:")
            width = max(len(tag) for tag, _ in counts)
            for tag, count in counts:
                noun = "case" if count == 1 else "cases"
                print(f"  {tag:<{width}}  {count} {noun}")
        return 0
    agent = load_agent(args.agent)
    judges = dict(DEFAULT_JUDGES)
    if args.judge:
        judges["custom"] = load_callable(args.judge, "judge")
    if args.judge_url:
        judges["http"] = make_http_judge(args.judge_url)
    if args.repeat > 1:
        if args.junit:
            parser.error("--junit is not supported with --repeat")
        stability = repeat_suite(
            suite,
            agent,
            args.repeat,
            workers=args.workers,
            judges=judges,
            tags=args.tag or None,
            tag_mode=args.tag_mode,
            exclude_tags=args.exclude_tag or None,
        )
        print(format_stability(stability))
        if args.report:
            stability.save(args.report)
            print(f"\nStability report written to {args.report}")
        # A repeated run only passes when every case passed every time.
        return 0 if not stability.flaky_cases and not stability.always_failing else 1
    report = run_suite(suite, agent, workers=args.workers, judges=judges)
    print(format_summary(report))
    if args.report:
        report.save(args.report)
        print(f"\nReport written to {args.report}")
    if args.junit:
        report.save_junit(args.junit)
        print(f"JUnit report written to {args.junit}")
    if args.fail_on_score is not None:
        # Only a case that passed its checks can trip this gate: a
        # case that failed, timed out, or errored already fails the
        # run, and a budget/cost violation is not a scored result
        # (its score was reset to 0.0 when it became a violation),
        # so the gate looks at passed cases only. A score exactly
        # at the threshold passes, the same as a per-case threshold.
        below = [
            r
            for r in report.results
            if r.passed and r.score < args.fail_on_score
        ]
        if below:
            print(
                f"\nScore gate FAILED: {len(below)} case(s) passed "
                f"but scored below {args.fail_on_score:g}: "
                + ", ".join(
                    f"{r.case_id} ({r.score:.2f})" for r in below
                ),
                file=sys.stderr,
            )
            return 1
    if args.fail_on_budget or args.fail_on_cost:
        return 0 if report.passed == report.total else 1
    # Without --fail-on-budget / --fail-on-cost, only genuine check
    # failures fail the run; budget and cost violations are reported
    # but do not fail it.
    genuine_failures = [
        r
        for r in report.results
        if not r.passed and not r.budget_violated and not r.cost_violated
    ]
    return 0 if not genuine_failures else 1


if __name__ == "__main__":
    sys.exit(main())
