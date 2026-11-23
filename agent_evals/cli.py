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
unless ``--fail-on-budget`` is given. ``--workers N`` runs cases
concurrently on a thread pool (default 1); the agent must be
thread-safe for more than 1. ``--repeat N`` runs the whole suite N
times and reports per-case pass rates, flagging flaky cases; the run
only passes when every case passed every time.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from .evals import format_summary, load_suite, run_suite
from .stability import format_stability, repeat_suite


def load_agent(spec: str):
    """Import an agent from a ``module:attribute`` spec."""
    if ":" not in spec:
        raise ValueError(
            f"agent must be given as module:attribute, got {spec!r}"
        )
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    agent = getattr(module, attr)
    if not callable(agent):
        raise ValueError(f"agent {spec!r} is not callable")
    return agent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-evals",
        description="Run an eval suite against a Python agent callable.",
    )
    parser.add_argument("--suite", required=True, help="path to a suite JSON file")
    parser.add_argument(
        "--agent",
        required=True,
        help="agent to test, as module:attribute (e.g. examples.calculator_agent:agent)",
    )
    parser.add_argument(
        "--report", help="optional path to write the JSON report to"
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
        "--repeat",
        type=int,
        default=1,
        help="run the suite N times and report per-case pass rates, "
        "flagging flaky cases that pass sometimes and fail other "
        "times (default 1, a single run)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    if args.repeat < 1:
        parser.error("--repeat must be a positive integer")

    suite = load_suite(args.suite)
    agent = load_agent(args.agent)
    if args.repeat > 1:
        stability = repeat_suite(suite, agent, args.repeat, workers=args.workers)
        print(format_stability(stability))
        if args.report:
            stability.save(args.report)
            print(f"\nStability report written to {args.report}")
        # A repeated run only passes when every case passed every time.
        return 0 if not stability.flaky_cases and not stability.always_failing else 1
    report = run_suite(suite, agent, workers=args.workers)
    print(format_summary(report))
    if args.report:
        report.save(args.report)
        print(f"\nReport written to {args.report}")
    if args.fail_on_budget:
        return 0 if report.passed == report.total else 1
    # Without --fail-on-budget, only genuine check failures fail the run;
    # budget violations are reported but do not fail it.
    genuine_failures = [
        r for r in report.results if not r.passed and not r.budget_violated
    ]
    return 0 if not genuine_failures else 1


if __name__ == "__main__":
    sys.exit(main())
