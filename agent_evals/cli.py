"""CLI: run an eval suite (JSON) against an agent (dotted import path).

Usage:
    python -m agent_evals.cli --suite examples/calculator.json \
        --agent examples.calculator_agent:agent
    python -m agent_evals.cli --suite examples/kv_store.json \
        --agent examples.kv_agent:agent --report reports/kv.json

The agent is named as ``module:attribute`` and must be a callable taking a
prompt string. Exit code is 0 when every case passes, 1 otherwise.
``--workers N`` runs cases concurrently on a thread pool (default 1);
the agent must be thread-safe for more than 1.
"""

from __future__ import annotations

import argparse
import importlib
import sys

from .evals import format_summary, load_suite, run_suite


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
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be a positive integer")

    suite = load_suite(args.suite)
    agent = load_agent(args.agent)
    report = run_suite(suite, agent, workers=args.workers)
    print(format_summary(report))
    if args.report:
        report.save(args.report)
        print(f"\nReport written to {args.report}")
    return 0 if report.passed == report.total else 1


if __name__ == "__main__":
    sys.exit(main())
