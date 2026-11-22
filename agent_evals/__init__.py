"""agent-evals: a lightweight, deterministic eval harness for AI agents.

Pure Python 3 standard library. No model calls, no API keys, no network.
The agent under test is a plain Python callable.
"""

__version__ = "0.1.4"

from .evals import (
    Case,
    Result,
    RunReport,
    Suite,
    format_summary,
    load_report,
    load_suite,
    run_suite,
    score_case,
)

__all__ = [
    "Case",
    "Result",
    "RunReport",
    "Suite",
    "format_summary",
    "load_report",
    "load_suite",
    "run_suite",
    "score_case",
]
