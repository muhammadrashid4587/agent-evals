"""agent-evals: a lightweight, deterministic eval harness for AI agents.

Pure Python 3 standard library. No model calls, no API keys, no network.
The agent under test is a plain Python callable.
"""

__version__ = "0.4.0"

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
from .stability import (
    CaseStability,
    StabilityReport,
    format_stability,
    repeat_suite,
)

__all__ = [
    "Case",
    "CaseStability",
    "ReportComparison",
    "Result",
    "RunReport",
    "StabilityReport",
    "Suite",
    "compare_reports",
    "format_comparison",
    "format_stability",
    "format_summary",
    "load_report",
    "load_suite",
    "repeat_suite",
    "run_suite",
    "score_case",
]


def __getattr__(name):
    # Lazy import: importing .compare eagerly would make
    # `python -m agent_evals.compare` warn about the module already
    # being in sys.modules (it is imported by this package __init__).
    if name in {"ReportComparison", "compare_reports", "format_comparison"}:
        from .compare import (
            ReportComparison,
            compare_reports,
            format_comparison,
        )

        return {
            "ReportComparison": ReportComparison,
            "compare_reports": compare_reports,
            "format_comparison": format_comparison,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
