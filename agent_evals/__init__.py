"""agent-evals: a lightweight, deterministic eval harness for AI agents.

Pure Python 3 standard library. No model calls, no API keys, no network —
unless you register a judge that calls one (see ``agent_evals.judge``).
The agent under test is a plain Python callable.
"""

__version__ = "0.7.0"

from .evals import (
    Case,
    Result,
    RunReport,
    Suite,
    format_summary,
    load_report,
    load_suite,
    make_output_diff,
    run_suite,
    score_case,
)
from .judge import (
    DEFAULT_JUDGES,
    JudgeVerdict,
    coerce_verdict,
    make_http_judge,
    stub_judge,
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
    "DEFAULT_JUDGES",
    "JudgeVerdict",
    "ReportComparison",
    "Result",
    "RunReport",
    "StabilityReport",
    "Suite",
    "coerce_verdict",
    "compare_reports",
    "format_comparison",
    "format_stability",
    "format_summary",
    "load_report",
    "load_suite",
    "make_http_judge",
    "make_output_diff",
    "repeat_suite",
    "run_suite",
    "score_case",
    "stub_judge",
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
