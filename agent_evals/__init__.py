"""agent-evals: a lightweight, deterministic eval harness for AI agents.

Pure Python 3 standard library. No model calls, no API keys, no network —
unless you register a judge that calls one (see ``agent_evals.judge``).
The agent under test is a plain Python callable.
"""

__version__ = "0.61.0"

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
from .junit import to_junit_xml
from .stability import (
    CaseStability,
    StabilityReport,
    format_stability,
    repeat_suite,
)

__all__ = [
    "Case",
    "CaseComparison",
    "CaseFlakyHistory",
    "CaseStability",
    "CaseTrend",
    "DEFAULT_JUDGES",
    "FlakyHistoryReport",
    "JudgeVerdict",
    "ModelComparison",
    "ReportComparison",
    "Result",
    "RunReport",
    "RunSummary",
    "StabilityReport",
    "StabilityRunSummary",
    "Suite",
    "TrendReport",
    "coerce_verdict",
    "compare_flaky_history",
    "compare_history",
    "compare_models",
    "compare_reports",
    "format_comparison",
    "format_flaky_history",
    "format_model_comparison",
    "format_stability",
    "format_summary",
    "format_trend",
    "load_report",
    "load_reports_dir",
    "load_stability_reports_dir",
    "load_suite",
    "make_http_judge",
    "make_output_diff",
    "repeat_suite",
    "run_suite",
    "score_case",
    "stub_judge",
    "to_junit_xml",
]


def __getattr__(name):
    # Lazy import: importing .compare/.trends eagerly would make
    # `python -m agent_evals.compare` (or `.trends`) warn about the
    # module already being in sys.modules (it is imported by this
    # package __init__).
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
    if name in {
        "CaseTrend",
        "RunSummary",
        "TrendReport",
        "compare_history",
        "format_trend",
        "load_reports_dir",
    }:
        from .trends import (
            CaseTrend,
            RunSummary,
            TrendReport,
            compare_history,
            format_trend,
            load_reports_dir,
        )

        return {
            "CaseTrend": CaseTrend,
            "RunSummary": RunSummary,
            "TrendReport": TrendReport,
            "compare_history": compare_history,
            "format_trend": format_trend,
            "load_reports_dir": load_reports_dir,
        }[name]
    if name in {
        "CaseComparison",
        "ModelComparison",
        "compare_models",
        "format_model_comparison",
    }:
        from .model_compare import (
            CaseComparison,
            ModelComparison,
            compare_models,
            format_model_comparison,
        )

        return {
            "CaseComparison": CaseComparison,
            "ModelComparison": ModelComparison,
            "compare_models": compare_models,
            "format_model_comparison": format_model_comparison,
        }[name]
    if name in {
        "CaseFlakyHistory",
        "FlakyHistoryReport",
        "StabilityRunSummary",
        "compare_flaky_history",
        "format_flaky_history",
        "load_stability_reports_dir",
    }:
        from .flaky_history import (
            CaseFlakyHistory,
            FlakyHistoryReport,
            StabilityRunSummary,
            compare_flaky_history,
            format_flaky_history,
            load_stability_reports_dir,
        )

        return {
            "CaseFlakyHistory": CaseFlakyHistory,
            "FlakyHistoryReport": FlakyHistoryReport,
            "StabilityRunSummary": StabilityRunSummary,
            "compare_flaky_history": compare_flaky_history,
            "format_flaky_history": format_flaky_history,
            "load_stability_reports_dir": load_stability_reports_dir,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
