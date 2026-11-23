"""JUnit XML export for run reports, so CI systems can ingest eval results.

A :class:`~agent_evals.evals.RunReport` becomes one ``<testsuite>``:
each case is a ``<testcase>`` (``classname`` is the suite name, ``name``
is the case id, ``time`` is the latency in seconds). A case that failed
its checks becomes a ``<failure>`` whose message and text are the
result's detail; a case that could not be scored at all — a timeout,
an agent crash, or a scorer error — becomes an ``<error>`` instead,
because the failure was in the harness or the agent, not in the
answer. A budget violation is a ``<failure>``: the answer was right
but too slow, and the detail says by how much. Suite version,
fingerprint, and weighted score ride along as ``<properties>``.

The XML is built with :mod:`xml.etree.ElementTree`, so text and
attribute values are escaped for free. Agent output can contain
control characters that are illegal in XML 1.0, so details are
sanitised before they are written — anything below U+0020 other than
tab, newline, and carriage return becomes U+FFFD.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

__all__ = ["to_junit_xml"]

# XML 1.0 allows tab, newline, and carriage return below U+0020; every
# other control character is illegal and would make the document
# unparseable. Agent output is untrusted text, so sanitise it.
_ILLEGAL_XML_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(text: str) -> str:
    """Replace characters illegal in XML 1.0 with U+FFFD."""
    return _ILLEGAL_XML_CHARS.sub("\ufffd", text)


def _is_error(result) -> bool:
    """True when the case could not be scored at all: a timeout, an
    agent crash, or a scorer error. Everything else that did not pass
    is a check failure."""
    return (
        result.timed_out
        or result.detail.startswith("agent error:")
        or result.detail.startswith("scorer error:")
    )


def to_junit_xml(report) -> str:
    """Render a :class:`~agent_evals.evals.RunReport` as JUnit XML.

    ``report`` is duck-typed (a ``RunReport``): the suite name,
    ``started_at``, version, fingerprint, weighted score, and one
    result per case with ``case_id``, ``passed``, ``detail``,
    ``latency_ms``, ``timed_out``, and ``budget_violated``.
    """
    failures = sum(
        1
        for r in report.results
        if not r.passed and not _is_error(r)
    )
    errors = sum(1 for r in report.results if not r.passed and _is_error(r))
    suite_el = ET.Element(
        "testsuite",
        {
            "name": report.suite,
            "tests": str(len(report.results)),
            "failures": str(failures),
            "errors": str(errors),
            "time": f"{report.total_latency_ms / 1000:.3f}",
            "timestamp": report.started_at,
        },
    )
    properties = ET.SubElement(suite_el, "properties")
    for name, value in (
        ("suite_version", report.suite_version),
        ("suite_fingerprint", report.suite_fingerprint),
        ("weighted_score", f"{report.weighted_score:.4f}"),
    ):
        if value is not None:
            prop = ET.SubElement(properties, "property")
            prop.set("name", name)
            prop.set("value", _clean(str(value)))
    for r in report.results:
        case_el = ET.SubElement(
            suite_el,
            "testcase",
            {
                "classname": report.suite,
                "name": r.case_id,
                "time": f"{r.latency_ms / 1000:.3f}",
            },
        )
        if r.passed:
            continue
        detail = _clean(r.detail)
        if _is_error(r):
            child = ET.SubElement(case_el, "error")
        else:
            child = ET.SubElement(case_el, "failure")
        child.set("message", detail)
        child.text = detail
    ET.indent(suite_el)
    return ET.tostring(suite_el, encoding="unicode", xml_declaration=True)
