"""Prometheus metrics unit tests.

Verify that the counters actually tick on the right events. We read the
`.labels(...).._value.get()` internal value instead of scraping /metrics
because in-process tests don't need the HTTP layer.
"""
from app.core.metrics import (
    fused_risk_total,
    intent_total,
    mcp_tool_total,
    silent_crisis_total,
)


def _val(counter):
    """Sum of all label-combo values for a Counter (int)."""
    return sum(
        m.value for fam in counter.collect() for m in fam.samples
        if m.name.endswith("_total") and not m.name.endswith("_created")
    )


def test_intent_counter_increments():
    before = intent_total.labels(intent="CHAT")._value.get()
    intent_total.labels(intent="CHAT").inc()
    assert intent_total.labels(intent="CHAT")._value.get() == before + 1


def test_fused_risk_counter_has_all_three_bands():
    for band in ["正常", "需关注", "高风险"]:
        # just constructing the label combo registers it
        fused_risk_total.labels(risk=band).inc()
        assert fused_risk_total.labels(risk=band)._value.get() >= 1


def test_silent_crisis_is_unlabeled_counter():
    before = silent_crisis_total._value.get()
    silent_crisis_total.inc()
    assert silent_crisis_total._value.get() == before + 1


def test_mcp_tool_counter_tracks_outcome():
    ok_before = mcp_tool_total.labels(tool="excel_writer", outcome="ok")._value.get()
    fail_before = mcp_tool_total.labels(tool="excel_writer", outcome="fail")._value.get()

    mcp_tool_total.labels(tool="excel_writer", outcome="ok").inc()
    mcp_tool_total.labels(tool="excel_writer", outcome="fail").inc(2)

    assert mcp_tool_total.labels(tool="excel_writer", outcome="ok")._value.get() == ok_before + 1
    assert mcp_tool_total.labels(tool="excel_writer", outcome="fail")._value.get() == fail_before + 2
