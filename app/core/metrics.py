"""Business-level Prometheus metrics.

HTTP-level metrics (latency, status codes, in-progress) are handled
automatically by prometheus_fastapi_instrumentator — see app/main.py.

What we add here is the stuff the instrumentator can't know about:
  - which intent we routed to (CHAT / CONSULT / RISK)
  - what the fused risk band looked like (正常 / 需关注 / 高风险)
  - whether excel_writer and mail_alert actually fired
  - how long each pipeline stage took

On the ops side these are the numbers a counselor dashboard would alert
on: a sudden spike in 高风险 turns, or mail_alert errors climbing, or
intent=CHAT with fused=高风险 (the silent-crisis rate).
"""

from prometheus_client import Counter, Histogram


# ---- counters ----

intent_total = Counter(
    "mindwise_intent_total",
    "Count of chat turns by classified intent.",
    ["intent"],  # CHAT / CONSULT / RISK
)

fused_risk_total = Counter(
    "mindwise_fused_risk_total",
    "Count of chat turns by fused risk band.",
    ["risk"],  # 正常 / 需关注 / 高风险
)

silent_crisis_total = Counter(
    "mindwise_silent_crisis_total",
    "Turns where intent was CHAT but fused risk was 高风险 "
    "(text-only looked casual, multimodal flagged crisis).",
)

mcp_tool_total = Counter(
    "mindwise_mcp_tool_total",
    "MCP tool invocations by tool name and outcome.",
    ["tool", "outcome"],  # tool=excel_writer|mail_alert, outcome=ok|fail
)


# ---- histograms ----

stage_latency_seconds = Histogram(
    "mindwise_stage_latency_seconds",
    "Latency of each pipeline stage, seconds.",
    ["stage"],  # text_emotion / fuse / classify_intent / rag / mcp_dispatch
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0),
)
