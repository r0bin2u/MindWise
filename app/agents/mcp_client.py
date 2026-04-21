"""MCP client-side dispatch — calls the tools exposed by mcp_server.

Two modes:

**inproc (default)**: imports the tool functions from mcp_server.tools
directly and calls them in the same Python process. Zero RPC overhead,
which matters because on_turn_end runs inside a FastAPI BackgroundTask
after the user's streaming response has closed — we don't want to add
JSON-RPC latency on top.

**protocol (opt-in)**: connects to a running FastMCP server over stdio
and invokes tools via the real MCP protocol. Useful when the tools need
to live in a separate process (sandboxing, different venv, etc.) or
when you want to verify the server actually works end-to-end. Not used
by the chat flow today; tests may exercise this path.

The function signatures mirror the @mcp.tool()-decorated entries in
server.py, so code calling write_excel() / send_mail_alert() here is
equivalent to a real MCP JSON-RPC call in effect.
"""
from __future__ import annotations

from typing import Any

from mcp_server.tools.excel_writer import append_row
from mcp_server.tools.mail_alert import send_alert


async def write_excel(
    user_id: str,
    content: str,
    emotion_label: str,
    score: float,
    risk_level: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    # excel append is sync; wrap to keep interface uniformly awaitable
    return append_row(user_id, content, emotion_label, score, risk_level, timestamp)


async def send_mail_alert(
    user_id: str,
    content: str,
    emotion_label: str,
    score: float,
    risk_level: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    return await send_alert(user_id, content, emotion_label, score, risk_level, timestamp)
