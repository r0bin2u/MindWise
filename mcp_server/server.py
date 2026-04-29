"""FastMCP server exposing mindwise tools.

Two tools:
  - excel_writer: log every non-CHAT turn
  - mail_alert:   send high-risk alert to the counselor admin

Run standalone:
    python -m mcp_server.server

This is the MCP "Server" side of the architecture image — the Qwen
model acts as the MCP "Client" deciding when to call which tool; this
server receives the JSON-RPC tool call, dispatches to the pure-function
tool implementations in mcp_server.tools, and returns the result.

The same tool functions are imported directly by app/agents/mcp_client
for the in-process low-latency path; running this standalone server is
what lets Claude Desktop / IDEs / any MCP client also invoke the tools.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.tools.excel_writer import append_row
from mcp_server.tools.mail_alert import send_alert


mcp = FastMCP("mindwise-tools")


@mcp.tool()
def excel_writer(
    user_id: str,
    content: str,
    emotion_label: str,
    score: float,
    risk_level: str,
    timestamp: str | None = None,
) -> dict:
    """追加一条咨询记录到 Excel 日志。

    参数:
      user_id: 匿名学生 ID
      content: 用户本轮对话内容
      emotion_label: 情绪标签 (正常/焦虑/低落/高风险)
      score: 多模态融合情绪总分
      risk_level: 风险等级 (正常/需关注/高风险)
      timestamp: 对话时间, ISO 或 'YYYY-MM-DD HH:MM:SS', 不给则取现在
    """
    return append_row(user_id, content, emotion_label, score, risk_level, timestamp)


@mcp.tool()
async def mail_alert(
    user_id: str,
    content: str,
    emotion_label: str,
    score: float,
    risk_level: str,
    timestamp: str | None = None,
) -> dict:
    """向管理员发送高危心理预警邮件。

    仅当 risk_level 为 '高风险' 或意图层判定 RISK 时调用。
    参数同 excel_writer。
    """
    return await send_alert(user_id, content, emotion_label, score, risk_level, timestamp)


if __name__ == "__main__":
    mcp.run()
