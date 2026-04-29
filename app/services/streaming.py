"""Streaming helpers shared by the chat route.

Two pieces here:
  - sse_frame: turns a piece of text into a Server-Sent Events data frame
    that the browser's EventSource can parse. Multi-line text gets split
    onto multiple `data:` lines per the SSE spec (the browser auto-joins
    them with \n), so no escaping needed.
  - stream_plain / stream_reply: token-level async iterators over an
    Ollama chat call. Used for the CHAT and RISK branches where no RAG
    retrieval is required — straight prompt → streaming reply.

All functions here yield **already-SSE-framed strings**, so chat.py can
pass them straight to StreamingResponse without further wrapping.
"""

from __future__ import annotations

from typing import AsyncIterator

from app.core.config import settings
from app.services.ollama_client import get_client


# ---------------------------------------------------------------------------
# SSE framing
# ---------------------------------------------------------------------------


def sse_frame(text: str) -> str:
    """Format a piece of text as an SSE data frame.

    Multi-line text is split into multiple 'data:' lines per spec; the
    browser auto-joins them with \n on the client side.
    """
    if not text:
        return ""
    lines = text.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"


def sse_event(event: str, data: str = "") -> str:
    """Named SSE event (e.g., 'done', 'error') — frontend can switch on it."""
    return f"event: {event}\ndata: {data}\n\n"


# ---------------------------------------------------------------------------
# token-level streaming from Ollama
# ---------------------------------------------------------------------------

PLAIN_SYSTEM = (
    "你是校园心理咨询助手 MindWise。用温和、共情的口吻回应学生，"
    "语气自然贴近年轻人，不要医学术语。回答控制在 150 字以内。"
)

RISK_COMFORT_SYSTEM = (
    "你是校园心理咨询助手 MindWise。用户刚刚表达了非常强烈的负面情绪，"
    "需要你立即做第一时间的情感安抚。请做三件事："
    "① 先肯定他愿意说出来的勇气；"
    "② 表达你在听、不会评判；"
    "③ 告诉他学校心理中心 24 小时电话，并建议现在就联系信任的人。"
    "语气温和坚定，回答控制在 200 字以内。不要做任何心理诊断。"
)


async def stream_reply(
    user_message: str,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> AsyncIterator[str]:
    """Stream a reply from the fine-tuned model, token by token.

    Yields **SSE-framed** strings ready to push to the client. Catches
    Ollama failures and yields a graceful fallback frame instead of
    raising — we never want an exception to surface as a broken HTTP
    connection mid-stream.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})

    try:
        stream = await get_client().chat(
            model=settings.ollama_model,
            messages=messages,
            stream=True,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        async for chunk in stream:
            token = chunk.get("message", {}).get("content") or ""
            if token:
                yield sse_frame(token)
    except Exception as e:
        yield sse_frame(f"[系统暂时有点忙，请稍后再试一下] ({type(e).__name__})")
    finally:
        yield sse_event("done")


async def stream_plain(user_message: str) -> AsyncIterator[str]:
    """CHAT branch: light, friendly reply, no crisis framing."""
    async for frame in stream_reply(user_message, system=PLAIN_SYSTEM):
        yield frame


async def stream_risk_comfort(user_message: str) -> AsyncIterator[str]:
    """RISK branch: crisis comfort mode. Mail alert is fired via
    BackgroundTask in parallel with this stream."""
    async for frame in stream_reply(user_message, system=RISK_COMFORT_SYSTEM):
        yield frame
