"""Langfuse integration — no-op when not configured.

Langfuse gives us per-request traces with nested spans: parent span is
the /chat turn, children are text_emotion / fuse / classify_intent /
agentic_rag / on_turn_end. Each span captures input, output, duration,
and any tags we attach (intent, fused risk, etc.).

If LANGFUSE_PUBLIC_KEY isn't set we skip initialization entirely and
the decorator becomes a pass-through — no overhead, no errors in
local-only runs.
"""

from __future__ import annotations

from typing import Any, Callable

from app.core.config import settings


_client = None
_enabled = False


def init_langfuse() -> None:
    """Best-effort Langfuse client init. Called once at app startup."""
    global _client, _enabled
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return  # silently disabled
    try:
        from langfuse import Langfuse

        _client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        _enabled = True
    except Exception as e:
        # never crash startup over observability
        import logging

        logging.getLogger("mindwise").warning(f"Langfuse init failed: {e}")


def is_enabled() -> bool:
    return _enabled


def observe(name: str | None = None):
    """Decorator that creates a Langfuse span around a function.

    When Langfuse isn't configured this is a zero-cost pass-through
    (returns the original function unchanged). When configured, it
    wraps the function in a trace / span and records input + output.

    Works on both sync and async functions.
    """

    def decorator(func: Callable) -> Callable:
        if not _enabled:
            # bind at decoration time — if langfuse wasn't up at app
            # startup, we never pay the wrapper overhead afterwards
            return func
        try:
            from langfuse.decorators import observe as _observe

            return _observe(name=name or func.__name__)(func)
        except Exception:
            return func

    return decorator


def trace_event(name: str, **payload: Any) -> None:
    """Emit a one-shot event (no duration). Used for fire-and-forget
    logging inside BackgroundTasks where we don't want a full span."""
    if not _enabled or _client is None:
        return
    try:
        _client.event(name=name, metadata=payload)
    except Exception:
        pass  # observability never breaks business flow


def flush() -> None:
    """Force-flush pending traces. Call on graceful shutdown."""
    if _client is not None:
        try:
            _client.flush()
        except Exception:
            pass
