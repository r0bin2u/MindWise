"""Langfuse tracing unit tests.

The key property is graceful no-op when Langfuse isn't configured (no
keys in env). We don't need to test the actual Langfuse SDK — just that
our wrapper doesn't crash and doesn't add overhead when disabled.
"""
from app.core import tracing


def test_is_enabled_defaults_to_false():
    # with no LANGFUSE_* env vars set, the module starts disabled
    assert tracing.is_enabled() is False


def test_observe_is_passthrough_when_disabled():
    """@observe decorator should return the original function unchanged
    when Langfuse isn't configured, so there's zero overhead."""
    @tracing.observe(name="foo")
    def plain_fn(x):
        return x * 2

    assert plain_fn(5) == 10
    # when disabled, the decorator returns the same function object
    assert plain_fn.__name__ == "plain_fn"


def test_trace_event_does_nothing_when_disabled():
    # should not raise even though no client is initialized
    tracing.trace_event("test_event", user="u1", foo=42)


def test_flush_does_nothing_when_disabled():
    tracing.flush()


def test_init_langfuse_no_op_without_keys(monkeypatch):
    """init_langfuse with no keys in settings must not raise or enable."""
    monkeypatch.setattr(tracing.settings, "langfuse_public_key", "")
    monkeypatch.setattr(tracing.settings, "langfuse_secret_key", "")
    tracing.init_langfuse()
    assert tracing.is_enabled() is False
