import pytest

from app.agents.orchestrator import (
    INTENTS,
    _extract_intent,
    classify_intent,
    on_turn_end,
)


# ---------------- _extract_intent ----------------

def test_extract_exact_label():
    for lbl in INTENTS:
        assert _extract_intent(lbl) == lbl


def test_extract_lowercase():
    assert _extract_intent("chat") == "CHAT"
    assert _extract_intent("consult") == "CONSULT"


def test_extract_from_verbose_output():
    # model adds noise around the label — still should resolve
    assert _extract_intent("判断：CHAT") == "CHAT"
    assert _extract_intent("标签 = RISK，已识别") == "RISK"


def test_extract_picks_last_occurrence():
    # "这是 CONSULT 不是 CHAT" -> last token CHAT... but we want to prefer the
    # model's final verdict which is usually at the end. That happens to
    # match rfind behavior; here the test is that last token wins.
    assert _extract_intent("CONSULT 但不是 CHAT") == "CHAT"


def test_extract_returns_none_on_unknown():
    assert _extract_intent("") is None
    assert _extract_intent("我不知道") is None


# ---------------- classify_intent ----------------

@pytest.mark.asyncio
async def test_classify_intent_empty_short_circuits_to_chat():
    assert await classify_intent("") == "CHAT"
    assert await classify_intent("   ") == "CHAT"


@pytest.mark.asyncio
async def test_classify_intent_risk_keyword_fast_path(monkeypatch):
    """Explicit distress phrases must not require the LLM — short-circuit."""
    from app.agents import orchestrator

    called = {"n": 0}

    async def tracker(text):
        called["n"] += 1
        return "CHAT"

    monkeypatch.setattr(orchestrator, "_call_model", tracker)

    for text in [
        "我不想活了",
        "我想死",
        "感觉活不下去了",
        "想去自杀",
        "想结束生命",
    ]:
        assert await classify_intent(text) == "RISK"

    assert called["n"] == 0  # LLM never invoked


@pytest.mark.asyncio
async def test_classify_intent_uses_llm_when_no_keyword(monkeypatch):
    from app.agents import orchestrator

    async def fake_llm(text):
        return "CONSULT"

    monkeypatch.setattr(orchestrator, "_call_model", fake_llm)

    assert await classify_intent("我最近压力好大，睡不好") == "CONSULT"


@pytest.mark.asyncio
async def test_classify_intent_fallback_on_llm_failure(monkeypatch):
    """LLM dead → default CONSULT (safer than CHAT)."""
    from app.agents import orchestrator

    async def boom(text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(orchestrator, "_call_model", boom)

    # no risk keyword, so fast path doesn't fire
    assert await classify_intent("今天吃啥") == "CONSULT"


@pytest.mark.asyncio
async def test_classify_intent_unparseable_output_fallback(monkeypatch):
    """Model returned gibberish without any CHAT/CONSULT/RISK token."""
    from app.agents import orchestrator

    async def garbage(text):
        return "我不知道怎么分类"

    monkeypatch.setattr(orchestrator, "_call_model", garbage)

    assert await classify_intent("今天吃啥") == "CONSULT"


@pytest.mark.asyncio
async def test_classify_intent_chat_routing(monkeypatch):
    from app.agents import orchestrator

    async def fake_llm(text):
        return "CHAT"

    monkeypatch.setattr(orchestrator, "_call_model", fake_llm)

    assert await classify_intent("今天天气怎么样？") == "CHAT"


# ---------------- on_turn_end (MCP dispatch matrix) ----------------
#
# Doc 9.3 trigger matrix:
#   CHAT                           → no excel, no mail
#   CONSULT + risk in {正常,需关注} → excel only
#   CONSULT + risk=高风险          → excel + mail
#   RISK (intent fast-path)        → excel + mail

def _patch_mcp(monkeypatch):
    """Replace write_excel / send_mail_alert with recording fakes."""
    from app.agents import orchestrator as mod

    calls = {"excel": [], "mail": []}

    async def fake_excel(*args, **kwargs):
        calls["excel"].append((args, kwargs))
        return {"ok": True}

    async def fake_mail(*args, **kwargs):
        calls["mail"].append((args, kwargs))
        return {"ok": True}

    monkeypatch.setattr(mod, "write_excel", fake_excel)
    monkeypatch.setattr(mod, "send_mail_alert", fake_mail)
    return calls


@pytest.mark.asyncio
async def test_on_turn_end_chat_skips_everything(monkeypatch):
    calls = _patch_mcp(monkeypatch)
    r = await on_turn_end("u1", "hi", "CHAT", "正常", 0.0, "正常")
    assert r == {"excel": False, "mail": False}
    assert calls["excel"] == []
    assert calls["mail"] == []


@pytest.mark.asyncio
async def test_on_turn_end_consult_normal_writes_excel_only(monkeypatch):
    calls = _patch_mcp(monkeypatch)
    r = await on_turn_end("u2", "stressed", "CONSULT", "焦虑", 1.8, "需关注")
    assert r == {"excel": True, "mail": False}
    assert len(calls["excel"]) == 1
    assert calls["mail"] == []


@pytest.mark.asyncio
async def test_on_turn_end_consult_high_risk_writes_both(monkeypatch):
    calls = _patch_mcp(monkeypatch)
    r = await on_turn_end("u3", "没意思", "CONSULT", "高风险", 2.5, "高风险")
    assert r == {"excel": True, "mail": True}
    assert len(calls["excel"]) == 1
    assert len(calls["mail"]) == 1


@pytest.mark.asyncio
async def test_on_turn_end_risk_intent_writes_both(monkeypatch):
    """Intent-level RISK (fast-path keyword hit) always alerts, even if
    fused risk band says 需关注."""
    calls = _patch_mcp(monkeypatch)
    r = await on_turn_end("u4", "我想死", "RISK", "高风险", 1.2, "需关注")
    assert r == {"excel": True, "mail": True}


@pytest.mark.asyncio
async def test_on_turn_end_swallows_excel_failure(monkeypatch):
    """A failing excel write must not abort the mail alert — both are tried."""
    from app.agents import orchestrator as mod

    async def boom_excel(*a, **kw):
        raise RuntimeError("disk full")

    mail_calls = []

    async def record_mail(*a, **kw):
        mail_calls.append(a)
        return {"ok": True}

    monkeypatch.setattr(mod, "write_excel", boom_excel)
    monkeypatch.setattr(mod, "send_mail_alert", record_mail)

    r = await on_turn_end("u5", "x", "RISK", "高风险", 2.1, "高风险")
    assert r == {"excel": False, "mail": True}
    assert len(mail_calls) == 1
