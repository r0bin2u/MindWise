"""Chat endpoint routing tests — mock every downstream service and verify
the three-branch dispatch (CHAT / CONSULT / RISK) wires the right pieces.

We're not testing the downstream services themselves here (those have
their own tests); we're testing that /chat picks the right stream
function + the right BackgroundTask given each intent.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.services.fusion import FusionResult


client = TestClient(app)


def _patch_all(monkeypatch, *, intent, fused=None):
    """Replace text_emotion / fuse / classify_intent / streams / on_turn_end
    with trackable fakes. Returns a calls dict."""
    from app.api import chat as chat_mod

    calls = {
        "intent": None,
        "fuse": 0,
        "text_emotion": 0,
        "stream_plain": 0,
        "stream_risk": 0,
        "rag_stream": 0,
        "on_turn_end_args": [],
    }

    async def fake_text_emotion(text):
        calls["text_emotion"] += 1
        return "焦虑"

    async def fake_fuse(**kwargs):
        calls["fuse"] += 1
        return fused or FusionResult(score=1.2, label="焦虑", risk="需关注", source="deterministic")

    async def fake_classify(text):
        calls["intent"] = intent
        return intent

    async def fake_plain(msg):
        calls["stream_plain"] += 1
        yield "data: hi\n\n"

    async def fake_risk(msg):
        calls["stream_risk"] += 1
        yield "data: crisis\n\n"

    async def fake_rag(msg):
        calls["rag_stream"] += 1
        yield "tok1"
        yield "tok2"

    async def fake_on_turn_end(*args):
        calls["on_turn_end_args"].append(args)
        return {"excel": True, "mail": False}

    monkeypatch.setattr(chat_mod, "text_emotion", fake_text_emotion)
    monkeypatch.setattr(chat_mod, "fuse", fake_fuse)
    monkeypatch.setattr(chat_mod, "classify_intent", fake_classify)
    monkeypatch.setattr(chat_mod, "stream_plain", fake_plain)
    monkeypatch.setattr(chat_mod, "stream_risk_comfort", fake_risk)
    monkeypatch.setattr(chat_mod, "agentic_rag_stream", fake_rag)
    monkeypatch.setattr(chat_mod, "on_turn_end", fake_on_turn_end)
    return calls


def _body(msg="hi"):
    return {"user_id": "u1", "message": msg}


def test_chat_branch_picks_plain_stream_and_forwards_intent(monkeypatch):
    """Normal CHAT + fused.risk=正常: on_turn_end is still called (policy
    lives inside the dispatcher now), but it'll early-exit internally.
    At the chat.py layer we just need to verify the right stream fired
    and the dispatcher got the correct intent."""
    calls = _patch_all(monkeypatch, intent="CHAT")
    r = client.post("/v1/chat", json=_body())
    assert r.status_code == 200
    body = r.text
    assert "hi" in body
    assert calls["stream_plain"] == 1
    assert calls["stream_risk"] == 0
    assert calls["rag_stream"] == 0
    # dispatcher is ALWAYS called; it decides policy based on (intent, risk)
    assert len(calls["on_turn_end_args"]) == 1
    assert calls["on_turn_end_args"][0][2] == "CHAT"


def test_chat_with_high_risk_fused_triggers_mcp(monkeypatch):
    """Silent-crisis case: user types 'hi' casually but video/audio fusion
    flagged 高风险. intent=CHAT but dispatcher must still receive
    risk='高风险' so the orchestrator fires excel + mail."""
    fused = FusionResult(score=2.5, label="高风险", risk="高风险", source="deterministic")
    calls = _patch_all(monkeypatch, intent="CHAT", fused=fused)
    r = client.post("/v1/chat", json=_body("今天吃什么"))
    assert r.status_code == 200
    # still streams the plain chat reply to the user (don't alarm them)
    assert calls["stream_plain"] == 1
    # but dispatcher gets the full context and the risk=高风险 flag
    assert len(calls["on_turn_end_args"]) == 1
    args = calls["on_turn_end_args"][0]
    assert args[2] == "CHAT"
    assert args[5] == "高风险"


def test_risk_branch_streams_comfort_and_queues_alert(monkeypatch):
    calls = _patch_all(monkeypatch, intent="RISK")
    r = client.post("/v1/chat", json=_body("我不想活了"))
    assert r.status_code == 200
    body = r.text
    assert "crisis" in body
    assert calls["stream_risk"] == 1
    assert calls["stream_plain"] == 0
    assert calls["rag_stream"] == 0
    # BackgroundTask fired with RISK intent — on_turn_end itself force-
    # escalates emotion_label and risk to 高风险 internally.
    assert len(calls["on_turn_end_args"]) == 1
    args = calls["on_turn_end_args"][0]
    assert args[2] == "RISK"


def test_consult_branch_streams_rag_and_logs_with_fused_risk(monkeypatch):
    fused = FusionResult(score=2.3, label="低落", risk="高风险", source="deterministic")
    calls = _patch_all(monkeypatch, intent="CONSULT", fused=fused)
    r = client.post("/v1/chat", json=_body("最近很焦虑"))
    assert r.status_code == 200
    body = r.text
    # both RAG tokens should be in the SSE body
    assert "tok1" in body
    assert "tok2" in body
    assert calls["rag_stream"] == 1
    assert calls["stream_plain"] == 0
    assert calls["stream_risk"] == 0
    # CONSULT branch forwards fused fields as-is to on_turn_end
    assert len(calls["on_turn_end_args"]) == 1
    args = calls["on_turn_end_args"][0]
    assert args[2] == "CONSULT"
    assert args[3] == "低落"
    assert args[4] == 2.3
    assert args[5] == "高风险"


def test_chat_response_emits_meta_event(monkeypatch):
    _patch_all(monkeypatch, intent="CHAT")
    r = client.post("/v1/chat", json=_body())
    assert r.status_code == 200
    body = r.text
    # SSE 'event: meta' line with JSON payload
    assert "event: meta" in body
    assert "CHAT" in body  # intent shows up in meta payload
    assert "session_id" in body
