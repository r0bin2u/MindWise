import pytest

from mcp_server.tools.mail_alert import _build_message, send_alert


def test_build_message_subject_and_body():
    msg = _build_message(
        user_id="S2025007",
        content="我觉得活着没什么意思，每天都很痛苦",
        emotion_label="高风险",
        score=2.93,
        risk_level="高风险",
        timestamp="2026-03-13 22:01:10",
        sender="ops@school.edu",
        recipients=["counselor@school.edu"],
    )
    assert "【高危心理预警】" in msg["Subject"]
    assert "S2025007" in msg["Subject"]

    body = msg.get_content()
    # all 6 required fields must be in the body
    for token in [
        "S2025007",
        "活着没什么意思",
        "高风险",
        "2.93",
        "2026-03-13 22:01:10",
        "自动存档",
    ]:
        assert token in body

    assert msg["From"] == "ops@school.edu"
    assert msg["To"] == "counselor@school.edu"


@pytest.mark.asyncio
async def test_send_alert_skips_when_no_recipients(monkeypatch):
    from mcp_server.tools import mail_alert as mod

    monkeypatch.setattr(mod.settings, "alert_to", "")

    r = await send_alert("U1", "x", "高风险", 2.1, "高风险")
    assert r["ok"] is False
    assert "ALERT_TO" in r["error"]


@pytest.mark.asyncio
async def test_send_alert_uses_injected_sender(monkeypatch):
    """send_fn dependency injection bypasses SMTP — pure unit test."""
    from mcp_server.tools import mail_alert as mod

    monkeypatch.setattr(mod.settings, "alert_to", "a@x,b@x")

    captured = {}

    async def fake_send(msg):
        captured["to"] = msg["To"]
        captured["subject"] = msg["Subject"]
        captured["body"] = msg.get_content()

    r = await send_alert(
        user_id="U9",
        content="不想活了",
        emotion_label="高风险",
        score=3.5,
        risk_level="高风险",
        timestamp="2026-04-21 10:00:00",
        send_fn=fake_send,
    )
    assert r["ok"] is True
    assert captured["to"] == "a@x, b@x"
    assert "U9" in captured["subject"]
    assert "不想活了" in captured["body"]


@pytest.mark.asyncio
async def test_send_alert_reports_missing_smtp_host(monkeypatch):
    """No send_fn, no SMTP_HOST → graceful error, no exception."""
    from mcp_server.tools import mail_alert as mod

    monkeypatch.setattr(mod.settings, "alert_to", "c@x")
    monkeypatch.setattr(mod.settings, "smtp_host", "")

    r = await send_alert("U1", "x", "高风险", 2.1, "高风险")
    assert r["ok"] is False
    assert "SMTP_HOST" in r["error"]
