"""High-risk alert email — body template matches doc section 9.5.

Uses aiosmtplib for async TLS send; kept pure (takes SMTP config via
args, falls back to settings) so tests can pass a MagicMock sender
and not hit a real server.
"""
from __future__ import annotations

from datetime import datetime
from email.message import EmailMessage
from typing import Any, Callable, Awaitable

import aiosmtplib

from app.core.config import settings


SUBJECT_TMPL = "【高危心理预警】学生用户 {user_id} 存在自伤风险"

BODY_TMPL = """系统在对话中监测到 1 名学生出现高风险心理状态，请及时关注并干预。

【预警信息如下】
用户ID：{user_id}
对话内容：{content}
情绪判定：{emotion_label}
综合情绪得分：{score}
风险等级：{risk_level}
对话时间：{timestamp}

系统已自动存档对话记录至 Excel，请尽快处理。
"""


def _build_message(
    user_id: str,
    content: str,
    emotion_label: str,
    score: float,
    risk_level: str,
    timestamp: str,
    sender: str,
    recipients: list[str],
) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = SUBJECT_TMPL.format(user_id=user_id)
    msg.set_content(BODY_TMPL.format(
        user_id=user_id,
        content=content,
        emotion_label=emotion_label,
        score=score,
        risk_level=risk_level,
        timestamp=timestamp,
    ))
    return msg


async def send_alert(
    user_id: str,
    content: str,
    emotion_label: str,
    score: float,
    risk_level: str,
    timestamp: str | None = None,
    # dependency injection for tests
    send_fn: Callable[[EmailMessage], Awaitable[Any]] | None = None,
) -> dict[str, Any]:
    """Send the high-risk alert email. Returns a result dict; does not raise
    on transient SMTP errors — caller treats a failed alert as a log event,
    not a chat-break."""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sender = settings.smtp_user or "mindwise@localhost"
    recipients = [r.strip() for r in (settings.alert_to or "").split(",") if r.strip()]

    if not recipients:
        return {"ok": False, "error": "ALERT_TO not configured"}

    msg = _build_message(
        user_id=user_id,
        content=content,
        emotion_label=emotion_label,
        score=score,
        risk_level=risk_level,
        timestamp=ts,
        sender=sender,
        recipients=recipients,
    )

    if send_fn is not None:
        # test path / dependency injection
        await send_fn(msg)
        return {"ok": True, "recipients": recipients, "subject": msg["Subject"]}

    if not settings.smtp_host:
        return {"ok": False, "error": "SMTP_HOST not configured"}

    try:
        await aiosmtplib.send(
            msg,
            hostname=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user or None,
            password=settings.smtp_password or None,
            use_tls=(settings.smtp_port == 465),
            start_tls=(settings.smtp_port in (25, 587)),
        )
    except Exception as e:
        return {"ok": False, "error": f"smtp_failed: {e!s}"[:200]}

    return {"ok": True, "recipients": recipients, "subject": msg["Subject"]}
