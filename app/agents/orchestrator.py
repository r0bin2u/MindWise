"""First-layer intent classifier + turn-end dispatcher.

Per doc section 7, after multimodal fusion we ask the same fine-tuned
Qwen to decide whether the turn is:

    CHAT     → just reply, skip RAG, skip Excel, skip alert
    CONSULT  → enter Agentic RAG → Excel log → maybe alert
    RISK     → Fast Path: immediate alert + Excel (doc section 2/7)

Two inference-time safety layers, in order:
  1. explicit RISK keyword scan (doc's "极端关键词 fast path") — short-circuits
     the LLM so an obvious distress phrase can never be misclassified as CHAT.
  2. LLM call with the strict-output prompt from the design image.

Keyword matching here is inference-time only. Doc-labeling pipelines must
NOT reuse this regex to auto-tag training data — we already learned that
lesson the hard way (see TODO_stage1_rework.md).
"""
import re
from typing import Literal

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.services.ollama_client import get_client


Intent = Literal["CHAT", "CONSULT", "RISK"]
INTENTS = {"CHAT", "CONSULT", "RISK"}


# Conservative set: only phrases that are unambiguous distress. We'd rather
# miss a subtle case (LLM still runs) than false-alarm on "我都快累死了".
RISK_KEYWORDS = re.compile(
    r"(想死|想去死|要自杀|去自杀|想自杀|自残|自伤|割腕|跳楼|"
    r"结束生命|活不下去|不想活了|不想活着|一了百了)"
)


# Verbatim from the design image — do not edit without re-evaluating the
# classifier on the test set, prompt changes drift accuracy.
INTENT_PROMPT = """你是一个用户意图分类器，只做意图识别，不回答问题。
用户输入内容：{user_input}

请将用户意图严格分为以下三类之一，只输出标签，不要其他任何内容：
- CHAT：日常闲聊、问候、天气、娱乐、无关内容
- CONSULT：心理咨询、情绪倾诉、压力、焦虑、低落、失眠、亲密关系、学习压力等心理相关
- RISK：自杀、自残、绝望、自伤、伤人、严重抑郁等高危内容"""


def _extract_intent(raw: str) -> Intent | None:
    """Pull a CHAT/CONSULT/RISK token out of the model's reply.

    Prefer the LAST occurrence so a chatty model saying
    '这是 CONSULT 内容,不是 CHAT' still resolves to CONSULT. Also
    case-insensitive in case the model lower-cases it.
    """
    s = (raw or "").strip().upper()
    if s in INTENTS:
        return s  # type: ignore[return-value]
    best_pos, best_lbl = -1, None
    for lbl in INTENTS:
        pos = s.rfind(lbl)
        if pos > best_pos:
            best_pos, best_lbl = pos, lbl
    return best_lbl  # type: ignore[return-value]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _call_model(text: str) -> str:
    r = await get_client().chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": INTENT_PROMPT.format(user_input=text)}],
        options={"temperature": 0, "num_predict": 16},
    )
    return r["message"]["content"]


async def classify_intent(text: str) -> Intent:
    """Classify a user turn into CHAT / CONSULT / RISK.

    Empty input → CHAT (nothing to route).
    LLM failure → CONSULT (conservative: run the full pipeline rather than
    silently dropping a potentially distressed turn).
    """
    if not text or not text.strip():
        return "CHAT"

    t = text.strip()

    # pre-LLM fast path — doc section 7
    if RISK_KEYWORDS.search(t):
        return "RISK"

    try:
        raw = await _call_model(t)
    except Exception:
        return "CONSULT"

    lbl = _extract_intent(raw)
    return lbl if lbl in INTENTS else "CONSULT"


# ---------------------------------------------------------------------------
# turn-end dispatcher — MCP tool calls per doc 9.3 trigger matrix
# ---------------------------------------------------------------------------
#
# intent=CHAT                         → no excel, no mail
# intent=CONSULT + risk in {正常,需关注} → excel only
# intent=CONSULT + risk=高风险          → excel + mail
# intent=RISK                          → excel + mail (fast-path distress)
#
# Called from the chat route as a BackgroundTask so it runs AFTER the
# streamed answer has already been delivered to the user. Any exception
# here is logged-not-raised because failing to write Excel should never
# take down the chat response to the user.

import logging

from app.agents.mcp_client import send_mail_alert, write_excel

log = logging.getLogger("mindwise.orchestrator")


async def on_turn_end(
    user_id: str,
    message: str,
    intent: Intent,
    emotion_label: str,
    score: float,
    risk: str,
) -> dict[str, bool]:
    """Dispatch post-turn side-effects.

    Returns {"excel": bool, "mail": bool} indicating which tools actually
    ran (useful for logging / tests). Exceptions are caught and logged,
    never propagated.
    """
    actions = {"excel": False, "mail": False}

    if intent == "CHAT":
        return actions  # never touch Excel or mail for chat

    # RISK intent is an explicit distress signal (user said something like
    # "我想死") — force-escalate regardless of what the fused risk band
    # claims. Belt-and-suspenders: doc 16 says the caller should already
    # pass risk="高风险" in this case, but don't trust callers silently.
    if intent == "RISK":
        risk = "高风险"
        if emotion_label != "高风险":
            emotion_label = "高风险"

    should_alert = intent == "RISK" or risk == "高风险"

    try:
        await write_excel(user_id, message, emotion_label, score, risk)
        actions["excel"] = True
    except Exception as e:
        log.exception("excel_writer failed: %s", e)

    if should_alert:
        try:
            res = await send_mail_alert(user_id, message, emotion_label, score, risk)
            actions["mail"] = bool(res.get("ok"))
            if not actions["mail"]:
                log.warning("mail_alert skipped: %s", res.get("error"))
        except Exception as e:
            log.exception("mail_alert failed: %s", e)

    return actions
