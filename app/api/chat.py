"""Main chat entry point — the /chat endpoint that ties all stages together.

Flow (matches the project-overview diagram exactly):

    user POST /chat
        │
        ▼
    ① text_emotion   → emotion label from the typed text
        │
        ▼
    ② fuse           → vision/audio/text into fused {score, label, risk}
        │
        ▼
    ③ classify_intent → CHAT / CONSULT / RISK
        │
        ├── CHAT    → stream plain friendly reply, NO side effects
        │
        ├── RISK    → background: write excel + mail (fast-path alert)
        │              and stream crisis-comfort reply
        │
        └── CONSULT → stream agentic_rag answer (may retrieve Chroma)
                       then background: write excel (+ mail if fused risk=高风险)

Every branch returns a StreamingResponse with text/event-stream media type
so browsers can consume it with native EventSource. Side effects always
run AFTER the stream closes via FastAPI BackgroundTasks — a failed tool
call never breaks the chat reply.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.orchestrator import classify_intent, on_turn_end
from app.agents.rag_agent import agentic_rag_stream
from app.core.metrics import (
    fused_risk_total,
    intent_total,
    silent_crisis_total,
    stage_latency_seconds,
)
from app.services.fusion import ModalOut, fuse
from app.services.streaming import (
    sse_event,
    sse_frame,
    stream_plain,
    stream_risk_comfort,
)
from app.services.text_emotion import text_emotion


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None
    # already-processed modality outputs from /emotion/audio and /emotion/video
    audio_emotion: Optional[str] = None     # label
    video_emotion: Optional[str] = None     # label
    video_score: Optional[float] = None     # raw MediaPipe score


class ChatResponseMeta(BaseModel):
    """Header sent as the first SSE event so frontends know routing details."""
    session_id: str
    intent: str
    emotion_label: str
    risk: str
    fused_score: float


@router.post("")
async def chat(req: ChatRequest, bg: BackgroundTasks):
    session_id = req.session_id or str(uuid4())

    # ① text emotion on the typed message (quick LLM call, 1 token output)
    with stage_latency_seconds.labels(stage="text_emotion").time():
        t_label = await text_emotion(req.message)

    # ② multimodal fusion — deterministic Python arithmetic.
    # Missing modalities contribute 0 per the conservative policy.
    with stage_latency_seconds.labels(stage="fuse").time():
        fused = await fuse(
            vision=ModalOut(label=req.video_emotion, score=req.video_score),
            audio=ModalOut(label=req.audio_emotion),
            text=ModalOut(label=t_label),
        )

    # ③ first-layer intent routing
    with stage_latency_seconds.labels(stage="classify_intent").time():
        intent = await classify_intent(req.message)

    # business metrics — used by the ops dashboard to spot spikes in
    # high-risk turns or silent-crisis rates
    intent_total.labels(intent=intent).inc()
    fused_risk_total.labels(risk=fused.risk).inc()
    if intent == "CHAT" and fused.risk == "高风险":
        silent_crisis_total.inc()

    # meta frame first so frontend knows which branch we took
    meta = ChatResponseMeta(
        session_id=session_id,
        intent=intent,
        emotion_label=fused.label,
        risk=fused.risk,
        fused_score=fused.score,
    )

    # Schedule the post-turn dispatcher UNCONDITIONALLY. on_turn_end itself
    # decides what to actually do based on intent + fused.risk:
    #   - CHAT + risk=正常/需关注 → no-op
    #   - CHAT + risk=高风险        → excel + mail (silent-crisis case:
    #                                  benign text but multimodal score is high)
    #   - CONSULT + risk=正常/需关注 → excel only
    #   - CONSULT + risk=高风险      → excel + mail
    #   - RISK (any risk)            → excel + mail (force-escalated inside)
    # This lets the orchestrator own the policy — chat.py just hands it
    # the full context of the turn.
    bg.add_task(
        on_turn_end,
        req.user_id, req.message, intent,
        fused.label, fused.score, fused.risk,
    )

    # ------- CHAT: lightweight reply (side-effect dispatch above) -------
    if intent == "CHAT":
        async def chat_gen() -> AsyncIterator[str]:
            yield sse_event("meta", meta.model_dump_json())
            async for frame in stream_plain(req.message):
                yield frame
        return StreamingResponse(chat_gen(), media_type="text/event-stream")

    # ------- RISK: fast-path crisis comfort stream -------
    if intent == "RISK":
        async def risk_gen() -> AsyncIterator[str]:
            yield sse_event("meta", meta.model_dump_json())
            async for frame in stream_risk_comfort(req.message):
                yield frame
        return StreamingResponse(risk_gen(), media_type="text/event-stream")

    # ------- CONSULT: agentic RAG stream -------
    async def consult_gen() -> AsyncIterator[str]:
        yield sse_event("meta", meta.model_dump_json())
        async for token in agentic_rag_stream(req.message):
            yield sse_frame(token)
        yield sse_event("done")

    return StreamingResponse(consult_gen(), media_type="text/event-stream")
