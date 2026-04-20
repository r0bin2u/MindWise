from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.orchestrator import classify_intent


router = APIRouter(prefix="/intent", tags=["intent"])


class IntentRequest(BaseModel):
    text: str


class IntentResponse(BaseModel):
    intent: str


@router.post("", response_model=IntentResponse)
async def classify(req: IntentRequest):
    """Classify a user text as CHAT / CONSULT / RISK.

    Used by the front-end or the main /chat route's first layer.
    """
    intent = await classify_intent(req.text)
    return IntentResponse(intent=intent)
