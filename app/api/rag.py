from fastapi import APIRouter
from pydantic import BaseModel

from app.agents.rag_agent import agentic_rag


router = APIRouter(prefix="/rag", tags=["rag"])


class RagRequest(BaseModel):
    query: str


class DocRef(BaseModel):
    source: str
    hit_idx: int


class RagResponse(BaseModel):
    answer: str
    steps: int
    docs: list[DocRef]


@router.post("/consult", response_model=RagResponse)
async def rag_consult(req: RagRequest):
    """Run the Agentic RAG loop on a user query.

    Used by the main chat route after intent is classified as CONSULT.
    Returns the final answer + how many retrieval cycles happened + the
    source chunks cited, so the caller can log provenance.
    """
    result = await agentic_rag(req.query)
    return RagResponse(**result)
