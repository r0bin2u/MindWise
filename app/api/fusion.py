from fastapi import APIRouter

from app.services.fusion import FusionResult, ModalOut, fuse


router = APIRouter(prefix="/emotion", tags=["emotion"])


@router.post("/fuse", response_model=FusionResult)
async def emotion_fuse(
    vision: ModalOut | None = None,
    audio: ModalOut | None = None,
    text: ModalOut | None = None,
):
    """Fuse three modalities into a final risk/emotion verdict.

    Typical call flow from the frontend:
      1. POST /emotion/audio  -> {text, label}
      2. POST /emotion/video  -> {label, score, risk}
      3. POST /emotion/fuse   with the labels above + the user text's label
    """
    return await fuse(vision=vision, audio=audio, text=text)
