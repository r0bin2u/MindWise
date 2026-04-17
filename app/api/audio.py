import os
import tempfile

from fastapi import APIRouter, File, UploadFile

from app.services.audio_emotion import audio_to_emotion


router = APIRouter(prefix="/emotion", tags=["emotion"])


@router.post("/audio")
async def emotion_audio(audio: UploadFile = File(...), language: str = "zh"):
    suffix = os.path.splitext(audio.filename or "clip.mp3")[1] or ".mp3"
    data = await audio.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return await audio_to_emotion(path, language=language)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
