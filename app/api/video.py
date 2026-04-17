import os
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from app.services.face_emotion import face_to_emotion, face_to_emotion_video


router = APIRouter(prefix="/emotion", tags=["emotion"])

VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def _is_video(filename: str | None, content_type: str | None) -> bool:
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext in VIDEO_EXTS:
            return True
    if content_type and content_type.startswith("video/"):
        return True
    return False


def _fallback(err: str) -> dict:
    return {"label": "正常", "score": 0, "risk": "正常", "error": err}


@router.post("/video")
async def emotion_video(file: UploadFile = File(...)):
    raw = await file.read()
    if _is_video(file.filename, file.content_type):
        suffix = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            r = face_to_emotion_video(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return r if r is not None else _fallback("no_face_or_bad_video")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return _fallback("decode_failed")
    r = face_to_emotion(img)
    return r if r is not None else _fallback("no_face_detected")
