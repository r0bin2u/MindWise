import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from app.services.face_emotion import face_to_emotion


router = APIRouter(prefix="/emotion", tags=["emotion"])


@router.post("/video")
async def emotion_video(image: UploadFile = File(...)):
    raw = await image.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"label": "正常", "score": 0, "risk": "正常", "error": "decode_failed"}
    r = face_to_emotion(img)
    if r is None:
        return {"label": "正常", "score": 0, "risk": "正常", "error": "no_face_detected"}
    return r
