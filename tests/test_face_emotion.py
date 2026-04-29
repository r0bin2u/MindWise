import mediapipe as mp
import numpy as np
import pytest

from app.services.face_emotion import _score_label, face_to_emotion


# Newer mediapipe builds (and some Linux/CI wheels) split the Tasks API and no
# longer expose `mp.solutions.face_mesh`. Tests that actually invoke FaceMesh
# need to skip under those builds; the score-band tests below don't need it.
_HAS_FACE_MESH = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def test_score_label_boundaries():
    # face-score → label/risk bands: 0~2 正常 / 2~3 焦虑 / 3~4 低落 / ≥4 高风险
    assert _score_label(0.0) == ("正常", "正常")
    assert _score_label(1.99) == ("正常", "正常")
    assert _score_label(2.0) == ("焦虑", "需关注")
    assert _score_label(2.99) == ("焦虑", "需关注")
    assert _score_label(3.0) == ("低落", "需关注")
    assert _score_label(3.99) == ("低落", "需关注")
    assert _score_label(4.0) == ("高风险", "高风险")
    assert _score_label(5.5) == ("高风险", "高风险")


def test_face_to_emotion_handles_none():
    assert face_to_emotion(None) is None


def test_face_to_emotion_handles_empty():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert face_to_emotion(empty) is None


@pytest.mark.skipif(
    not _HAS_FACE_MESH, reason="mediapipe.solutions.face_mesh unavailable in this build"
)
def test_face_to_emotion_returns_none_on_no_face():
    # pure noise → MediaPipe won't find a face
    noise = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    result = face_to_emotion(noise)
    assert result is None
