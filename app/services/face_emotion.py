import cv2
import mediapipe as mp
import numpy as np


# FaceMesh landmark indices (478 points with refine_landmarks=True)
# picked by eyeballing the MediaPipe diagram, not tuned
L_EYE_OUTER = 33
R_EYE_OUTER = 263

# eyebrow inner / outer / mid
L_BROW_INNER, L_BROW_OUTER = 107, 70
R_BROW_INNER, R_BROW_OUTER = 336, 300

# mouth corners, upper/lower lip center
MOUTH_L, MOUTH_R = 61, 291
LIP_UP, LIP_DOWN = 0, 17

# eye upper/lower
R_EYE_UP, R_EYE_DOWN = 159, 145
L_EYE_UP, L_EYE_DOWN = 386, 374

# cheek + lip keypoint bundle for tension dispersion
CHEEK_LIP = [
    # left cheek
    50, 101, 205, 207, 214,
    # right cheek
    280, 330, 425, 427, 434,
    # upper & lower lip contour
    61, 84, 17, 314, 291, 0, 13, 14, 81, 178,
]


_mesh = None


def _mesh_lazy():
    global _mesh
    if _mesh is None:
        _mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
    return _mesh


def _d(lm, a, b):
    pa = np.array([lm[a].x, lm[a].y])
    pb = np.array([lm[b].x, lm[b].y])
    return float(np.linalg.norm(pa - pb))


def _score_label(score: float):
    # doc maps: 0~2 normal, 2~3 anxious, 3~4 down, >=4 high-risk
    if score >= 4:
        return "高风险", "高风险"
    if score >= 3:
        return "低落", "需关注"
    if score >= 2:
        return "焦虑", "需关注"
    return "正常", "正常"


def face_to_emotion(bgr_image: np.ndarray) -> dict | None:
    if bgr_image is None or bgr_image.size == 0:
        return None
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    res = _mesh_lazy().process(rgb)
    if not res.multi_face_landmarks:
        return None

    lm = res.multi_face_landmarks[0].landmark
    face_w = _d(lm, L_EYE_OUTER, R_EYE_OUTER)
    if face_w < 1e-6:
        return None

    # eyebrow: inner end dropping below outer end = frown
    brow_drop = (
        (lm[L_BROW_INNER].y - lm[L_BROW_OUTER].y)
        + (lm[R_BROW_INNER].y - lm[R_BROW_OUTER].y)
    ) / 2 / face_w

    # mouth corners vs mouth midpoint (positive = drooping)
    mouth_mid_y = (lm[LIP_UP].y + lm[LIP_DOWN].y) / 2
    mouth_drop = (
        (lm[MOUTH_L].y - mouth_mid_y) + (lm[MOUTH_R].y - mouth_mid_y)
    ) / 2 / face_w

    # eye opening: smaller = more closed
    eye_open = (_d(lm, R_EYE_UP, R_EYE_DOWN) + _d(lm, L_EYE_UP, L_EYE_DOWN)) / 2 / face_w

    # cheek + lip tension: std of displacement vectors from the cluster
    # centroid (scale-invariant via face_w). muscles pulling unevenly
    # → higher std
    pts = np.array([[lm[i].x, lm[i].y] for i in CHEEK_LIP], dtype=float)
    pts -= pts.mean(axis=0)
    cheek_tension = float(np.linalg.norm(pts, axis=1).std()) / face_w

    score = 0.0
    if brow_drop > 0.02:
        score += 1.5
    if mouth_drop > 0.05:
        score += 1.0
    if eye_open < 0.04:
        score += 1.0
    # threshold is a rough starting point; needs calibration on real samples
    if cheek_tension > 0.25:
        score += 1.5

    label, risk = _score_label(score)

    return {
        "label": label,
        "score": round(score, 2),
        "risk": risk,
        "features": {
            "brow_drop": round(brow_drop, 4),
            "mouth_drop": round(mouth_drop, 4),
            "eye_open": round(eye_open, 4),
            "cheek_tension": round(cheek_tension, 4),
        },
    }


def face_to_emotion_video(video_path: str, sample_fps: float = 1.0) -> dict | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / max(sample_fps, 0.1))))

    scores = []
    labels = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            r = face_to_emotion(frame)
            if r is not None:
                scores.append(r["score"])
                labels.append(r["label"])
        idx += 1
    cap.release()

    if not scores:
        return None

    # aggregate: mean damps single-frame noise; peak surfaces worst moment
    mean_score = sum(scores) / len(scores)
    peak_score = max(scores)
    # escalate on single-frame 高风险 but otherwise use mean
    final = peak_score if peak_score >= 4 else mean_score
    label, risk = _score_label(final)

    return {
        "label": label,
        "score": round(final, 2),
        "risk": risk,
        "frames_sampled": len(scores),
        "mean_score": round(mean_score, 2),
        "peak_score": round(peak_score, 2),
    }
