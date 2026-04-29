"""Multimodal emotion fusion engine.

Original design intent: "let Qwen2.5 do the weighted sum so the rule stays
in natural language." We implemented both paths and empirically found that
7B-class models are **unreliable at multi-step arithmetic** (e.g. it turned
2×0.1 into 0.8 and (3×0.5+2×0.4+2×0.1)=2.5 into 2.3, crossing the 高风险
threshold the wrong way).

Engineering decision: default to **deterministic Python compute**. Keep the
LLM path behind an opt-in flag with a cross-check — if the LLM result
diverges from deterministic by more than 0.3 on score, fall back to the
deterministic result. This preserves the "LLM in the loop" narrative for
future label disambiguation while guaranteeing correctness in production.

Formula:
    final = 0.5 * vision + 0.4 * audio + 0.1 * text
    score mapping: 正常=0, 焦虑=2, 低落=3, 高风险=4

Missing-modality policy is **conservative**: missing channel contributes 0,
weights stay fixed (no normalization). Better to under-estimate from a
lucky channel than to over-escalate and burn counselor trust.
"""
import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.services.ollama_client import get_client


EMOTIONS = {"正常", "焦虑", "低落", "高风险"}
RISKS = {"正常", "需关注", "高风险"}

SCORE_MAP = {"正常": 0, "焦虑": 2, "低落": 3, "高风险": 4}

# fusion weights — vision dominates because facial cues are hardest to fake;
# text is lowest because typed messages are the easiest to consciously mask
W_VISION, W_AUDIO, W_TEXT = 0.5, 0.4, 0.1

# max allowed divergence between LLM score and deterministic score before we
# discard the LLM's opinion and use deterministic. 0.3 is the 1/3 of the
# smallest risk-band width (0→1, 1→2, 2→4).
LLM_SCORE_TOLERANCE = 0.3


class ModalOut(BaseModel):
    """Output of a single modality (text / audio / vision).

    label: one of 4 emotion labels; None when the modality was unavailable
           (no face detected, empty transcript, text-emotion service failed).
    score: optional raw numeric score; today only vision produces one
           (from MediaPipe geometric rules). text/audio pass through the
           SCORE_MAP inside the fusion formula.
    """
    label: Optional[str] = None
    score: Optional[float] = None


class FusionResult(BaseModel):
    score: float = Field(ge=0, le=4)
    label: str
    risk: str
    source: Literal["deterministic", "llm"] = "deterministic"


# ----------------------------------------------------------------------------
# deterministic path (production default)
# ----------------------------------------------------------------------------

def _risk_band(score: float) -> str:
    if score >= 2.0:
        return "高风险"
    if score >= 1.0:
        return "需关注"
    return "正常"


def compute_deterministic(
    vision: ModalOut | None,
    audio: ModalOut | None,
    text: ModalOut | None,
) -> FusionResult:
    def s(m: ModalOut | None) -> float:
        if m is None or not m.label or m.label not in SCORE_MAP:
            return 0.0
        return float(SCORE_MAP[m.label])

    sv, sa, st = s(vision), s(audio), s(text)
    score = W_VISION * sv + W_AUDIO * sa + W_TEXT * st

    # label = modality with the highest weighted score; ties broken by
    # the priority order vision > audio > text (same as weight order)
    candidates = [
        (W_VISION * sv, vision),
        (W_AUDIO * sa, audio),
        (W_TEXT * st, text),
    ]
    candidates.sort(key=lambda x: -x[0])
    top = candidates[0][1]
    label = top.label if (top and top.label) else "正常"

    return FusionResult(
        score=round(score, 2),
        label=label,
        risk=_risk_band(score),
        source="deterministic",
    )


# ----------------------------------------------------------------------------
# LLM path (opt-in, cross-checked against deterministic)
# ----------------------------------------------------------------------------

FUSION_PROMPT = """你是多模态情绪融合引擎。按以下规则计算最终得分和风险等级：

情绪分数映射：正常=0，焦虑=2，低落=3，高风险=4

最终得分 = 视觉情绪分数 × 0.5
        + 语音情绪分数 × 0.4
        + 文本情绪分数 × 0.1

若某一路模态缺失，该模态分数直接记为 0，权重不变，不做归一化。

风险分类规则：
- 最终得分 ≥ 2.0 → 高风险
- 1.0 ≤ 得分 < 2.0 → 需关注
- 得分 < 1.0 → 正常

输入：
- 视觉情绪：{vision_label}
- 语音情绪：{audio_label}
- 文本情绪：{text_label}

情绪标签(label)取三路中分数最大的那一路的标签；若全部缺失输出「正常」。

只输出 JSON，不要解释：
{{"score": 浮点数, "label": "正常|焦虑|低落|高风险", "risk": "正常|需关注|高风险"}}"""


def _render_label(m: ModalOut | None) -> str:
    if m is None or not m.label or m.label not in EMOTIONS:
        return "缺失"
    return m.label


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    retry=retry_if_exception_type((json.JSONDecodeError, ValidationError, KeyError)),
    reraise=True,
)
async def _llm_fuse(vision: ModalOut, audio: ModalOut, text: ModalOut) -> FusionResult:
    prompt = FUSION_PROMPT.format(
        vision_label=_render_label(vision),
        audio_label=_render_label(audio),
        text_label=_render_label(text),
    )
    r = await get_client().chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0},
    )
    obj = json.loads(r["message"]["content"])
    result = FusionResult(source="llm", **obj)
    if result.label not in EMOTIONS or result.risk not in RISKS:
        raise ValidationError.from_exception_data("FusionResult", [])
    return result


# ----------------------------------------------------------------------------
# public entrypoint
# ----------------------------------------------------------------------------

async def fuse(
    vision: ModalOut | None = None,
    audio: ModalOut | None = None,
    text: ModalOut | None = None,
    mode: Literal["deterministic", "llm"] = "deterministic",
) -> FusionResult:
    """Fuse three modalities into a final risk/emotion verdict.

    - mode='deterministic' (default): pure Python, always correct.
    - mode='llm': ask Qwen to do the arithmetic, cross-check against
      deterministic; if LLM score diverges by > LLM_SCORE_TOLERANCE we
      discard it and return the deterministic result.
    """
    v = vision or ModalOut()
    a = audio or ModalOut()
    t = text or ModalOut()

    det = compute_deterministic(v, a, t)
    if mode == "deterministic":
        return det

    try:
        llm = await _llm_fuse(v, a, t)
    except Exception:
        return det

    if abs(llm.score - det.score) > LLM_SCORE_TOLERANCE:
        # LLM arithmetic drifted — trust the deterministic answer instead
        return det
    if llm.risk != _risk_band(llm.score):
        # LLM output is internally inconsistent (score says one band, risk
        # field says another). Seen in practice: score=2.3 but risk=需关注.
        return det
    return llm
