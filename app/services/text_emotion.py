from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import settings
from app.services.ollama_client import get_client


LABELS = ["正常", "焦虑", "低落", "高风险"]

# Identical to the instruction used during QLoRA fine-tuning.
# Keep training and inference prompts in lockstep; drift silently kills accuracy.
PROMPT = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险\n\n用户文本：{text}"


def _extract_label(raw: str) -> str | None:
    raw = (raw or "").strip()
    if raw in LABELS:
        return raw
    # prefer the LAST occurrence: verbose models put the verdict at the end
    best_pos, best_lbl = -1, None
    for lbl in LABELS:
        pos = raw.rfind(lbl)
        if pos > best_pos:
            best_pos, best_lbl = pos, lbl
    return best_lbl


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _call_model(text: str) -> str:
    r = await get_client().chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        options={"temperature": 0, "num_predict": 8},
    )
    return r["message"]["content"]


async def text_emotion(text: str) -> str:
    """Classify user text into one of {正常, 焦虑, 低落, 高风险}.

    Returns '正常' on empty input or if every retry fails — this is the
    conservative fallback per doc 13.2 (never raise from a perception module).
    """
    if not text or not text.strip():
        return "正常"
    try:
        raw = await _call_model(text.strip())
    except Exception:
        return "正常"
    return _extract_label(raw) or "正常"
