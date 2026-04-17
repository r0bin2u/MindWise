import ollama

from app.core.config import settings


LABELS = ["正常", "焦虑", "低落", "高风险"]

PROMPT = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险\n\n用户文本：{text}"

_client = None


def _client_lazy():
    global _client
    if _client is None:
        _client = ollama.AsyncClient(host=settings.ollama_host)
    return _client


async def text_emotion(text: str) -> str:
    if not text or not text.strip():
        return "正常"
    r = await _client_lazy().chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": PROMPT.format(text=text.strip())}],
        options={"temperature": 0, "num_predict": 8},
    )
    out = r["message"]["content"].strip()
    for lbl in LABELS:
        if lbl in out:
            return lbl
    return "正常"
