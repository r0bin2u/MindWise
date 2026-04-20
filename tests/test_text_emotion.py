import pytest

from app.services.text_emotion import _extract_label, text_emotion


def test_extract_exact_label():
    for lbl in ["正常", "焦虑", "低落", "高风险"]:
        assert _extract_label(lbl) == lbl


def test_extract_from_verbose_output():
    assert _extract_label("根据文本判断，这是焦虑情绪") == "焦虑"
    assert _extract_label("标签：高风险。") == "高风险"


def test_extract_picks_last_occurrence():
    # model may mention '正常' in reasoning then conclude '焦虑'
    raw = "这不是正常情况，我判断为焦虑"
    assert _extract_label(raw) == "焦虑"


def test_extract_none_on_unknown():
    assert _extract_label("I don't know") is None
    assert _extract_label("") is None


@pytest.mark.asyncio
async def test_text_emotion_empty_short_circuit():
    # should not call the model for empty input
    assert await text_emotion("") == "正常"
    assert await text_emotion("   ") == "正常"


@pytest.mark.asyncio
async def test_text_emotion_fallback_on_failure(monkeypatch):
    from app.services import text_emotion as mod

    async def boom(text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(mod, "_call_model", boom)
    # even after all retries fail, we return the conservative fallback
    assert await text_emotion("压力好大") == "正常"
