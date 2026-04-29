import pytest

from app.services.fusion import (
    FusionResult,
    ModalOut,
    _render_label,
    compute_deterministic,
    fuse,
)


# ---------------- compute_deterministic ----------------


def test_compute_three_normal():
    r = compute_deterministic(
        ModalOut(label="正常"), ModalOut(label="正常"), ModalOut(label="正常")
    )
    assert r.score == 0
    assert r.risk == "正常"
    assert r.label == "正常"
    assert r.source == "deterministic"


def test_compute_canonical_example():
    # vision 低落(3)×0.5 + audio 焦虑(2)×0.4 + text 正常(0)×0.1 = 2.3
    r = compute_deterministic(
        ModalOut(label="低落"),
        ModalOut(label="焦虑"),
        ModalOut(label="正常"),
    )
    assert r.score == pytest.approx(2.3, abs=0.01)
    assert r.risk == "高风险"
    assert r.label == "低落"  # vision dominates (highest weighted score)


def test_compute_missing_modalities_stay_conservative():
    # only vision 低落(3) × 0.5 = 1.5 → 需关注 (NOT high-risk)
    r = compute_deterministic(ModalOut(label="低落"), ModalOut(), ModalOut())
    assert r.score == pytest.approx(1.5, abs=0.01)
    assert r.risk == "需关注"


def test_compute_all_missing():
    r = compute_deterministic(ModalOut(), ModalOut(), ModalOut())
    assert r.score == 0
    assert r.risk == "正常"
    assert r.label == "正常"


def test_compute_text_only_high_risk_below_threshold():
    # text 高风险(4)×0.1 = 0.4 → 正常, because text weight is small.
    r = compute_deterministic(ModalOut(), ModalOut(), ModalOut(label="高风险"))
    assert r.score == pytest.approx(0.4, abs=0.01)
    assert r.risk == "正常"


def test_compute_label_picks_dominant_modality():
    # vision 焦虑(2)×0.5=1.0, audio 高风险(4)×0.4=1.6, text 正常(0)×0.1=0
    # audio wins → label=高风险
    r = compute_deterministic(
        ModalOut(label="焦虑"),
        ModalOut(label="高风险"),
        ModalOut(label="正常"),
    )
    assert r.label == "高风险"
    assert r.score == pytest.approx(2.6, abs=0.01)
    assert r.risk == "高风险"


# ---------------- _render_label ----------------


def test_render_label_missing():
    assert _render_label(None) == "缺失"
    assert _render_label(ModalOut()) == "缺失"
    assert _render_label(ModalOut(label="")) == "缺失"


def test_render_label_valid():
    assert _render_label(ModalOut(label="焦虑")) == "焦虑"


def test_render_label_garbage_falls_back():
    assert _render_label(ModalOut(label="unknown")) == "缺失"


# ---------------- fuse() ----------------


@pytest.mark.asyncio
async def test_fuse_default_is_deterministic():
    r = await fuse(
        vision=ModalOut(label="低落"),
        audio=ModalOut(label="焦虑"),
        text=ModalOut(label="正常"),
    )
    assert isinstance(r, FusionResult)
    assert r.source == "deterministic"
    assert r.score == pytest.approx(2.3, abs=0.01)
    assert r.risk == "高风险"


@pytest.mark.asyncio
async def test_fuse_llm_mode_falls_back_on_exception(monkeypatch):
    from app.services import fusion as fusion_mod

    async def boom(*a, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(fusion_mod, "_llm_fuse", boom)

    r = await fuse(
        vision=ModalOut(label="低落"),
        audio=ModalOut(label="焦虑"),
        text=ModalOut(label="正常"),
        mode="llm",
    )
    assert r.source == "deterministic"
    assert r.score == pytest.approx(2.3, abs=0.01)


@pytest.mark.asyncio
async def test_fuse_llm_mode_overrides_bad_arithmetic(monkeypatch):
    """If LLM score diverges > tolerance from deterministic, discard LLM."""
    from app.services import fusion as fusion_mod

    async def wrong_math(*a, **kw):
        # LLM hallucinates 0.8 when correct answer is 0.2
        return FusionResult(score=0.8, label="焦虑", risk="正常", source="llm")

    monkeypatch.setattr(fusion_mod, "_llm_fuse", wrong_math)

    r = await fuse(text=ModalOut(label="焦虑"), mode="llm")
    assert r.source == "deterministic"
    assert r.score == pytest.approx(0.2, abs=0.01)


@pytest.mark.asyncio
async def test_fuse_llm_mode_accepts_close_result(monkeypatch):
    """If LLM score is within tolerance AND internally consistent, use LLM."""
    from app.services import fusion as fusion_mod

    async def close_math(*a, **kw):
        # deterministic would produce 2.3; LLM returns 2.2 (within 0.3 tolerance)
        # and risk band matches score (2.2 >= 2.0 → 高风险)
        return FusionResult(score=2.2, label="低落", risk="高风险", source="llm")

    monkeypatch.setattr(fusion_mod, "_llm_fuse", close_math)

    r = await fuse(
        vision=ModalOut(label="低落"),
        audio=ModalOut(label="焦虑"),
        text=ModalOut(label="正常"),
        mode="llm",
    )
    assert r.source == "llm"
    assert r.score == pytest.approx(2.2, abs=0.01)


@pytest.mark.asyncio
async def test_fuse_llm_mode_rejects_internal_inconsistency(monkeypatch):
    """LLM self-contradictions (score in a band, risk in another) → reject."""
    from app.services import fusion as fusion_mod

    async def inconsistent(*a, **kw):
        # score 2.3 but risk '需关注' — inconsistent (2.3 >= 2.0 → 高风险)
        return FusionResult(score=2.3, label="焦虑", risk="需关注", source="llm")

    monkeypatch.setattr(fusion_mod, "_llm_fuse", inconsistent)

    r = await fuse(
        vision=ModalOut(label="低落"),
        audio=ModalOut(label="焦虑"),
        text=ModalOut(label="正常"),
        mode="llm",
    )
    assert r.source == "deterministic"
    assert r.risk == "高风险"
