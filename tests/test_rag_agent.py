import json

import pytest

from app.agents.rag_agent import (
    MAX_STEPS,
    AgentStep,
    route_after_think,
)


# ---------------- AgentStep schema ----------------

def test_agent_step_parses_valid_json():
    raw = json.dumps({"thought": "分析中", "action": "RETRIEVE", "query": "失眠"})
    s = AgentStep.model_validate_json(raw)
    assert s.action == "RETRIEVE"
    assert s.query == "失眠"


def test_agent_step_answer_no_query():
    raw = json.dumps({"thought": "直接回答", "action": "ANSWER", "query": ""})
    s = AgentStep.model_validate_json(raw)
    assert s.action == "ANSWER"


def test_agent_step_rejects_invalid_action():
    raw = json.dumps({"thought": "x", "action": "FOO", "query": ""})
    with pytest.raises(Exception):
        AgentStep.model_validate_json(raw)


# ---------------- route_after_think ----------------

def _state(step=0, action="RETRIEVE"):
    return {
        "user_q": "x",
        "history": [],
        "docs": [],
        "step": step,
        "last_action": action,
        "last_query": None,
        "last_thought": None,
        "final_answer": None,
    }


def test_route_answer_when_llm_says_answer():
    assert route_after_think(_state(action="ANSWER")) == "answer"


def test_route_retrieve_when_llm_says_retrieve():
    assert route_after_think(_state(action="RETRIEVE")) == "retrieve"


def test_route_answer_on_max_steps_cap():
    # even if LLM wants to keep retrieving, hard stop
    assert route_after_think(_state(step=MAX_STEPS, action="RETRIEVE")) == "answer"


# ---------------- full graph with mocked LLM + retrieval ----------------

@pytest.mark.asyncio
async def test_rag_answer_path_no_retrieval(monkeypatch):
    """Model says ANSWER on first turn → skip retrieve, generate answer."""
    from app.agents import rag_agent as mod

    turns = iter([
        json.dumps({"thought": "闲聊,不需要查库", "action": "ANSWER", "query": ""}),
        "嗨，今天怎么样？",  # the answer_node generation call
    ])

    class FakeClient:
        async def chat(self, **kwargs):
            return {"message": {"content": next(turns)}}

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient())

    result = await mod.agentic_rag("你好")
    assert result["steps"] == 0
    assert result["docs"] == []
    assert "嗨" in result["answer"]


@pytest.mark.asyncio
async def test_rag_retrieve_then_answer(monkeypatch):
    """Model says RETRIEVE, then ANSWER, then generate answer."""
    from app.agents import rag_agent as mod

    turns = iter([
        json.dumps({"thought": "先查库", "action": "RETRIEVE", "query": "失眠"}),
        json.dumps({"thought": "资料够了", "action": "ANSWER", "query": ""}),
        "试试 4-7-8 呼吸法。",
    ])

    class FakeClient:
        async def chat(self, **kwargs):
            return {"message": {"content": next(turns)}}

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient())
    monkeypatch.setattr(mod, "retrieve",
                        lambda q, k=3, neighbors=1: [
                            {"text": "呼吸法介绍", "source": "insomnia.md", "hit_idx": 0}
                        ])

    result = await mod.agentic_rag("我失眠怎么办")
    assert result["steps"] == 1
    assert len(result["docs"]) == 1
    assert "4-7-8" in result["answer"]


@pytest.mark.asyncio
async def test_rag_max_steps_cap(monkeypatch):
    """Model keeps saying RETRIEVE — hard cap kicks in after MAX_STEPS.

    Trace: think→retrieve (step=1) → think→retrieve (step=2) → ... →
    think→retrieve (step=MAX_STEPS) → think → route sees step>=MAX_STEPS
    → answer. That's (MAX_STEPS + 1) think calls before the answer call.
    """
    from app.agents import rag_agent as mod

    def think_retrieve():
        return {"message": {"content": json.dumps(
            {"thought": "再查一次", "action": "RETRIEVE", "query": "x"}
        )}}

    responses = iter(
        [think_retrieve() for _ in range(MAX_STEPS + 1)]
        + [{"message": {"content": "fallback reply"}}]  # answer_node call
    )

    class FakeClient:
        async def chat(self, **kwargs):
            return next(responses)

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient())
    monkeypatch.setattr(mod, "retrieve",
                        lambda q, k=3, neighbors=1: [
                            {"text": "x", "source": "a.md", "hit_idx": 0}
                        ])

    result = await mod.agentic_rag("复杂问题")
    assert result["steps"] == MAX_STEPS
    assert result["answer"] == "fallback reply"


@pytest.mark.asyncio
async def test_rag_unparseable_llm_output_terminates_cleanly(monkeypatch):
    """Bad JSON from LLM → parse failure forces ANSWER, graceful exit."""
    from app.agents import rag_agent as mod

    turns = iter([
        "not json at all 💥",  # think_node parse fails → forced ANSWER
        "兜底答复",              # answer_node
    ])

    class FakeClient:
        async def chat(self, **kwargs):
            return {"message": {"content": next(turns)}}

    monkeypatch.setattr(mod, "get_client", lambda: FakeClient())

    result = await mod.agentic_rag("hi")
    assert result["steps"] == 0
    assert result["answer"] == "兜底答复"
