"""Agentic RAG via LangGraph StateGraph.

The agent emits {think, action, query} JSON each turn — `think` is the
reasoning trace, `action` is RETRIEVE or ANSWER, `query` is the search
key when retrieving.

Graph shape:
    think ── route ──► retrieve ──► think
                   │
                   └──► answer ──► END

Loop invariants:
  - think is the only node that calls the LLM for the JSON decision
  - retrieve is deterministic (Chroma + neighbor splicing)
  - answer generates the final user-facing reply with accumulated context
  - max_steps=4 is a hard cap on retrieve cycles; prevents runaway loops

Why LangGraph here: a while-loop would hide the state transitions in
local variables; LangGraph makes the graph explicit and easy to extend
(add a new node + edge = one-line change) and trace.
"""
from __future__ import annotations

import json
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from app.core.config import settings
from app.services.ollama_client import get_client
from app.services.retrieval import format_passages, retrieve


MAX_STEPS = 4


class AgentStep(BaseModel):
    think: str
    action: Literal["RETRIEVE", "ANSWER"]
    query: str = ""


# Locked prompt — Qwen must reply with strict JSON; we enforce via
# `format="json"` and pydantic-validate the output.
AGENT_PROMPT = """你是智心AI心理咨询助手，必须严格按照以下步骤执行多步推理：

步骤1：理解用户问题与情绪状态
步骤2：判断是否需要查询心理知识库（Chroma）
   - 需要：返回 action = "RETRIEVE"
   - 不需要：返回 action = "ANSWER"
步骤3：如果需要检索，生成精准的检索关键词
步骤4：如果问题复杂，支持分步骤、多轮检索
步骤5：结合知识库内容，生成专业、温和、安全的回答

【输出格式：严格 JSON，不要其他任何内容】
{"think": "你的思考过程", "action": "RETRIEVE | ANSWER", "query": "检索关键词（不需要则为空）"}"""


# Used by answer_node to turn the accumulated state into a natural-language
# reply. Kept separate from AGENT_PROMPT so the decision step and the
# answer step don't collide in format expectations.
ANSWER_PROMPT_WITH_DOCS = """基于用户问题和检索到的心理学资料，生成一段温和、专业的心理咨询回答。

要求：
- 先共情用户的感受，再给出建议
- 用贴近学生的口语，不要医学术语
- 回答控制在 200 字以内
- 如果资料中的信息相关，请引用其中的具体建议

用户问题：{user_q}

检索到的资料：
{docs}

请直接输出自然语言回答。"""


ANSWER_PROMPT_NO_DOCS = """请以温和、共情的校园心理咨询助手身份，回答用户的问题。

要求：
- 如果是闲聊或问候，自然简短地回应
- 如果是情绪倾诉但不需要专业知识，给出共情+简单支持
- 语气贴近学生，不要医学术语，不要元描述（不要说"这是问候"这种话）
- 控制在 150 字以内

用户问题：{user_q}

请直接输出自然语言回答。"""


class AgentState(TypedDict):
    user_q: str
    history: list[dict]
    docs: list[dict]  # retrieved passages across all steps
    step: int          # number of completed retrieve cycles
    last_action: Optional[str]
    last_query: Optional[str]
    last_think: Optional[str]
    final_answer: Optional[str]


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------

async def think_node(state: AgentState) -> AgentState:
    """Ask the LLM for the next {think, action, query} decision."""
    resp = await get_client().chat(
        model=settings.ollama_model,
        messages=state["history"],
        format="json",
        options={"temperature": 0, "num_predict": 256},
    )
    raw = resp["message"]["content"]
    try:
        parsed = AgentStep.model_validate_json(raw)
    except (ValidationError, ValueError, json.JSONDecodeError):
        # malformed output → force ANSWER so the loop terminates cleanly
        parsed = AgentStep(
            think="模型输出不可解析，终止推理以免死循环。",
            action="ANSWER",
            query="",
        )

    state["history"].append({"role": "assistant", "content": raw})
    state["last_action"] = parsed.action
    state["last_query"] = parsed.query
    state["last_think"] = parsed.think
    return state


async def retrieve_node(state: AgentState) -> AgentState:
    """Hit Chroma with last_query (fallback to user_q); append docs to state."""
    q = (state["last_query"] or "").strip() or state["user_q"]
    passages = retrieve(q, k=3, neighbors=1)
    state["docs"].extend(passages)

    state["history"].append({
        "role": "user",
        "content": f"第 {state['step'] + 1} 轮检索结果：\n{format_passages(passages)}\n\n请继续推理，决定下一步是再次检索还是给出最终答案。",
    })
    state["step"] += 1
    return state


async def answer_node(state: AgentState) -> AgentState:
    """Produce the final natural-language reply.

    Whether or not docs were retrieved, we always make a dedicated
    generation call — otherwise the user would see the model's
    meta-reasoning ("用户情绪低落, 涉及人际冲突...") as the answer,
    which is terrible UX. Docs (if any) are appended to the prompt;
    otherwise we use the no-docs variant.
    """
    if state["docs"]:
        prompt = ANSWER_PROMPT_WITH_DOCS.format(
            user_q=state["user_q"],
            docs=format_passages(state["docs"][:4]),  # cap to avoid prompt bloat
        )
    else:
        prompt = ANSWER_PROMPT_NO_DOCS.format(user_q=state["user_q"])

    try:
        resp = await get_client().chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 512},
        )
        state["final_answer"] = resp["message"]["content"].strip()
    except Exception:
        # LLM failure in the final generation step — conservative fallback.
        # Don't use last_thought here, that's model meta-reasoning, not a reply.
        state["final_answer"] = "我在这里，请再告诉我一些你的情况，我们慢慢聊。"
    return state


def route_after_think(state: AgentState) -> str:
    """Decide whether to retrieve more or wrap up."""
    if state["step"] >= MAX_STEPS:
        return "answer"
    if state["last_action"] == "ANSWER":
        return "answer"
    return "retrieve"


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("think", think_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("answer", answer_node)

    g.set_entry_point("think")
    g.add_conditional_edges(
        "think", route_after_think,
        {"retrieve": "retrieve", "answer": "answer"},
    )
    g.add_edge("retrieve", "think")
    g.add_edge("answer", END)
    return g.compile()


_app = None


def _get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


async def agentic_rag(user_q: str) -> dict:
    """Run the state machine on a user query.

    Returns {"answer": str, "steps": int, "docs": list[dict]} so the caller
    can log / stream / show sources.
    """
    initial: AgentState = {
        "user_q": user_q,
        "history": [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": user_q},
        ],
        "docs": [],
        "step": 0,
        "last_action": None,
        "last_query": None,
        "last_think": None,
        "final_answer": None,
    }
    final = await _get_app().ainvoke(initial)
    return {
        "answer": final["final_answer"] or "暂时无法给出答案，请换种表达。",
        "steps": final["step"],
        "docs": [{"source": d["source"], "hit_idx": d["hit_idx"]} for d in final["docs"]],
    }


# ---------------------------------------------------------------------------
# streaming variant for the /chat endpoint
# ---------------------------------------------------------------------------

async def agentic_rag_stream(user_q: str):
    """Streaming variant of agentic_rag.

    The think/retrieve cycles run to completion as normal (they're short
    LLM calls with JSON outputs, not worth streaming token-by-token), but
    the final answer generation is streamed token-by-token so the user
    sees typewriter output.

    Yields plain text tokens. The chat route wraps them into SSE frames.
    """
    history: list[dict] = [
        {"role": "system", "content": AGENT_PROMPT},
        {"role": "user", "content": user_q},
    ]
    docs: list[dict] = []
    step = 0
    last_action: Optional[str] = None
    last_query: Optional[str] = None

    # ---- think/retrieve cycles (unchanged logic, just inline) ----
    while step < MAX_STEPS:
        resp = await get_client().chat(
            model=settings.ollama_model,
            messages=history,
            format="json",
            options={"temperature": 0, "num_predict": 256},
        )
        raw = resp["message"]["content"]
        try:
            parsed = AgentStep.model_validate_json(raw)
        except (ValidationError, ValueError, json.JSONDecodeError):
            parsed = AgentStep(
                think="模型输出不可解析，直接给出最终回答。",
                action="ANSWER",
                query="",
            )
        history.append({"role": "assistant", "content": raw})
        last_action = parsed.action
        last_query = parsed.query

        if parsed.action == "ANSWER":
            break

        passages = retrieve(parsed.query or user_q, k=3, neighbors=1)
        docs.extend(passages)
        history.append({
            "role": "user",
            "content": f"第 {step + 1} 轮检索结果：\n{format_passages(passages)}\n\n请继续推理，决定下一步是再次检索还是给出最终答案。",
        })
        step += 1

    # ---- stream the final answer generation ----
    if docs:
        prompt = ANSWER_PROMPT_WITH_DOCS.format(
            user_q=user_q,
            docs=format_passages(docs[:4]),
        )
    else:
        prompt = ANSWER_PROMPT_NO_DOCS.format(user_q=user_q)

    try:
        stream = await get_client().chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": 0.3, "num_predict": 512},
        )
        async for chunk in stream:
            token = chunk.get("message", {}).get("content") or ""
            if token:
                yield token
    except Exception:
        yield "我在这里，请再告诉我一些你的情况，我们慢慢聊。"
