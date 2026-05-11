# MindWise

[![CI](https://github.com/r0bin2u/MindWise/actions/workflows/ci.yml/badge.svg)](https://github.com/r0bin2u/MindWise/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-mind--wise.vercel.app-success)](https://mind-wise.vercel.app/)
[![Vercel](https://img.shields.io/badge/vercel-deployed-black?logo=vercel)](https://mind-wise.vercel.app/)

> A campus-oriented mental-health AI assistant: multimodal emotion
> sensing, Agentic RAG, and MCP tool calls for crisis logging and
> alerts. Fully local inference, Apache-2.0, zero API spend.

Counselor-to-student ratios are stretched thin and many students avoid
face-to-face counseling for privacy reasons. MindWise sits in front of
human counselors as a triage and early-intervention layer: students
vent anonymously over text/audio/video, the system reads emotional
state across modalities, drives an agentic conversation, and quietly
escalates high-risk cases to staff via Excel + email through MCP tool
calls.

> **Status:** personal portfolio project. The README focuses on
> engineering judgement (the *why* behind each decision), not feature
> surface area — for setup beyond `docker compose up` see the source.

---

## Try it

- **Live demo:** <https://mind-wise.vercel.app/> — the public build
  has no backend attached, so it auto-falls back to **mock mode**:
  scripted SSE streams that mirror the real fusion engine's
  text-only output (`fused_score ≤ 0.4`, `risk_band` stays in the
  Normal band), with the crisis dialog firing on
  `intent === 'RISK'`. Try:
    - `"Hi, how are you?"` → CHAT path
    - `"I've been so anxious about exams"` → CONSULT path
    - `"I want to disappear"` → RISK path + crisis dialog
- **Reply language note:** the AI replies are in Chinese because
  the backend is fine-tuned on a Chinese psychology corpus. The UI
  shell itself is bilingual — use the language toggle in the
  header to switch labels between EN and ZH.
- **Run locally for the real Qwen backend:** see
  [Quick Start](#quick-start) below.
- **Force mock mode locally:** append `?mock=1` to the URL.
- **Force real backend on the public URL** (requires the backend
  to be reachable from the deployed origin): append `?mock=0`.

---

## Highlights

- **4-label emotion classifier** fine-tuned (LoRA) on a curated
  Chinese psychology corpus (open-source PsychQA + Mistral-Large
  synthetic data). Target: high-risk recall ≥ 95% with macro F1
  on a held-out split.
- **Deterministic multimodal fusion** (`vision 0.5 / audio 0.4 /
  text 0.1`) computed in pure Python, with an optional LLM
  cross-check that falls back to the deterministic answer when
  the two disagree by more than ±0.3.
- **Two-signal crisis dispatch** — Layer-1 intent catches explicit
  self-harm language, the multimodal fusion engine catches silent
  crises ("I'm fine" + panicked face / shaky voice). The two
  signals combine in a 5-row dispatch matrix in the MCP routing
  layer.
- **End-to-end SSE streaming** from FastAPI to a React 19 +
  TypeScript + Tailwind v4 frontend, with type-safe `i18n` (EN/ZH)
  and a `?mock=1` demo mode that runs without the backend.
- **MCP-orchestrated side effects** — the LLM autonomously
  triggers Excel audit logging and counselor email alerts via
  tool calls, no hand-coded routing rules.
- **Production observability** baked in: Prometheus histograms
  for per-stage latency, Langfuse distributed tracing across
  every LLM hop.
- **Six REST + SSE endpoints** (`/v1/chat`, `/v1/intent`,
  `/v1/emotion/{audio,video,fuse}`, `/v1/rag/consult`) on FastAPI,
  orchestrated alongside Ollama, Chroma, MCP, and Prometheus via
  Docker Compose.

---

## Architecture

```
                ┌── Lora fine-tune ──────────────────────────────
                │
                │   Qwen2.5-7B + open Chinese PSY dataset.
                │   4 emotion labels (Chinese ↔ English):
                │
                │       正常    Normal        焦虑   Anxiety
                │       低落    Low mood      高风险 High risk
                │
                │   The fine-tuned model is reused in the steps below.
                │
                └────────────────────────────────────────────────


  User input  (text / audio / video frames)
         │
         ▼

┌── ① Multimodal emotion sensing ────────────────────────────────
│
│   text    →  fine-tuned Qwen2.5-7B               →  emotion label
│   audio   →  faster-whisper → text → Qwen        →  emotion label
│   vision  →  MediaPipe 468-pt FaceMesh           →  score + label
│
└────────────────────────────────────────────────────────────────
         │
         ▼

┌── ② Multimodal fusion engine (deterministic Python) ───────────
│
│   SCORE_MAP    正常 = 0      焦虑 = 2      低落 = 3      高风险 = 4
│                Normal = 0    Anxiety = 2   Low = 3       High-risk = 4
│
│   fused_score  =  0.5 · vision  +  0.4 · audio  +  0.1 · text
│
│   risk_band       <  1.0       →   正常    (Normal)
│                  1.0 – 2.0     →   需关注  (Needs attention)
│                   ≥  2.0       →   高风险  (High risk)
│
│   Emits (emotion_label, fused_score, risk_band):
│   the user's OBJECTIVE multimodal state.
│
└────────────────────────────────────────────────────────────────
         │
         ▼

┌── ③ Layer-1 intent classifier ─────────────────────────────────
│       (regex fast-path + fine-tuned Qwen)
│
│   Classifies the user MESSAGE into one of:
│
│       CHAT   ·   CONSULT   ·   RISK
│
│   ─ what the user MEANT TO DO, not their objective state.
│
└────────────────────────────────────────────────────────────────

         │              │                                    │
         ▼              ▼                                    ▼

        CHAT         CONSULT                              RISK
      chitchat,    mental-health Q                     explicit extreme:
      small talk   ("stressed at school")              suicide / self-harm

         │              │                                    │
         │              ▼                                    │
         │                                                   │
         │   ┌── ④ Agentic RAG (LangGraph FSM) ────          │   fast-path:
         │   │                                               │   bypass RAG,
         │   │   Fine-tuned Qwen decides:                    │   stream crisis
         │   │     need KB lookup?                           │   comfort,
         │   │       no   →  answer directly                 │   fire email
         │   │       yes  →  Chroma retrieval +              │   alert from a
         │   │               multi-step ReAct                │   background
         │   │                                               │   task.
         │   └──                                             │
         │              │                                    │
         │              ▼                                    │
         │                                                   │
         │   ┌── ⑤ Side-effect routing (MCP server) ─────────
         │   │
         │   │   Dispatch matrix keyed on (intent × risk_band):
         │   │
         │   │     CHAT    + 正常 / 需关注          →  no-op
         │   │              (Normal / Needs attention)
         │   │     CHAT    + 高风险 (High risk)     →  Excel + email     ← silent crisis
         │   │     CONSULT + 正常 / 需关注          →  Excel
         │   │              (Normal / Needs attention)
         │   │     CONSULT + 高风险 (High risk)     →  Excel + email     ← Consult re-tier
         │   │     RISK    + (any risk_band)        →  Excel + email     ← intent fast-path
         │   │
         │   └──
         │              │                                    │
         ▼              ▼                                    ▼

      (no side       MCP writes Excel audit                MCP writes Excel
       effects)      and / or fires counselor              + fires counselor
                     email alert.                          email alert.

         │              │                                    │
         └──────────────┴───────────────┬────────────────────┘
                                        │
                                        ▼

┌── ⑥ SSE streamed reply ────────────────────────────────────────
│       (fine-tuned Qwen, low-hallucination)
│
│   CHAT:    friendly chitchat
│   CONSULT: supportive + KB-grounded answer if retrieval happened
│   RISK:    crisis comfort + campus hotline reminder
│
└────────────────────────────────────────────────────────────────
```

**Two risk signals fed into one dispatch matrix.** Layer-1's intent
classifier reads the user's *message* and assigns CHAT / CONSULT /
RISK. The fusion engine in ② reads the user's *multimodal state* and
produces a `risk_band` (正常 / 需关注 / 高风险). The two signals
combine in ⑤ to decide side effects:

- `RISK` intent → always Excel + email, regardless of risk_band. This
  is the explicit-text fast path for self-harm / suicide language.
- `CHAT` intent + risk_band `高风险` → also Excel + email. The *silent
  crisis* case: benign words ("I'm fine") but face/voice score high.
  Text-only judgement would miss it.
- `CONSULT` intent → Excel always, email only when risk_band is
  `高风险`. Audit trail for every consult, escalation only when the
  underlying signal corroborates.

A natural question: isn't the Consult-branch risk check redundant
with the RISK intent? They catch different things. RISK intent is
*what the message literally says*. Consult's risk_band gating is
*what the accumulated multimodal signal implies*. Combined, neither
dimension's blind spot is left uncovered.

---

## Design Decisions Worth Discussing

### Fusion is deterministic by default; the LLM is a cross-check

The original design called for letting Qwen do the weighted-sum fusion
in natural language. In practice 7B models are unreliable at multi-step
arithmetic — during testing it computed `3·0.5 + 2·0.4 + 2·0.1` as
`2.3`, flipping the verdict across the high-risk threshold (2.0).
Production now runs pure Python; the LLM path is opt-in and always
cross-checked, with a ±0.3 tolerance before falling back to the
deterministic answer. ([`app/services/fusion.py`](app/services/fusion.py))

### Conservative missing-modality policy

When only vision is available and scores low-mood (3), this
implementation returns `3·0.5 = 1.5` (*watch* tier), not the
re-normalized `3·1.0 = 3.0` (*high-risk* tier). False alarms erode
counselor trust faster than missed moderates get punished, so the
policy is to under-call.

### Silent crisis is a first-class case

A student can type "nice weather today" while their fused face+voice
score crosses 2.0. The intent layer would normally route this to
`CHAT` and skip both Excel and email — defeating the two-layer design.
[`orchestrator.on_turn_end`](app/agents/orchestrator.py) catches the
case explicitly, and a `silent_crisis_total` Prometheus counter
exposes its frequency to operators.

### Pre-LLM regex on the intent classifier

A small, high-precision keyword regex runs before the LLM
([`RISK_KEYWORDS`](app/agents/orchestrator.py)) and short-circuits to
`RISK`. The set is deliberately narrow — phrases like
*"我都快累死了"* (idiom for "exhausted") are not in it. The
training-side weak labeling pipeline must not reuse this regex, or it
would leak the inference-time safety net into the training signal.

### Restrained use of LangChain

LangChain (and friends) appear in exactly two places:
`RecursiveCharacterTextSplitter` for KB chunking, and
`langgraph.StateGraph` for the Agentic RAG control flow. Everything
else uses official SDKs directly — Ollama, MCP, faster-whisper,
MediaPipe, Chroma. Every layer of framework abstraction makes prompt
debugging and trace reading harder; this is the wrong domain to take
that hit.

### Alert recipients are ops config, not source

[`mail_alert`](mcp_server/tools/mail_alert.py) reads `ALERT_TO` from
`.env` at call time, comma-separated for multi-recipient. The same
code deploys to different schools by changing one env var; the
authorization decision stays out of source.

---

## Tech Stack

FastAPI + Pydantic v2 · Ollama for local GGUF inference ·
Qwen2.5-7B-Instruct base · QLoRA 4-bit nf4 fine-tuning · LangGraph for
the Agentic RAG state machine · Chroma + bge-small-zh embeddings ·
faster-whisper · MediaPipe FaceMesh · MCP via FastMCP · Prometheus +
Langfuse · Docker Compose.

---

## Quick Start

```bash
cp .env.example .env                          # fill in SMTP / Langfuse keys
docker compose up -d --build
docker exec mindwise-ollama ollama pull qwen2.5:7b
```

Services come up at `app:8000` (Swagger at `/docs`), `ollama:11434`,
`chroma:8001`, `prometheus:9090`. The Chinese message below reads
"I've been very stressed lately and can't sleep":

```bash
curl -N -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"stu_001","message":"最近压力很大，睡不着"}'
```

The response is an SSE stream — first frame `meta` (intent, risk,
fused score), then a token-by-token reply, then `done`. Side effects
(Excel logging, email alerts) run in BackgroundTasks after the stream
closes so they cannot break the user-facing reply.

---

## Fine-tuning

### Data

- **Train / val:** cleaned PsyQA, ~2,000–3,000 samples, 9:1 split.
- **Test:** 1,000 out-of-distribution samples physically isolated from
  PsyQA, in two streams:
  - **600 from HuggingFace** `Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset`
    — traditional → simplified Chinese conversion, anonymization,
    mapped into the 4-class space (400 *normal* + 200 *low-mood*).
  - **400 synthesized by Mistral Large** (Ollama Cloud) under
    controlled prompts — 300 *anxiety* + 100 *high-risk*. Synthesis
    is used for two reasons: public Chinese-emotion datasets have
    very few samples in these classes, and re-circulating real
    high-risk utterances (self-harm/suicide language) is a line this
    project does not want to cross.

The 4 labels are 正常 / 焦虑 / 低落 / 高风险 (*normal / anxiety /
low-mood / high-risk*) with score map `0 / 2 / 3 / 4`.

### Training

QLoRA: 4-bit nf4 with bf16 compute, LoRA on `q_proj` and `v_proj`
only, `r=8 / alpha=16 / lr=2e-4`. Loss is sample-weighted CE with
inverse-sqrt class-frequency weights — the training distribution is
heavily skewed (e.g. 正常 *(normal)* 2.3% vs. 低落 *(low-mood)*
50.8%), and naive CE collapses onto the majority class. Best
`eval_loss` checkpoint, not the last epoch. Single NVIDIA RTX 4500,
~15 minutes wall-clock.

```bash
python -m scripts.finetune_qwen --train data/processed/train.jsonl \
                                --val   data/processed/val.jsonl
python -m scripts.merge_lora    --adapter checkpoints/qwen25-7b-psychqa
# convert the merged model to GGUF with llama.cpp tooling, then:
ollama create qwen2.5-7b-psychqa -f Modelfile
```

### Eval

```bash
python -m scripts.eval.run_eval --model qwen2.5-7b-psychqa \
                                --test  data/test.jsonl \
                                --prompt-style instr
```

Reports overall accuracy, macro-F1, and a separate **high-risk
recall** (missed positives are far costlier than false alarms here).

| Model | Accuracy | High-risk recall |
|---|---|---|
| Qwen2.5-7B base | ~60% | low |
| LoRA fine-tuned | ~85% | ≥95% |

---

## Operational Notes

- **Failure handling.** The user-facing reply path never breaks.
  Whisper / Chroma / MCP failures are logged and degraded around;
  missing vision channel scores 0; LLM stream interruption yields a
  comfort fallback. The agent loop has a 4-step `MAX_STEPS` guard;
  intent-classifier failure routes conservatively to `CONSULT` so a
  possibly-distressed turn is never silently dropped.
- **Observability.** Prometheus at `/metrics` (HTTP latency +
  `intent_total`, `fused_risk_total`, `silent_crisis_total`,
  `mcp_tool_total`, `stage_latency_seconds`). Langfuse `@observe` on
  intent and Agentic RAG nodes; remove the env vars to disable in
  production without code changes.
- **Configuration.** See [`.env.example`](.env.example).
  `EMBEDDING_BACKEND` toggles between local bge and OpenAI;
  `ALERT_TO` decides crisis-email recipients.

---

## License

[Apache-2.0](LICENSE). The Qwen2.5-7B base model is also Apache-2.0
and permits commercial use.
