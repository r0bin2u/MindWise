import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import ollama
from tqdm import tqdm


LABELS = ["正常", "焦虑", "低落", "高风险"]

# PsyQA has a lot of knowledge QA with no first-person emotion; filter those out
EMO_HINT = re.compile(
    r"(我|自己|最近|感到|觉得|压力|焦虑|抑郁|难过|失眠|害怕|担心|绝望|"
    r"想死|自残|委屈|孤独|崩溃|想不开|没意思|痛苦|烦|累|空虚|无助|心累)"
)

# skip LLM for obvious high-risk phrases
RISK_RE = re.compile(r"(想死|自杀|自残|跳楼|割腕|结束生命|活不下去|不想活了)")


PROMPT = """你是情绪分类器。严格按 JSON 输出，不解释、不扩展。
标签只能从以下四选一：正常、焦虑、低落、高风险。
confidence 是 0~1 的浮点，代表你对标签的确信度。

输出格式：{{"label": "<标签>", "confidence": <浮点>}}

文本：{text}"""


def prefilter(text):
    return bool(EMO_HINT.search(text))


def call_llm(client, model, text, retries=2):
    prompt = PROMPT.format(text=text)
    for i in range(retries + 1):
        try:
            r = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0},
            )
            raw = r["message"]["content"]
            obj = json.loads(raw)
            lbl = obj.get("label", "").strip()
            conf = float(obj.get("confidence", 0.0))
            if lbl in LABELS and 0.0 <= conf <= 1.0:
                return lbl, conf
        except Exception:
            if i < retries:
                time.sleep(0.5 * (i + 1))
    return None, 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/candidate.jsonl")
    ap.add_argument("--output", default="data/labeled.jsonl")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap on labeled records for a quick run")
    args = ap.parse_args()

    client = ollama.Client(host=args.host)

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    label_dist = Counter()

    lines = [l for l in inp.read_text(encoding="utf-8").splitlines() if l.strip()]

    with out.open("w", encoding="utf-8") as w:
        for line in tqdm(lines, desc="labeling"):
            if args.max_samples and stats["kept"] >= args.max_samples:
                break
            rec = json.loads(line)
            text = rec["text"]
            stats["total"] += 1

            if not prefilter(text):
                stats["filtered"] += 1
                continue

            if RISK_RE.search(text):
                label, conf = "高风险", 0.95
                stats["risk_shortcut"] += 1
            else:
                label, conf = call_llm(client, args.model, text)

            if label is None:
                stats["llm_fail"] += 1
                continue
            if conf < args.threshold:
                stats["low_conf"] += 1
                continue

            w.write(json.dumps(
                {"text": text, "label": label, "confidence": round(conf, 3)},
                ensure_ascii=False
            ) + "\n")
            stats["kept"] += 1
            label_dist[label] += 1

    print(f"stats={dict(stats)}")
    print(f"label_dist={dict(label_dist)}")


if __name__ == "__main__":
    main()
