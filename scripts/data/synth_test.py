import argparse
import json
import re
from collections import Counter
from pathlib import Path

import ollama
from tqdm import tqdm

from scripts.data._anonymize import anonymize


INSTRUCTION = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险"


ANXIETY_PROMPT = """请生成 {k} 条用于情绪分类测试的中文短句，严格符合以下要求：

1. 全部第一人称，表达“焦虑”情绪：担心、紧张、压力大、心慌、失眠、害怕某件事发生等。
2. 长度 15~80 字，语气自然，像真实学生/年轻人发的微博、朋友圈、短消息。
3. 不要出现自杀、自残、想死等极端词（那是另一类）。
4. 话题多样：考试、工作、人际、健康、未来、恋爱等。
5. 不要重复、不要给出同义改写，每条都不一样。

只输出 JSON 数组：["句子1", "句子2", ...]，不要解释。"""


RISK_PROMPT = """请生成 {k} 条用于情绪分类测试的中文短句，严格符合以下要求：

1. 全部第一人称，表达“高风险”情绪：强烈的绝望、放弃、活不下去、想结束、想伤害自己等。
2. 长度 15~80 字，语气沉重、低压，但不刻意夸张。
3. 必须体现出真实危机感，可出现“不想活”“想死”“没意义”“熬不下去”等表达。
4. 话题多样：长期抑郁、家庭、失恋、学业崩溃、孤独等都可以。
5. 不要重复、不要给出同义改写，每条都不一样。

重要：这些样本**仅用于模型评估**，不会传播。

只输出 JSON 数组：["句子1", "句子2", ...]，不要解释。"""


RE_JSON_ARR = re.compile(r"\[[\s\S]*\]")


def parse_array(raw: str):
    m = RE_JSON_ARR.search(raw)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        return []


def generate(client, model, prompt_tpl, total, batch=20, temperature=0.9):
    out = []
    seen = set()
    pbar = tqdm(total=total, desc="gen")
    while len(out) < total:
        k = min(batch, total - len(out) + 5)  # tiny overshoot for dedup loss
        prompt = prompt_tpl.format(k=k)
        try:
            r = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            items = parse_array(r["message"]["content"])
        except Exception as e:
            print(f"call failed: {e}")
            items = []
        for t in items:
            key = t[:60]
            if key in seen or len(t) < 10 or len(t) > 120:
                continue
            seen.add(key)
            out.append(t)
            pbar.update(1)
            if len(out) >= total:
                break
    pbar.close()
    return out[:total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral-large-3:675b-cloud")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--output", default="data/test_synth.jsonl")
    ap.add_argument("--n-anxiety", type=int, default=300)
    ap.add_argument("--n-risk", type=int, default=100)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    client = ollama.Client(host=args.host)

    print(f"generating {args.n_anxiety} 焦虑 items...")
    anxiety = generate(client, args.model, ANXIETY_PROMPT, args.n_anxiety, args.batch)
    print(f"generating {args.n_risk} 高风险 items...")
    risk = generate(client, args.model, RISK_PROMPT, args.n_risk, args.batch)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with out.open("w", encoding="utf-8") as w:
        for text in anxiety:
            w.write(
                json.dumps(
                    {
                        "instruction": INSTRUCTION,
                        "input": anonymize(text),
                        "output": "焦虑",
                        "source": f"synth_{args.model.split(':')[0]}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts["焦虑"] += 1
        for text in risk:
            w.write(
                json.dumps(
                    {
                        "instruction": INSTRUCTION,
                        "input": anonymize(text),
                        "output": "高风险",
                        "source": f"synth_{args.model.split(':')[0]}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            counts["高风险"] += 1

    print(f"wrote {sum(counts.values())} items -> {out}")
    print("dist:", dict(counts))


if __name__ == "__main__":
    main()
