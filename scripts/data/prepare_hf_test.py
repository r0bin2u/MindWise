import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from zhconv import convert

from scripts.data._anonymize import anonymize


INSTRUCTION = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险"

LABEL_MAP = {
    "平淡語氣": "正常",
    "開心語調": "正常",
    "悲傷語調": "低落",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset")
    ap.add_argument("--output", default="data/test_real.jsonl")
    ap.add_argument("--n-normal", type=int, default=400)
    ap.add_argument("--n-low", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train")

    buckets = {"正常": [], "低落": []}
    for r in ds:
        tgt = LABEL_MAP.get(r["emotion"])
        if tgt is None:
            continue
        text = convert(r["text"], "zh-cn").strip()
        text = anonymize(text)
        if text:
            buckets[tgt].append(text)

    rnd = random.Random(args.seed)
    for k in buckets:
        rnd.shuffle(buckets[k])

    picked = {
        "正常": buckets["正常"][: args.n_normal],
        "低落": buckets["低落"][: args.n_low],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as w:
        for label, items in picked.items():
            for t in items:
                w.write(json.dumps({
                    "instruction": INSTRUCTION,
                    "input": t,
                    "output": label,
                    "source": "hf_johnson8187",
                }, ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {n} items -> {out}")
    print("dist:", {k: len(v) for k, v in picked.items()})
    print("pool sizes:", {k: len(v) for k, v in buckets.items()})


if __name__ == "__main__":
    main()
