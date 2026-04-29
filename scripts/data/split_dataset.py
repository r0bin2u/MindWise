import argparse
import json
import random
from collections import Counter
from pathlib import Path


INSTRUCTION = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险"


def stratified_split(rows, ratio, seed):
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    rnd = random.Random(seed)
    train, val = [], []
    for lbl, items in by_label.items():
        rnd.shuffle(items)
        k = int(len(items) * ratio)
        train.extend(items[:k])
        val.extend(items[k:])
    rnd.shuffle(train)
    rnd.shuffle(val)
    return train, val


def dump_sft(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "instruction": INSTRUCTION,
                        "input": r["text"],
                        "output": r["label"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/labeled.jsonl")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.input).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train, val = stratified_split(rows, args.ratio, args.seed)

    dump_sft(train, args.train)
    dump_sft(val, args.val)

    print(f"train={len(train)} val={len(val)}")
    print("train dist:", dict(Counter(r["label"] for r in train)))
    print("val   dist:", dict(Counter(r["label"] for r in val)))


if __name__ == "__main__":
    main()
