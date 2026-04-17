import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import ollama
from tqdm import tqdm


LABELS = ["正常", "焦虑", "低落", "高风险"]

EVAL_PROMPT = """你是情绪分类器。严格输出 JSON：{{"label": "<标签>"}}
标签只能从四选一：正常、焦虑、低落、高风险。
不解释，不输出其它字段。

用户文本：{text}"""


def predict(client, model, text, retries=2):
    prompt = EVAL_PROMPT.format(text=text)
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
            if lbl in LABELS:
                return lbl
        except Exception:
            if i < retries:
                time.sleep(0.3 * (i + 1))
    return None


def prf(cm, label):
    tp = cm[label][label]
    fp = sum(cm[o][label] for o in LABELS if o != label)
    fn = sum(cm[label][o] for o in LABELS if o != label)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="ollama model tag, e.g. qwen2.5:7b or qwen2.5-7b-psychqa")
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--pred-out", default=None,
                    help="optional path to dump per-item predictions")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = ollama.Client(host=args.host)

    rows = [json.loads(l) for l in Path(args.test).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    cm = {t: Counter() for t in LABELS}  # cm[true][pred]
    preds = []
    parse_fail = 0

    for r in tqdm(rows, desc=f"eval {args.model}"):
        text = r["input"]
        truth = r["output"]
        pred = predict(client, args.model, text)
        if pred is None:
            parse_fail += 1
            pred = "正常"  # fall back; still counted against accuracy
        cm[truth][pred] += 1
        preds.append({"input": text, "truth": truth, "pred": pred,
                      "source": r.get("source")})

    if args.pred_out:
        Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.pred_out, "w", encoding="utf-8") as w:
            for p in preds:
                w.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- report ----
    total = len(rows)
    correct = sum(cm[t][t] for t in LABELS)
    acc = correct / total

    print(f"\n==== {args.model} on {args.test} ====")
    print(f"total={total}  correct={correct}  accuracy={acc*100:.2f}%  parse_fail={parse_fail}")
    print(f"\n{'class':<8}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}")
    f1s = []
    for lbl in LABELS:
        n = sum(cm[lbl].values())
        p, r, f1 = prf(cm, lbl)
        f1s.append(f1)
        print(f"{lbl:<8}{n:>6}{p*100:>7.1f}%{r*100:>7.1f}%{f1*100:>7.1f}%")
    print(f"macro-F1 = {sum(f1s)/len(f1s)*100:.2f}%")

    print("\nconfusion matrix (row=truth, col=pred):")
    print(" " * 6 + "".join(f"{c:>8}" for c in LABELS))
    for t in LABELS:
        print(f"{t:<6}" + "".join(f"{cm[t][c]:>8}" for c in LABELS))

    hr_recall = prf(cm, "高风险")[1]
    print(f"\nHIGH-RISK RECALL = {hr_recall*100:.1f}%  (漏判代价最大，目标 >= 95%)")


if __name__ == "__main__":
    main()
