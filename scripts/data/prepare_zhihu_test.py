import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.data._anonymize import anonymize


INSTRUCTION = "分析用户文本情绪，只能输出：正常、焦虑、低落、高风险"
LABELS = {"正常", "焦虑", "低落", "高风险"}


def iter_records(path: Path):
    suf = path.suffix.lower()
    if suf == ".csv":
        with path.open(encoding="utf-8") as f:
            yield from csv.DictReader(f)
    elif suf in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suf == ".json":
        yield from json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"unsupported suffix {suf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="csv/jsonl with 'text' and 'label'")
    ap.add_argument("--output", default="data/test.jsonl")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = Counter()
    dist = Counter()

    with out.open("w", encoding="utf-8") as w:
        for r in iter_records(inp):
            text = (r.get("text") or r.get("content") or "").strip()
            label = (r.get("label") or "").strip()

            if not text:
                dropped["empty"] += 1
                continue
            if label not in LABELS:
                dropped["bad_label"] += 1
                continue

            w.write(
                json.dumps(
                    {
                        "instruction": INSTRUCTION,
                        "input": anonymize(text),
                        "output": label,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            dist[label] += 1

    print(f"kept={kept} dropped={dict(dropped)}")
    print("dist:", dict(dist))


if __name__ == "__main__":
    main()
