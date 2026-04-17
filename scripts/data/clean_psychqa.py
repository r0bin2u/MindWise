import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from scripts.data._anonymize import anonymize

MIN_LEN = 10
MAX_LEN = 2000

RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"https?://\S+")
RE_WS = re.compile(r"\s+")


def normalize(text):
    if not text:
        return ""
    t = RE_HTML.sub(" ", text)
    t = RE_URL.sub(" ", t)
    t = t.replace("\u3000", " ").replace("\xa0", " ")
    t = RE_WS.sub(" ", t).strip()
    return t


def iter_records(path: Path):
    suf = path.suffix.lower()
    if suf in {".jsonl", ".ndjson"}:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or data.get("items") or list(data.values())
        yield from data
    elif suf == ".csv":
        with path.open(encoding="utf-8") as f:
            yield from csv.DictReader(f)
    else:
        raise ValueError(f"unsupported suffix: {suf}")


def extract_text(rec, field):
    # PsyQA: title in `question`, body in `description`; concat when both exist
    if not isinstance(rec, dict):
        return ""
    q = (rec.get("question") or "").strip()
    d = (rec.get("description") or "").strip()
    if q and d:
        return q + "。" + d if not q.endswith(("。", "？", "！", "?", "!")) else q + d
    if q or d:
        return q or d
    for k in (field, "query", "user", "input", "text", "content"):
        v = rec.get(k)
        if v:
            return v
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="data/candidate.jsonl")
    ap.add_argument("--text-field", default="question")
    ap.add_argument("--min-len", type=int, default=MIN_LEN)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--no-anonymize", action="store_true",
                    help="skip the PII anonymizer (default: on)")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    kept = 0
    dropped = Counter()

    with out.open("w", encoding="utf-8") as w:
        for rec in iter_records(inp):
            text = normalize(extract_text(rec, args.text_field))
            if not text:
                dropped["empty"] += 1
                continue
            n = len(text)
            if n < args.min_len or n > args.max_len:
                dropped["length"] += 1
                continue
            key = text[:200]
            if key in seen:
                dropped["dup"] += 1
                continue
            seen.add(key)
            if not args.no_anonymize:
                text = anonymize(text)
            w.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept={kept} dropped={dict(dropped)} -> {out}")


if __name__ == "__main__":
    main()
