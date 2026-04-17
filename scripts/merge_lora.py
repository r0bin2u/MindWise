import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/qwen25-7b-psychqa")
    ap.add_argument("--output", default="checkpoints/qwen25-7b-psychqa-merged")
    args = ap.parse_args()

    print(f"loading base: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("merging LoRA into base weights...")
    model = model.merge_and_unload()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print(f"saved merged model to {out}")


if __name__ == "__main__":
    main()
