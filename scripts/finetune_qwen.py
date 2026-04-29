"""QLoRA fine-tune Qwen2.5-7B-Instruct on PsyQA emotion-classification data.

Key choices:
  - 4bit nf4 quantization, bf16 compute
  - LoRA r=8 alpha=16 on q_proj / v_proj only
  - Sample-weighted CE loss with inverse-sqrt class frequency (handles
    class imbalance: 正常 2.3% vs 低落 50.8%)
  - Prompt tokens masked as -100 so loss is computed only on the label
  - Save best checkpoint by eval_loss, not last
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


LABELS = ["正常", "焦虑", "低落", "高风险"]


def load_rows(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def inv_sqrt_weights(rows):
    cnt = Counter(r["output"] for r in rows)
    total = sum(cnt.values())
    raw = {c: 1.0 / math.sqrt(cnt[c] / total) for c in cnt}
    mn = min(raw.values())
    return {c: round(raw[c] / mn, 3) for c in raw}


def encode_row(row, tokenizer, weight, max_len):
    user_msg = {
        "role": "user",
        "content": f"{row['instruction']}\n\n用户文本：{row['input']}",
    }
    prompt_str = tokenizer.apply_chat_template(
        [user_msg], add_generation_prompt=True, tokenize=False
    )
    full_str = tokenizer.apply_chat_template(
        [user_msg, {"role": "assistant", "content": row["output"]}],
        add_generation_prompt=False,
        tokenize=False,
    )
    prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    input_ids = full_ids[:max_len]
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    labels = labels[:max_len]
    while len(labels) < len(input_ids):
        labels.append(-100)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "sample_weight": weight,
    }


def build_dataset(rows, tokenizer, weights, max_len):
    return Dataset.from_list(
        [encode_row(r, tokenizer, weights[r["output"]], max_len) for r in rows]
    )


def pad_collate(batch, pad_id):
    mx = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "sample_weight": []}
    for b in batch:
        pad = mx - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * pad)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        out["labels"].append(b["labels"] + [-100] * pad)
        out["sample_weight"].append(b["sample_weight"])
    return {
        "input_ids": torch.tensor(out["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(out["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(out["labels"], dtype=torch.long),
        "sample_weight": torch.tensor(out["sample_weight"], dtype=torch.float32),
    }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        w = inputs.pop("sample_weight")
        labels = inputs.pop("labels")
        out = model(**inputs)
        logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        ce = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat = ce(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).float()
        per_sample = (flat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        loss = (per_sample * w.to(per_sample.device)).mean()
        return (loss, out) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--output-dir", default="checkpoints/qwen25-7b-psychqa")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="if >0, cap training to this many steps (smoke-test)",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print(f"[1/5] loading tokenizer: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[2/5] loading 4bit base model")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print("[3/5] attaching LoRA adapters")
    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[4/5] building datasets")
    tr_rows = load_rows(args.train)
    va_rows = load_rows(args.val)
    weights = inv_sqrt_weights(tr_rows)
    print(f"      class weights: {weights}")
    print(f"      train={len(tr_rows)}  val={len(va_rows)}")

    train_ds = build_dataset(tr_rows, tok, weights, args.max_len)
    val_ds = build_dataset(va_rows, tok, weights, args.max_len)

    def collate(batch):
        return pad_collate(batch, tok.pad_token_id)

    print("[5/5] starting training")
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    meta = {
        "base_model": args.base_model,
        "lora": {
            "r": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "targets": ["q_proj", "v_proj"],
        },
        "optim": {
            "lr": args.lr,
            "scheduler": "cosine",
            "warmup_ratio": 0.05,
            "optimizer": "paged_adamw_8bit",
        },
        "batch": {
            "micro": args.micro_batch,
            "grad_accum": args.grad_accum,
            "effective": args.micro_batch * args.grad_accum,
        },
        "epochs": args.epochs,
        "class_weights": weights,
        "best_eval_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    with open(Path(args.output_dir) / "run_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nsaved adapter + meta to {args.output_dir}")


if __name__ == "__main__":
    main()
