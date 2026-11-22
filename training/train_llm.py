#!/usr/bin/env python3
"""CPU-only LoRA fine-tuning for LionAI's single Qwen2.5-0.5B-Instruct LLM.

This script trains an adapter from JSONL records with instruction/input/output.
It does not download or train automatically; run it explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def read_records(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("instruction", "input", "output"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"Line {line_number}: missing non-empty {key}")
            rows.append(item)
    if len(rows) < 10:
        raise ValueError("At least 10 examples are required; use a larger, diverse dataset.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training/lionai_sft.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="artifacts/lionai-qwen-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if not 0.05 <= args.eval_ratio <= 0.3:
        raise ValueError("--eval-ratio must be between 0.05 and 0.30")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    device = torch.device("cpu")
    print(f"Training device: {device}; torch threads: {torch.get_num_threads()}")

    records = read_records(Path(args.data))
    dataset = Dataset.from_list(records).train_test_split(
        test_size=args.eval_ratio, seed=42
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def encode(row: dict[str, str]) -> dict[str, list[int]]:
        prompt_messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": row["output"]}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(
            full, max_length=args.max_length, truncation=True, add_special_tokens=False
        )
        labels = list(encoded["input_ids"])
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        encoded["labels"] = labels
        return encoded

    tokenized = dataset.map(
        encode,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing LionAI examples",
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        no_cuda=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"Adapter saved to: {output / 'adapter'}")
    print(f"Evaluation metrics: {metrics}")
    print("This is a LoRA adapter. Merge/convert it for native GGUF inference separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
