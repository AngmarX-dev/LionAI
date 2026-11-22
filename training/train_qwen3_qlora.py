#!/usr/bin/env python3
"""CUDA QLoRA fine-tuning for LionAI's Qwen3-8B checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def load_rows(path: Path) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 20:
        raise ValueError(f"{path}: at least 20 examples are required.")
    for index, row in enumerate(rows, 1):
        for key in ("instruction", "input", "output"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{path}:{index}: missing non-empty {key}")
        target = json.loads(row["output"])
        if target.get("schema_version") != "1.0":
            raise ValueError(f"{path}:{index}: unsupported output schema")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-8B")
    parser.add_argument("--data", default="training/lionai_sft.jsonl")
    parser.add_argument("--output", default="artifacts/lionai-qwen3-8b-qlora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-8B QLoRA requires a CUDA GPU; CPU-only training is not supported by this pipeline.")

    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / (1024 ** 3)
    print(f"GPU: {props.name}; VRAM: {vram_gib:.1f} GiB")
    if vram_gib < 16:
        raise RuntimeError(
            f"Insufficient VRAM ({vram_gib:.1f} GiB). This 8B QLoRA profile requires "
            "at least 16 GiB VRAM. Use a suitable GPU runner; do not silently fall back to CPU."
        )

    rows = load_rows(Path(args.data))
    split = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat-template prompt is not a token prefix of the full conversation.")
        if len(prompt_ids) >= args.max_length:
            raise ValueError("Prompt leaves no room for supervised assistant tokens.")
        target_ids = full_ids[len(prompt_ids):][: args.max_length - len(prompt_ids)]
        if not target_ids:
            raise ValueError("Training example has no supervised assistant tokens.")
        input_ids = prompt_ids + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": ([-100] * len(prompt_ids)) + target_ids,
        }

    tokenized = split.map(
        encode,
        remove_columns=split["train"].column_names,
        desc="Tokenizing Qwen3-8B examples",
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
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
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Qwen3-8B adapter saved to {output / 'adapter'}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
