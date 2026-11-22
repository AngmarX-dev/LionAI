#!/usr/bin/env python3
"""Generate predictions on the deterministic held-out split and score them."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from datasets import Dataset
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def words(text: str) -> list[str]:
    return re.findall(r"[\w.\-/]+", text.casefold(), flags=re.UNICODE)

def f1_score(prediction: str, reference: str) -> float:
    from collections import Counter
    p, r = Counter(words(prediction)), Counter(words(reference))
    overlap = sum((p & r).values())
    if not p or not r or not overlap:
        return 0.0
    precision, recall = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * precision * recall / (precision + recall)

def normalize(text: str) -> str:
    return " ".join(words(text))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="training/lionai_sft.jsonl")
    ap.add_argument("--output", default="artifacts/evaluation")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 10:
        raise ValueError("Need at least 10 examples for held-out evaluation.")
    split = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=42)
    heldout = list(split["test"])
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1))) if False else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    results = []
    for index, row in enumerate(heldout, 1):
        messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        reference = row["output"].strip()
        results.append({
            "case": index, "input": row["input"], "reference": reference,
            "prediction": answer, "exact_match": normalize(answer) == normalize(reference),
            "token_f1": round(f1_score(answer, reference), 4),
        })

    count = len(results)
    summary = {
        "model": args.model, "adapter": args.adapter, "evaluation_examples": count,
        "exact_match_accuracy": round(sum(x["exact_match"] for x in results) / count, 4),
        "mean_token_f1": round(sum(x["token_f1"] for x in results) / count, 4),
        "scoring_note": "Token-overlap F1 is a rough lexical metric, not semantic correctness. Review predictions manually.",
        "results": results,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    import os
    raise SystemExit(main())
