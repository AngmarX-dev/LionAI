#!/usr/bin/env python3
"""Schema-aware, deterministic evaluation for LionAI's JSON-only file proposals."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_json_object(text: str) -> dict | None:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(candidate[start:end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and "\\\\" not in value


def score_prediction(prediction: str, reference_text: str, input_text: str) -> dict:
    predicted = parse_json_object(prediction)
    reference = json.loads(reference_text)
    input_names = [x.strip() for x in input_text.split("files=", 1)[-1].split(",") if x.strip()]
    result = {
        "valid_json": predicted is not None,
        "valid_schema": False,
        "input_file_count": len(input_names),
        "filename_coverage": 0.0,
        "path_match_rate": 0.0,
        "extension_preservation_rate": 0.0,
        "safe_paths": False,
        "needs_review_count": 0,
        "hallucinated_names": [],
        "missing_names": [],
    }
    if predicted is None:
        return result

    mappings = predicted.get("files")
    schema_ok = (
        predicted.get("schema_version") == "1.0"
        and isinstance(predicted.get("folders"), list)
        and isinstance(mappings, list)
        and all(isinstance(x, dict) for x in mappings)
    )
    result["valid_schema"] = schema_ok
    if not schema_ok:
        return result

    by_name = {}
    for item in mappings:
        name = item.get("original_name")
        if isinstance(name, str) and name not in by_name:
            by_name[name] = item
    expected = set(input_names)
    actual = set(by_name)
    result["missing_names"] = sorted(expected - actual)
    result["hallucinated_names"] = sorted(actual - expected)
    matched = expected & actual
    result["filename_coverage"] = len(matched) / max(1, len(expected))

    ref_map = {x["original_name"]: x for x in reference.get("files", [])}
    path_matches, ext_matches, safe_count = 0, 0, 0
    review_count = 0
    for name in matched:
        item = by_name[name]
        path = item.get("suggested_path")
        if isinstance(item.get("needs_review"), bool) and item["needs_review"]:
            review_count += 1
        if safe_relative_path(path):
            safe_count += 1
        if path == ref_map.get(name, {}).get("suggested_path"):
            path_matches += 1
        predicted_ext = Path(name).suffix.casefold()
        path_name = Path(path).name if isinstance(path, str) else ""
        if Path(path_name).suffix.casefold() == predicted_ext:
            ext_matches += 1
    denominator = max(1, len(matched))
    result["path_match_rate"] = path_matches / max(1, len(expected))
    result["extension_preservation_rate"] = ext_matches / denominator
    result["safe_paths"] = safe_count == len(matched) and len(matched) > 0
    result["needs_review_count"] = review_count
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="training/lionai_eval.jsonl")
    ap.add_argument("--output", default="artifacts/evaluation")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    args = ap.parse_args()

    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(rows) < 20:
        raise ValueError("Independent evaluation requires at least 20 curated examples.")
    heldout = rows

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
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
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        reference = row["output"].strip()
        metrics = score_prediction(answer, reference, row["input"])
        results.append({
            "case": index,
            "input": row["input"],
            "prediction": answer,
            "reference": reference,
            **metrics,
        })

    count = len(results)
    averages = {}
    for key in ("valid_json", "valid_schema", "filename_coverage", "path_match_rate",
                "extension_preservation_rate", "safe_paths"):
        averages[key] = round(
            sum(float(item[key]) for item in results) / count, 4
        )
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "evaluation_examples": count,
        "metrics": averages,
        "warning": (
            "The held-out set is small; these results are diagnostic, not a reliable "
            "generalization-accuracy estimate. Path match is exact against the reference."
        ),
        "results": results,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
