#!/usr/bin/env python3
"""Independent QLoRA evaluation for LionAI's Qwen3-1.7B adapter."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def score_prediction(prediction: str, reference_text: str, input_text: str) -> dict:
    predicted = parse_json_object(prediction)
    reference = json.loads(reference_text)
    input_names = [part.strip() for part in input_text.split("files=", 1)[-1].split(",") if part.strip()]
    result = {
        "valid_json": predicted is not None, "valid_schema": False,
        "input_file_count": len(input_names), "filename_coverage": 0.0,
        "path_match_rate": 0.0, "extension_preservation_rate": 0.0,
        "safe_paths": False, "needs_review_count": 0,
        "hallucinated_names": [], "missing_names": [],
    }
    if predicted is None:
        return result
    mappings = predicted.get("files")
    schema_ok = (
        predicted.get("schema_version") == "1.0"
        and isinstance(predicted.get("folders"), list)
        and isinstance(mappings, list)
        and all(isinstance(item, dict) for item in mappings)
    )
    result["valid_schema"] = schema_ok
    if not schema_ok:
        return result
    by_name = {}
    for item in mappings:
        name = item.get("original_name")
        if isinstance(name, str) and name not in by_name:
            by_name[name] = item
    expected, actual = set(input_names), set(by_name)
    matched = expected & actual
    result["missing_names"] = sorted(expected - actual)
    result["hallucinated_names"] = sorted(actual - expected)
    result["filename_coverage"] = len(matched) / max(1, len(expected))
    ref_map = {item["original_name"]: item for item in reference.get("files", [])}
    path_matches = ext_matches = safe_count = review_count = 0
    for name in matched:
        item = by_name[name]
        path = item.get("suggested_path")
        if item.get("needs_review") is True:
            review_count += 1
        if safe_relative_path(path):
            safe_count += 1
        if path == ref_map.get(name, {}).get("suggested_path"):
            path_matches += 1
        predicted_name = Path(path).name if isinstance(path, str) else ""
        if Path(predicted_name).suffix.casefold() == Path(name).suffix.casefold():
            ext_matches += 1
    denominator = max(1, len(matched))
    result["path_match_rate"] = path_matches / max(1, len(expected))
    result["extension_preservation_rate"] = ext_matches / denominator
    result["safe_paths"] = safe_count == len(matched) and bool(matched)
    result["needs_review_count"] = review_count
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-1.7B")
    parser.add_argument("--adapter", default="artifacts/lionai-qwen3-1.7b-qlora/adapter")
    parser.add_argument("--data", default="training/lionai_eval.jsonl")
    parser.add_argument("--output", default="artifacts/lionai-qwen3-1.7b-evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen3-1.7B evaluation requires CUDA.")
    props = torch.cuda.get_device_properties(0)
    vram_gib = props.total_memory / (1024 ** 3)
    if vram_gib < 5.5:
        raise RuntimeError(f"Evaluation requires at least 5.5 GiB VRAM; detected {vram_gib:.2f} GiB.")
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 20:
        raise ValueError("Independent evaluation requires at least 20 held-out examples.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quantization, device_map={"": 0},
        torch_dtype=compute_dtype, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    results = []
    for index, row in enumerate(rows, 1):
        messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        results.append({
            "case": index, "input": row["input"], "prediction": answer,
            "reference": row["output"], **score_prediction(answer, row["output"], row["input"]),
        })
    keys = (
        "valid_json", "valid_schema", "filename_coverage", "path_match_rate",
        "extension_preservation_rate", "safe_paths",
    )
    averages = {
        key: round(sum(float(item[key]) for item in results) / len(results), 4)
        for key in keys
    }
    summary = {
        "model": args.model, "adapter": args.adapter,
        "evaluation_examples": len(results), "metrics": averages,
        "warning": "Small held-out set; diagnostic only. path_match_rate is exact string match.",
        "results": results,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "generation_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
