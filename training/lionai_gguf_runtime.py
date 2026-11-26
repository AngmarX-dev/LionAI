#!/usr/bin/env python3
"""Native Python/PyTorch inference runtime for LionAI's custom GGUF v3 model."""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from lionai_scratch import LionAIConfig, LionAITokenizer, build_model
from training.metadata_rules import extension_rule

GGUF_VALUE_FORMATS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}


class GGUFError(ValueError):
    """Raised when a GGUF file is malformed or incompatible with LionAI."""


class GGUFReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.size = self.path.stat().st_size
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, tuple[tuple[int, ...], int, int]] = {}
        try:
            self._parse()
        finally:
            self.file.close()

    def _read(self, count: int) -> bytes:
        if count < 0 or self.file.tell() + count > self.size:
            raise GGUFError("Unexpected end of GGUF file.")
        data = self.file.read(count)
        if len(data) != count:
            raise GGUFError("Truncated GGUF file.")
        return data

    def _unpack(self, fmt: str):
        return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]

    def _string(self) -> str:
        length = self._unpack("<Q")
        if length > self.size:
            raise GGUFError("Invalid GGUF string length.")
        try:
            return self._read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("GGUF contains invalid UTF-8 metadata.") from exc

    def _value(self, value_type: int):
        if value_type == 8:
            return self._string()
        if value_type == 9:
            item_type = self._unpack("<I")
            count = self._unpack("<Q")
            if count > 10_000_000:
                raise GGUFError("GGUF metadata array is unreasonably large.")
            return [self._value(item_type) for _ in range(count)]
        if value_type not in GGUF_VALUE_FORMATS:
            raise GGUFError(f"Unsupported GGUF metadata type: {value_type}")
        fmt, _ = GGUF_VALUE_FORMATS[value_type]
        return self._unpack(fmt)

    def _parse(self):
        if self._read(4) != b"GGUF":
            raise GGUFError("Not a GGUF file.")
        version = self._unpack("<I")
        if version != 3:
            raise GGUFError(f"Unsupported GGUF version {version}; expected v3.")
        tensor_count = self._unpack("<Q")
        metadata_count = self._unpack("<Q")
        if tensor_count > 100_000 or metadata_count > 100_000:
            raise GGUFError("GGUF header counts are unreasonable.")
        for _ in range(metadata_count):
            key = self._string()
            value_type = self._unpack("<I")
            if key in self.metadata:
                raise GGUFError(f"Duplicate GGUF metadata key: {key}")
            self.metadata[key] = self._value(value_type)
        for _ in range(tensor_count):
            name = self._string()
            dimensions = self._unpack("<I")
            if dimensions < 1 or dimensions > 4:
                raise GGUFError(f"Invalid dimensions for tensor {name!r}.")
            gguf_shape = tuple(self._unpack("<Q") for _ in range(dimensions))
            tensor_type = self._unpack("<I")
            offset = self._unpack("<Q")
            if name in self.tensors:
                raise GGUFError(f"Duplicate tensor name: {name}")
            self.tensors[name] = (gguf_shape, tensor_type, offset)
        alignment = int(self.metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment & (alignment - 1):
            raise GGUFError("Invalid GGUF tensor alignment.")
        self.data_start = (self.file.tell() + alignment - 1) // alignment * alignment
        for name, (shape, tensor_type, offset) in self.tensors.items():
            if tensor_type != 0:
                raise GGUFError(f"Tensor {name!r} is not F32; quantized tensors are not supported.")
            byte_count = math.prod(shape) * 4
            if self.data_start + offset + byte_count > self.size:
                raise GGUFError(f"Tensor {name!r} points outside the GGUF file.")

    def load_tensors(self) -> dict[str, torch.Tensor]:
        result = {}
        with self.path.open("rb") as handle:
            for name, (gguf_shape, tensor_type, offset) in self.tensors.items():
                if tensor_type != 0:
                    raise GGUFError(f"Unsupported tensor type for {name}.")
                handle.seek(self.data_start + offset)
                count = math.prod(gguf_shape)
                raw = handle.read(count * 4)
                if len(raw) != count * 4:
                    raise GGUFError(f"Truncated tensor data for {name}.")
                torch_shape = tuple(reversed(gguf_shape))
                tensor = torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()
                result[name] = tensor.reshape(torch_shape)
        return result


def load_gguf_model(path: str | Path, device: str | torch.device = "cpu"):
    reader = GGUFReader(path)
    metadata = reader.metadata
    if metadata.get("general.architecture") != "lionai":
        raise GGUFError("GGUF architecture is not 'lionai'.")
    if metadata.get("lionai.initialized_from") != "random":
        raise GGUFError("Expected a LionAI from-scratch checkpoint.")
    if metadata.get("lionai.tokenizer") != "utf8-byte-v1":
        raise GGUFError("Unsupported LionAI tokenizer.")
    config = LionAIConfig(
        vocab_size=int(metadata["lionai.vocab_size"]),
        d_model=int(metadata["lionai.d_model"]),
        n_heads=int(metadata["lionai.n_heads"]),
        n_layers=int(metadata["lionai.n_layers"]),
        feedforward=int(metadata["lionai.feedforward"]),
        max_length=int(metadata["lionai.max_length"]),
        num_classes=int(metadata["lionai.num_classes"]),
        dropout=float(metadata.get("lionai.dropout", 0.25)),
    )
    labels = metadata.get("lionai.labels")
    if not isinstance(labels, list) or len(labels) != config.num_classes or not all(isinstance(x, str) for x in labels):
        raise GGUFError("GGUF class labels do not match model configuration.")
    model = build_model(config)
    raw_state = reader.load_tensors()
    prefix = "lionai."
    if not all(name.startswith(prefix) for name in raw_state):
        raise GGUFError("GGUF tensor names do not use the expected lionai. namespace.")
    state = {name[len(prefix):]: tensor for name, tensor in raw_state.items()}
    expected = model.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise GGUFError(f"Tensor-name mismatch; missing={missing}, unexpected={extra}")
    for name, target in expected.items():
        if state[name].shape != target.shape:
            raise GGUFError(f"Tensor shape mismatch for {name}: {tuple(state[name].shape)} != {tuple(target.shape)}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, labels, config, metadata


class LionAIGGUFSession:
    """Load one GGUF model once and reuse it for single or batched predictions."""

    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.75,
                 margin_threshold: float = 0.10, temperature: float = 1.0,
                 device: str = "auto"):
        if not 0 <= confidence_threshold <= 1 or not 0 <= margin_threshold <= 1:
            raise ValueError("confidence and margin thresholds must be between 0 and 1")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        self.model, self.labels, self.config, self.metadata = load_gguf_model(
            model_path, self.device
        )
        self.tokenizer = LionAITokenizer()
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.temperature = temperature

    @staticmethod
    def _validate_record(record: dict, index: int) -> tuple[str, str, str]:
        if not isinstance(record, dict):
            raise ValueError(f"Batch item {index} must be a JSON object.")
        if set(record) != {"topic", "filename"}:
            raise ValueError(
                f"Batch item {index} must contain exactly 'topic' and 'filename'."
            )
        topic, filename = record["topic"], record["filename"]
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"Batch item {index}: topic must be a non-empty string.")
        if (not isinstance(filename, str) or not filename.strip()
                or "/" in filename or "\\" in filename or "\x00" in filename):
            raise ValueError(f"Batch item {index}: filename must be a plain filename, not a path.")
        extension = Path(filename).suffix.casefold().lstrip(".")
        return topic, filename, extension

    def predict_batch(self, records: list[dict], batch_size: int = 128) -> list[dict]:
        if not isinstance(records, list):
            raise TypeError("records must be a list of metadata objects.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or not 1 <= batch_size <= 1024:
            raise ValueError("batch_size must be an integer between 1 and 1024.")
        validated = [
            self._validate_record(record, index)
            for index, record in enumerate(records)
        ]
        results: list[dict] = []
        model_name = str(self.metadata.get(
            "lionai.model",
            self.metadata.get("general.name", "lionai-scratch-transformer-custom"),
        ))
        for offset in range(0, len(validated), batch_size):
            chunk = validated[offset:offset + batch_size]
            token_rows = [
                {"topic": topic, "filename": filename, "extension": extension}
                for topic, filename, extension in chunk
            ]
            ids, mask = self.tokenizer.batch(
                token_rows, max_length=self.config.max_length
            )
            with torch.inference_mode():
                logits = self.model(ids.to(self.device), mask.to(self.device))
                probabilities = torch.softmax(logits / self.temperature, dim=-1)
                top_probs, top_ids = probabilities.topk(k=2, dim=-1)
            for row_index, (topic, filename, extension) in enumerate(chunk):
                confidence = float(top_probs[row_index, 0].item())
                margin = float(
                    (top_probs[row_index, 0] - top_probs[row_index, 1]).item()
                )
                raw_predicted = self.labels[int(top_ids[row_index, 0].item())]
                rule_category = extension_rule(filename)
                predicted = rule_category or raw_predicted
                rule_override = (
                    rule_category is not None and rule_category != raw_predicted
                )
                needs_review = (
                    confidence < self.confidence_threshold
                    or margin < self.margin_threshold
                    or raw_predicted == "Other/Review"
                    or rule_override
                )
                results.append({
                    "model": model_name,
                    "format": "GGUF v3 / F32",
                    "topic": topic,
                    "filename": filename,
                    "extension": extension,
                    "predicted_category": "Other/Review" if needs_review else predicted,
                    "suggested_category": predicted,
                    "raw_top_category": raw_predicted,
                    "rule_category": rule_category,
                    "decision_source": "extension_rule" if rule_category is not None else "transformer",
                    "rule_override": rule_override,
                    "confidence": round(confidence, 6),
                    "confidence_margin": round(margin, 6),
                    "needs_review": needs_review,
                    "device": str(self.device),
                    "safety": {
                        "metadata_only": True,
                        "file_contents_read": False,
                        "filesystem_modified": False,
                        "path_generation_authority": False,
                        "preview_only": True,
                    },
                })
        return results

    def predict(self, topic: str, filename: str) -> dict:
        return self.predict_batch([{"topic": topic, "filename": filename}])[0]


def predict_gguf_batch(records: list[dict], model_path: str | Path,
                       confidence_threshold: float = 0.75,
                       margin_threshold: float = 0.10,
                       temperature: float = 1.0, device: str = "auto",
                       batch_size: int = 128) -> list[dict]:
    """Load the model once and classify a list of metadata-only records."""
    session = LionAIGGUFSession(
        model_path, confidence_threshold, margin_threshold, temperature, device
    )
    return session.predict_batch(records, batch_size=batch_size)


def predict_gguf(topic: str, filename: str, model_path: str | Path,
                 confidence_threshold: float = 0.75,
                 margin_threshold: float = 0.10,
                 temperature: float = 1.0,
                 device: str = "auto") -> dict:
    """Backward-compatible single-record API backed by the batch runtime."""
    session = LionAIGGUFSession(
        model_path, confidence_threshold, margin_threshold, temperature, device
    )
    return session.predict(topic, filename)


def _iter_jsonl_batches(input_path: Path, batch_size: int):
    batch = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(line.encode("utf-8")) > 1_048_576:
                raise ValueError(f"Input line {line_number} exceeds the 1 MiB limit.")
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on input line {line_number}.") from exc
            batch.append(record)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a LionAI custom GGUF v3 model")
    parser.add_argument("--topic", help="Topic for one metadata record")
    parser.add_argument("--filename", help="Plain filename for one metadata record")
    parser.add_argument("--input-jsonl", help="JSONL input; each row must contain topic and filename only")
    parser.add_argument("--output-jsonl", help="Write batch predictions as JSONL (default: stdout)")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size (1..1024)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--margin-threshold", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    single_mode = args.topic is not None or args.filename is not None
    if args.input_jsonl and single_mode:
        parser.error("Use either --input-jsonl or --topic/--filename, not both.")
    if bool(args.topic) != bool(args.filename):
        parser.error("--topic and --filename must be supplied together.")
    if not args.input_jsonl and not single_mode:
        parser.error("Supply --input-jsonl or both --topic and --filename.")
    if not 1 <= args.batch_size <= 1024:
        parser.error("--batch-size must be between 1 and 1024.")

    session = LionAIGGUFSession(
        args.model, args.confidence_threshold, args.margin_threshold,
        args.temperature, args.device
    )
    if single_mode:
        print(json.dumps(session.predict(args.topic, args.filename), ensure_ascii=False, indent=2))
        return 0

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None and input_path.resolve() == output_path.resolve():
        parser.error("Input and output JSONL paths must be different.")
    output_handle = None
    temporary_path = None
    try:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = output_path.with_name(output_path.name + ".tmp")
            output_handle = temporary_path.open("w", encoding="utf-8", newline="\n")
        for records in _iter_jsonl_batches(input_path, args.batch_size):
            results = session.predict_batch(records, batch_size=args.batch_size)
            for result in results:
                line = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                if output_handle is None:
                    print(line)
                else:
                    output_handle.write(line + "\n")
        if output_handle is not None:
            output_handle.close()
            temporary_path.replace(output_path)
    except Exception:
        if output_handle is not None and not output_handle.closed:
            output_handle.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
