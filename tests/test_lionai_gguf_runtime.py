from __future__ import annotations

import struct
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from lionai_scratch import LionAIConfig, LionAITokenizer, build_model
from training.lionai_gguf_runtime import (
    GGUFError, GGUFReader, LionAIGGUFSession, load_gguf_model, main as runtime_main, predict_gguf,
)
from training.metadata_rules import extension_rule


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _meta_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _meta_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _meta_float(key: str, value: float) -> bytes:
    return _string(key) + struct.pack("<If", 6, value)


def _meta_string_array(key: str, values: list[str]) -> bytes:
    return (_string(key) + struct.pack("<IIQ", 9, 8, len(values))
            + b"".join(_string(value) for value in values))


def _write_test_gguf(path: Path, model, labels: list[str], config: LionAIConfig):
    metadata = [
        _meta_string("general.architecture", "lionai"),
        _meta_string("lionai.initialized_from", "random"),
        _meta_string("lionai.tokenizer", "utf8-byte-v1"),
        _meta_u32("general.alignment", 32),
        _meta_u32("lionai.vocab_size", config.vocab_size),
        _meta_u32("lionai.d_model", config.d_model),
        _meta_u32("lionai.n_heads", config.n_heads),
        _meta_u32("lionai.n_layers", config.n_layers),
        _meta_u32("lionai.feedforward", config.feedforward),
        _meta_u32("lionai.max_length", config.max_length),
        _meta_float("lionai.dropout", config.dropout),
        _meta_u32("lionai.num_classes", config.num_classes),
        _meta_string_array("lionai.labels", labels),
    ]
    tensors = []
    for name, value in model.state_dict().items():
        raw = bytes(value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        tensors.append(("lionai." + name, tuple(reversed(value.shape)), raw))
    tensor_infos = bytearray()
    tensor_data = bytearray()
    for name, shape, raw in tensors:
        tensor_data.extend(b"\0" * ((-len(tensor_data)) % 32))
        offset = len(tensor_data)
        tensor_infos.extend(_string(name))
        tensor_infos.extend(struct.pack("<I", len(shape)))
        tensor_infos.extend(struct.pack("<" + "Q" * len(shape), *shape))
        tensor_infos.extend(struct.pack("<IQ", 0, offset))
        tensor_data.extend(raw)
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    header.extend(b"".join(metadata))
    header.extend(tensor_infos)
    header.extend(b"\0" * ((-len(header)) % 32))
    path.write_bytes(header + tensor_data)


class LionAIGGUFTests(unittest.TestCase):
    def test_round_trip_weights_preserve_and_inference_runs(self):
        torch.manual_seed(1234)
        config = LionAIConfig(dropout=0.0)
        original = build_model(config).eval()
        with torch.no_grad():
            for parameter in original.parameters():
                parameter.fill_(0.01)
        labels = [f"class-{i}" for i in range(config.num_classes)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, original, labels, config)
            restored, restored_labels, _, _ = load_gguf_model(path)
            ids, mask = LionAITokenizer().batch([{
                "topic": "photos", "filename": "image.jpg", "extension": "jpg"
            }])
            with torch.inference_mode():
                expected = original(ids, mask)
                actual = restored(ids, mask)
            self.assertEqual(labels, restored_labels)
            for name, tensor in original.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(expected).all().item())
            self.assertTrue(torch.isfinite(actual).all().item())
            self.assertEqual(actual.shape, expected.shape)

    def test_predict_returns_metadata_only_safety_contract(self):
        torch.manual_seed(1234)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.01)
        labels = [f"class-{i}" for i in range(config.num_classes)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, model, labels, config)
            result = predict_gguf("documents", "notes.txt", path, device="cpu")
        self.assertEqual(result["format"], "GGUF v3 / F32")
        self.assertTrue(result["safety"]["metadata_only"])
        self.assertFalse(result["safety"]["file_contents_read"])
        self.assertFalse(result["safety"]["filesystem_modified"])

    def test_high_precision_extension_rules_and_ambiguous_suffixes(self):
        self.assertEqual(extension_rule("summer.jpg"), "Images")
        self.assertEqual(extension_rule("PHOTO.PNG"), "Images")
        self.assertEqual(extension_rule("recording.mp3"), "Audio")
        self.assertIsNone(extension_rule("settings.json"))
        self.assertIsNone(extension_rule("schema.sql"))
        self.assertIsNone(extension_rule("stream.ts"))
        self.assertIsNone(extension_rule("folder\\\\summer.jpg"))

    def test_prediction_exposes_rule_override_for_image_extension(self):
        torch.manual_seed(1234)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, model, [f"class-{i}" for i in range(config.num_classes)], config)
            result = predict_gguf("photography", "summer.jpg", path, device="cpu")
        self.assertEqual(result["predicted_category"], "Other/Review")
        self.assertEqual(result["suggested_category"], "Images")
        self.assertEqual(result["rule_category"], "Images")
        self.assertEqual(result["decision_source"], "extension_rule")
        self.assertTrue(result["rule_override"])
        self.assertTrue(result["needs_review"])

    def test_batch_predictions_match_single_predictions_and_keep_order(self):
        torch.manual_seed(4321)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        labels = [f"class-{i}" for i in range(config.num_classes)]
        records = [
            {"topic": "photos", "filename": "summer.jpg"},
            {"topic": "documents", "filename": "notes.txt"},
            {"topic": "audio", "filename": "voice.mp3"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, model, labels, config)
            session = LionAIGGUFSession(path, device="cpu")
            batch_results = session.predict_batch(records, batch_size=2)
            single_results = [
                session.predict(row["topic"], row["filename"]) for row in records
            ]
        self.assertEqual(len(batch_results), len(records))
        self.assertEqual(batch_results, single_results)
        self.assertEqual(
            [row["filename"] for row in batch_results],
            ["summer.jpg", "notes.txt", "voice.mp3"],
        )

    def test_batch_rejects_paths_and_extra_metadata_fields(self):
        torch.manual_seed(4321)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        labels = [f"class-{i}" for i in range(config.num_classes)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, model, labels, config)
            session = LionAIGGUFSession(path, device="cpu")
            with self.assertRaises(ValueError):
                session.predict_batch([{"topic": "docs", "filename": "folder\\\\note.txt"}])
            with self.assertRaises(ValueError):
                session.predict_batch([{
                    "topic": "docs", "filename": "note.txt", "content": "must not be accepted"
                }])
            with self.assertRaises(ValueError):
                session.predict_batch([{"topic": "", "filename": "note.txt"}])

    def test_batch_size_is_bounded_and_empty_batch_is_supported(self):
        torch.manual_seed(4321)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        labels = [f"class-{i}" for i in range(config.num_classes)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, model, labels, config)
            session = LionAIGGUFSession(path, device="cpu")
            self.assertEqual(session.predict_batch([]), [])
            with self.assertRaises(ValueError):
                session.predict_batch([], batch_size=0)
            with self.assertRaises(ValueError):
                session.predict_batch([], batch_size=1025)

    def test_jsonl_cli_writes_ordered_batch_results_atomically(self):
        torch.manual_seed(9876)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        labels = [f"class-{i}" for i in range(config.num_classes)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "fixture.gguf"
            input_path = root / "metadata.jsonl"
            output_path = root / "predictions.jsonl"
            _write_test_gguf(model_path, model, labels, config)
            input_path.write_text(
                "\n".join([
                    '{"topic":"photos","filename":"one.jpg"}',
                    '{"topic":"documents","filename":"two.txt"}',
                    '{"topic":"audio","filename":"three.mp3"}',
                ]) + "\n",
                encoding="utf-8",
            )
            with patch("sys.argv", [
                "lionai_gguf_runtime.py", "--model", str(model_path),
                "--input-jsonl", str(input_path), "--output-jsonl", str(output_path),
                "--batch-size", "2", "--device", "cpu",
            ]):
                self.assertEqual(runtime_main(), 0)
            rows = [
                __import__("json").loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["filename"] for row in rows],
                ["one.jpg", "two.txt", "three.mp3"],
            )
            self.assertFalse(output_path.with_name(output_path.name + ".tmp").exists())

    def test_rejects_non_gguf_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            path.write_bytes(b"not a model")
            with self.assertRaises(GGUFError):
                GGUFReader(path)

    def test_rejects_path_instead_of_plain_filename(self):
        with self.assertRaises(ValueError):
            predict_gguf("documents", "folder/notes.txt", "unused.gguf")


if __name__ == "__main__":
    unittest.main()
