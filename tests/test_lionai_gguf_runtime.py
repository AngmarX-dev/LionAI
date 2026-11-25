from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import torch

from lionai_scratch import LionAIConfig, LionAITokenizer, build_model
from training.lionai_gguf_runtime import GGUFError, GGUFReader, load_gguf_model, predict_gguf


def _string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _meta_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _meta_u32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


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
        _meta_u32("lionai.num_classes", config.num_classes),
        _meta_string_array("lionai.labels", labels),
    ]
    tensors = []
    for name, value in model.state_dict().items():
        raw = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False).tobytes()
        tensors.append(("lionai." + name, tuple(reversed(value.shape)), raw))
    tensor_infos = bytearray()
    tensor_data = bytearray()
    for name, shape, raw in tensors:
        tensor_data.extend(b"\\0" * ((-len(tensor_data)) % 32))
        offset = len(tensor_data)
        tensor_infos.extend(_string(name))
        tensor_infos.extend(struct.pack("<I", len(shape)))
        tensor_infos.extend(struct.pack("<" + "Q" * len(shape), *shape))
        tensor_infos.extend(struct.pack("<IQ", 0, offset))
        tensor_data.extend(raw)
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    header.extend(b"".join(metadata))
    header.extend(tensor_infos)
    header.extend(b"\\0" * ((-len(header)) % 32))
    path.write_bytes(header + tensor_data)


class LionAIGGUFTests(unittest.TestCase):
    def test_round_trip_logits_match_original_model(self):
        torch.manual_seed(1234)
        config = LionAIConfig(dropout=0.0)
        original = build_model(config).eval()
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
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_predict_returns_metadata_only_safety_contract(self):
        torch.manual_seed(1234)
        config = LionAIConfig(dropout=0.0)
        model = build_model(config).eval()
        labels = [f"class-{i}" for i in range(config.num_classes)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.gguf"
            _write_test_gguf(path, model, labels, config)
            result = predict_gguf("documents", "notes.txt", path, device="cpu")
        self.assertEqual(result["format"], "GGUF v3 / F32")
        self.assertTrue(result["safety"]["metadata_only"])
        self.assertFalse(result["safety"]["file_contents_read"])
        self.assertFalse(result["safety"]["filesystem_modified"])

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
