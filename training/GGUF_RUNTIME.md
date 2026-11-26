# LionAI GGUF runtime

LionAI's `lionai` architecture is not implemented by stock llama.cpp/LM Studio. This project includes a small Python + PyTorch runtime that reads the project's GGUF v3 F32 tensors directly and executes the exact custom Transformer classifier.

## Install

```bash
python -m pip install -r training/requirements-scratch.txt
```

## Run a prediction

```bash
python training/lionai_gguf_runtime.py \
  --model ./LionAI-scratch-v2.gguf \
  --topic "photography" \
  --filename "summer.jpg" \
  --device auto
```

Use `--device cpu` to force CPU, or `--device cuda` when a CUDA-enabled PyTorch build and compatible NVIDIA driver are installed. The runtime uses CUDA automatically when available.

The program accepts a plain filename, not a path. It uses only the topic, filename, and extension supplied on the command line. It never opens the named file, reads file contents, creates folders, moves files, deletes files, or overwrites anything. Low-confidence predictions are returned as `Other/Review` and must be reviewed by the user.

## Format support

This runtime currently supports LionAI's own GGUF v3 architecture with F32 tensors. It validates metadata, tensor names, shapes, offsets, and file bounds before loading. It does not claim compatibility with generic GGUF models, llama.cpp, or LM Studio, and it intentionally rejects quantized tensor types until those are implemented and tested.

## Tests

```bash
python -m unittest discover -s tests -p 'test_lionai_gguf_runtime.py' -v
```

The tests write a tiny temporary GGUF fixture and verify that inference logits after GGUF reload exactly match the original PyTorch model.


## Hybrid decision layer

The runtime applies a conservative extension rule only for high-precision, unambiguous suffixes (for example, `.jpg` and `.png`). Ambiguous suffixes such as `.json`, `.sql`, `.html`, and `.ts` remain model-only. The JSON result preserves `raw_top_category`, exposes `suggested_category`, and records `decision_source` and `rule_override`. If a rule conflicts with the Transformer, the result remains `Other/Review` with `needs_review=true`; the suggestion is not silently treated as a confirmed classification.

## Export from a trained checkpoint

After training, export and verify the custom GGUF file:

```bash
python training/export_gguf.py --checkpoint artifacts/lionai-scratch.pt --output artifacts/LionAI-scratch-v2.gguf
```

The exporter validates the GGUF header and metadata, reloads all tensors, requires bit-exact tensor round-trip equality, and checks that a fixed inference input produces exactly matching PyTorch and GGUF-runtime logits. The generated JSON sidecar records the SHA-256 digest and validation status.

A successful serialization check does not certify model quality. Promote a checkpoint only after repository-disjoint evaluation meets the project's acceptance criteria. The v2 benchmark currently remains below the earlier independent-evaluation baseline, so its GGUF is a technical candidate rather than a production release.

## Batch inference (JSONL)

The runtime loads the GGUF model once per process, then reuses it across chunks. Each input line must be a JSON object containing exactly `topic` and `filename`; the filename must be a plain name, not a path. File contents are never opened.

Example input (`metadata.jsonl`):

```json
{"topic":"photography","filename":"summer.jpg"}
{"topic":"documents","filename":"notes.txt"}
{"topic":"audio","filename":"voice.mp3"}
```

Run batched inference and write one JSON result per input row:

```bash
python training/lionai_gguf_runtime.py \
  --model ./LionAI-scratch-v2.gguf \
  --input-jsonl ./metadata.jsonl \
  --output-jsonl ./predictions.jsonl \
  --batch-size 128 \
  --device auto
```

Use `--batch-size 1` for effectively sequential inference or increase it (maximum 1024) when memory permits. Output order matches input order. The output is written to a temporary sibling file and renamed only after the complete run succeeds, so a failed run does not leave a partial output at the requested destination. Input and output paths must differ. The model is loaded once, not once per row.

For Python callers, reuse `LionAIGGUFSession` and call `predict_batch(records, batch_size=128)`. The backward-compatible `predict_gguf(...)` API remains available.
