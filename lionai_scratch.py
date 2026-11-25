#!/usr/bin/env python3
"""LionAI's small task-specific Transformer, initialized from random weights."""
from __future__ import annotations

from dataclasses import dataclass

PAD_ID = 0
BYTE_OFFSET = 1
SEP_ID = 257
VOCAB_SIZE = 258


class LionAITokenizer:
    """Deterministic UTF-8 byte tokenizer; no pretrained vocabulary or weights."""

    def encode(self, topic: str, filename: str, extension: str, max_length: int = 256) -> list[int]:
        if not all(isinstance(x, str) for x in (topic, filename, extension)):
            raise TypeError("topic, filename and extension must be strings")
        payload = topic.encode("utf-8") + bytes([10]) + filename.encode("utf-8") + bytes([SEP_ID - BYTE_OFFSET]) + extension.encode("utf-8")
        return [min(byte + BYTE_OFFSET, VOCAB_SIZE - 1) for byte in payload[:max_length]]

    def batch(self, rows: list[dict[str, str]], max_length: int = 256):
        import torch
        encoded = [self.encode(row["topic"], row["filename"], row.get("extension", ""), max_length) for row in rows]
        width = max((len(item) for item in encoded), default=1)
        ids = torch.full((len(encoded), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.bool)
        for index, item in enumerate(encoded):
            ids[index, :len(item)] = torch.tensor(item, dtype=torch.long)
            mask[index, :len(item)] = True
        return ids, mask


@dataclass(frozen=True)
class LionAIConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 3
    feedforward: int = 512
    max_length: int = 256
    dropout: float = 0.1
    num_classes: int = 13


def build_model(config: LionAIConfig | None = None):
    """Build a randomly initialized encoder classifier; no model download occurs."""
    import torch
    from torch import nn

    cfg = config or LionAIConfig()
    if cfg.d_model % cfg.n_heads:
        raise ValueError("d_model must be divisible by n_heads")

    class LionAITransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD_ID)
            self.position_embedding = nn.Embedding(cfg.max_length, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=cfg.feedforward, dropout=cfg.dropout,
                batch_first=True, norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, cfg.num_classes))

        def forward(self, input_ids, attention_mask):
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError("input_ids and attention_mask must be matching [batch, sequence] tensors")
            if input_ids.shape[1] > cfg.max_length:
                raise ValueError("sequence exceeds configured max_length")
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask.bool())
            weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.classifier(self.norm(pooled))

    return LionAITransformer()
