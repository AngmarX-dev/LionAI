"""Pure-Python class-aware sampling weights for metadata training."""
from __future__ import annotations

from collections import Counter
import math


def compute_sample_weights(rows: list[dict[str, str]], power: float = 0.5) -> list[float]:
    """Return per-row weights with a tunable, bounded class-frequency correction.

    power=0 gives ordinary uniform-row sampling; power=1 gives equal expected
    class mass. The default square-root correction is deliberately moderate.
    """
    if not math.isfinite(power) or not 0.0 <= power <= 1.0:
        raise ValueError("sampling power must be finite and between 0 and 1")
    if not rows:
        raise ValueError("cannot compute sample weights for an empty dataset")
    labels = [row.get("label") for row in rows]
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("every row must contain a non-empty string label")
    counts = Counter(labels)
    return [1.0 / (counts[label] ** power) for label in labels]
