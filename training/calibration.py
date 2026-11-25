"""Temperature scaling helpers for LionAI confidence calibration."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor,
                    max_iter: int = 100) -> float:
    """Fit one positive scalar temperature on a dedicated calibration split."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.numel():
        raise ValueError("logits must be [N, C] and targets must be [N]")
    if logits.shape[0] < 2 or logits.shape[1] < 2:
        raise ValueError("at least two calibration examples and classes are required")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError("calibration tensors must contain only finite values")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("targets must contain integer class indices")
    if int(targets.min()) < 0 or int(targets.max()) >= logits.shape[1]:
        raise ValueError("target class index is outside the logits dimension")
    logits = logits.detach().to(dtype=torch.float64, device="cpu")
    targets = targets.detach().to(dtype=torch.long, device="cpu")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def write_calibration(path: Path, temperature: float, labels: list[str],
                      examples: int) -> None:
    if not math.isfinite(temperature) or not 0.05 <= temperature <= 20.0:
        raise ValueError("temperature must be finite and between 0.05 and 20")
    if examples < 2 or not labels or not all(isinstance(label, str) for label in labels):
        raise ValueError("calibration metadata is invalid")
    payload = {
        "format": "lionai-temperature-calibration-v1",
        "method": "scalar-temperature-scaling",
        "temperature": temperature,
        "labels": labels,
        "calibration_examples": examples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_temperature(path: Path, labels: list[str]) -> float:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Unable to read calibration file") from exc
    if not isinstance(payload, dict) or payload.get("format") != "lionai-temperature-calibration-v1":
        raise ValueError("Unsupported calibration file format")
    temperature = payload.get("temperature")
    if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)
            or not math.isfinite(temperature) or not 0.05 <= temperature <= 20.0):
        raise ValueError("Invalid calibration temperature")
    if payload.get("labels") != labels:
        raise ValueError("Calibration labels do not match checkpoint labels")
    return float(temperature)
