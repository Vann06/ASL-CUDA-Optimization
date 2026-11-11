"""Selección explícita y estricta de CPU/CUDA."""

from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Se solicitó CUDA, pero torch.cuda.is_available() es False. "
                "Instale la rueda CUDA en una computadora NVIDIA con driver compatible; "
                "no se usará CPU silenciosamente."
            )
        return torch.device("cuda")
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError("device debe ser 'auto', 'cpu' o 'cuda'")
