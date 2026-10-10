"""Loading and caching of ESM-2 backbones."""

from __future__ import annotations

import torch
from transformers import EsmForMaskedLM, EsmTokenizer

ESM_CHECKPOINTS: dict[str, str] = {
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}

_CACHE: dict[tuple[str, str, torch.dtype], tuple[EsmForMaskedLM, EsmTokenizer]] = {}


def resolve_device(prefer: str | None = None) -> torch.device:
    """Return `prefer` if given, else mps when available, else cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_checkpoint(name: str) -> str:
    return ESM_CHECKPOINTS.get(name, name)


def load_esm(
    name: str,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[EsmForMaskedLM, EsmTokenizer]:
    """Load an ESM-2 masked LM in eval mode with gradients disabled.

    `name` is a short alias (35M, 150M, 650M) or a full HuggingFace id. Results are cached per
    (checkpoint, device, dtype) so repeated calls within a sweep are free.

    MPS has no float64 support and lacks kernels for some ops, so the default dtype is float32
    on every device. Scoring code upcasts logits to float32 before log_softmax.
    """
    dev = resolve_device(device) if not isinstance(device, torch.device) else device
    dt = dtype or torch.float32
    ckpt = resolve_checkpoint(name)
    key = (ckpt, str(dev), dt)
    if key not in _CACHE:
        tokenizer = EsmTokenizer.from_pretrained(ckpt)
        model = EsmForMaskedLM.from_pretrained(ckpt, torch_dtype=dt)
        model.eval().to(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        _CACHE[key] = (model, tokenizer)
    return _CACHE[key]
