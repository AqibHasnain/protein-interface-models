"""Process-wide seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # warn_only: some MPS ops have no deterministic implementation and would otherwise raise.
    torch.use_deterministic_algorithms(True, warn_only=True)
