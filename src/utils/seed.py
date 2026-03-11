from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    """Best-effort seeding across random / numpy / torch.

    Note: On GPU, exact bitwise reproducibility may still depend on hardware,
    driver/CUDA versions, and operator determinism. We only set RNG seeds here.
    """
    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate while preserving first occurrence order."""
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out
