"""Model factory and registry for ProAgg variants.

Currently scoped to ProAgg only, but built on a generic Registry so it can
expand to other components later if needed.
"""

from __future__ import annotations

from typing import Any, Callable, Type

from src.utils.registry import Registry


PROAGG_MODEL_REGISTRY: Registry[Type] = Registry("proagg_model")


def register_proagg_model(model_name: str) -> Callable[[Type], Type]:
    return PROAGG_MODEL_REGISTRY.register(model_name)


def build_proagg_model(cfg: Any):
    """Build a ProAgg model from config.

    Config contract (backwards-compatible):
      - cfg.model.model_name (optional): registry key
      - if missing, defaults to 'proagg_mlp_v1'
    """
    # Ensure all default model variants are registered.
    # New variants can live in src/models/ProAgg.py (or other modules) as long
    # as they import `register_proagg_model` and register themselves.
    import src.models.ProAgg  # noqa: F401

    model_name = None
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None:
        if hasattr(model_cfg, "get"):
            model_name = model_cfg.get("model_name", None)
        else:
            model_name = getattr(model_cfg, "model_name", None)

    if not model_name:
        model_name = "proagg_mlp_v1"

    model_cls = PROAGG_MODEL_REGISTRY.require(model_name)
    return model_cls(cfg)

