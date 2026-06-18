"""Attach offline EIC scores to ONLY's contrastive (CD) branch head selection."""
from __future__ import annotations

import os
from typing import Optional

import torch

def _resolve_attn(model, layer_idx: int):
    """Resolve the decoder self-attention at ``layer_idx`` across architectures.

    Handles LLaVA (``model.model.layers``) and the Qwen3-VL / InternVL HF
    layouts (``model.model.language_model.layers`` or ``model.language_model``).
    """
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer_idx].self_attn
    if hasattr(model, "language_model"):
        return model.language_model.layers[layer_idx].self_attn
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx].self_attn
    raise ValueError("Could not resolve decoder self-attention for this model")

def inject_eic_for_only(
    *,
    model,
    scores_path: str,
    layer_index: Optional[int] = None,
    pure_eic: bool = False,
    a: float = 3.0,
    b: float = 1.0,
    require_match: bool = True,
) -> int:
    """
    Attach offline C-scores for ONLY c_head_select at ``layer_index``.

    When ``pure_eic=True`` (ONLY+EIC ablation), high-EIC heads (C>0) are zeroed
    in the CD branch instead of the default ratio-lambda*C rule.
    """
    if not os.path.exists(scores_path):
        raise FileNotFoundError(scores_path)

    payload = torch.load(scores_path, map_location="cpu")
    if not isinstance(payload, dict) or "C" not in payload:
        raise ValueError("scores file must be a dict with key 'C'")

    chosen_layer = int(payload.get("chosen_layer", layer_index if layer_index is not None else 0))
    layer_idx = chosen_layer if layer_index is None else int(layer_index)

    if require_match and layer_idx != chosen_layer:
        raise ValueError(
            f"C-score layer mismatch: calibrated at {chosen_layer}, target {layer_idx}"
        )

    c = payload["C"]
    if not torch.is_tensor(c):
        c = torch.tensor(c)
    c = torch.nan_to_num(c.float())

    device = next(model.parameters()).device
    attn = _resolve_attn(model, layer_idx)
    attn._causal_C = c.to(device=device)
    attn._causal_a = a
    attn._causal_b = b
    attn._causal_source_layer = chosen_layer
    attn._use_c_head_select = True
    attn._use_c_soft_weight = False
    attn._use_pure_eic = bool(pure_eic)

    n_eic = int((c > 0).sum().item())
    mode = "pure EIC (C>0)" if pure_eic else "c_head_select"
    print(
        f"[ONLY+EIC] layer={layer_idx} mode={mode} "
        f"high-EIC heads={n_eic}/{c.numel()} from {scores_path}"
    )
    return layer_idx
