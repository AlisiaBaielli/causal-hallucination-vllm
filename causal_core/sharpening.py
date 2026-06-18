"""Adaptive logit sharpening based on the grounding score.

    tau_eff(t) = max(1 - alpha * (1 - GS(t)), tau_floor)
    z_tilde   = z / tau_eff(t)
"""
from __future__ import annotations
import torch

def effective_temperature(grounding_score: float,
                          alpha: float,
                          tau_floor: float = 0.3) -> float:
    """Compute tau_eff. Scalar in, scalar out."""
    return max(1.0 - alpha * (1.0 - grounding_score), tau_floor)

def sharpen_logits(logits: torch.Tensor,
                   grounding_score: float,
                   alpha: float,
                   tau_floor: float = 0.3,
                   noop_threshold: float = 0.99) -> torch.Tensor:
    """Apply temperature-based logit sharpening.

    When tau_eff is essentially 1.0 (grounded), the logits are returned
    unchanged (no division). Otherwise: ``logits / tau_eff``.
    """
    tau = effective_temperature(grounding_score, alpha, tau_floor)
    if tau < noop_threshold:
        return logits / tau
    return logits
