"""TVER aggregation, EIC scoring, and intervention-layer selection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch

def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:

    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p.clamp_min(0.0)
    return -(p * (p + eps).log()).sum(dim=-1)

def tver_from_attn(
    attn: torch.Tensor,
    text_mask: torch.Tensor,
    vision_mask: torch.Tensor,
    eps: float = 1e-12,
    renormalize_within_subset: bool = True,
) -> torch.Tensor:
    """
    Paper Eq. 1: TVER_{l,i} = H(a^text_{l,i}) / (H(a^vis_{l,i}) + eps)

    attn: [B, H, KV] attention distribution for a single query token
    masks: [KV] or [B, KV] boolean
    returns: TVER [B, H]

    Note: Entropy H(p) requires p to be a proper probability distribution (sum to 1).
    We renormalize the attention subsets to ensure valid entropy computation.
    """
    assert attn.dim() == 3, f"Expected [B,H,KV], got {attn.shape}"
    B, H, KV = attn.shape
    device = attn.device

    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    text_mask = text_mask.to(device=device)
    vision_mask = vision_mask.to(device=device)

    if text_mask.dim() == 1:
        text_mask = text_mask.unsqueeze(0).expand(B, KV)
    if vision_mask.dim() == 1:
        vision_mask = vision_mask.unsqueeze(0).expand(B, KV)
    assert text_mask.shape == (B, KV), f"text_mask shape {text_mask.shape} != {(B, KV)}"
    assert vision_mask.shape == (B, KV), f"vision_mask shape {vision_mask.shape} != {(B, KV)}"

    aT = torch.stack([attn[b, :, text_mask[b]] for b in range(B)], dim=0)
    aV = torch.stack([attn[b, :, vision_mask[b]] for b in range(B)], dim=0)

    if aT.size(-1) == 0 or aV.size(-1) == 0:
        raise ValueError(
            f"Empty subset when computing TVER: text={aT.size(-1)} vision={aV.size(-1)}. "
            "Check your KV masks and multimodal tokenization."
        )

    if renormalize_within_subset:
        aT = aT / (aT.sum(dim=-1, keepdim=True).clamp_min(eps))
        aV = aV / (aV.sum(dim=-1, keepdim=True).clamp_min(eps))

    HT = entropy(aT, eps=eps)
    HV = entropy(aV, eps=eps)

    tver = HT / (HV.clamp_min(eps))
    return torch.nan_to_num(tver, nan=0.0, posinf=0.0, neginf=0.0)

def normalize_across_heads(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Min-max normalize per batch row across heads.

    x: [B, H] -> returns [B, H] in [0,1]
    """
    if x.dim() != 2:
        raise ValueError(f"Expected [B,H], got {tuple(x.shape)}")
    xmin = x.min(dim=1, keepdim=True).values
    xmax = x.max(dim=1, keepdim=True).values
    return (x - xmin) / (xmax - xmin + eps)

@dataclass
class RunningStats:

    n: int
    mean: torch.Tensor
    m2: torch.Tensor

    @classmethod
    def create(cls, num_heads: int, device="cpu"):
        return cls(n=0, mean=torch.zeros(num_heads, device=device), m2=torch.zeros(num_heads, device=device))

    def update(self, x: torch.Tensor):
        """
        x: [H] or [B,H] -> averaged over batch
        """
        if x.dim() == 2:
            x = x.mean(dim=0)
        x = x.detach()
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def finalize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n == 0:
            var = torch.zeros_like(self.mean)
        else:

            var = self.m2 / max(self.n, 1)
        return self.mean, var

def minmax_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vmin = v.min()
    vmax = v.max()
    return (v - vmin) / (vmax - vmin + eps)

def compute_C(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    mean,var: [H]
    C = (1 - norm(mean)) * (1 - norm(var))
    """
    mu = minmax_norm(mean)
    vv = minmax_norm(var)
    return (1.0 - mu) * (1.0 - vv)

def choose_intervention_layer(mu_by_layer: torch.Tensor, eps: float = 1e-12) -> int:
    """Select the intervention layer per the screenshot.

    mu_by_layer: [L, H] (means per head per layer)
    Returns: layer index ℓ̃ = argmax_ℓ Std_i(minmax(mu_{ℓ,*}))
    """
    if mu_by_layer.dim() != 2:
        raise ValueError(f"Expected [L,H], got {tuple(mu_by_layer.shape)}")
    mu_norm = torch.stack([minmax_norm(mu_by_layer[l], eps=eps) for l in range(mu_by_layer.size(0))], dim=0)
    stds = mu_norm.std(dim=1)
    return int(torch.argmax(stds).item())

