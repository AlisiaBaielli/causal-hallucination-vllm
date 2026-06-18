"""Online grounding score (GS) computation.

GS = 1 - sum_{i in H_EIC} (EIC_i / sum(EIC)) * H_hat_i(t)

where H_hat_i is the normalised entropy of head i's attention over image
tokens. Low GS = diffuse attention = hallucination risk; high GS = grounded.
"""
from __future__ import annotations
import math
import torch

def normalized_attention_entropy(attn_over_image: torch.Tensor,
                                 num_image_tokens: int,
                                 eps: float = 1e-12) -> torch.Tensor:
    """Per-head normalised entropy H_hat in [0, 1].

    Args:
        attn_over_image: (num_heads, num_image_tokens) renormalised attention.
        num_image_tokens: denominator for normalisation (log N_img).
        eps: numerical floor.
    """
    p = attn_over_image.clamp_min(eps)
    H = -(p * p.log()).sum(dim=-1)
    H_hat = H / math.log(max(num_image_tokens, 2))
    return H_hat.clamp(0.0, 1.0)

def grounding_score(per_head_entropy: torch.Tensor,
                    eic_weights: torch.Tensor) -> torch.Tensor:
    """EIC-weighted grounding score in [0, 1] from a full per-head entropy vector.

    Args:
        per_head_entropy: (num_heads,) normalised entropy per head.
        eic_weights:      (num_heads,) EIC values; heads with EIC=0 are ignored.

    Returns:
        Scalar GS in [0, 1]. ~1 = focused (grounded), ~0 = diffuse (risk).
    """
    mask = eic_weights > 0
    if not mask.any():
        return torch.tensor(1.0, device=per_head_entropy.device)
    w = eic_weights[mask]
    w = w / w.sum()
    h = per_head_entropy[mask]
    return (1.0 - (w * h).sum()).clamp(0.0, 1.0)

def compute_grounding_score_batched(img_attn: torch.Tensor,
                                    c_weights_filtered: torch.Tensor,
                                    eps: float = 1e-10) -> tuple[float, float]:
    """Batched GS computation used inside the decode-time hook.

    Args:
        img_attn: (B, n_high_c, img_len) raw attention weights over image tokens
                  for the high-EIC heads only. Will be re-normalised here.
        c_weights_filtered: (n_high_c,) EIC weights for the same heads,
                            already normalised so they sum to 1.
        eps: numerical floor.

    Returns:
        (grounding_score, mean_normalised_entropy) as plain Python floats.
    """

    img_attn_sum = img_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    img_attn_norm = img_attn / img_attn_sum

    entropy = -(img_attn_norm * (img_attn_norm + eps).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(img_attn.shape[-1]),
                                         device=img_attn.device))
    norm_entropy = entropy / max_entropy.clamp(min=1e-8)

    c_w = c_weights_filtered.to(norm_entropy.device)
    mean_norm_entropy = (norm_entropy * c_w.unsqueeze(0)).sum(dim=-1).mean().item()

    return 1.0 - mean_norm_entropy, mean_norm_entropy
