"""Apply z-score pre-filter to raw C-scores.

  1. z_{l,i} = (mu_{l,i} - mu_bar_l) / Std_i(mu_{l,i})
  2. Heads with z >= 0 are zeroed (above-average TVER = text-heavy)
  3. C = (1 - minmax(mu)) * (1 - minmax(var)) on remaining heads only
"""
from __future__ import annotations
import argparse
import torch

def minmax_norm(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (v - v.min()) / (v.max() - v.min() + eps)

def apply_zscore_filter(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    mean_f = mean.float()
    var_f = var.float()
    z = (mean_f - mean_f.mean()) / (mean_f.std() + 1e-12)
    mask = z < 0

    C = torch.zeros_like(mean_f)
    if mask.sum() > 1:
        C[mask] = (1.0 - minmax_norm(mean_f[mask])) * (1.0 - minmax_norm(var_f[mask]))
    elif mask.sum() == 1:
        C[mask] = 1.0
    return C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw .pt from compute_scores.py")
    ap.add_argument("--output", required=True, help="Output .pt with z-score filtered C")
    args = ap.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    mean = payload["mean"]
    var = payload["var"]

    C_filtered = apply_zscore_filter(mean, var)
    payload["C"] = C_filtered

    n_kept = int((C_filtered > 0).sum())
    n_total = len(C_filtered)
    print(f"Z-score filter: {n_kept}/{n_total} heads retained")
    print(f"Top-5 heads: {torch.argsort(C_filtered, descending=True)[:5].tolist()}")

    torch.save(payload, args.output)
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
