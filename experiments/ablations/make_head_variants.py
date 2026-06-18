"""Build the head-set variants used in the head-selection ablation (Table 4.5).
"""
import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Calibrated EIC .pt (key 'C').")
    ap.add_argument("--layer", type=int, default=1, help="Layer index if 'C' is 2-D [L,H].")
    ap.add_argument("--out_dir", default="scores")
    ap.add_argument("--prefix", default="llava_L1")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    C = payload["C"] if isinstance(payload, dict) else payload
    C = C.float()
    if C.dim() == 2:
        C = C[args.layer]
    H = C.numel()
    selected = C > 1e-6
    k = int(selected.sum())
    print(f"EIC set: {k}/{H} heads active")

    g = torch.Generator().manual_seed(args.seed)
    rand_idx = torch.randperm(H, generator=g)[:k]

    variants = {
        "all":      torch.ones(H),
        "inverted": (~selected).float(),
        "random":   torch.zeros(H).index_fill_(0, rand_idx, 1.0),
    }
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    for name, Cv in variants.items():
        out = os.path.join(args.out_dir, f"{args.prefix}_{name}.pt")
        torch.save({"C": Cv}, out)
        print(f"  saved {out}  ({int((Cv > 0).sum())} active heads)")

if __name__ == "__main__":
    main()
