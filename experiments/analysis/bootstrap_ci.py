"""
Paired bootstrap 95% CI on the CHAIR delta (vanilla -> causal steering).
"""
import json, argparse, numpy as np
from pathlib import Path

RNG = np.random.default_rng(42)
B = 10000

def load_per_image(p):
    d = json.load(open(p))
    out = {}
    for s in d['sentences']:
        out[s['image_id']] = (int(s['metrics']['CHAIRs']),
                              float(s['metrics']['CHAIRi']))
    return out

def paired_delta_ci(van_vals, ste_vals, B=B, alpha=0.05):
    """Paired bootstrap on the difference of means."""
    assert len(van_vals) == len(ste_vals)
    diffs = ste_vals - van_vals
    n = len(diffs)
    boot = np.array([diffs[RNG.integers(0, n, n)].mean()
                     for _ in range(B)])
    med = float(np.median(boot))
    lo = float(np.percentile(boot, 100 * alpha/2))
    hi = float(np.percentile(boot, 100 * (1-alpha/2)))
    point = float(diffs.mean())
    return point, med, lo, hi, boot

def run(label, van_path, ste_path):
    van = load_per_image(van_path)
    ste = load_per_image(ste_path)
    common = sorted(set(van) & set(ste))
    print(f"\n══════════════════════════════════════════════════════════════")
    print(f"  {label}")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"n images (matched): {len(common)}")

    van_s = np.array([van[i][0] for i in common])
    ste_s = np.array([ste[i][0] for i in common])
    van_i = np.array([van[i][1] for i in common])
    ste_i = np.array([ste[i][1] for i in common])

    print(f"\nCHAIRs (sentence-level: image had any hallucination?)")
    print(f"  vanilla  : {van_s.mean()*100:.2f}%")
    print(f"  Causal   : {ste_s.mean()*100:.2f}%")
    pt, med, lo, hi, boot = paired_delta_ci(van_s, ste_s)
    print(f"  Δ (point): {pt*100:+.2f} pp")
    print(f"  Δ (boot) : median {med*100:+.2f} pp  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")
    p = (boot >= 0).mean()
    print(f"  one-sided p(Δ ≥ 0): {p:.4f}")

    print(f"\nCHAIRi (instance-level: fraction of objects hallucinated per image)")
    print(f"  vanilla  : {van_i.mean()*100:.2f}%")
    print(f"  Causal   : {ste_i.mean()*100:.2f}%")
    pt, med, lo, hi, boot = paired_delta_ci(van_i, ste_i)
    print(f"  Δ (point): {pt*100:+.2f} pp")
    print(f"  Δ (boot) : median {med*100:+.2f} pp  95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")
    p = (boot >= 0).mean()
    print(f"  one-sided p(Δ ≥ 0): {p:.4f}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_results", type=str, required=True,
                   help="Path to vanilla CHAIR result JSON")
    p.add_argument("--chall_results", type=str, required=True,
                   help="Path to CHALL (ours) CHAIR result JSON")
    p.add_argument("--label", type=str, default="CHAIR delta (vanilla → CHALL)")
    args = p.parse_args()

    print(f"Paired bootstrap 95% CI (B={B})")
    run(args.label, args.vanilla_results, args.chall_results)

if __name__ == '__main__':
    main()
