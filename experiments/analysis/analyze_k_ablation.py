"""Summarize K-environment ablation CHAIR results into a table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

def load_chair_metrics(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    if "overall_metrics" in data:
        return data["overall_metrics"]
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, required=True,
                   help="Directory containing k_ablation/internvl_K*/chair_results.json")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root = Path(args.results_dir)
    rows = []
    for k in (1, 3, 5, 7):
        metric_path = root / f"internvl_K{k}" / "chair_results.json"
        if not metric_path.exists():
            print(f"[warn] missing {metric_path}")
            continue
        m = load_chair_metrics(metric_path)
        rows.append({
            "K": k,
            "CHAIRs": round(m["CHAIRs"] * 100, 2),
            "CHAIRi": round(m["CHAIRi"] * 100, 2),
            "Recall": round(m["Recall"] * 100, 2),
        })

    print(f"{'K':>3}  {'CHAIRs':>8}  {'CHAIRi':>8}  {'Recall':>8}")
    for r in rows:
        print(f"{r['K']:>3}  {r['CHAIRs']:8.2f}  {r['CHAIRi']:8.2f}  {r['Recall']:8.2f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"[saved] {out}")

if __name__ == "__main__":
    main()
