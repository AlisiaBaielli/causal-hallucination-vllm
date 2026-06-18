"""
Run ONLY with offline EIC head selection (ONLY+EIC ablation).

Forwards to the standard LLaVA benchmark scripts with --use_only --use_eic_heads.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

BENCHMARKS = {
    "chair": REPO / "experiments" / "chair" / "llava.py",
    "pope": REPO / "experiments" / "pope" / "llava.py",
    "amber": REPO / "experiments" / "amber" / "llava.py",
    "mme": REPO / "experiments" / "mme" / "llava.py",
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BENCHMARKS:
        print(
            "Usage: python -m experiments.ablations.only_eic_runner "
            "{chair|pope|amber|mme} <benchmark-args>",
            file=sys.stderr,
        )
        sys.exit(2)

    bench = sys.argv[1]
    script = str(BENCHMARKS[bench])
    forwarded = ["--use_only", "--use_eic_heads", "--no_hook"] + sys.argv[2:]
    if "--method_name" not in forwarded:
        forwarded += ["--method_name", "only_eic"]

    sys.argv = [script] + forwarded
    runpy.run_path(script, run_name="__main__")

if __name__ == "__main__":
    main()
