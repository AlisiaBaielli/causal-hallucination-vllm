"""Shared helpers for benchmark evaluation scripts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

def import_vcd_baseline(model: str):
    """Import VCD/M3ID helpers for Qwen3-VL or InternVL eval scripts."""
    baselines = str(REPO_ROOT / "experiments" / "baselines")
    if baselines not in sys.path:
        sys.path.insert(0, baselines)
    if model == "qwen3":
        from vcd_m3id_qwen3 import add_diffusion_noise, contrastive_generate
    elif model == "internvl":
        from vcd_m3id_internvl import add_diffusion_noise, contrastive_generate
    else:
        raise ValueError(f"unknown model for VCD import: {model}")
    return contrastive_generate, add_diffusion_noise

def load_c_scores(
    scores_path: str,
    layer_index: Optional[int] = None,
) -> torch.Tensor:
    """Load per-head C/EIC scores from a calibration checkpoint."""
    payload = torch.load(scores_path, map_location="cpu")
    if isinstance(payload, dict):
        c_scores = payload.get("C", payload.get("scores", None))
        if c_scores is None:
            c_scores = next(v for v in payload.values() if torch.is_tensor(v))
        if layer_index is None and "chosen_layer" in payload:
            layer_index = int(payload["chosen_layer"])
    else:
        c_scores = payload

    if not torch.is_tensor(c_scores):
        c_scores = torch.tensor(c_scores)

    if c_scores.dim() == 2:
        if layer_index is None:
            raise ValueError("layer_index required for multi-layer score tensors")
        c_scores = c_scores[layer_index]

    return c_scores.float()

def resolve_method(args) -> Tuple[str, bool]:
    """Return (method_name, needs_c_scores) from CLI flags."""
    if getattr(args, "use_only", False):
        if getattr(args, "use_eic_heads", False):
            return "only_eic", True
        return "only", False
    if getattr(args, "use_vcd", False):
        return "vcd", False
    if getattr(args, "use_m3id", False):
        return "m3id", False
    if getattr(args, "no_hook", False):
        return "vanilla", False
    return "chall", True

def caption_output_path(out_path: str, method: str) -> str:
    """Resolve caption jsonl path; out_path may be a directory or .jsonl file."""
    p = Path(out_path)
    if p.suffix == ".jsonl":
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    p.mkdir(parents=True, exist_ok=True)
    return str(p / f"{method}.jsonl")
