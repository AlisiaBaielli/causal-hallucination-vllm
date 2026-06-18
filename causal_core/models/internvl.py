from __future__ import annotations
"""ONLY-baseline sampling patch entry point for InternVL3.5-8B-HF.
"""

import transformers

from causal_core.models.qwen3 import _sample_only

_ALLOWED_EXTRA_KWARGS = {
    "use_only",
    "enhance_layer_index",
    "ritual_alpha_pos",
    "ritual_alpha_neg",
    "ritual_beta",
    "js_gamma",
}

def _patched_validate_model_kwargs(self, model_kwargs):
    """Strip known ONLY-baseline kwargs before validation, then delegate."""
    filtered = {k: v for k, v in model_kwargs.items() if k not in _ALLOWED_EXTRA_KWARGS}
    return _orig_validate_model_kwargs(self, filtered)

_orig_validate_model_kwargs = None

def evolve_only_sampling_internvl():
    """Install the ONLY-baseline _sample override + relaxed kwargs validation.
    """
    global _orig_validate_model_kwargs

    GM = transformers.generation.utils.GenerationMixin
    GM._sample = _sample_only

    if _orig_validate_model_kwargs is None:
        _orig_validate_model_kwargs = GM._validate_model_kwargs
    GM._validate_model_kwargs = _patched_validate_model_kwargs

from typing import Dict, Tuple

import torch
from PIL import Image

def build_inputs(bundle, image: Image.Image, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Build InternVL-HF chat-template inputs for calibration.
.
    """
    processor = bundle["processor"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        crop_to_patches=False,
    )

    inputs = inputs.to(device)
    return dict(inputs)

def get_kv_masks(model, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (vision_mask, text_mask) over KV positions.
    """
    if "input_ids" not in inputs:
        raise ValueError("InternVL adapter expected input_ids in inputs")

    input_ids = inputs["input_ids"]
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Model config is missing image_token_id")

    vision_mask = input_ids.eq(int(image_token_id))
    text_mask = ~vision_mask

    attn_mask = inputs.get("attention_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask.bool()
        vision_mask = vision_mask & attn_mask
        text_mask = text_mask & attn_mask

    if vision_mask.sum().item() == 0:
        raise ValueError("No visual tokens detected in InternVL inputs; cannot compute TVER")
    if text_mask.sum().item() == 0:
        raise ValueError("No text tokens detected in InternVL inputs; cannot compute TVER")

    return vision_mask, text_mask
