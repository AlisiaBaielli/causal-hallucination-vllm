from __future__ import annotations

from typing import Dict, Tuple

import torch
from PIL import Image

def build_inputs(bundle, image: Image.Image, text: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Build Qwen3-VL chat-template inputs for calibration."""
    processor = bundle["processor"]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image.convert("RGB")},
                {"type": "text", "text": "Answer yes or no only. " + text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)
    return dict(inputs)

def get_kv_masks(model, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (vision_mask, text_mask) over KV positions.
    """
    if "input_ids" not in inputs:
        raise ValueError("Qwen adapter expected input_ids in inputs")

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
        raise ValueError("No visual tokens detected in Qwen inputs; cannot compute TVER")
    if text_mask.sum().item() == 0:
        raise ValueError("No text tokens detected in Qwen inputs; cannot compute TVER")

    return vision_mask, text_mask
