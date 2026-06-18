import os
import sys
from typing import Dict, Tuple

import torch
from PIL import Image

def _ensure_llava_on_path():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

def build_inputs(bundle, image: Image.Image, text: str, device: torch.device) -> Dict[str, torch.Tensor]:

    _ensure_llava_on_path()
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    tokenizer = bundle["tokenizer"]
    image_processor = bundle["image_processor"]
    conv_mode = bundle.get("conv_mode", "llava_v1")
    model = bundle["model"]

    qs = text
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    model_dtype = next(model.parameters()).dtype
    image_tensor = image_tensor.to(dtype=model_dtype)

    return {
        "input_ids": input_ids.to(device),
        "images": image_tensor.unsqueeze(0).to(device),
    }

def get_kv_masks(model, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (vision_mask, text_mask) over KV positions for the current prompt forward.

    """
    _ensure_llava_on_path()
    from llava.constants import IMAGE_TOKEN_INDEX

    input_ids = inputs["input_ids"]
    images = inputs.get("images", None)
    B, S = input_ids.shape
    device = input_ids.device

    if images is None or (input_ids == IMAGE_TOKEN_INDEX).sum().item() == 0:
        vision_mask = torch.zeros((B, S), dtype=torch.bool, device=device)
        text_mask = ~vision_mask
        return vision_mask, text_mask

    if not hasattr(model, "encode_images"):
        raise RuntimeError("Expected LLaVA model to expose encode_images(images).")
    with torch.no_grad():
        images = images.to(dtype=next(model.parameters()).dtype)
        feats = model.encode_images(images)
    P = int(feats.shape[1])
    if P <= 0:
        raise RuntimeError(f"Unexpected image feature length P={P}.")

    vision_masks = []
    text_masks = []
    for b in range(B):
        seq_vis = []
        seq_txt = []
        for tid in input_ids[b].tolist():
            if tid == IMAGE_TOKEN_INDEX:
                seq_vis.extend([True] * P)
                seq_txt.extend([False] * P)
            else:
                seq_vis.append(False)
                seq_txt.append(True)
        vision_masks.append(torch.tensor(seq_vis, dtype=torch.bool, device=device))
        text_masks.append(torch.tensor(seq_txt, dtype=torch.bool, device=device))

    max_len = max(m.numel() for m in vision_masks)
    def _pad(mask_list, value: bool):
        out = []
        for m in mask_list:
            if m.numel() < max_len:
                pad = torch.full((max_len - m.numel(),), value, dtype=torch.bool, device=device)
                out.append(torch.cat([m, pad], dim=0))
            else:
                out.append(m)
        return torch.stack(out, dim=0)

    vision_mask = _pad(vision_masks, False)
    text_mask = _pad(text_masks, True)
    return vision_mask, text_mask
