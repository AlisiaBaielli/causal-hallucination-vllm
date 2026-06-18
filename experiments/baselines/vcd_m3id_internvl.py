"""VCD and M3ID baselines for InternVL3.5-8B-HF.
"""

import torch
import torch.nn.functional as F
import logging
from transformers import DynamicCache

log = logging.getLogger(__name__)

def add_diffusion_noise(pixel_values, noise_step=500):
    """Add diffusion noise to pixel tensor (VCD paper recipe)."""
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alpha_bar = alphas_prod[noise_step].to(pixel_values.device, pixel_values.dtype)
    noise = torch.randn_like(pixel_values)
    return torch.sqrt(alpha_bar) * pixel_values + torch.sqrt(1 - alpha_bar) * noise

@torch.inference_mode()
def contrastive_generate(
    model,
    inputs_pos,
    inputs_neg,
    max_new_tokens=128,
    do_sample=True,
    temperature=1.0,
    top_p=1.0,
    cd_alpha=1.0,
    cd_beta=0.1,
    eos_token_id=None,
    pad_token_id=None,
):
    device = inputs_pos["input_ids"].device

    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
        if eos_token_id is None and hasattr(model.config, "text_config"):
            eos_token_id = getattr(model.config.text_config, "eos_token_id", 151643)
        if eos_token_id is None:
            eos_token_id = 151643
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    pos_cache = DynamicCache()
    neg_cache = DynamicCache()

    pos_out = model(**inputs_pos, past_key_values=pos_cache)
    pos_cache = pos_out.past_key_values
    pos_logits = pos_out.logits[:, -1, :].float()

    neg_out = model(**inputs_neg, past_key_values=neg_cache)
    neg_cache = neg_out.past_key_values
    neg_logits = neg_out.logits[:, -1, :].float()

    pos_len = inputs_pos["input_ids"].shape[1]
    neg_len = inputs_neg["input_ids"].shape[1]

    cutoff = pos_logits.max(dim=-1, keepdim=True).values +             torch.log(torch.tensor(cd_beta, device=device, dtype=pos_logits.dtype))
    cd_logits = (1 + cd_alpha) * pos_logits - cd_alpha * neg_logits
    mask = pos_logits >= cutoff
    final_logits = torch.where(mask, cd_logits, pos_logits)

    if do_sample and temperature > 0:
        probs = F.softmax(final_logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            remove = cumsum - sorted_probs > top_p
            sorted_probs[remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
        else:
            next_token = torch.multinomial(probs, 1)
    else:
        next_token = final_logits.argmax(dim=-1, keepdim=True)

    generated_ids = [next_token.squeeze(-1)]

    for step in range(1, max_new_tokens):
        if next_token.item() in eos_token_id:
            break

        pos_cache_pos = torch.tensor([pos_len + step - 1], device=device)
        pos_out = model(
            input_ids=next_token,
            past_key_values=pos_cache,
            cache_position=pos_cache_pos,
        )
        pos_cache = pos_out.past_key_values
        pos_logits = pos_out.logits[:, -1, :].float()

        neg_cache_pos = torch.tensor([neg_len + step - 1], device=device)
        neg_out = model(
            input_ids=next_token,
            past_key_values=neg_cache,
            cache_position=neg_cache_pos,
        )
        neg_cache = neg_out.past_key_values
        neg_logits = neg_out.logits[:, -1, :].float()

        cutoff = pos_logits.max(dim=-1, keepdim=True).values +                 torch.log(torch.tensor(cd_beta, device=device, dtype=pos_logits.dtype))
        cd_logits = (1 + cd_alpha) * pos_logits - cd_alpha * neg_logits
        mask = pos_logits >= cutoff
        final_logits = torch.where(mask, cd_logits, pos_logits)

        if do_sample and temperature > 0:
            probs = F.softmax(final_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = final_logits.argmax(dim=-1, keepdim=True)

        generated_ids.append(next_token.squeeze(-1))

    gen_tensor = torch.stack(generated_ids).squeeze(-1) if generated_ids[0].dim() > 0 else torch.stack(generated_ids)
    all_ids = torch.cat([inputs_pos["input_ids"][0], gen_tensor], dim=0)
    return all_ids.unsqueeze(0)
