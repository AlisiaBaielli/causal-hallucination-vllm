"""Shared core for the EIC-suppression ablation.
"""
import torch

def _resolve_attn(model, layer_index):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers[layer_index].self_attn
    if hasattr(model, "language_model"):
        return model.language_model.layers[layer_index].self_attn
    return model.model.layers[layer_index].self_attn

def install_eic_suppress(model, layer_index, c_scores):
    attn = _resolve_attn(model, layer_index)
    n_heads = int(getattr(attn.config, "num_attention_heads"))
    head_mask = (c_scores.float() > 0).float()
    attn._eic_head_mask = head_mask

    def _pre_hook(module, args, kwargs):
        x = args[0] if args else kwargs.get("input")
        if x is None or x.shape[1] != 1:
            return None
        bsz, q_len, hidden = x.shape
        head_dim = hidden // n_heads
        mask = head_mask.to(device=x.device, dtype=x.dtype)
        x = (x.view(bsz, q_len, n_heads, head_dim) * mask.view(1, 1, -1, 1)).reshape(bsz, q_len, hidden)
        if args:
            return (x,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs); kwargs["input"] = x
        return args, kwargs

    handle = attn.o_proj.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    return handle, head_mask

def restore_eic_suppress(model, layer_index, handle):
    if handle is not None:
        handle.remove()

def load_c_scores(c_scores_path: str, layer_index: int) -> torch.Tensor:
    payload = torch.load(c_scores_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        c = payload.get("C", payload.get("scores"))
        if c is None:
            c = next(iter(payload.values()))
    else:
        c = payload
    if c.dim() == 2:
        c = c[layer_index]
    return c.float()

def load_c_scores(c_scores_path: str, layer_index: int) -> torch.Tensor:
    payload = torch.load(c_scores_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        c = payload.get("C", payload.get("scores"))
        if c is None:
            c = next(iter(payload.values()))
    else:
        c = payload
    if c.dim() == 2:
        c = c[layer_index]
    return c.float()
