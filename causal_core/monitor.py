"""
causal_core/monitor.py
"""
import os
import re
import logging

import torch
from causal_core.grounding import compute_grounding_score_batched
from causal_core.sharpening import sharpen_logits

log = logging.getLogger(__name__)

class CausalMonitor:
    """
    Hooks into a target LLaMA layer's attention to monitor high-C heads'
    attention entropy over image tokens during decode.

    At each decode step, computes:
      - per-head attention entropy over image tokens (for high-C heads)
      - C-weighted mean entropy across high-C heads
      - a "grounding score" = 1 - normalized_entropy (0=diffuse, 1=focused)

    The grounding score is consumed by :class:`CausalLogitsProcessor` to
    modulate output logits.
    """

    def __init__(self, model, layer_idx, c_scores, img_start=35, img_len=576):
        self.layer = model.model.layers[layer_idx]
        attn = self.layer.self_attn
        cfg = getattr(attn, "config", getattr(model, "config", None))
        self.num_heads = getattr(attn, "num_heads", None) or cfg.num_attention_heads
        self.head_dim = getattr(attn, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )

        C = c_scores.float()
        self.high_c_mask = (C > 0)
        self.high_c_indices = torch.where(self.high_c_mask)[0]
        self.c_weights = C[self.high_c_mask]
        self.c_weights = self.c_weights / (self.c_weights.sum() + 1e-8)

        self.img_start = img_start
        self.img_len = img_len

        self.grounding_score = 1.0
        self.mean_entropy = 0.0
        self._handle = None

        log.info(f"[Causal] layer={layer_idx} monitoring {len(self.high_c_indices)}/{self.num_heads} heads, "
                 f"img_tokens=[{img_start}:{img_start+img_len}]")

    def install_qk_hook(self):
        """
        Wrap the LlamaAttention forward to force ``output_attentions=True``
        for this layer only, then read the attention weights to compute
        per-head entropy over image tokens during decode (Q=1).

        Returns the original forward so the caller can restore it later.
        """
        layer = self.layer
        monitor = self

        original_forward = layer.self_attn.forward

        def hooked_forward(*args, **kwargs):

            kwargs_copy = dict(kwargs)
            kwargs_copy["output_attentions"] = True
            result = original_forward(*args, **kwargs_copy)

            if len(result) >= 2 and result[1] is not None:
                attn_weights = result[1]
                B, H, Q, KV = attn_weights.shape

                if Q == 1 and KV > monitor.img_start + monitor.img_len:

                    img_end = monitor.img_start + monitor.img_len

                    img_attn = attn_weights[:, monitor.high_c_indices.to(attn_weights.device), :,
                                            monitor.img_start:img_end]

                    img_attn = img_attn.squeeze(2)

                    gs, mean_norm_entropy = compute_grounding_score_batched(
                        img_attn, monitor.c_weights
                    )
                    monitor.grounding_score = gs
                    monitor.mean_entropy = mean_norm_entropy

            return result

        layer.self_attn.forward = hooked_forward
        return original_forward

    def restore(self, original_forward):
        self.layer.self_attn.forward = original_forward

class CausalMonitorQwen3:
    """
    Hooks into Qwen3VLTextAttention.forward at ``layer_idx`` and, during
    decode (Q == 1), manually computes Q * K attention weights for the
    high-C heads over image-token positions, derives a mean normalised
    entropy, and exposes ``grounding_score`` (1 = focused, 0 = diffuse).

    Because Qwen3 uses GQA (32 heads / 8 KV-heads) and may use Flash
    Attention internally, we wrap ``forward()`` to run the ORIGINAL kernel
    first (so the KV cache is populated) and then read the cached keys to
    compute a lightweight entropy from raw Q*K scores.
    """

    def __init__(self, model, layer_idx, c_scores, image_token_id):

        if hasattr(model.model, "language_model"):
            self.attn = model.model.language_model.layers[layer_idx].self_attn
        elif hasattr(model, "language_model"):
            self.attn = model.language_model.layers[layer_idx].self_attn
        else:
            raise ValueError("Cannot resolve Qwen3 attention layer")

        self.num_heads = self.attn.config.num_attention_heads
        self.num_kv_heads = self.attn.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.attn.head_dim
        self.scaling = self.attn.scaling

        C = c_scores.float()
        self.high_c_mask = C > 0
        self.high_c_indices = torch.where(self.high_c_mask)[0]
        self.c_weights = C[self.high_c_mask]
        self.c_weights = self.c_weights / (self.c_weights.sum() + 1e-8)

        self.image_token_id = image_token_id
        self.img_positions = None

        self.grounding_score = 1.0
        self.mean_entropy = 0.0

        self._orig_forward = None

        log.info(
            f"[Causal-Qwen3] layer={layer_idx}  monitoring "
            f"{len(self.high_c_indices)}/{self.num_heads} heads  "
            f"(GQA groups={self.num_kv_groups})"
        )

    def install_hook(self):
        """Monkey-patch self_attn.forward to compute entropy via manual Q*K.

        Runs the original forward first (so the KV cache is populated),
        then reads the cached keys to compute attention entropy over
        image tokens for high-C heads.
        """
        monitor = self
        original_forward = self.attn.forward
        self._orig_forward = original_forward

        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        from transformers.models.qwen3.modeling_qwen3 import repeat_kv

        def hooked_forward(hidden_states, position_embeddings,
                           attention_mask=None, past_key_values=None, **kwargs):

            result = original_forward(hidden_states, position_embeddings,
                                      attention_mask=attention_mask,
                                      past_key_values=past_key_values, **kwargs)

            B, S_q = hidden_states.shape[:2]
            is_decode = S_q == 1

            if is_decode and monitor.img_positions is not None and len(monitor.img_positions) > 0                    and len(monitor.high_c_indices) > 0:

                q_shape = (B, S_q, monitor.num_heads, monitor.head_dim)
                q = monitor.attn.q_norm(
                    monitor.attn.q_proj(hidden_states).view(q_shape)
                ).transpose(1, 2)

                cos, sin = position_embeddings

                k_dummy = monitor.attn.k_norm(
                    monitor.attn.k_proj(hidden_states).view(B, S_q, monitor.num_kv_heads, monitor.head_dim)
                ).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, k_dummy, cos, sin)

                full_k = None
                if past_key_values is not None:
                    layer_idx = monitor.attn.layer_idx

                    if hasattr(past_key_values, "layers") and layer_idx < len(past_key_values.layers):
                        lc = past_key_values.layers[layer_idx]
                        if hasattr(lc, "keys") and lc.keys is not None:
                            full_k = lc.keys

                    elif hasattr(past_key_values, "key_cache") and layer_idx < len(past_key_values.key_cache):
                        full_k = past_key_values.key_cache[layer_idx]

                if full_k is not None:
                    KV_len = full_k.shape[2]
                    img_pos = monitor.img_positions
                    device = q.device

                    if img_pos is not None and len(img_pos) > 0 and img_pos[-1] < KV_len:

                        full_k_exp = repeat_kv(full_k, monitor.num_kv_groups)

                        scores = torch.matmul(q, full_k_exp.transpose(2, 3)) * monitor.scaling

                        if attention_mask is not None and attention_mask.shape[-1] >= KV_len:
                            scores = scores + attention_mask[:, :, :, :KV_len]

                        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32)

                        high_idx = monitor.high_c_indices.to(device)
                        ip = img_pos.to(device)

                        img_attn = attn_w[:, high_idx, 0, :][:, :, ip]

                        img_attn_sum = img_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        img_attn_norm = img_attn / img_attn_sum

                        entropy = -(img_attn_norm * (img_attn_norm + 1e-10).log()).sum(dim=-1)
                        max_entropy = torch.log(torch.tensor(float(len(ip)), device=device))
                        norm_entropy = entropy / max_entropy.clamp(min=1e-8)

                        c_w = monitor.c_weights.to(device)
                        mean_norm_entropy = (norm_entropy * c_w.unsqueeze(0)).sum(dim=-1).mean().item()

                        monitor.grounding_score = 1.0 - mean_norm_entropy
                        monitor.mean_entropy = mean_norm_entropy

            return result

        self.attn.forward = hooked_forward

    def restore(self):
        if self._orig_forward is not None:
            self.attn.forward = self._orig_forward

    def set_img_positions(self, input_ids, image_token_id):
        """Call before generate to detect image token positions in the input."""
        ids = input_ids[0] if input_ids.dim() > 1 else input_ids
        mask = ids == image_token_id
        if mask.any():
            self.img_positions = mask.nonzero(as_tuple=True)[0]
            log.info(
                f"[Causal-Qwen3] Detected {len(self.img_positions)} image tokens "
                f"at positions [{self.img_positions[0].item()}..{self.img_positions[-1].item()}]"
            )
        else:
            self.img_positions = None
            log.warning("[Causal-Qwen3] No image tokens found in input_ids!")

class CausalMonitorInternVL:
    """
    Hooks ``Qwen3Attention.forward`` at ``layer_idx`` of the InternVL language
    model and, during decode (Q == 1), manually computes Q*K attention
    weights for the high-C heads over image-token positions, derives a
    mean normalised entropy, and exposes ``grounding_score``
    (1 = focused, 0 = diffuse).

    GQA (32 query / 8 KV heads) is handled by expanding KV with
    ``repeat_kv()``. The cache is read AFTER the original forward updates
    it, via ``past_key_values.layers[idx].keys`` (DynamicCache modern API)
    with a fallback to the older ``.key_cache[idx]`` API.
    """

    def __init__(self, model, layer_idx, c_scores, image_token_id):

        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            self.attn = model.model.language_model.layers[layer_idx].self_attn
        elif hasattr(model, "language_model"):
            self.attn = model.language_model.layers[layer_idx].self_attn
        else:
            raise ValueError("Cannot resolve InternVL attention layer")

        self.num_heads = self.attn.config.num_attention_heads
        self.num_kv_heads = self.attn.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.attn.head_dim
        self.scaling = self.attn.scaling

        C = c_scores.float()
        self.high_c_mask = C > 0
        self.high_c_indices = torch.where(self.high_c_mask)[0]
        self.c_weights = C[self.high_c_mask]
        self.c_weights = self.c_weights / (self.c_weights.sum() + 1e-8)

        self.image_token_id = image_token_id
        self.img_positions = None

        self.grounding_score = 1.0
        self.mean_entropy = 0.0

        self._orig_forward = None

        log.info(
            f"[Causal-InternVL] layer={layer_idx}  monitoring "
            f"{len(self.high_c_indices)}/{self.num_heads} heads  "
            f"(GQA groups={self.num_kv_groups})"
        )

    def install_hook(self):
        """Monkey-patch self_attn.forward to compute entropy via manual Q*K."""
        monitor = self
        original_forward = self.attn.forward
        self._orig_forward = original_forward

        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

        def hooked_forward(hidden_states, position_embeddings,
                           attention_mask=None, past_key_values=None, **kwargs):

            result = original_forward(hidden_states, position_embeddings,
                                      attention_mask=attention_mask,
                                      past_key_values=past_key_values, **kwargs)

            B, S_q = hidden_states.shape[:2]
            is_decode = S_q == 1

            if is_decode and monitor.img_positions is not None and len(monitor.img_positions) > 0                    and len(monitor.high_c_indices) > 0:

                q_shape = (B, S_q, monitor.num_heads, monitor.head_dim)
                q = monitor.attn.q_norm(
                    monitor.attn.q_proj(hidden_states).view(q_shape)
                ).transpose(1, 2)

                cos, sin = position_embeddings

                k_dummy = monitor.attn.k_norm(
                    monitor.attn.k_proj(hidden_states).view(B, S_q, monitor.num_kv_heads, monitor.head_dim)
                ).transpose(1, 2)
                q, _ = apply_rotary_pos_emb(q, k_dummy, cos, sin)

                full_k = None
                if past_key_values is not None:
                    layer_idx = monitor.attn.layer_idx
                    if hasattr(past_key_values, "layers") and layer_idx < len(past_key_values.layers):
                        lc = past_key_values.layers[layer_idx]
                        if hasattr(lc, "keys") and lc.keys is not None:
                            full_k = lc.keys
                    elif hasattr(past_key_values, "key_cache") and layer_idx < len(past_key_values.key_cache):
                        full_k = past_key_values.key_cache[layer_idx]

                if full_k is not None:
                    KV_len = full_k.shape[2]
                    img_pos = monitor.img_positions
                    device = q.device

                    if img_pos is not None and len(img_pos) > 0 and img_pos[-1] < KV_len:

                        full_k_exp = repeat_kv(full_k, monitor.num_kv_groups)

                        scores = torch.matmul(q, full_k_exp.transpose(2, 3)) * monitor.scaling

                        if attention_mask is not None and attention_mask.shape[-1] >= KV_len:
                            scores = scores + attention_mask[:, :, :, :KV_len]

                        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32)

                        high_idx = monitor.high_c_indices.to(device)
                        ip = img_pos.to(device)

                        img_attn = attn_w[:, high_idx, 0, :][:, :, ip]

                        img_attn_sum = img_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                        img_attn_norm = img_attn / img_attn_sum

                        entropy = -(img_attn_norm * (img_attn_norm + 1e-10).log()).sum(dim=-1)
                        max_entropy = torch.log(torch.tensor(float(len(ip)), device=device))
                        norm_entropy = entropy / max_entropy.clamp(min=1e-8)

                        c_w = monitor.c_weights.to(device)
                        mean_norm_entropy = (norm_entropy * c_w.unsqueeze(0)).sum(dim=-1).mean().item()

                        monitor.grounding_score = 1.0 - mean_norm_entropy
                        monitor.mean_entropy = mean_norm_entropy

            return result

        self.attn.forward = hooked_forward

    def restore(self):
        if self._orig_forward is not None:
            self.attn.forward = self._orig_forward

    def set_img_positions(self, input_ids, image_token_id):
        """Call before generate to detect image token positions in the input."""
        ids = input_ids[0] if input_ids.dim() > 1 else input_ids
        mask = ids == image_token_id
        if mask.any():
            self.img_positions = mask.nonzero(as_tuple=True)[0]
            log.info(
                f"[Causal-InternVL] Detected {len(self.img_positions)} image tokens "
                f"at positions [{self.img_positions[0].item()}..{self.img_positions[-1].item()}]"
            )
        else:
            self.img_positions = None
            log.warning("[Causal-InternVL] No image tokens found in input_ids!")

class CausalLogitsProcessor:
    """
    Logits processor that uses a Causal monitor's grounding score to
    modulate logits at each decode step.

    When grounding is low (high entropy in visual heads):
      - Sharpen the logit distribution (reduce effective temperature)
      - This forces the model to commit to high-confidence tokens,
        reducing hallucinated low-confidence completions.

    When grounding is high (low entropy):
      - Leave logits unchanged (model is visually grounded).

    The sharpening strength is:
        effective_temp = 1 - alpha * (1 - grounding_score)
    So at grounding=1: temp=1 (unchanged), at grounding=0: temp=1-alpha (sharper).
    A 0.3 floor is enforced to avoid over-sharpening.
    """

    def __init__(self, monitor, alpha=0.5):
        self.monitor = monitor
        self.alpha = alpha

    def __call__(self, input_ids, scores):
        gs = self.monitor.grounding_score if self.monitor is not None else 1.0
        return sharpen_logits(scores, gs, self.alpha, tau_floor=0.3)

def parse_image_id(filename: str) -> int:
    """Parse a numeric image id from a COCO-style filename, e.g.
    ``COCO_val2014_000000123456.jpg`` -> ``123456``.
    """
    m = re.search(r"(\d+)\.jpg$", filename)
    if m:
        return int(m.group(1))
    stem = os.path.splitext(filename)[0]
    m2 = re.search(r"(\d+)$", stem)
    if m2:
        return int(m2.group(1))
    raise ValueError(f"Cannot parse image id from filename: {filename}")
