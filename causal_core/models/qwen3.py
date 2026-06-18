import json
import os

import torch
from torch import nn

import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList

def _resolve_qwen_attn_layer(model, layer_idx: int):
    layer_groups = []

    if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "layers"):
        layer_groups.append(model.model.language_model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_groups.append(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layer_groups.append(model.language_model.layers)
    if hasattr(model, "layers"):
        layer_groups.append(model.layers)

    for layers in layer_groups:
        try:
            if 0 <= layer_idx < len(layers):
                return layers[layer_idx].self_attn
        except Exception:
            continue

    for layers in layer_groups:
        try:
            if len(layers) > 0:
                return layers[0].self_attn
        except Exception:
            continue

    raise RuntimeError("Cannot resolve Qwen attention layer for ONLY/v2 controls")

def _sample_only(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = getattr(generation_config, "output_logits", False)
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    use_only = bool(model_kwargs.get("use_only", False))
    ritual_alpha_pos = float(model_kwargs.get("ritual_alpha_pos", 3.0))
    ritual_alpha_neg = float(model_kwargs.get("ritual_alpha_neg", 1.0))
    ritual_beta = float(model_kwargs.get("ritual_beta", 0.1))
    js_gamma = float(model_kwargs.get("js_gamma", 0.1))

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if self._valid_auto_compile_criteria(model_kwargs, generation_config)
        else self.__call__
    )

    if not generation_config.is_assistant:
        outputs = self._prefill(input_ids, generation_config, model_kwargs)
        prefill_consumed = False
    else:
        model_kwargs = self._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
        prefill_consumed = True

    t = 0
    total_overlapping_index_len = []

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            with self._optimize_model_for_decode():
                outputs = model_forward(**model_inputs, return_dict=True)
        prefill_consumed = True

        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        if use_only:
            logits_cd = getattr(self, "_logits_cd", None)
            if logits_cd is not None:
                next_token_logits_cd = logits_cd[:, -1, :].to(dtype=torch.float32, device=input_ids.device)

                probs_main = nn.functional.softmax(next_token_logits, dim=-1)
                probs_cd = nn.functional.softmax(next_token_logits_cd, dim=-1)
                tvd = torch.sum(torch.abs(probs_main - probs_cd), dim=-1).mean()
                total_overlapping_index_len.append(tvd.item())

                beta_safe = max(ritual_beta, 1e-8)
                cutoff = torch.log(torch.tensor(beta_safe, device=next_token_logits.device, dtype=next_token_logits.dtype))
                cutoff = cutoff + next_token_logits.max(dim=-1, keepdim=True).values

                if tvd < js_gamma:
                    diffs = next_token_logits + ritual_alpha_pos * next_token_logits_cd
                else:
                    diffs = (1.0 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_cd

                only_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
                next_token_scores = logits_processor(input_ids, only_logits)
                t += 1
            else:
                next_token_scores = logits_processor(input_ids, next_token_logits)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                )

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum(dim=-1).min() < 1e-10:
                probs = torch.ones_like(probs) / probs.shape[-1]
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del outputs

    if streamer is not None:
        streamer.end()

    tvd_path = os.environ.get("TVD_LOG_PATH", "")
    if tvd_path and total_overlapping_index_len:
        with open(tvd_path, "w") as f:
            json.dump(total_overlapping_index_len, f)

    if return_dict_in_generate:
        cache = None
        cache_keys = getattr(transformers.generation.utils, "ALL_CACHE_NAMES", [])
        if any(cache_key in model_kwargs for cache_key in cache_keys):
            cache_key = next(cache_key for cache_key in cache_keys if cache_key in model_kwargs)
            cache = model_kwargs[cache_key]

        if self.config.is_encoder_decoder:
            return transformers.generation.utils.GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=cache,
            )

        return transformers.generation.utils.GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=cache,
        )

    return input_ids

_ALLOWED_EXTRA_KWARGS = {
    "use_only",
    "enhance_layer_index",
    "ritual_alpha_pos",
    "ritual_alpha_neg",
    "ritual_beta",
    "js_gamma",
}

_orig_validate_model_kwargs = None


def _patched_validate_model_kwargs(self, model_kwargs):
    filtered = {k: v for k, v in model_kwargs.items() if k not in _ALLOWED_EXTRA_KWARGS}
    return _orig_validate_model_kwargs(self, filtered)


def evolve_only_sampling_qwen3():
    global _orig_validate_model_kwargs

    gm = transformers.generation.utils.GenerationMixin
    gm._sample = _sample_only

    if _orig_validate_model_kwargs is None:
        _orig_validate_model_kwargs = gm._validate_model_kwargs
    gm._validate_model_kwargs = _patched_validate_model_kwargs
