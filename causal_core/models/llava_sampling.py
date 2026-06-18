"""Patched sample() for LLaVA generation supporting ONLY, VCD, M3ID, and RITUAL.

Adapted from Wan et al.'s ONLY codebase (https://github.com/zifuwan/ONLY).
VCD: Leng et al., CVPR 2024 (https://github.com/DAMO-NLP-SG/VCD).
M3ID: Favero et al., CVPR 2024 (https://arxiv.org/abs/2403.14003).
"""
import warnings
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers

try:
    from transformers.generation.utils import SampleEncoderDecoderOutput, SampleOutput
except ImportError:
    from transformers.generation.utils import (
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
    )

    SampleOutput = GenerateDecoderOnlyOutput
    SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False

    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()

    t = 0
    total_overlapping_index_len = []
    while True:
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_only = model_kwargs.get("use_only")

        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if use_only:
            outputs, logits_cd = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            outputs, _ = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        if use_ritual or use_vcd or use_m3id or use_only:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs.get("images_pos") is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos, _ = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs.get("images_neg") is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg, _ = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg, _ = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]

            ritual_alpha_pos = model_kwargs.get("ritual_alpha_pos") if model_kwargs.get("ritual_alpha_pos") is not None else 3
            ritual_alpha_neg = model_kwargs.get("ritual_alpha_neg") if model_kwargs.get("ritual_alpha_neg") is not None else 1
            ritual_beta = model_kwargs.get("ritual_beta") if model_kwargs.get("ritual_beta") is not None else 0.1
            js_gamma = model_kwargs.get("js_gamma") if model_kwargs.get("js_gamma") is not None else 0.1

            cutoff = torch.log(torch.tensor(ritual_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

            if use_ritual:
                diffs = next_token_logits + ritual_alpha_pos * next_token_logits_pos
            elif use_vcd:
                diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02 * t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg) * (1 - gamma_t) / gamma_t
                t += 1
            elif use_only:
                assert logits_cd is not None
                next_token_logits_cd = logits_cd[:, -1, :]
                tvd = torch.sum(torch.abs(
                    nn.functional.softmax(next_token_logits, dim=-1) -
                    nn.functional.softmax(next_token_logits_cd, dim=-1)
                ))
                total_overlapping_index_len.append(tvd.item())

                if tvd < js_gamma:
                    diffs = next_token_logits + ritual_alpha_pos * next_token_logits_cd
                else:
                    diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_cd

            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            logits = logits_processor(input_ids, logits)
            logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids, decoder_attentions
    else:
        return input_ids, total_overlapping_index_len

def _llava_forward(self, model_inputs, use_only, output_attentions, output_hidden_states):
    """Run LLaVA forward; ONLY mode returns (output, logits_cd)."""
    if use_only:
        return self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
    out = self(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    if isinstance(out, tuple):
        return out[0], None
    return out, None


def _sample_llava(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config,
    synced_gpus: bool = False,
    streamer=None,
    **model_kwargs,
):
    """Transformers >=5 _sample hook with ONLY / VCD / M3ID / RITUAL support."""
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    return_dict_in_generate = generation_config.return_dict_in_generate
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    this_peer_finished = False
    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()
    t = 0
    total_overlapping_index_len = []

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_only = model_kwargs.get("use_only")

        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs, logits_cd = _llava_forward(
            self, model_inputs, use_only, output_attentions, output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device,
        )

        if use_ritual or use_vcd or use_m3id or use_only:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs.get("images_pos") is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos, _ = _llava_forward(
                    self, model_inputs_pos, False, output_attentions, output_hidden_states,
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
            elif model_kwargs.get("images_neg") is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg, _ = _llava_forward(
                    self, model_inputs_neg, False, output_attentions, output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg, _ = _llava_forward(
                    self, model_inputs_neg, False, output_attentions, output_hidden_states,
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )

            ritual_alpha_pos = model_kwargs.get("ritual_alpha_pos", 3)
            ritual_alpha_neg = model_kwargs.get("ritual_alpha_neg", 1)
            ritual_beta = model_kwargs.get("ritual_beta", 0.1)
            js_gamma = model_kwargs.get("js_gamma", 0.1)

            beta_safe = max(float(ritual_beta), 1e-8)
            cutoff = torch.log(torch.tensor(beta_safe, device=next_token_logits.device, dtype=next_token_logits.dtype))
            cutoff = cutoff + next_token_logits.max(dim=-1, keepdim=True).values

            if use_ritual:
                diffs = next_token_logits + ritual_alpha_pos * next_token_logits_pos
            elif use_vcd:
                diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02 * t, device=next_token_logits.device))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg) * (1 - gamma_t) / gamma_t
                t += 1
            elif use_only:
                assert logits_cd is not None
                next_token_logits_cd = logits_cd[:, -1, :].to(
                    dtype=torch.float32, device=input_ids.device,
                )
                tvd = torch.sum(torch.abs(
                    nn.functional.softmax(next_token_logits, dim=-1) -
                    nn.functional.softmax(next_token_logits_cd, dim=-1)
                ))
                total_overlapping_index_len.append(tvd.item())
                if tvd < js_gamma:
                    diffs = next_token_logits + ritual_alpha_pos * next_token_logits_cd
                else:
                    diffs = (1 + ritual_alpha_neg) * next_token_logits - ritual_alpha_neg * next_token_logits_cd

            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            next_token_scores = logits_processor(input_ids, logits)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)

        if return_dict_in_generate and output_scores:
            scores += (next_token_scores,)

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder,
            )
        if use_vcd or use_m3id:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder,
            )

        this_peer_finished = stopping_criteria(input_ids, scores)

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        return transformers.generation.utils.GenerateDecoderOnlyOutput(
            sequences=input_ids,
            scores=scores,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
        )
    return input_ids


_LAVA_EXTRA_KWARGS = {
    "images", "images_pos", "images_neg",
    "use_ritual", "use_vcd", "use_m3id", "use_only",
    "enhance_layer_index",
    "ritual_alpha_pos", "ritual_alpha_neg", "ritual_beta", "js_gamma",
}
_orig_llava_validate_model_kwargs = None


def _patched_llava_validate_model_kwargs(self, model_kwargs):
    filtered = {k: v for k, v in model_kwargs.items() if k not in _LAVA_EXTRA_KWARGS}
    return _orig_llava_validate_model_kwargs(self, filtered)


def evolve_only_sampling():
    gm = transformers.generation.utils.GenerationMixin
    gm._sample = _sample_llava
    if hasattr(gm, "sample"):
        gm.sample = sample

    global _orig_llava_validate_model_kwargs
    if _orig_llava_validate_model_kwargs is None:
        _orig_llava_validate_model_kwargs = gm._validate_model_kwargs
    gm._validate_model_kwargs = _patched_llava_validate_model_kwargs
