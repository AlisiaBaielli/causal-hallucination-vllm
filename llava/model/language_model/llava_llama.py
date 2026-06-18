#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append(".") 
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from causal_core.transformers_fork import ensure_llama_fork
ensure_llama_fork()

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..utils import cache_has_contents


def _last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)
from transformers import AutoProcessor, AutoModelForCausalLM


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_pos: Optional[torch.FloatTensor] = None,
        images_neg: Optional[torch.FloatTensor] = None,
        use_ritual: Optional[bool] = None,
        use_vcd: Optional[bool] = None,
        use_m3id: Optional[bool] = None,
        use_only: Optional[bool] = None,
        enhance_layer_index: Optional[int] = 0,
        ritual_alpha_pos: Optional[torch.FloatTensor] = None,
        ritual_alpha_neg: Optional[torch.FloatTensor] = None,
        ritual_beta: Optional[torch.FloatTensor] = None,
        js_gamma: Optional[torch.FloatTensor] = None, 
        return_dict: Optional[bool] = None,
        tokenizer=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        logits_cd = None
        if not use_only:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                use_only=use_only,
                enhance_layer_index=enhance_layer_index,
            )
            hidden_states_cd = getattr(self.model, "_hidden_states_cd", None)
            if hidden_states_cd is not None:
                hidden_states_cd = hidden_states_cd + 0.5 * _last_hidden_state(outputs)
                logits_cd = self.lm_head(hidden_states_cd)

        hidden_states = _last_hidden_state(outputs)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            tail = outputs[1:] if isinstance(outputs, tuple) else ()
            output = (logits,) + tail
            return (loss,) + output if loss is not None else output

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
        if use_only:
            return out, logits_cd
        return out
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if cache_has_contents(past_key_values):
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and not cache_has_contents(past_key_values):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "use_only": kwargs.get("use_only", None),
                "enhance_layer_index": kwargs.get("enhance_layer_index", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_pos(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if cache_has_contents(past_key_values):
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and not cache_has_contents(past_key_values):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_pos", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_neg(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if cache_has_contents(past_key_values):
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and not cache_has_contents(past_key_values):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_neg", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_m3id(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if cache_has_contents(past_key_values):
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and not cache_has_contents(past_key_values):
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids[input_ids != -200].reshape(input_ids.shape[0], -1)}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask[:, :-1],
                "images": None,
            }
        )
        return model_inputs
    
    
try:
    AutoConfig.register("llava", LlavaConfig)
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except ValueError:
    pass
