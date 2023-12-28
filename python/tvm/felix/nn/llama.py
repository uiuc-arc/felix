from typing import List, Optional

import torch
from torch import FloatTensor, Tensor, nn
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    OPTForCausalLM,
    PretrainedConfig,
)
from transformers.generation import LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import CausalLMOutputWithPast

__all__ = ["CustomOPTForCausalLM", "CustomLlamaForCausalLM", "AutoTokenizer"]


def make_custom_forward(inner_model, lm_head, config: PretrainedConfig):
    def forward_(
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else config.output_hidden_states
        )
        # inner_model outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        output, *rest = inner_model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=False,
        )
        logits = lm_head(output).contiguous()
        return logits, *rest

    return forward_


class CustomOPTForCausalLM(OPTForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        output, *rest = self.model.decoder(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=False,
        )
        logits = self.lm_head(output).contiguous()
        return logits, *rest

    def sample(
        self,
        input_ids: torch.Tensor,
        logits_warper: Optional[LogitsProcessorList] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
        assert self.generation_config is not None
        # init values
        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()
        logits_warper = LogitsProcessorList()
        pad_token_id = self.generation_config.pad_token_id
        eos_token_id = self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        )
        # return_dict_in_generate = self.generation_config.return_dict_in_generate

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # forward pass to get next token
            logits, histories = self(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            outputs = CausalLMOutputWithPast(logits=logits, past_key_values=histories)
            next_token_logits = logits[:, -1, :]
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)  # type: ignore
            next_token_scores = logits_warper(input_ids, next_token_scores)
            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, None):  # type: ignore
                break
        return input_ids  # type: ignore


class CustomLlamaForCausalLM(LlamaForCausalLM):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output, *rest = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        logits = self.lm_head(output).contiguous()
        return logits, *rest


def llama():
    network = CustomLlamaForCausalLM(LlamaConfig(attn_implementation="eager"))
    inputs = torch.randint(0, 20000, (1, 100))
    return network, inputs
