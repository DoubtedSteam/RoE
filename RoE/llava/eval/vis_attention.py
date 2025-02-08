import torch
import transformers

from torch import nn
from typing import Optional, Tuple
from  torch.cuda.amp import autocast


def forward_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,
            use_cache=use_cache,
        )
        print(self_attn_weights.shape)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


# def forward_clip(
#         self,
#         hidden_states: torch.Tensor,
#         attention_mask: torch.Tensor,
#         causal_attention_mask: torch.Tensor,
#         output_attentions: Optional[bool] = False,
#     ) -> Tuple[torch.FloatTensor]:
#         """
#         Args:
#             hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
#             attention_mask (`torch.FloatTensor`): attention mask of size
#                 `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
#                 `(config.encoder_attention_heads,)`.
#             output_attentions (`bool`, *optional*):
#                 Whether or not to return the attentions tensors of all attention layers. See `attentions` under
#                 returned tensors for more detail.
#         """

#         residual = hidden_states

#         hidden_states = self.layer_norm1(hidden_states)
#         hidden_states, attn_weights = self.self_attn(
#             hidden_states=self.attn_adapter(hidden_states).type_as(residual),
#             attention_mask=attention_mask,
#             causal_attention_mask=causal_attention_mask,
#             output_attentions=output_attentions,
#         )
#         hidden_states = residual + hidden_states

#         residual = hidden_states
#         hidden_states = self.layer_norm2(hidden_states)
#         hidden_states = self.mlp(hidden_states)
#         hidden_states = residual + hidden_states

#         outputs = (hidden_states,)

#         if output_attentions:
#             outputs += (attn_weights,)

#         return outputs


def set_printMHA(model):
    for _ in model.children():
        if type(_) == transformers.models.llama.modeling_llama.LlamaDecoderLayer:
            print('here')
            bound_method = forward_llama.__get__(_, _.__class__)            
            setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_printMHA(_)


# from clip.model import ResidualAttentionBlock
# def set_Clip_Adapter(model, method, dim=8, s=1, set_forward=True, t=10.):
#     for _ in model.children():
#         if type(_) == ResidualAttentionBlock:
#             _.adapter = MoE_LoRA(1024, hidden_dim=dim, scale=s,  t=t)
#             _.s = s

#             if method == 'router_block':
#                 bound_method = forward_clip_full.__get__(_, _.__class__)
#             else:
#                 bound_method = forward_clip.__get__(_, _.__class__)

#             if set_forward:
#                 setattr(_, 'forward', bound_method)

#         elif len(list(_.children())) != 0:
#             set_Clip_Adapter(_, method, dim, s, set_forward=set_forward, t=t)
