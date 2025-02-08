import torch
import transformers

from torch import nn
from typing import Optional, Tuple
from  torch.cuda.amp import autocast

import jsonlines

import llava

import torch.nn.functional as F


class Skip_Router(nn.Module):
    def __init__(
        self,
        in_features=768,
        out_features=2,
    ):
        super().__init__()

        self.expert = nn.Sequential(
            nn.Linear(in_features * 2, out_features, bias=False)
        )

    def forward(self, x, question_mask):
        image_tokens = x[:, 0:1, :] 
        text_tokens = x[:, 1:question_mask.shape[1], :]
        weight = torch.cat([image_tokens.repeat(1, text_tokens.shape[1], 1), text_tokens], dim=-1)
        weight = self.expert(weight)

        return weight


class Adapter(nn.Module):
    def __init__(
        self,
        in_features=768,
        hidden_dim=8,
        scale=1,
    ):
        super().__init__()
        
        self.conv_A = nn.Linear(in_features, hidden_dim, bias=False)
        self.conv_B = nn.Linear(hidden_dim, in_features, bias=False)
        self.act = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p=0.1)
        
        nn.init.xavier_uniform_(self.conv_A.weight)
        nn.init.xavier_uniform_(self.conv_B.weight)
        
    def forward(self, x):
        x = self.conv_A(x)
        x = self.act(x)
        x = self.conv_B(x)
        
        return x


def forward_llama(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        question_mask: Optional[torch.Tensor] = None,
        weight_mask: Optional[torch.Tensor] = None,
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
        
        if (past_key_value is None):
            self.weight = self.adapter_MHA_router(hidden_states, question_mask) # B x T x 2
            self.weight = self.weight[:, -1, 0] > self.weight[:, -1, 1]
                
        if self.weight.any():
            # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = residual + hidden_states

            # Fully Connected
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        else:
            # Skip Path
            hidden_states = residual + self.adapter_MHA(hidden_states)
            self_attn_weights = torch.empty(0)
            present_key_value = torch.empty(0)
                
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def set_Adapter(model, dim=8, s=1, set_forward=True, t=10, gradient_checkpointing=False):
    for _ in model.children():
        if type(_) == llava.model.language_model.modeling_llama.LlamaDecoderLayer:
            _.adapter_MHA = Adapter(_.hidden_size, 1024).half()
            _.adapter_MHA_router = Skip_Router(_.hidden_size).half()
            _.weight = None
            _.s = s
            _.t = t
            _.times = 0

            bound_method = forward_llama.__get__(_, _.__class__)            
            setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_Adapter(_, dim, s, set_forward=set_forward, t=t, gradient_checkpointing=gradient_checkpointing)

