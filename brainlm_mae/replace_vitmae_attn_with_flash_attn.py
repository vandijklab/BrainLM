import torch
import transformers
from typing import Optional, Set, Tuple, Union
from flash_attn.flash_attn_interface import flash_attn_func

def forward(
    self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    sm_scale = self.attention_head_size**(-0.5)

    q = self.query(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
    k = self.key(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
    v = self.value(hidden_states).reshape(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)

    context_layer = flash_attn_func(q, k, v, dropout_p=self.dropout.p, softmax_scale=sm_scale, causal=False)
    context_layer = context_layer.reshape(batch_size, seq_len, -1)

    return (context_layer,)


def replace_vitmae_attn_with_flash_attn():
    transformers.models.vit_mae.modeling_vit_mae.ViTMAESelfAttention.forward = forward
