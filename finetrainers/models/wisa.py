import torch
from diffusers.models.attention_processor import Attention
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import math

class QuantifyPrioriEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        freq_dim: int,
        proj_dim: int,
    ):
        super().__init__()

        self.freq_dim = freq_dim
        self.quantify_proj = nn.Linear(freq_dim, dim)

        if dim > 1024:
            self.linear_adaln = nn.Sequential(
                    nn.Linear(dim, 512),
                    nn.SiLU(),
                    nn.Linear(512, proj_dim)
                )
        else:
            self.linear_adaln = nn.Linear(dim, proj_dim)

    def forward(self, quantify_prioris: torch.Tensor, timestep_proj: torch.Tensor, before: bool = False):
        quant_emb = encode_quantify_prioris(quantify_prioris, embedding_dim=self.freq_dim, flip_sin_to_cos=True)

        quantify_proj_dtype = next(iter(self.quantify_proj.parameters())).dtype

        if quant_emb.dtype != quantify_proj_dtype:
            quant_emb = quant_emb.to(quantify_proj_dtype)
        quant_proj = self.quantify_proj(quant_emb)

        if before:
            quant_timestep_proj = timestep_proj + quant_proj
            quant_timestep_proj = self.linear_adaln(quant_timestep_proj)
        else:
            batch_size, number, _ = timestep_proj.shape
            quant_proj = self.linear_adaln(quant_proj).reshape(batch_size, number, -1)
            quant_timestep_proj = timestep_proj + quant_proj

        return quant_timestep_proj

def encode_quantify_prioris(quantify_prioris: torch.Tensor, embedding_dim=512, max_period=10000, scale=1.0, downscale_freq_shift=0, flip_sin_to_cos=False):
    assert len(quantify_prioris.shape) == 2  # "Input should be (batch, d)"

    batch_size, input_dim = quantify_prioris.shape
    half_dim = embedding_dim // 2  
    per_dim = half_dim // input_dim 

    # calculate exponential decay frequency
    exponent = - math.log(max_period) * torch.arange(
        start=0, end=per_dim, dtype=torch.float32, device=quantify_prioris.device
    )
    exponent = exponent / (per_dim - downscale_freq_shift) 

    emb = torch.exp(exponent)
    emb = quantify_prioris[..., None] * emb[None, :]

    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, :, per_dim:], emb[:, :, :per_dim]], dim=-1)

    emb = emb.view(batch_size, -1)

    # padding
    if embedding_dim % 2 == 1 or emb.shape[-1] < embedding_dim:
        emb = torch.nn.functional.pad(emb, (0, embedding_dim - emb.shape[-1]), value=0)

    return emb


def computing_bce_loss(logits, priori):
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    loss = criterion(logits, priori)
    return loss

class MoHattention_vannile(Attention):
    def __init__(
        self,
        shared_head: int,
        expert_head: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.shared_head = shared_head

        self.expert_head = expert_head

        self.routed_head = 1 # activate flags

        self.head_dim = kwargs['dim_head']
        dim = kwargs['query_dim']
        out_bias = kwargs['out_bias']
        self.qk_norm = kwargs['qk_norm']

        self.total_head = self.shared_head + self.expert_head

        self.to_q = nn.Linear(dim, self.head_dim * self.total_head)
        self.to_k = nn.Linear(dim, self.head_dim * self.total_head)
        self.to_v = nn.Linear(dim, self.head_dim * self.total_head)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.head_dim * self.total_head, self.out_dim, bias=out_bias))
        # self.to_out.append(nn.Dropout(0.0))
    
        if self.expert_head > 0:
            if self.shared_head > 0:
                self.wg_0 = nn.Linear(dim, 2, bias=False)

        if self.shared_head > 1:
            self.wg_1 = nn.Linear(dim, self.shared_head, bias=False)

        self.processor = PhysAttnProcessor_2_0()

class PhysAttnProcessor_2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("PhysAttnProcessor_2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        priori: torch.Tensor,
        rotary_emb: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size, sequence_length, C = hidden_states.shape
        if attn.routed_head > 0:
            _x = hidden_states.reshape(batch_size * sequence_length, C)

            priori = priori.to(hidden_states.dtype)

            if priori.shape[-1] != attn.heads and attn.shared_head == 0:
                if attn.heads % priori.shape[-1] == 0:
                    num_priori = priori.shape[-1]
                    priori = priori.unsqueeze(-1).repeat(1, 1, attn.heads // num_priori)
                    priori = priori.reshape(batch_size, -1)
            
            random_prob = torch.rand_like(priori)
            soft_priori = priori.clone()

            soft_priori[(priori == 0) & (random_prob < 0.2)] = 1.0
            soft_priori[(priori == 0) & (random_prob >= 0.2)] = 0.1

            soft_priori[(priori == 1) & (random_prob < 0.2)] = 0.1
            soft_priori[(priori == 1) & (random_prob >= 0.2)] = 1.0

            soft_priori = soft_priori.unsqueeze(1).repeat(1, sequence_length, 1)
            routed_head_gates = soft_priori


        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.qk_norm == "layer_norm": # for cogvideox
            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

        else: # if attn.qk_norm == "rms_norm_across_heads"  for wanx
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            if isinstance(rotary_emb, tuple):
                from diffusers.models.embeddings import apply_rotary_emb
                query= apply_rotary_emb(query, rotary_emb)
                key= apply_rotary_emb(key, rotary_emb)
            else:
                def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                    x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                    x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                    return x_out.type_as(hidden_states)

                query = apply_rotary_emb(query, rotary_emb)
                key = apply_rotary_emb(key, rotary_emb)


        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        if attn.routed_head > 0:
            x = hidden_states.transpose(1, 2)
            priori = priori.to(hidden_states.dtype)
            if attn.shared_head > 0:
                shared_head_weight = attn.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(batch_size, sequence_length, -1) * attn.shared_head
                
                weight_0 = attn.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(batch_size, sequence_length, 2) * 2
        
                shared_head_gates = torch.einsum("bn,bnh->bnh", weight_0[:,:,0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bnh->bnh", weight_0[:,:,1], routed_head_gates)
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2) # [B, N, h]
            else:
                masked_gates = routed_head_gates

            x = torch.einsum("bnh,bnhd->bnhd", masked_gates, x)
            hidden_states = x.reshape(batch_size, sequence_length, attn.head_dim * attn.heads)
        else:
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        # hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
